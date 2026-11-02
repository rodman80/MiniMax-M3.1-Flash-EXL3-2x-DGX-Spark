#!/usr/bin/env python3
"""Experiment A2: the MiniMax-M3 indexer path -- block scores and block top-k.

The indexer runs once per layer per prefill chunk, in two kernels:

  minimax_m3_index_score   idx_q x index-K -> score [num_idx_heads, total_q, max_block] fp32
  minimax_m3_index_topk    that score matrix -> topk_idx [num_idx_heads, total_q, 16]

The score matrix is the whole cost story: it is materialized in fp32 at
num_idx_heads x total_q x ceil(max_seq_len/128) and read back in full by the top-k
kernel, which runs a bitonic merge over BLOCK_SIZE_K lanes per (query, head) chunk.
This script prices the two halves separately, reports the score matrix's memory traffic
against the node's measured bandwidth, and reports the GPU time of the whole selection
step, which is what the server pays.

Decode (the spec-verify path) is measured too, because at 4 rows the same two kernels
run with split-K over the sequence and their cost profile is completely different.
"""
import os
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/fixtures")

import torch
import vllm.models.minimax_m3.common.ops.index_topk as it
import vllm.models.minimax_m3.common.ops.sparse_attn as sa

from labutil import timeit, write_result, emit, env_record, rel_err
import sparse_selection as sel

DEV = "cuda"
KVH = 2                     # indexer heads per TP=2 rank (sparse_num_index_heads 4)
D = 128
TOPK = 16
SHAPES = [
    ("16k_chunk", [16384], [0]),
    ("16k_at_48k", [16384], [49152]),
    ("16k_at_192k", [16384], [196608]),
    ("8k_at_56k", [8192], [57344]),
]


def bandwidth():
    n = 512 * 1024 * 1024 // 2
    a = torch.empty(n, dtype=torch.float16, device=DEV)
    a.normal_()
    t = timeit(lambda: torch.sum(a), reps=15, warmup=8)
    gb = n * 2 / 1e9
    out = round(gb / (t["median_us"] / 1e6), 1)
    del a
    torch.cuda.empty_cache()
    return out


def bf16_ceiling():
    n = 8192
    a = torch.randn(n, n, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(n, n, dtype=torch.bfloat16, device=DEV)
    t = timeit(lambda: a @ b, reps=15, warmup=5)
    tf = round(2.0 * n ** 3 / (t["median_us"] / 1e6) / 1e12, 1)
    del a, b
    torch.cuda.empty_cache()
    return tf


def main() -> int:
    torch.manual_seed(0)
    bw = bandwidth()
    ceil = bf16_ceiling()
    print(f"measured streaming read bandwidth: {bw} GB/s; bf16 GEMM ceiling {ceil} TFLOPS",
          flush=True)
    rows_out = []
    for name, lens, prefixes in SHAPES:
        total_q = sum(lens)
        seq_max = max(p + l for p, l in zip(prefixes, lens))
        nblk = (seq_max + 127) // 128
        topk_idx, cu, sl, pl = sel.make_selection(
            lens, prefixes, KVH, 11, DEV, "drift" if False else "real_like",
            swap=int(os.environ.get("LAB_SWAP", "0")), window=64)
        idx_q = (torch.randn(total_q, KVH, D, device=DEV) * 0.5).bfloat16()
        ik = (torch.randn(nblk, 128, D, device=DEV) * 0.5).bfloat16()
        bt = torch.arange(nblk, dtype=torch.int32, device=DEV)[None, :].repeat(
            max(len(lens), 8), 1).contiguous()
        maxq = max(lens)
        max_seq = max(p + l for p, l in zip(prefixes, lens))

        def score():
            return it.minimax_m3_index_score(idx_q, ik, bt, cu, sl, pl, maxq, max_seq, KVH)

        s = score()
        t_score = timeit(score, reps=15, warmup=6)
        ti = it.minimax_m3_index_topk(s, cu, pl, maxq, TOPK, 0, 1)
        t_topk = timeit(lambda: it.minimax_m3_index_topk(s, cu, pl, maxq, TOPK, 0, 1),
                        reps=15, warmup=6)

        def both():
            sc = score()
            return it.minimax_m3_index_topk(sc, cu, pl, maxq, TOPK, 0, 1)

        t_both = timeit(both, reps=15, warmup=6)
        score_bytes = s.numel() * 4
        flops = 2.0 * total_q * KVH * nblk * 128 * 128
        # the top-k kernel reads the causal prefix of each row's score row, not the whole
        # padded stride, so the read volume is the same order as the write volume
        traffic = 2 * score_bytes
        rec = {
            "shape": name, "rows": total_q, "prefix": max(prefixes), "max_block": nblk,
            "score_MiB": round(score_bytes / 2**20, 1),
            "score_us": t_score["median_us"], "topk_us": t_topk["median_us"],
            "both_us": t_both["median_us"], "p10": t_both["p10_us"], "p90": t_both["p90_us"],
            "spread_pct": t_both["spread_pct"],
            "score_GBps": round(score_bytes / 1e9 / (t_score["median_us"] / 1e6), 1),
            "score_TFLOPS": round(flops / (t_score["median_us"] / 1e6) / 1e12, 1),
            "score_pct_of_gemm_ceiling": round(
                flops / (t_score["median_us"] / 1e6) / 1e12 / ceil * 100, 1),
            "topk_GBps": round(score_bytes / 1e9 / (t_topk["median_us"] / 1e6), 1),
            "traffic_floor_us": round(traffic / 1e9 / bw * 1e6, 1),
            "topk_share_pct": round(t_topk["median_us"] / t_both["median_us"] * 100, 1),
        }
        rows_out.append(rec)
        print(f"{name:14} rows {total_q:6d} max_block {nblk:5d} score {s.numel()*4/2**20:7.1f} MiB  "
              f"score {t_score['median_us']:8.1f} us  topk {t_topk['median_us']:8.1f} us  "
              f"both {t_both['median_us']:8.1f} us  floor {rec['traffic_floor_us']:7.1f} us", flush=True)
        del s, ti, idx_q, ik
        torch.cuda.empty_cache()

    # decode: the spec-verify batch. vLLM flattens decode request-major, so a k=3
    # verification step is ONE request with decode_query_len = rows = 4.
    for batch, rows, seq in ((1, 1, 65536), (1, 4, 65536), (1, 4, 262144), (6, 4, 65536)):
        total_q = batch * rows
        nblk = (seq + 127) // 128
        idx_q = (torch.randn(total_q, KVH, D, device=DEV) * 0.5).bfloat16()
        ik = (torch.randn(nblk, 128, D, device=DEV) * 0.5).bfloat16()
        bt = torch.arange(nblk, dtype=torch.int32, device=DEV)[None, :].repeat(batch, 1).contiguous()
        sl = torch.full((batch,), seq, dtype=torch.int32, device=DEV)
        sc = it.minimax_m3_index_decode_score(
            idx_q, ik, bt, sl, seq, 0, 1, KVH, rows, max(rows, 4), None)
        t_sc = timeit(lambda: it.minimax_m3_index_decode_score(
            idx_q, ik, bt, sl, seq, 0, 1, KVH, rows, max(rows, 4), None), reps=15, warmup=6)
        out = torch.empty(KVH, total_q, TOPK, dtype=torch.int32, device=DEV)
        t_all = timeit(lambda: it.minimax_m3_index_decode(
            idx_q, ik, bt, sl, seq, TOPK, 0, 1, KVH, rows, max(rows, 4), out, sc),
            reps=15, warmup=6)
        rec = {
            "shape": f"decode_b{batch}_q{rows}_seq{seq}", "rows": total_q, "prefix": seq - rows,
            "max_block": nblk, "score_MiB": round(sc.numel() * 4 / 2**20, 1),
            "score_us": t_sc["median_us"], "topk_us": round(t_all["median_us"] - t_sc["median_us"], 1),
            "both_us": t_all["median_us"], "spread_pct": t_all["spread_pct"],
        }
        rows_out.append(rec)
        print(f"decode b {batch:2d} q {rows:2d} seq {seq:7d} score {sc.numel()*4/2**20:7.1f} MiB  "
              f"score {t_sc['median_us']:8.1f} us  score+topk {t_all['median_us']:8.1f} us", flush=True)
        del sc, out, idx_q, ik
        torch.cuda.empty_cache()

    path = write_result("a2_indexer", {"env": env_record(), "bandwidth_GBps": bw,
                                       "bf16_ceiling_TFLOPS": ceil,
                                       "index_heads_per_rank": KVH, "rows": rows_out})
    emit({"rows": rows_out})
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
