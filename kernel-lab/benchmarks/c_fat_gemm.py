#!/usr/bin/env python3
"""Experiment C: the EXL3 grouped fat GEMM at the real TP=2 prefill shapes.

Per rank of MiniMax-M3.1 (hidden 6144, expert intermediate 3072 sharded to 1536):

  gate|up   6144 -> 1536 + 1536 fused, plain store
  down      1536 -> 6144, atomic scatter into the token rows

The q4-300k profile runs FAT_THRESHOLD_PREFILL=0, so in a prefill chunk EVERY routed
expert goes through this kernel: 16384 tokens x top-4 = 65536 routed rows over 128
experts. The kernel is launched once per (K, codebook) class per projection, so K2 and
K3 are timed separately, as production does.

The point of the table is the TFLOPS column against a measured bf16 GEMM ceiling on the
same node: it says whether the prefill MoE is compute-bound (and therefore whether a
kernel speedup is possible at all) or bound by something else.
"""
import os
import sys

sys.path.insert(0, "/lab")

import math
import torch
import vllm_exl3_c as nat

from labutil import timeit, write_result, emit, env_record

DEV = "cuda"
HIDDEN = 6144
INTER_RANK = 1536
N_EXPERTS = 128
TOPK = 4
CHUNKS = [2048, 8192, 16384]


def routing_counts(tokens, seed=5, sigma=0.6):
    """Skewed per-expert routed row counts, the way real routing is (the recipe records
    that a 16K chunk leaves a third of the experts with 1-32 rows)."""
    g = torch.Generator().manual_seed(seed)
    mean = tokens * TOPK / N_EXPERTS
    c = (torch.randn(N_EXPERTS, generator=g) * sigma + math.log(mean)).exp().round().long()
    c = torch.where(c < 1, torch.ones_like(c), c)
    # renormalize to the exact routed-row count
    scale = tokens * TOPK / int(c.sum())
    c = (c.float() * scale).round().long().clamp(min=1)
    return c.tolist()


def bounds_of(counts):
    starts = [0]
    for n in counts:
        starts.append(starts[-1] + n)
    return torch.tensor([b for g in range(len(counts)) for b in (starts[g], starts[g + 1])],
                        dtype=torch.int32, device=DEV), starts[-1]


def experts(counts, k_in, k_out, K):
    tr = [torch.randint(-32768, 32767, (k_in // 16, k_out // 16, 16 * K),
                        dtype=torch.int16, device=DEV) for _ in counts]
    sv = [(torch.rand(k_out, device=DEV) + 0.5).half() for _ in counts]
    tab = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=DEV)
    return tab(tr), tab(sv), tr, sv


def bf16_ceiling():
    """Achievable bf16 dense GEMM throughput on this node, measured."""
    n = 8192
    a = torch.randn(n, n, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(n, n, dtype=torch.bfloat16, device=DEV)
    t = timeit(lambda: a @ b, reps=15, warmup=5)
    tf = 2.0 * n ** 3 / (t["median_us"] / 1e6) / 1e12
    del a, b
    torch.cuda.empty_cache()
    return round(tf, 1)


def main() -> int:
    torch.manual_seed(0)
    ceil = bf16_ceiling()
    print(f"measured bf16 dense GEMM ceiling on this node: {ceil} TFLOPS", flush=True)
    rows_out = []
    for tokens in CHUNKS:
        counts = routing_counts(tokens)
        bounds, rows = bounds_of(counts)
        tok = torch.cat([torch.randperm(tokens, device=DEV)[:n] for n in counts]).long()
        wt = torch.rand(rows, device=DEV).half()
        a_g = (torch.randn(rows, HIDDEN, device=DEV) * 0.5).half()
        a_d = (torch.randn(rows, INTER_RANK, device=DEV) * 0.5).half()
        o_g = torch.empty(rows, 2 * INTER_RANK, dtype=torch.float32, device=DEV)
        o_d = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=DEV)
        el, eh = torch.empty(0, dtype=torch.long, device=DEV), torch.empty(0, dtype=torch.float16, device=DEV)
        for K in (2, 3):
            g_tp, g_sp, kg, sg = experts(counts, HIDDEN, 2 * INTER_RANK, K)
            d_tp, d_sp, kd, sd = experts(counts, INTER_RANK, HIDDEN, K)
            for name, fn, fl, bytes_ in (
                (f"gate|up K{K}", lambda: nat.exl3_fat_gemm_grouped(
                    a_g, g_tp, o_g, g_sp, bounds, el, eh, K, False, True, False,
                    2 * INTER_RANK, max(counts)),
                 2 * rows * HIDDEN * 2 * INTER_RANK,
                 rows * HIDDEN * 2 * INTER_RANK * K / 4.0),
                (f"down K{K} scatter", lambda: nat.exl3_fat_gemm_grouped(
                    a_d, d_tp, o_d, d_sp, bounds, tok, wt, K, False, True, True,
                    HIDDEN, max(counts)),
                 2 * rows * INTER_RANK * HIDDEN,
                 rows * INTER_RANK * HIDDEN * K / 4.0),
            ):
                t = timeit(fn, reps=15, warmup=5)
                rec = {
                    "chunk_tokens": tokens, "shape": name, "experts": len(counts),
                    "rows": rows, "max_expert_rows": max(counts),
                    "us": t["median_us"], "p10": t["p10_us"], "p90": t["p90_us"],
                    "spread_pct": t["spread_pct"],
                    "TFLOPS": round(fl / (t["median_us"] / 1e6) / 1e12, 1),
                    "pct_of_bf16_ceiling": round(
                        fl / (t["median_us"] / 1e6) / 1e12 / ceil * 100, 1),
                    "weight_MiB": round(bytes_ / 2**20, 1),
                    "weight_GBps": round(bytes_ / 1e9 / (t["median_us"] / 1e6), 1),
                }
                rows_out.append(rec)
                print(f"chunk {tokens:6d} {name:20} rows {rows:6d}  {t['median_us']:9.1f} us  "
                      f"{rec['TFLOPS']:6.1f} TFLOPS  {rec['pct_of_bf16_ceiling']:5.1f}% of "
                      f"ceiling  {rec['weight_GBps']:6.1f} GB/s weights", flush=True)
            del g_tp, g_sp, kg, sg, d_tp, d_sp, kd, sd
        del a_g, a_d, o_g, o_d, tok, wt
        torch.cuda.empty_cache()

    path = write_result("c_fat_gemm", {"env": env_record(), "bf16_ceiling_TFLOPS": ceil,
                                       "hidden": HIDDEN, "inter_per_rank": INTER_RANK,
                                       "rows": rows_out})
    emit({"rows": rows_out})
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
