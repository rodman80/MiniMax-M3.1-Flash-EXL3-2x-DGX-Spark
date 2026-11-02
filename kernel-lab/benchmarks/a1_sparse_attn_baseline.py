#!/usr/bin/env python3
"""Experiment A1 baseline: MiniMax-M3 prefill block-sparse attention.

Compares, at the shapes a TP=2 rank really runs (32 q heads, 2 kv heads, head_dim 128,
topk 16 of 128-token blocks):

  vLLM   _gqa_sparse_fwd_kernel            one program per (query token, kv head)
  tiled  _gqa_sparse_tiled_fwd_kernel      patches/0012, one program per 8 rows
  tiled-kernel-only / tiled-metadata       the same call split into its two halves

The split matters: patches/0014 moved the tile metadata onto the GPU but build_union is
still a chain of torch sort/searchsorted/scatter ops over int64. If the metadata is a
large share of the operation, a kernel-only speedup is not a win.

Selection families come from fixtures/sparse_selection.py; the union size per tile is
printed with every row because it is the cost driver.
"""
import json
import os
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/fixtures")

import torch
import vllm.models.minimax_m3.common.ops.sparse_attn as sa
from vllm_exl3.sparse_attn_tiled import sparse_attn_tiled, build_tiles, build_union

import labutil
from labutil import timeit, rel_err, row_rel_err, write_result, emit, env_record
import sparse_selection as sel

DEV = "cuda"
H, KVH, D, TOPK = 32, 2, 128, 16     # TP=2 per-rank shapes for MiniMax-M3.1 (64 q / 4 kv heads)
SCALE = D ** -0.5

SHAPES = [
    ("8k_first_chunk", [8192], [0]),
    ("8k_second_chunk", [8192], [8192]),
    ("16k_chunk", [16384], [0]),
    ("16k_at_48k", [16384], [49152]),
    ("4k_6req", [683] * 6, [0] * 6),
]
FAMILIES = ["real_like", "uniform", "disjoint"]


def pool(seq_max: int):
    nblk = (seq_max + 127) // 128 + 1
    kv = (torch.randn(nblk, KVH, 128, 2 * D, device=DEV) * 0.5).bfloat16()
    bt = torch.arange(nblk, dtype=torch.int32, device=DEV)[None, :].repeat(8, 1).contiguous()
    return kv, bt


def make_fixture(lens, prefixes, family, seed=11):
    if family == "uniform":
        return sel.make_uniform(lens, prefixes, KVH, seed, DEV, TOPK)
    if family == "disjoint":
        return sel.make_disjoint(lens, prefixes, KVH, seed, DEV, TOPK)
    return sel.make_selection(lens, prefixes, KVH, seed, DEV, "real_like",
                              swap=int(os.environ.get("LAB_SWAP", "2")),
                              window=int(os.environ.get("LAB_WINDOW", "64")))


def main() -> int:
    torch.manual_seed(0)
    rows = []
    cal = sel.calibrate([4096], [4096], KVH, DEV, 18.9, tile=8)
    print("fixture calibration (target = the live 18.9 blocks/tile of patches/0012):", cal, flush=True)
    for name, lens, prefixes in SHAPES:
        total_q = sum(lens)
        seq_max = max(p + l for p, l in zip(prefixes, lens))
        kv, bt = pool(seq_max)
        for family in FAMILIES:
            topk_idx, cu, sl, pl = make_fixture(lens, prefixes, family)
            maxq = max(lens)
            q = (torch.randn(total_q, H, D, device=DEV) * 0.5).bfloat16()
            ref = torch.zeros(total_q, H, D, dtype=torch.bfloat16, device=DEV)
            sa.minimax_m3_sparse_attn(
                torch.zeros(total_q, H, D, dtype=torch.bfloat16, device=DEV),
                kv, topk_idx, bt, cu, sl, pl, maxq, KVH, SCALE, ref)
            sa.minimax_m3_sparse_attn(q, kv, topk_idx, bt, cu, sl, pl, maxq, KVH, SCALE, ref)
            t_vllm = timeit(lambda: sa.minimax_m3_sparse_attn(
                q, kv, topk_idx, bt, cu, sl, pl, maxq, KVH, SCALE, ref))

            u8 = sel.union_stats(topk_idx, cu, pl, 8)
            for tile in (4, 8, 16):
                u = sel.union_stats(topk_idx, cu, pl, tile)
                for nw in (4, 8):
                    for ns in (1, 2, 3):
                        out = torch.zeros_like(ref)
                        try:
                            sparse_attn_tiled(q, kv, topk_idx, bt, cu, sl, pl, KVH, SCALE, out,
                                              tile=tile, num_warps=nw, num_stages=ns)
                        except Exception as exc:
                            rows.append({"shape": name, "family": family, "variant":
                                         f"tiled T{tile} w{nw} s{ns}", "error": str(exc)[:90]})
                            continue
                        torch.cuda.synchronize()
                        e = rel_err(out, ref)
                        rr = row_rel_err(out, ref)
                        t_full = timeit(lambda: sparse_attn_tiled(
                            q, kv, topk_idx, bt, cu, sl, pl, KVH, SCALE, out,
                            tile=tile, num_warps=nw, num_stages=ns))
                        t0, n0, r0, a0 = build_tiles(cu, pl, total_q, tile)
                        t_meta = timeit(lambda: (build_tiles(cu, pl, total_q, tile),
                                                 build_union(topk_idx, t0, n0, a0, tile)))
                        rows.append({
                            "shape": name, "family": family, "rows": total_q, "prefix": max(prefixes),
                            "variant": f"tiled T{tile} w{nw} s{ns}",
                            "mean_union": u["mean_union"], "max_union": u["max_union"],
                            "vllm_us": t_vllm["median_us"],
                            "tiled_op_us": t_full["median_us"],
                            "tiled_p10": t_full["p10_us"], "tiled_p90": t_full["p90_us"],
                            "tiled_spread_pct": t_full["spread_pct"],
                            "meta_us": t_meta["median_us"],
                            "kernel_us": round(t_full["median_us"] - t_meta["median_us"], 2),
                            "speedup_vs_vllm": round(t_vllm["median_us"] / t_full["median_us"], 3),
                            "rel_l2_vs_vllm": e["rel_l2"], "worst_row_rel": rr["worst_row_rel"],
                            "finite": e["finite"],
                        })
            rows.append({"shape": name, "family": family, "rows": total_q, "prefix": max(prefixes),
                         "variant": "vllm per-token", "vllm_us": t_vllm["median_us"],
                         "vllm_p10": t_vllm["p10_us"], "vllm_p90": t_vllm["p90_us"],
                         "vllm_spread_pct": t_vllm["spread_pct"], "mean_union": u8["mean_union"],
                         "speedup_vs_vllm": 1.0, "rel_l2_vs_vllm": 0.0})
            print(f"{name} {family}: vllm {t_vllm['median_us']:.0f} us  union(T8) {u8['mean_union']}", flush=True)
        del kv, bt
        torch.cuda.empty_cache()

    path = write_result("a1_baseline", {
        "env": env_record(),
        "fixture_calibration": cal,
        "shapes": [{"name": n, "rows": sum(l), "prefix": max(p)} for n, l, p in SHAPES],
        "heads": {"q": H, "kv": KVH, "head_dim": D, "topk": TOPK},
        "rows": rows,
    })
    emit({"rows": rows})
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
