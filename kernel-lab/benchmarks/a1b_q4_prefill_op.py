#!/usr/bin/env python3
"""Experiment A1b: the FULL production prefill sparse-attention operation on a q4 KV pool.

patches/0012 optimized the attention kernel. But on the served lane the KV pool is nvfp4,
and the prefill wrapper (vllm_exl3/minimax_q4kv.py) does three things per layer:

  1. plan_prefill_slices_fast  -- greedy row slices under a 48 MiB staging cap, with a
                                  host transfer per slice (a .tolist())
  2. q4_stage_pages            -- dequantize the union of selected pages into a bf16 pool
  3. sparse_attn_tiled         -- the 0012 kernel, on the staged bf16 pool

The module docstring itself says "a fused q4 prefill reader (R1) is milestone 2", i.e. the
staging round-trip is still there. This script prices each of the three parts separately
so the report can say what fraction of prefill attention is the kernel and what fraction
is the dequant round-trip that the kernel never needed.

Pool bytes are random uint8: the dequant cost does not depend on the values, and the
selection pattern (which does drive cost) comes from fixtures/sparse_selection.py.
"""
import os
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/fixtures")

import torch
import vllm.models.minimax_m3.common.ops.sparse_attn as sa
from vllm_exl3 import minimax_q4kv_reference as ref
from vllm_exl3 import minimax_q4kv_kernels as q4k
from vllm_exl3.minimax_q4kv import _make_prefill_wrapper
from vllm_exl3.sparse_attn_tiled import sparse_attn_tiled, build_tiles, build_union

import labutil
from labutil import timeit, timeit_wall, write_result, emit, env_record
import sparse_selection as sel
import q4pool

DEV = "cuda"
H, KVH, D, TOPK = 32, 2, 128, 16
SCALE = D ** -0.5
SHAPES = [
    ("8k_first_chunk", [8192], [0]),
    ("16k_chunk", [16384], [0]),
    ("16k_at_16k", [16384], [16384]),
    ("16k_at_48k", [16384], [49152]),
    ("16k_at_192k", [16384], [196608]),
    ("4k_6req_at_48k", [4096] * 6, [49152] * 6),
]
FAMILIES = ["real_like", "drift", "uniform"]


def q4_pool(nblk: int):
    """A q4 pool of nblk pages with dequantizable bytes (see fixtures/q4pool.py)."""
    return q4pool.make_q4_pool(nblk, KVH, seed=3)


def main() -> int:
    torch.manual_seed(0)
    wrapped = _make_prefill_wrapper(sa.minimax_m3_sparse_attn)
    max_pages = ref.staging_max_pages(KVH)
    # The planner iterates the PHYSICAL pool, not the logical block count. With
    # KVB=20 GB per rank and 60 layers, one layer's q4 container is 9,042 pages; the
    # reference planner's [rows, nb] int64 term is sized by that, not by the prompt
    # length. Using ceil(seq/128) here understated it by up to 70x in Phase 2.
    pool_pages = int(os.environ.get("LAB_POOL_PAGES", "9042"))
    print(f"staging cap: {max_pages} pages of {ref.q4_pool_page_bytes(KVH)} B "
          f"({max_pages * ref.q4_pool_page_bytes(KVH) / 2**20:.0f} MiB); "
          f"pool pages per layer nb={pool_pages}", flush=True)
    rows = []
    for name, lens, prefixes in SHAPES:
        total_q = sum(lens)
        seq_max = max(p + l for p, l in zip(prefixes, lens))
        nblk = (seq_max + 127) // 128
        if nblk > pool_pages:
            print(f"skipping {name}: needs {nblk} pages > pool {pool_pages}", flush=True)
            continue
        pool = q4_pool(pool_pages)
        bt = torch.arange(nblk, dtype=torch.int32, device=DEV)[None, :].repeat(
            max(len(lens), 8), 1).contiguous()
        q = (torch.randn(total_q, H, D, device=DEV) * 0.5).bfloat16()
        out = torch.zeros(total_q, H, D, dtype=torch.bfloat16, device=DEV)
        for family in FAMILIES:
            topk_idx, cu, sl, pl = sel.make_drift(lens, prefixes, KVH, 11, DEV, TOPK, drift=1) \
                if family == "drift" else (
                sel.make_selection(lens, prefixes, KVH, 11, DEV, "real_like",
                                   swap=int(os.environ.get("LAB_SWAP", "0")), window=64)
                if family == "real_like" else
                sel.make_uniform(lens, prefixes, KVH, 11, DEV, TOPK))
            maxq = max(lens)
            nreq = len(lens)

            def op():
                return wrapped(q, pool, topk_idx, bt, cu, sl, pl, maxq, KVH, SCALE, out)

            t_op = timeit_wall(op, reps=15)

            # --- part 1: the slice plan (host-visible: it ends in .tolist()) ---
            def plan():
                return ref.plan_prefill_slices_fast(pool, bt, topk_idx, cu, KVH, max_pages)

            t_plan = timeit_wall(plan, reps=15)
            slices = plan()

            # --- part 2: the staging fill, and the fake block table per slice ---
            buffer = torch.zeros(max(max_pages, max(s["n"] for s in slices)) + 1,
                                 KVH, 128, 2 * D, dtype=torch.bfloat16, device=DEV)

            def fill():
                for s in slices:
                    q4k.q4_stage_pages(buffer, pool, s["pages"])

            t_fill = timeit(fill, reps=15)

            def fake_bt():
                for s in slices:
                    ref.slice_fake_block_table(bt, s["reqs"], s["pages"],
                                               int(s["pages"].numel()), pool.shape[0])

            t_bt = timeit_wall(fake_bt, reps=15)

            # --- part 3: the attention kernel on a staged-shaped bf16 pool ---
            # Contents are irrelevant to the kernel's cost; the geometry is what
            # matters, so this is a bf16 pool of the same nblk pages.
            attn_buf = (torch.randn(nblk + 1, KVH, 128, 2 * D, device=DEV) * 0.5).bfloat16()
            bt_small = torch.arange(nblk, dtype=torch.int32, device=DEV)[None, :] \
                .repeat(max(nreq, 1), 1).contiguous()

            def attn_tiled():
                sparse_attn_tiled(q, attn_buf, topk_idx, bt_small, cu, sl, pl, KVH, SCALE,
                                  out, tile=8, num_warps=8, num_stages=1)

            def attn_vllm():
                sa.minimax_m3_sparse_attn(q, attn_buf, topk_idx, bt_small, cu, sl, pl,
                                          maxq, KVH, SCALE, out)

            t_attn_tiled = timeit(attn_tiled, reps=15)
            t_attn_vllm = timeit(attn_vllm, reps=15)

            staged_bytes = sum(int(s["n"]) for s in slices) * ref.staging_page_bytes(KVH)
            rows.append({
                "shape": name, "family": family, "rows": total_q, "prefix": max(prefixes),
                "nb": pool_pages,
                "n_slices": len(slices),
                "slice_rows": [int(s["q"][1] - s["q"][0]) for s in slices],
                "slice_pages": [int(s["n"]) for s in slices],
                "staged_MiB": round(staged_bytes / 2**20, 1),
                "op_us": t_op["median_us"], "op_p10": t_op["p10_us"], "op_p90": t_op["p90_us"],
                "plan_us": t_plan["median_us"], "fill_us": t_fill["median_us"],
                "fake_bt_us": t_bt["median_us"],
                "attn_tiled_us": t_attn_tiled["median_us"],
                "attn_vllm_us": t_attn_vllm["median_us"],
                "sum_parts_us": round(t_plan["median_us"] + t_fill["median_us"]
                                      + t_bt["median_us"] + t_attn_tiled["median_us"], 1),
                "attn_share_of_op_pct": round(
                    t_attn_tiled["median_us"] / t_op["median_us"] * 100, 1),
                "fill_share_of_op_pct": round(
                    t_fill["median_us"] / t_op["median_us"] * 100, 1),
                "plan_share_of_op_pct": round(
                    t_plan["median_us"] / t_op["median_us"] * 100, 1),
            })
            print(f"{name:16} {family:10} op {t_op['median_us']:8.0f} us  plan {t_plan['median_us']:7.0f}  "
                  f"fill {t_fill['median_us']:7.0f}  attn {t_attn_tiled['median_us']:7.0f}  "
                  f"slices {len(slices)}", flush=True)
        del pool, buffer, attn_buf
        torch.cuda.empty_cache()

    path = write_result("a1b_q4_prefill_op", {
        "env": env_record(),
        "staging_cap_pages": max_pages,
        "page_bytes": ref.q4_pool_page_bytes(KVH),
        "rows": rows,
    })
    emit({"rows": rows})
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
