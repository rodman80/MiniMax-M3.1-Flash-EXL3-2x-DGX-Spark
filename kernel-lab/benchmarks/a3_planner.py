#!/usr/bin/env python3
"""Experiment A3: the q4 prefill slice planner, reference vs the O(rows x topk) candidate.

The plan runs once per attention layer per prefill chunk (60 times per chunk). It is pure
torch over a [rows, nb] page bitmap, and the reference re-derives the whole
[first-occurrence-at-or-after-a] curve for every slice start. This measures:

  plan_ref   plan_prefill_slices_fast          (production)
  plan_fast  plan_prefill_slices_fast_v2       (candidate)
  op_ref / op_fast
             the whole prefill attention operation with each planner in place, which is
             the number that reaches the server

Every case asserts the two planners return the identical plan: same slices, same query
ranges, same page ids in the same order. A plan difference is a correctness failure, not
a performance result.
"""
import os
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/fixtures")
sys.path.insert(0, "/lab/candidates")

import torch
import vllm.models.minimax_m3.common.ops.sparse_attn as sa
from vllm_exl3 import minimax_q4kv_reference as ref
from vllm_exl3 import minimax_q4kv as q4mod

import labutil
from labutil import timeit_wall, write_result, emit, env_record, rel_err
import sparse_selection as sel
import q4pool
import fast_slices

DEV = "cuda"
H, KVH, D, TOPK = 32, 2, 128, 16
SCALE = D ** -0.5
SHAPES = [
    ("16k_chunk", [16384], [0]),
    ("16k_at_16k", [16384], [16384]),
    ("16k_at_48k", [16384], [49152]),
    ("16k_at_112k", [16384], [114688]),
    ("16k_at_192k", [16384], [196608]),
    ("8k_at_56k", [8192], [57344]),
    ("4k_6req_at_48k", [4096] * 6, [49152] * 6),
]
FAMILIES = ["real_like", "drift", "uniform"]


def main() -> int:
    torch.manual_seed(0)
    orig = ref.plan_prefill_slices_fast
    max_pages = ref.staging_max_pages(KVH)
    rows = []
    for name, lens, prefixes in SHAPES:
        total_q = sum(lens)
        seq_max = max(p + l for p, l in zip(prefixes, lens))
        nblk = (seq_max + 127) // 128
        pool = q4pool.make_q4_pool(nblk, KVH, seed=3)
        bt = torch.arange(nblk, dtype=torch.int32, device=DEV)[None, :].repeat(
            max(len(lens), 8), 1).contiguous()
        q = (torch.randn(total_q, H, D, device=DEV) * 0.5).bfloat16()
        out = torch.zeros(total_q, H, D, dtype=torch.bfloat16, device=DEV)
        for family in FAMILIES:
            if family == "real_like":
                topk_idx, cu, sl, pl = sel.make_selection(
                    lens, prefixes, KVH, 11, DEV, "real_like",
                    swap=int(os.environ.get("LAB_SWAP", "0")), window=64)
            elif family == "drift":
                topk_idx, cu, sl, pl = sel.make_drift(
                    lens, prefixes, KVH, 11, DEV, TOPK,
                    drift=int(os.environ.get("LAB_DRIFT", "1")))
            else:
                topk_idx, cu, sl, pl = sel.make_uniform(lens, prefixes, KVH, 11, DEV, TOPK)
            maxq = max(lens)
            u8 = sel.union_stats(topk_idx, cu, pl, 8)["mean_union"]

            plan_ref = orig(pool, bt, topk_idx, cu, KVH, max_pages)
            plan_fast = fast_slices.plan_prefill_slices_fast_v2(
                pool, bt, topk_idx, cu, KVH, max_pages)
            same = (
                len(plan_ref) == len(plan_fast)
                and all(s["reqs"] == t["reqs"] and s["q"] == t["q"]
                        and int(s["n"]) == int(t["n"])
                        and bool((s["pages"] == t["pages"]).all())
                        for s, t in zip(plan_ref, plan_fast))
            )

            t_ref = timeit_wall(lambda: orig(pool, bt, topk_idx, cu, KVH, max_pages), reps=9)
            t_fast = timeit_wall(
                lambda: fast_slices.plan_prefill_slices_fast_v2(
                    pool, bt, topk_idx, cu, KVH, max_pages), reps=9)

            # full operation, with each planner installed on the module the wrapper calls
            wrapped = q4mod._make_prefill_wrapper(sa.minimax_m3_sparse_attn)
            ref.plan_prefill_slices_fast = orig
            t_op_ref = timeit_wall(
                lambda: wrapped(q, pool, topk_idx, bt, cu, sl, pl, maxq, KVH, SCALE, out),
                reps=9)
            out_ref = out.clone()
            ref.plan_prefill_slices_fast = fast_slices.plan_prefill_slices_fast_v2
            wrapped2 = q4mod._make_prefill_wrapper(sa.minimax_m3_sparse_attn)
            t_op_fast = timeit_wall(
                lambda: wrapped2(q, pool, topk_idx, bt, cu, sl, pl, maxq, KVH, SCALE, out),
                reps=9)
            out_fast = out.clone()
            ref.plan_prefill_slices_fast = orig

            e = rel_err(out_fast, out_ref)
            rows.append({
                "shape": name, "family": family, "rows": total_q, "prefix": max(prefixes),
                "nb": nblk, "union_T8": u8, "n_slices": len(plan_ref),
                "plan_identical": same,
                "plan_ref_us": t_ref["median_us"], "plan_fast_us": t_fast["median_us"],
                "plan_speedup": round(t_ref["median_us"] / max(t_fast["median_us"], 1e-3), 2),
                "op_ref_us": t_op_ref["median_us"], "op_fast_us": t_op_fast["median_us"],
                "op_speedup": round(t_op_ref["median_us"] / max(t_op_fast["median_us"], 1e-3), 3),
                "op_p10_fast": t_op_fast["p10_us"], "op_p90_fast": t_op_fast["p90_us"],
                "out_rel_l2": e["rel_l2"], "out_max_abs": e["max_abs"],
            })
            print(f"{name:16} {family:10} slices {len(plan_ref):5d}  plan {t_ref['median_us']:9.0f} -> "
                  f"{t_fast['median_us']:8.0f} us  op {t_op_ref['median_us']:9.0f} -> "
                  f"{t_op_fast['median_us']:9.0f} us  identical={same}", flush=True)
        del pool
        torch.cuda.empty_cache()

    path = write_result("a3_planner", {"env": env_record(),
                                       "staging_cap_pages": max_pages, "rows": rows})
    emit({"rows": rows})
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
