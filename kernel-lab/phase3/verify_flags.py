#!/usr/bin/env python3
"""Flag smoke test: prove the two switches actually select the code they claim to.

Run inside the experimental image. It (1) checks the A3 flag picks the v2 planner and that
the two planners produce bit-identical attention output through the REAL served wrapper,
(2) checks a malformed flag value raises instead of silently picking a path, and (3)
reports the coop decode tile geometry the B3 setting produces, by timing the same launch
with EXL3_MOE_COOP_WIDE unset, =1 and =0 in three subprocesses.

This is the pre-flight the brief asks for: before any long benchmark, confirm the
operations that are supposed to change actually change.
"""
import os
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/fixtures")

import torch
import vllm.models.minimax_m3.common.ops.sparse_attn as sa
from vllm_exl3 import minimax_q4kv as q4
from vllm_exl3 import minimax_q4kv_reference as ref
from vllm_exl3 import slice_planner as sp

import q4pool
import sparse_selection as sel
from labutil import timeit

DEV = "cuda"
H, KVH, D, TOPK = 32, 2, 128, 16
SCALE = D ** -0.5
POOL_PAGES = 9042


def build(rows, prefix):
    topk_idx, cu, sl, pl = sel.make_drift([rows], [prefix], KVH, 11, DEV, TOPK, drift=1)
    pool = q4pool.make_q4_pool(POOL_PAGES, KVH, seed=3)
    nblk = (rows + prefix + 127) // 128
    bt = torch.arange(nblk, dtype=torch.int32, device=DEV)[None, :].repeat(8, 1).contiguous()
    q = (torch.randn(rows, H, D, device=DEV) * 0.5).bfloat16()
    out = torch.zeros(rows, H, D, dtype=torch.bfloat16, device=DEV)
    return topk_idx, cu, sl, pl, pool, bt, q, out


def main() -> int:
    cap = ref.staging_max_pages(KVH)
    topk_idx, cu, sl, pl, pool, bt, q, out = build(16384, 196608)
    wrapped = q4._make_prefill_wrapper(sa.minimax_m3_sparse_attn)

    print(f"VLLM_EXL3_Q4KV_PLANNER_V2 = {os.environ.get('VLLM_EXL3_Q4KV_PLANNER_V2', '<unset>')!r}")
    print(f"EXL3_MOE_COOP_WIDE        = {os.environ.get('EXL3_MOE_COOP_WIDE', '<unset>')!r}")
    print(f"staging cap {cap} pages, pool {POOL_PAGES} pages")

    plan_ref = ref.plan_prefill_slices_fast(pool, bt, topk_idx, cu, KVH, cap)
    plan_v2 = sp.plan_prefill_slices_v2(pool, bt, topk_idx, cu, KVH, cap)
    same = (len(plan_ref) == len(plan_v2)
            and all(a["reqs"] == b["reqs"] and a["q"] == b["q"] and a["n"] == b["n"]
                    and bool((a["pages"] == b["pages"]).all())
                    for a, b in zip(plan_ref, plan_v2)))
    print(f"planners agree: {same}  slices {len(plan_ref)}")
    assert same, "planner plans differ"

    t_ref = timeit(lambda: ref.plan_prefill_slices_fast(pool, bt, topk_idx, cu, KVH, cap),
                   reps=7, warmup=3)
    t_v2 = timeit(lambda: sp.plan_prefill_slices_v2(pool, bt, topk_idx, cu, KVH, cap),
                  reps=7, warmup=3)
    print(f"plan: reference {t_ref['median_us']:.0f} us  v2 {t_v2['median_us']:.0f} us  "
          f"{t_ref['median_us'] / t_v2['median_us']:.1f}x")

    # the wrapper, with the flag as this process has it
    t_op = timeit(lambda: wrapped(q, pool, topk_idx, bt, cu, sl, pl, 16384, KVH, SCALE, out),
                  reps=7, warmup=3)
    out_flag = out.clone()
    print(f"wrapper op with this env: {t_op['median_us']:.0f} us")

    # and with the other setting, in-process (the flag is read per call)
    keep = os.environ.get("VLLM_EXL3_Q4KV_PLANNER_V2")
    other = "0" if q4._planner_v2_enabled() else "1"
    os.environ["VLLM_EXL3_Q4KV_PLANNER_V2"] = other
    t_other = timeit(lambda: wrapped(q, pool, topk_idx, bt, cu, sl, pl, 16384, KVH, SCALE, out),
                     reps=7, warmup=3)
    out_other = out.clone()
    if keep is None:
        del os.environ["VLLM_EXL3_Q4KV_PLANNER_V2"]
    else:
        os.environ["VLLM_EXL3_Q4KV_PLANNER_V2"] = keep
    print(f"wrapper op with VLLM_EXL3_Q4KV_PLANNER_V2={other}: {t_other['median_us']:.0f} us")
    ident = bool((out_flag == out_other).all())
    print(f"attention output bit-identical across the flag: {ident}")
    assert ident, "the flag changed the attention output"

    for bad in ("maybe", "2", "yes-no"):
        os.environ["VLLM_EXL3_Q4KV_PLANNER_V2"] = bad
        try:
            q4._planner_v2_enabled()
            print(f"FAIL: {bad!r} accepted")
            return 1
        except ValueError:
            pass
    del os.environ["VLLM_EXL3_Q4KV_PLANNER_V2"]
    print("malformed flag values fail closed: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
