# Where the baselines actually live

The baselines are not copied into this lab: they are the code the serving image runs, and
the lab imports them from inside that image. Verified identical to the staged patched
source (`md5sum` of `sparse_attn_tiled.py` and `minimax_q4kv.py` in
`/usr/local/lib/python3.12/dist-packages/vllm_exl3/` vs
`/home/rodman/build/m31/src/vllm-exl3/src/vllm_exl3/`).

| Operation | Module in the image | Reached by |
|---|---|---|
| prefill block-sparse attention (vLLM baseline) | `vllm/models/minimax_m3/common/ops/sparse_attn.py::_gqa_sparse_fwd_kernel` | `benchmarks/a1_sparse_attn_baseline.py` |
| prefill block-sparse attention (production) | `vllm_exl3/sparse_attn_tiled.py::_gqa_sparse_tiled_fwd_kernel` (patches 0012/0013/0014) | same |
| q4 prefill staging + slicing | `vllm_exl3/minimax_q4kv.py::_make_prefill_wrapper`, `minimax_q4kv_reference.plan_prefill_slices_fast`, `minimax_q4kv_kernels.q4_stage_pages` | `benchmarks/a1b_q4_prefill_op.py` |
| indexer scores / top-k (prefill and decode) | `vllm/models/minimax_m3/common/ops/index_topk.py` | `benchmarks/a2_indexer.py` |
| decode MoE | `exllamav3_ext.exl3_moe_coop_mk` (exllamav3 0001/0002 + plugin 0011/0018), call shape copied from `vllm_exl3/exl3.py:3700-3755` | `benchmarks/_coop_harness.py` |
| prefill MoE grouped fat GEMM | `vllm_exl3_c.exl3_fat_gemm_grouped` (plugin 0008/0010/0015/0017/0019) | `benchmarks/c_fat_gemm.py` |

The vLLM sources under `_vendor/minimax_m3/` are a read-only extract of the image for
auditing; nothing imports them.
