# kernel-lab — Phase 2 kernel optimization lab

Read `REPORT.md` first: it is the audit, the measurements and the recommendation.

This directory is a laboratory. It does not participate in serving, it does not import
the running container, and nothing here writes to the pack, the pins or the weight
snapshots. Every GPU run is its own `--rm`, memory-capped container from the serving
image, so the kernels under test are the served ones (md5-verified against the staged
patched source).

## Layout

| Path | What it is |
|---|---|
| `REPORT.md` | the phase report: kernel map, diagnoses, results, ranking, integration plan |
| `labutil.py` | CUDA-event timing with median/p10/p90/spread, memory budget guard, result writer |
| `runner/run.sh` | run one lab script in the serving image on the local GB10 |
| `runner/sweep_coop_knobs.sh` | one container per `exl3_moe_coop` knob (they latch per process) |
| `runner/sweep_wide_abba.sh` | A/B/A/B over `EXL3_MOE_COOP_WIDE` |
| `references/` | notes on the baseline call sites (the real code lives in the image) |
| `candidates/fast_slices.py` | **accepted candidate A3**: exact, O(rows × topk) q4 slice planner |
| `fixtures/census_pack.py` | reads the pack's safetensors headers → real per-expert K geometry |
| `fixtures/pack_shapes.json` | its output |
| `fixtures/pack_weights.py` | loads one MoE layer's real EXL3 expert weights from the pack |
| `fixtures/sparse_selection.py` | selection families (`real_like` calibrated to the live 18.9 blocks/tile, `drift`, `uniform`, `disjoint`) |
| `fixtures/q4pool.py` | a dequantizable nvfp4 pool container |
| `benchmarks/a1_sparse_attn_baseline.py` | tiled vs vLLM prefill attention, kernel and metadata split |
| `benchmarks/a1b_q4_prefill_op.py` | the whole production prefill attention operation: plan / stage-fill / attention |
| `benchmarks/a2_indexer.py` | indexer block scores and block top-k, prefill and decode |
| `benchmarks/a3_planner.py` | reference planner vs candidate, plan-equality asserted on every case |
| `benchmarks/b_coop_decode.py` | coop mixed-K decode baseline on real weights, with a bandwidth ceiling column |
| `benchmarks/_coop_harness.py` | shared fixture for the B benches (production call shape) |
| `benchmarks/b2_coop_knobs.py` | the coop tuning surface |
| `benchmarks/b3_wide_abba.py` | the A/B/A/B on the wide down tile |
| `benchmarks/c_fat_gemm.py` | grouped fat GEMM at real TP=2 prefill shapes, vs a measured bf16 ceiling |
| `tests/test_planner.py` | plan-equality of the candidate planner against the reference, 10 randomized cases |
| `results/*.json` | every measurement, with the environment block that produced it |
| `_vendor/minimax_m3/` | the vLLM MiniMax-M3 ops extracted from the image, for reading only |

## Rules this lab follows

* Kernel cost and operation cost are separate numbers. A kernel win the operation does
  not show is not a win.
* Nothing is cached or reused across reps to make a later rep cheaper.
* A candidate that changes a plan, a selection or an ordering must be proven identical,
  or the difference must be measured against a higher-precision reference and justified.
* Negative results are reported with their numbers. Three of the ten ranked items in the
  report are rejections backed by a measured ceiling.
* The lab refuses to start under an 8 GiB host `MemAvailable` floor and caps its own CUDA
  allocation: on GB10 a host OOM takes the node down.
