# Kernel Optimization Lab — MiniMax M3.1 Flash EXL3 on 2× DGX Spark

Phase 2. Everything below was **executed on this node** unless a row says otherwise. No
number in this report is invented; every measured figure has a JSON record under
`kernel-lab/results/`, produced by the scripts under `kernel-lab/benchmarks/`.

Reference tree: `main` @ `42d4ea1`, image `m31-flash-exl3-tp2:053c634d-8d9c23f6`
(vllm-exl3 `053c634db54afc981becfb46c0482e7358cc00ba` + patches 0001-0019, exllamav3
`fdd77361971144e8d175e09e7b50b7100739fd95` + exllamav3 patches 0001-0003). The installed
`vllm_exl3` inside that image is byte-identical to the patched source staged at
`/home/rodman/build/m31/src/vllm-exl3/src` (md5-checked), so the lab benches the served
code, not a copy of it.

Work branch: `kernel-lab`. Nothing in the serving container, the pack, the pins or the
DFlash2 configuration was modified.

---

## 1. Environment (measured, from `results/*.json` → `env`)

| Item | Value |
|---|---|
| GPU | NVIDIA GB10 (DGX Spark), `sm_121`, **48 SMs**, 24 MiB L2, 124,608 MiB unified |
| Driver / CUDA | 580.178.04 / CUDA 13.0 (nvcc V13.0.88) |
| PyTorch / Triton | 2.14.0+cu130 / 3.8.0 |
| vLLM | `0.1.dev23341+g8744dd2fb.d20261022` |
| **Measured streaming read bandwidth** | **239–245 GB/s** (512 MiB `torch.sum`, fp16) |
| **Measured bf16 dense GEMM ceiling** | **88.4–91.6 TFLOPS** (8192³ `a @ b`) |
| Contention | the serving container `m31-flash-tp2` was up and idle (84 GiB resident, 0% util) for every run; lab containers are `--memory 10g --oom-score-adj 1000` and refuse to start under an 8 GiB `MemAvailable` floor |

Two architecture facts that shape everything below:

* `sm_121` is the consumer/GB10 Blackwell family. It has `mma.sync` HMMA but **not** the
  `sm_100a`/`sm_101a` `tcgen05` tensor-memory path, and no `wgmma` (`sm_90a`). Published
  B200/GB300 sparse-attention and grouped-GEMM numbers do not transfer.
* 48 SMs and ~240 GB/s. Decode is a bandwidth problem; prefill is a compute problem.

---

## 2. What production actually runs (audit)

Model: 60 layers, hidden 6144, 64 q heads / 4 kv heads / head_dim 128, `topk_blocks=16`
of 128-token blocks, `sparse_score_type=max`, `init_block=0`, `local_block=1`, 4 index
heads, 128 routed experts × top-4 (sigmoid + routing bias, `routed_scaling_factor=2.0`),
1 shared expert, `swigluoai` with `alpha=1.702`, `limit=7.0`, 3 dense-MLP layers
(0,1,2) and 57 MoE layers.

Real pack geometry (`fixtures/census_pack.py` → `fixtures/pack_shapes.json`, read from
the safetensors headers of all 16 shards, 89,952 tensors):

| Group | K (trellis width) | count |
|---|---|---|
| routed `w1` (gate) | 2 / 3 | 7176 / 120 |
| routed `w2` (down) | 2 / 3 | 6096 / 1200 |
| routed `w3` (up) | 2 / 3 | 7176 / 120 |
| attention linears | 4 (q) / 5 (k, v, o) | 60 / 180 |
| shared experts | 4 / 5 | 114 / 57 |
| dense MLPs (layers 0-2) | 4 / 5 | 6 / 3 |
| lm_head | 6 | 1 |

So every MoE layer is **mixed-K over {2,3}**, mul1 codebook, with 12 of the 57 layers
carrying 10 K=3 gate/up experts and 40 K=3 down experts, and the other 45 carrying 0 and
16. That is exactly the `[2,3]` mixed instance `exl3_moe_coop_mk` was built for.

Per-layer prefill path (q4 KV, `VLLM_EXL3_Q4KV=1`, `q4-300k` profile):

1. **Indexer** — `vllm/models/minimax_m3/common/ops/index_topk.py`:
   `_index_block_score_kernel` (MXFP8 `index_q` × fp8 index-K cache, materializes
   `score [2, total_q, ceil(max_seq/128)]` fp32) then `_topk_index_kernel` (bitonic
   top-k, autotuned `BLOCK_SIZE_K` 64…2048).
2. **Sparse attention** — `vllm_exl3/minimax_q4kv.py::_make_prefill_wrapper`:
   `plan_prefill_slices_fast` → `q4_stage_pages` (nvfp4 → bf16 staging, cap
   `VLLM_EXL3_Q4KV_STAGING_MAX_MB=48` → **384 pages per rank**) →
   `vllm_exl3/sparse_attn_tiled.py` (patches 0012/0013/0014).
3. **MoE prefill** — grouped fat GEMM (`csrc/exl3_fat_gemm.cu`, patches 0008/0010/0015/
   0017/0019), `FAT_THRESHOLD_PREFILL=0` so **every** routed expert of a prefill chunk
   goes through it.
4. **Dense linears** — bf16 wide prefill (patch 0009).

Per-layer decode path (≤ 24 rows):

1. Indexer decode score + top-k (split-K, CUDA-graph safe).
2. Sparse decode attention (`_gqa_sparse_decode_kernel` + merge, q4 dequant via the
   plugin's R2 kernel).
3. MoE decode: **one `exl3_moe_coop_mk` launch per stage** (patch 0018 on 0011 on
   exllamav3 0001/0002), experts renumbered into ascending K-tuple order.
4. LM head rank blocks (patch 0006).

### Already optimized — do not re-propose

Query tiling of prefill attention (0012), GPU-side tile metadata (0014), mixed-K coop
decode with one launch per stage (0011/0018 + exllamav3 0001/0002), device-side split-k
and weight prefetch in coop (exllamav3 0001), split-k decode GEMV for small dense
weights (exllamav3 0003), `cp.async` 5-stage pipeline in the grouped fat GEMM with 2 CTAs
per SM and a full-tile MMA loop (0015/0017), fused prefill MoE glue (0019), 64-row
mixed-K tiles for ≥ 2048-token batches (0010), bf16 dense prefill (0009), fast q4
staging (upstream `VLLM_EXL3_Q4KV_PREFILL_FAST`).

### Open, and now measured

The q4 **slice planner** (A3 below), the **indexer top-k** kernel, the **down-projection
epilogue** of the grouped fat GEMM, and the coop kernel's **wide-tile heuristic on
`sm_121`**.

---

## 3. Method

* `kernel-lab/labutil.py`: CUDA-event timing, warmup outside the measured region,
  15-60 independent samples, median / p10 / p90 / spread reported. Kernel cost and
  operation cost are measured as separate numbers.
* Every lab process is capped (`LAB_BUDGET`, `--memory`) and refuses to start when host
  `MemAvailable` is under 8 GiB. On GB10 a host OOM takes the node down.
* Fixtures are three selection families, because the tiled attention's premise is
  selection correlation and it must be tested at both ends (`fixtures/sparse_selection.py`):
  * `real_like` — calibrated so the mean union of an 8-token tile is **18.0 blocks**,
    against the 18.9 the recipe records in `patches/0012` (calibration recorded in
    `results/a1_baseline.json`).
  * `drift` — the hot set moves one block per token: same local union, much larger
    *chunk* union. This is the more plausible long-context model.
  * `uniform` / `disjoint` — 15 random valid blocks per token, and fully disjoint
    neighbours. Adversarial upper bounds on union size.
* Routing families for MoE (`indep`, `corr0.5`, `zipf`, `hot4`) because traffic is per
  **distinct** expert, not per slot, and a DFlash2 block of k+1 near-duplicate tokens
  routes very differently from independent tokens.
* A/B/A/B drivers (`runner/sweep_wide_abba.sh`) where a single sweep could be drift.

### Not executed this phase (stated, not guessed)

* **No Nsight Compute / Systems profiles of the live server.** The service was up for the
  whole phase; profiling it would have violated the "no long runs competing with the
  active service" rule. All end-to-end fractions below are **arithmetic from measured
  per-operation times and the recorded `PERFORMANCE.md` throughput**, and are labelled as
  such.
* **No end-to-end `m31.sh bench` A/B of any candidate.** That needs a controlled restart
  of the pair; the commands are in §7.
* **No real captured `topk_idx` or routing histogram.** The live server was not started
  with `VLLM_EXL3_ROUTE_STATS=1`, and restarting it to capture was out of scope. The
  fixture generator is calibrated to the one live statistic the repo records; §7 lists
  the exact capture commands.
* KernelBench itself was **not** used as the harness. Its task format (single
  `Model`/`CustomModel` Python files, fixed input shapes, correctness against a
  reference forward) does not fit a paged-KV sparse attention kernel whose contract is
  `(cu_seqlens, block_table, topk_idx, q4 pool)` plus a C++ extension with device pointer
  tables. Forcing it through KernelBench would have meant re-implementing the plugin's
  wrappers and losing fidelity to the served code. The lab instead keeps KernelBench's
  methodology — reference vs candidate, correctness first, repeated timing, per-shape
  tables — against the real kernels.

---

## 4. Experiment A — MiniMax sparse attention

### A1 — tiled prefill attention kernel (`results/a1_baseline.json`)

Baseline: `sparse_attn_tiled` at the production launch (`T=8, warps=8, stages=1`)
against vLLM's `_gqa_sparse_fwd_kernel`, TP=2 per-rank shapes (32 q heads, 2 kv heads,
D=128, topk 16). Best-of-sweep over `T ∈ {4,8,16} × warps ∈ {4,8} × stages ∈ {1,2,3}`.

| shape | family | vLLM µs | tiled op µs | metadata µs | kernel µs | speedup | mean union (T=8) |
|---|---|---|---|---|---|---|---|
| 8k_first_chunk | real_like | 12,369 | 6,243 | 434 | 5,809 | **1.98×** | 21.6 |
| 8k_second_chunk | real_like | 14,660 | 8,273 | 422 | 7,852 | **1.77×** | 29.4 |
| 16k_chunk | real_like | 27,333 | 14,156 | 680 | 13,476 | **1.93×** | 25.6 |
| 16k_at_48k | real_like | 28,907 | 17,681 | 687 | 16,994 | **1.63×** | 30.1 |
| 8k_second_chunk | uniform | 14,250 | 18,951 | 423 | 18,528 | **0.75×** | 48.0 |
| 16k_at_48k | uniform | 32,788 | 46,927 | 711 | 46,216 | **0.70×** | 58.1 |
| 16k_at_48k | disjoint | 49,731 | 50,808 | 720 | 50,087 | **0.98×** | 61.0 |

Findings:

1. **Patch 0012's win is entirely conditional on selection correlation.** With the
   calibrated `real_like` family it is 1.6-2.0×. With `uniform`/`disjoint` selections it
   is **0.70-0.98×, i.e. slower than the kernel it replaced.** The recipe's live figure
   (18.9 blocks/tile) sits in the favourable regime, but the margin is not structural:
   it is a property of the workload, and a workload with more diverse retrieval would
   regress. This is worth knowing before anyone quotes "2× on prefill attention".
2. **The production launch is already the best in the sweep.** `T=4` and `T=16` are worse
   everywhere; `T=16` fails outright with `out of resource: shared memory, Required:
   131328, Hardware limit: 101376` (60 of 270 configs errored). `num_stages=2` helps only
   at small row counts (4k_6req: 776 → 736 µs, +5%) and is neutral-to-worse at 8K/16K.
   **No kernel-level win found here; the existing tuning is at a local optimum.**
3. **The metadata build is a real cost, not a rounding error**: `build_tiles` +
   `build_union` is 350-720 µs per call, 45% of the operation at 4K rows and 5-7% at
   16K rows. It is pure torch (`sort`/`searchsorted`/`scatter` over int64). A fused
   Triton union-build would recover most of it at short chunks. Not implemented this
   phase; ranked in §6.

### A1b — the whole prefill attention operation on the q4 pool (`results/a1b_q4_prefill_op.json`)

The kernel is only one third of what the server pays. Per layer:

| shape | family | op µs | plan µs | stage-fill µs | attention µs | slices |
|---|---|---|---|---|---|---|
| 8k_first_chunk | real_like | 5,109 | 228 (4%) | 61 | 4,714 | 1 |
| 16k_chunk | real_like | 10,010 | 375 (4%) | 108 | 9,446 | 1 |
| 16k_at_48k | real_like | 18,125 | **3,596 (20%)** | 1,872 | 10,202 | 6 |
| 64k_at_192k | real_like | 32,892 | **13,720 (42%)** | 2,899 | 10,411 | 10 |
| 16k_at_48k | drift | 45,876 | **10,035 (22%)** | — | — | 20 |
| 64k_at_192k | drift | 97,843 | **50,029 (51%)** | — | — | 40 |
| 16k_at_192k | uniform | 1,086,795 | **725,370 (67%)** | 177,208 | 67,107 | 573 |

**The slice planner, not the attention kernel, is the largest single component of the
prefill attention operation at long context.** The staging cap is 384 pages per rank
(48 MiB / 128 KiB), so any chunk whose page union exceeds 384 needs several slices, and
`plan_prefill_slices_fast` costs O(slices × rows × nb) bytes: `_union_curve` builds an
`[(rows-a), nb]` **int64** tensor for every slice start `a`. At 16,384 rows and 1,664
pages that is 218 MB per slice.

### A2 — indexer block scores and block top-k (`results/a2_indexer.py`, `results/a2_indexer.json`)

| shape | rows | max_block | score MiB | score µs | score TFLOPS | % of bf16 ceiling | top-k µs | top-k GB/s |
|---|---|---|---|---|---|---|---|---|
| 16k_chunk | 16384 | 128 | 16.0 | 1,246 | 110.3 | 120% | 287 | 58.4 |
| 16k_at_48k | 16384 | 512 | 64.0 | 7,773 | 70.7 | 77% | 1,556 | 43.1 |
| 16k_at_192k | 16384 | 1664 | 208.0 | 26,854 | 66.5 | 73% | 5,359 | 40.7 |
| 8k_at_56k | 8192 | 512 | 32.0 | 4,305 | 63.8 | 70% | 845 | 39.7 |

Decode (one request, `decode_query_len` = k+1):

| shape | score µs | score+top-k µs |
|---|---|---|
| b=1 q=1 seq=65536 | 34.2 | 64.8 |
| b=1 q=4 seq=65536 | 34.9 | 64.6 |
| b=1 q=4 seq=262144 | 314.6 | 323.7 |
| b=6 q=4 seq=65536 | 99.6 | 107.8 |

Findings:

1. **The index-score kernel is compute-bound and near the machine's dense bf16 rate**
   (66-110 TFLOPS against a measured 88-92 TFLOPS ceiling; the >100% figure is the FLOP
   count over-attributing blocks that the causal per-tile bound `hi = min(seq_len,
   prefix + (pid_q+1)·64)` never scores). MSA's indexer is a dense O(n²/128) score pass;
   at 192K prefix it is 1.8 TFLOP per layer. **There is no kernel-level headroom here.**
   The only lever is algorithmic — scoring fewer blocks — which changes what the model
   attends to and is out of bounds for this phase.
2. **The block top-k kernel is 4-6× off its memory floor.** It reads the 208 MiB score
   matrix at 40.7 GB/s against a measured 239 GB/s (floor 1.83 ms, actual 5.36 ms at
   16k_at_192k). It is 16% of the indexer step. This is the clearest unclaimed win in
   Experiment A. The kernel bitonic-merges `BLOCK_SIZE_K` (autotuned up to 2048) lanes
   per (query, head) to select 16 of ≤ 1664; a threshold/partition selection would be
   O(n) instead of O(n log n) per row. **Not implemented this phase** — a correct
   replacement has to reproduce the reference's exact tie-breaks, the `init_blocks`/
   `local_blocks` forced-selection (+1e30 / +1e29 score boosts) and the `-1` padding,
   and there was not enough of the phase left to build and validate it to the standard
   the repo holds itself to.
3. **Long-context decode indexer is not free**: 324 µs per layer at 262K context
   (b=1, q=4) → 18.5 ms per verify step over 57 layers, ~25% of a decode step, running at
   ~107 GB/s (45% of the bandwidth floor) because it must read the whole index-K cache
   every layer. At 65K context it is 65 µs/layer (3.7 ms/step, ~5%).

### A3 — the slice planner: implemented, validated, measured (`candidates/fast_slices.py`)

The identity that kills the O(rows × nb) term: for a slice starting at row `a`,

```
cov(a + k) = |union of pages of rows [a, a+k)| = #{ page : first[page] < a + k }
```

is nondecreasing in `k`, so the greedy's answer — the largest `k` with `cov(a+k) ≤ cap` —
is exactly `first_sorted[cap] - a`. One ascending sort of `nb` elements per slice, and the
slice's page list is the set of pages the same sort already ordered. No `[rows, nb]`
tensor is ever built.

Correctness (`tests/test_planner.py`, 10 randomized cases from 1 row to 16,384 rows,
`nb` 20 to 4096, caps 4 to 384, with and without `-1` sentinels, with hot/cold page
Zipf): **the partition and every slice's page list are identical to the reference
planner in all 10 cases.** `benchmarks/a3_planner.py` re-asserts plan equality on every
bench case and reports the attention output difference: `rel_l2 = 0.0`,
`max_abs = 0.0` — bit-identical output, because the plan is bit-identical.

| shape | family | slices | plan ref µs | plan fast µs | plan speedup | op ref µs | op fast µs | **op speedup** |
|---|---|---|---|---|---|---|---|---|
| 16k_chunk | real_like | 1 | 399 | 372 | 1.07 | 9,997 | 9,984 | 1.00 |
| 16k_at_16k | drift | 1 | 406 | 406 | 1.00 | 13,216 | 13,568 | 0.97 |
| 16k_at_48k | real_like | 6 | 3,630 | 1,730 | 2.10 | 19,029 | 16,185 | **1.18** |
| 16k_at_48k | drift | 20 | 10,035 | 4,584 | 2.19 | 45,876 | 39,572 | **1.16** |
| 16k_at_112k | real_like | 10 | 9,169 | 2,093 | 4.38 | 27,965 | 20,136 | **1.39** |
| 16k_at_112k | drift | 36 | 31,381 | 6,636 | 4.73 | 76,666 | 51,014 | **1.50** |
| 16k_at_192k | real_like | 10 | 14,243 | 1,903 | 7.48 | 32,855 | 20,072 | **1.64** |
| 16k_at_192k | drift | 40 | 50,029 | 6,469 | 7.73 | 97,843 | 52,908 | **1.85** |
| 8k_at_56k | drift | 12 | 3,583 | 2,241 | 1.60 | 22,930 | 21,632 | 1.06 |
| 16k_at_192k | uniform | 573 | 725,981 | 94,582 | 7.68 | 1,085,786 | 454,421 | 2.39 |

Host syncs are unchanged (2 per slice in both), so CUDA-graph and eager behaviour is
unchanged. Memory goes **down**: the reference's 218 MB-per-slice int64 tensor
disappears.

**Decision: accept.** This is the only candidate in the phase that is both a large
measured win and a provably identical result.

---

## 5. Experiment B — EXL3 mixed-K cooperative MoE decode

### B baseline (`results/b_coop_decode.json`)

Real weights: all 128 experts of layer 10 loaded from the pack (1,764 MiB of trellis
plus scales), K gate/up = {2}, K down = {2,3}, mul1 codebook, hidden 6144, intermediate
3072, top-4. Production call shape: one `exl3_moe_coop_mk` per stage, experts renumbered
into K-tuple order, `min_expert=0, max_expert=128`, SwiGLU-OAI clamped at 7.0.

| rows | family | distinct experts | weight MiB | coop µs | effective GB/s | ceiling µs | **% of ceiling** |
|---|---|---|---|---|---|---|---|
| 1 | indep | 4 | 54.2 | 302 | 188 | 233 | 77% |
| 4 | indep | 15 | 207.8 | 1,082 | 201 | 893 | 83% |
| 4 | corr0.5 | 9 | 122.0 | 851 | 150 | 524 | 62% |
| 4 | hot4 | 4 | 54.2 | 347 | 164 | 233 | 67% |
| 8 | indep | 27 | 374.9 | 1,788 | 220 | 1,612 | 90% |
| 16 | indep | 52 | 718.2 | 3,569 | 211 | 3,088 | 87% |
| 24 | indep | 69 | 962.1 | 4,468 | 226 | 4,136 | 93% |
| 32 | indep | 81 | 1,118.0 | 5,435 | 216 | 4,807 | 88% |

Correctness against a `reconstruct` + `had_r_128` + `hgemm` reference built from the same
trellis: **rel L2 = 8.6-8.8e-4 for every family at rows 1, 2, 4 and 8**, i.e. the coop
path is where the recipe says it is (patches/0011 quotes 6.4-6.5e-4 against an fp32
reference; this reference is itself fp16-accumulating, so 8.7e-4 is the same error bar).

**Two caveats on this table, both important.**

1. *Process-to-process spread is large on this node.* The identical configuration
   (rows=4, corr0.5, same seeds) measured **851 µs** in this run, **683 µs** in
   `b2_knobs_dflt_dflt_dflt.json` and **692 µs / 692 µs** in the two B3 auto passes. The
   within-process spreads are 1-5%, so this is not measurement noise inside a run; it is
   the process, the node state or the resident serving container. **That is exactly why
   the `WIDE` decision was made on a paired A/B/A inside the same process rather than on
   any cross-run comparison**, and why no claim in this report rests on comparing two
   different containers.
2. *The ceiling column is a DRAM-traffic model, and it over-predicts when L2 helps.* It
   charges every distinct expert's full weight bytes to DRAM. With 24 MiB of L2 and 4
   rows sharing 4 experts (`hot4`, `corr0.5`), a large part of that traffic is served
   from L2, so the "62% of ceiling" rows are not 38% of headroom — they are the model
   being wrong. The honest reading is the **independent-routing column**, where no
   sharing helps: **77% at 1 row rising to 87-93% at 16-32 rows.**

### The bandwidth answer the brief asked for

**On independent routing the coop decode kernel runs at 77% of this node's measured
streaming read bandwidth at 1 row and 87-93% at 16-32 rows, and time scales linearly
with the number of *distinct* experts a step touches.** Therefore:

* A "2× faster decode MoE kernel" is **not physically available** on GB10 with this
  quantization. The bytes are `distinct_experts × 14.1 MiB` per layer (K=2 experts) and
  the node moves 239-245 GB/s. At 4 rows and independent routing that is 208 MiB per
  layer, 893 µs, and the kernel takes 1,082 µs.
* The remaining 7-23% is launch/setup, the fp32 `d_out` scratch and imperfect DRAM
  efficiency. Even recovering all of it is ≤ 1.3× at 1 row and ≤ 1.10× at ≥ 8 rows.
* The lever that actually moves decode is **bytes, not kernels**: fewer distinct experts
  per verify step. That is a property of the drafter's acceptance and correlation, which
  this phase was told not to touch. The measured spread between `indep` (15 distinct at
  4 rows) and `corr0.5` (9 distinct) is **1.58×** — that is the size of the prize sitting
  in speculative-block routing correlation, and it is why DFlash2's better correlation
  shows up as decode throughput.

### B2/B3 — the tuning surface, with an A/B/A (`results/b2_knobs_*.json`, `results/b3_wide_*.json`)

`exl3_moe_coop` latches three knobs per process. B2 swept them; B3 re-tested the one
that looked real, in the order auto, wide=1, auto, wide=1, with 60 reps and a numerics
check.

| rows | family | auto pass 1 | wide=1 pass 1 | auto pass 3 | wide=1 pass 3 | gain 1 | gain 2 |
|---|---|---|---|---|---|---|---|
| 1 | indep | 305.1 | 283.6 | 311.4 | 283.4 | **1.076** | **1.099** |
| 1 | corr0.5 | 304.3 | 283.2 | 311.9 | 283.6 | **1.074** | **1.100** |
| 4 | indep | 1,102.1 | 1,008.1 | 1,093.3 | 1,037.4 | **1.093** | **1.054** |
| 4 | corr0.5 | 692.5 | 640.1 | 692.3 | 631.0 | **1.082** | **1.097** |
| 8 | indep | 1,801.1 | 1,774.3 | 1,824.0 | 1,779.1 | 1.015 | 1.025 |
| 8 | corr0.5 | 1,265.7 | 1,283.8 | 1,266.6 | 1,270.8 | 0.986 | 0.997 |
| 16 | indep | 3,367.3 | 3,343.6 | 3,358.6 | 3,360.3 | 1.007 | 1.000 |
| 16 | corr0.5 | 1,989.8 | 1,994.4 | 1,998.4 | 2,025.6 | 0.998 | 0.987 |
| 24 | indep | 4,511.3 | 4,514.9 | 4,546.5 | 4,509.7 | 0.999 | 1.008 |
| 24 | corr0.5 | 2,541.7 | 2,522.1 | 2,515.7 | 2,515.2 | 1.008 | 1.000 |

The effect is real and it is **exactly explained by the auto rule**
(`exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu:119`):

```c
static bool moe_coop_pick_wide(int kslices, int slots, int device) {
  if (mode >= 0) return mode != 0;
  if (DevCtx::instance().get_cc(device) < CC_BLACKWELL) return true;
  return kslices >= 256 || (kslices >= 128 && slots >= 32);
}
```

called with `kslices = Hi/16` for the gate/up stage and `kslices = I/16` for the down
stage. For MiniMax-M3.1 per rank: gate/up `kslices = 6144/16 = 384` → already wide;
down `kslices = 3072/16 = 192` → wide **only when `slots ≥ 32`**, i.e. only at ≥ 8 rows
at top-4. So `WIDE=1` changes one thing: **the down projection at 1, 2 and 4 rows.**
That is precisely where the measured gain is (+5.4% to +11.0%) and precisely where rows
≥ 8 shows nothing (auto already picks wide).

Numerics: `rel_l2` against the reconstruct+GEMM reference is identical to three digits
between auto and wide=1 at every checked shape (8.61e-4 / 8.61e-4, 8.73e-4 / 8.73e-4,
8.77e-4 / 8.77e-4, 8.65e-4 / 8.65e-4, 8.59e-4 / 8.59e-4).

Other knobs: `EXL3_MOE_COOP_KSPLIT` forced splits are **worse everywhere** (−6% to −17%),
confirming the upstream comment that a static split is ineffective. `DSPLIT=1` (disable
device-side split) is neutral at 1/4 rows and reaches the highest utilization of all
configs at 16/24 rows (98.1% / 98.7% of ceiling) but the medians are inside the spread —
not a claim. `WIDE=0` is worse than auto everywhere, which independently confirms that
auto is already picking wide for the gate/up stage.

**Decision: accept `WIDE=1` for the decode shapes, as a heuristic change rather than a
global env flag.** It is +5-11% on the MoE kernel at 1-4 rows, neutral at ≥ 8 rows, and
numerically indistinguishable.

---

## 6. Experiment C — EXL3 grouped fat GEMM prefill (`results/c_fat_gemm.json`)

Real per-rank shapes: gate|up fused `6144 → 1536+1536` (plain store), down
`1536 → 6144` (fp32 atomic scatter), 128 experts, skewed routing renormalized to exactly
`tokens × 4` routed rows, K2 and K3 launched separately as production does.
`FAT_THRESHOLD_PREFILL=0` in `q4-300k` means every expert of a prefill chunk is fat.

| chunk tokens | shape | rows | max rows/expert | µs | TFLOPS | **% of bf16 ceiling** |
|---|---|---|---|---|---|---|
| 2048 | gate\|up K2 | 8,195 | 272 | 5,561 | 55.6 | 62.9% |
| 2048 | down K2 scatter | 8,195 | 272 | 3,722 | 41.6 | 47.0% |
| 8192 | gate\|up K2 | 32,768 | 1,087 | 16,321 | 75.8 | 85.7% |
| 8192 | down K2 scatter | 32,768 | 1,087 | 12,259 | 50.4 | 57.1% |
| 8192 | gate\|up K3 | 32,768 | 1,087 | 16,136 | 76.7 | 86.7% |
| 8192 | down K3 scatter | 32,768 | 1,087 | 13,289 | 46.5 | 52.6% |
| 16384 | gate\|up K2 | 65,535 | 2,174 | 30,965 | **79.9** | **90.4%** |
| 16384 | down K2 scatter | 65,535 | 2,174 | 24,513 | 50.5 | 57.1% |
| 16384 | gate\|up K3 | 65,535 | 2,174 | 31,093 | 79.6 | 90.0% |
| 16384 | down K3 scatter | 65,535 | 2,174 | 24,685 | 50.1 | 56.7% |

Findings:

1. **The gate/up grouped fat GEMM is at 90% of this node's achievable bf16 GEMM rate at
   16K chunks.** There is no meaningful prefill win there. The `cp.async` 5-stage
   pipeline (0015), the 2-CTA occupancy bound (0010) and the full-tile MMA loop (0017)
   have taken it to the wall. Sweeping `FAT_G_KSUB` / tile shapes would be moving deck
   chairs: the kernel is already issuing near-peak HMMA.
2. **The down projection with the atomic scatter epilogue sits at 50-57% of the same
   ceiling** — 24.5 ms where the compute says 14.0 ms. That gap is the only real prefill
   target this lab found. Its shape is consistent with an epilogue/atomic-bound kernel
   rather than a GEMM-bound one: the scatter is `red.global.add.v4.f32` into fp32 output
   rows, and the down projection's K (1536 per rank) gives fewer MMA cycles per byte of
   epilogue.
3. Sanity anchor: summing gate|up K2 + down K2 at 16K rows gives 55.5 ms per layer for
   the K=2 classes, and the pack puts 16 of 128 down experts in the K=3 class, which
   matches the 72.5 ms per MoE layer that patches/0019 records on real 16K-chunk
   routing. The fixture is therefore in the right regime.
4. **Amdahl, estimated not profiled**: at ~72 ms per MoE layer × 57 layers, the grouped
   fat GEMM is ~4.1 s of a ~10.5 s 16K prefill chunk (the `PERFORMANCE.md` 1,561 tok/s
   figure) — roughly **39% of prefill**. The down projection is ~34% of that, i.e. ~13%
   of the chunk. A 1.5× on the down epilogue alone would be ~4.4% prefill.

**Decision: no candidate accepted in C.** The honest result is a diagnosis: gate/up is
done, down-scatter is the target, and attacking it means changing the scatter epilogue
(e.g. per-expert contiguous output plus a segmented reduce, trading memory for atomic
traffic), which is a larger change than this phase's remaining budget and needs a
numerics argument (fp32 atomics are order-nondeterministic; the repo already accepts
that, "≤ 2.4e-7, from atomic order only", in patches/0019).

---

## 7. Server-impact estimate (Amdahl, from measured per-operation times)

These are **arithmetic on measured per-operation medians plus the throughput recorded in
`PERFORMANCE.md`**, not a profiled end-to-end run. They are the reason for the ranking in
§8.

**Decode, c=1, DFlash2 k=3 → 4 rows per verify step.**
Routed MoE: 851 µs (correlated routing, `b_coop_decode.json`) to 1,082 µs (independent)
per layer × 57 MoE layers = **49-62 ms per step**; the paired B3 passes measured the
correlated case at 692 µs, i.e. 39 ms, so the honest band is **39-62 ms per step**.
Against a step time of ~75-86 ms implied by 30.8 tok/s at 2.3-2.7 accepted tokens per
step, routed MoE is **~50-80% of the decode step**.
→ `WIDE=1` at +5-11% on that kernel ⇒ **+2.7% to +8.8% decode**, best at c=1.

**Prefill, 16K chunk.**
Grouped fat GEMM ≈ 4.1 s (~39%); sparse attention operation 10.0 ms/layer at prefix 0
(~6%) rising to 32.9 ms/layer = 1.97 s at 192K prefix; indexer 1.46 ms/layer at prefix 0
(~1%) rising to 32.6 ms/layer = 1.96 s at 192K prefix.
→ the planner fix saves 12.8 ms/layer = **0.77 s per 16K chunk at 192K prefix**, i.e.
**~5-6% prefill at long context**, and nothing at short context (one slice).

**Decode at long context.** Indexer score+top-k at 262K context is 324 µs/layer =
18.5 ms/step (~25% of the step) vs 65 µs/layer = 3.7 ms (~5%) at 65K. If the top-k kernel
were brought to its memory floor, that is ~2.5 ms/step at 262K — ~3% of a long-context
decode step.

---

## 7b. Comparison table

Every row is a measured pair from `results/`. "Original" is the served code path. µs are
medians. Rows marked *not implemented* have a measured headroom, not a measured gain.

| Kernel / operation | Shape | Original (µs) | Candidate (µs) | Speedup | Correction | Decision |
|---|---|---|---|---|---|---|
| q4 slice planner `plan_prefill_slices_fast` | 16K rows, 192K prefix, real_like, 10 slices | 14,243 | 1,903 | **7.48×** | plan identical, output bit-identical (rel 0.0) | **accept** |
| q4 slice planner | 16K rows, 112K prefix, drift, 36 slices | 31,381 | 6,636 | **4.73×** | identical | **accept** |
| full prefill sparse-attention operation | 16K rows, 192K prefix, drift, 40 slices | 97,843 | 52,908 | **1.85×** | bit-identical output | **accept** |
| full prefill sparse-attention operation | 16K rows, 112K prefix, drift, 36 slices | 76,666 | 51,014 | **1.50×** | bit-identical output | **accept** |
| full prefill sparse-attention operation | 16K rows, 48K prefix, real_like, 6 slices | 19,029 | 16,185 | **1.18×** | bit-identical output | **accept** |
| full prefill sparse-attention operation | 16K rows, prefix 0, 1 slice | 9,997 | 9,984 | 1.00 | bit-identical | accept (no regression) |
| `exl3_moe_coop_mk` (down stage tile) | 4 rows, indep routing, 15 experts | 1,102.1 / 1,093.3 | 1,008.1 / 1,037.4 | **1.09 / 1.05** | rel L2 8.73e-4 both | **accept** |
| `exl3_moe_coop_mk` (down stage tile) | 4 rows, corr0.5, 9 experts | 692.5 / 692.3 | 640.1 / 631.0 | **1.08 / 1.10** | rel L2 8.77e-4 both | **accept** |
| `exl3_moe_coop_mk` (down stage tile) | 1 row, indep, 4 experts | 305.1 / 311.4 | 283.6 / 283.4 | **1.08 / 1.10** | rel L2 8.61e-4 both | **accept** |
| `exl3_moe_coop_mk` (down stage tile) | 16 rows, corr0.5 | 1,989.8 / 1,998.4 | 1,994.4 / 2,025.6 | 1.00 / 0.99 | — | neutral, no regression |
| `exl3_moe_coop_mk` with forced `KSPLIT=4` | 1 row, indep | 299.0 | 362.5 | **0.82** | — | **reject** |
| `exl3_moe_coop_mk` with `WIDE=0` | 16 rows, indep | 3,314.9 | 3,886.2 | **0.85** | — | **reject** |
| tiled prefill attention `T=8 w8 s1` (production) vs vLLM per-token | 8K rows, real_like | 12,369 | 6,243 | 1.98 | rel L2 1.9e-3, worst row 3.1e-3 | already shipped |
| tiled prefill attention `T=8 w8 s1` vs vLLM per-token | 16K rows, uniform selection | 26,809 | 27,428 | **0.98** | — | shipped, but workload-conditional |
| tiled prefill attention `T=16` | 16K rows | — | — | — | — | **fails**: shared memory 131,328 > 101,376 |
| tiled prefill attention `T=4` / `T=16` / `stages=2,3` | all shapes | — | best 1.02× over production | ≤ 1.02 | — | **reject: production launch is the sweep optimum** |
| indexer `_index_block_score_kernel` | 16K rows, 192K prefix | 26,854 | *not implemented* | headroom ≈ 1.0-1.4× | — | **reject: 66-110 TFLOPS = 73-120% of measured bf16 ceiling** |
| indexer `_topk_index_kernel` | 16K rows, 192K prefix | 5,359 | *not implemented* | headroom **4-6×** (40.7 vs 239 GB/s) | — | **next phase** |
| grouped fat GEMM gate\|up K2 | 16K tokens, 65,535 rows | 30,965 | *not implemented* | headroom ≈ 1.1× | — | **reject: 79.9 TFLOPS = 90.4% of ceiling** |
| grouped fat GEMM down K2 scatter | 16K tokens, 65,535 rows | 24,513 | *not implemented* | headroom ≈ 1.75× | — | **next phase (epilogue redesign)** |
| `exl3_moe_coop_mk` general decode speedup | 4-32 rows | 1,082-5,435 | *not implementable* | **≤ 1.10-1.30** | — | **reject: 77-93% of measured DRAM bandwidth under independent routing** |

## 8. Final ranking

| # | Candidate | Measured gain | Integration | Numerics risk | Memory | Verdict |
|---|---|---|---|---|---|---|
| 1 | **A3 q4 slice planner** (`candidates/fast_slices.py`) | plan 2.1-7.7×; whole attention op **1.16-1.85×** at ≥ 48K prefix, 1.00× at short context | low: one new function, swap one call in `minimax_q4kv.py` | **none** — plan and output bit-identical, 10/10 + 14/14 equality tests | **lower** (removes a 218 MB/slice int64 tensor) | **accept** |
| 2 | **B3 `WIDE=1` down tile at ≤ 4 rows** | coop decode **+5.4% to +11.0%** at 1-4 rows, neutral ≥ 8 rows, A/B/A twice | low: one predicate in `moe_coop_pick_wide` | **none observed** — rel L2 identical to 3 digits | unchanged | **accept** |
| 3 | Indexer block top-k | **not implemented**; 4-6× headroom measured (40.7 GB/s vs 239 GB/s floor) | medium: new Triton kernel + exact tie-break parity tests | medium (selection order, forced init/local blocks, `-1` pad) | unchanged | **next phase** |
| 4 | Grouped fat GEMM down epilogue | **not implemented**; 57% of ceiling vs 90% for gate/up | high: CUDA epilogue redesign | medium-high (atomic order, fp32 vs bf16) | may rise | next phase |
| 5 | Fused Triton `build_union` for tiled attention | **not implemented**; metadata is 5-45% of the op depending on rows | medium | low (pure metadata) | lower | next phase, low priority at 16K chunks |
| 6 | Tiled attention launch tuning | **none found** — production `T=8 w8 s1` is the sweep optimum; `T=16` fails on shared memory | — | — | — | **rejected, no headroom** |
| 7 | Forced split-k (`EXL3_MOE_COOP_KSPLIT`) | **negative**: −6% to −17% | — | — | — | **rejected** |
| 8 | `EXL3_MOE_COOP_DSPLIT=1` | within spread at 16/24 rows (98.1%/98.7% of ceiling), neutral elsewhere | trivial | none | unchanged | **not claimed**; retest with more reps if decode at 16-24 rows ever becomes the target |
| 9 | Faster coop decode kernel in general | **physically unavailable**: 77-93% of measured DRAM bandwidth under independent routing | — | — | — | **rejected with evidence** |
| 10 | Gate/up grouped fat GEMM tuning | **no headroom**: 90% of measured bf16 GEMM ceiling | — | — | — | **rejected with evidence** |

---

## 9. Exact integration for the two accepted candidates

### A3 — q4 slice planner

Files: `vcruz305/vllm-exl3` at `053c634d`.

* New module `src/vllm_exl3/slice_planner.py` containing `greedy_row_slices` and
  `plan_prefill_slices_fast_v2` from `kernel-lab/candidates/fast_slices.py` verbatim.
* `src/vllm_exl3/minimax_q4kv.py`, inside `_make_prefill_wrapper`:
  `plan = ref.plan_prefill_slices_fast` → `plan = slice_planner.plan_prefill_slices_fast_v2`,
  behind `VLLM_EXL3_Q4KV_PLANNER_V2` (default `1`, `=0` restores the reference planner).
* Tests to carry over: `kernel-lab/tests/test_planner.py` (10 plan-equality cases) into
  the plugin's `tests/test_minimax_q4kv.py` suite, alongside the existing CPU pin of the
  legacy planner.
* Reversibility: one env var, one call site, no data-format change, no pin change. The
  reference planner stays in the tree.

### B3 — coop wide down tile on `sm_121`

File: `vcruz305/exllamav3` at `fdd77361`, `exllamav3/exllamav3_ext/quant/exl3_moe_coop.cu`.

```c
static bool moe_coop_pick_wide(int kslices, int slots, int device) {
  if (mode >= 0) return mode != 0;
  // GB10 (sm_121): the 128-column, 4-way k-split tile wins for the down projection
  // (kslices = I/16 = 192 at MiniMax-M3.1's 3072 intermediate) at every slot count,
  // not only from 32 slots up. Measured on GB10: +5.4% to +11.0% at 1-4 rows, neutral
  // at 8-24 rows, identical accuracy (kernel-lab/results/b3_wide_*.json).
  const int cc = DevCtx::instance().get_cc(device);
  if (cc == CC_BLACKWELL_121) return kslices >= 128;
  if (cc < CC_BLACKWELL) return true;
  return kslices >= 256 || (kslices >= 128 && slots >= 32);
}
```

(`CC_BLACKWELL_121` = 121; check the `DevCtx` cc encoding before writing the patch.)
This is a new patch in the `patches/exllamav3/` series (MIT), so it changes the image tag
and `local.m31.patch-sha256`, and therefore requires the full release ritual: re-run
`tests/test_recipe.sh`, `tests/test_pack_layout.py`, rebuild, re-verify, new
`bench/records/` entry.

### Gate before either lands

```bash
./m31.sh bench decode --conc 1,2,4 --types prose,code --seconds 30   # baseline, DFlash2 k=3
./m31.sh bench prefill --lengths 8k,16k,64k
./m31.sh bench check                                                 # NLL drift
./m31.sh bench needle --tokens 65536 --depths 10,50,90
./m31.sh quality ; ./m31.sh tools ; ./m31.sh api-test
```

For the NLL gate, do not stop at the 10% limit: the repo's own records show repeat-to-
repeat drift of up to 2.26% and 0.66% across pairs, and one bimodal passage that flips
between 0% and 6.88%. A3 predicts **exactly zero** change (bit-identical plan and output)
— any nonzero NLL movement from it is a bug. B3 predicts movement of the same order as
the coop-vs-mixedk 1e-3 per-layer difference already accepted.

---

## 10. Reproducing this report

```bash
# one-off: the pack geometry (host, headers only, no GPU)
python3 kernel-lab/fixtures/census_pack.py

# every measurement (each starts its own memory-capped container on the local GB10)
kernel-lab/runner/run.sh benchmarks/a1_sparse_attn_baseline.py
kernel-lab/runner/run.sh benchmarks/a1b_q4_prefill_op.py
kernel-lab/runner/run.sh benchmarks/a2_indexer.py
kernel-lab/runner/run.sh benchmarks/a3_planner.py
kernel-lab/runner/run.sh benchmarks/b_coop_decode.py
kernel-lab/runner/run.sh benchmarks/c_fat_gemm.py
bash kernel-lab/runner/sweep_coop_knobs.sh
bash kernel-lab/runner/sweep_wide_abba.sh

# correctness
kernel-lab/runner/run.sh tests/test_planner.py
```

Capture commands for the fixtures this phase could not take from the live server:

```bash
# real routing marginals, without restarting anything else
VLLM_EXL3_ROUTE_STATS=1 VLLM_EXL3_ROUTE_STATS_DIR=/cache/routes ./m31.sh start
# ... drive traffic, then:
touch /cache/routes/dump.now          # the daemon thread writes the histogram
# real topk_idx: dump it from a hook on minimax_m3_sparse_attn during one 16K chunk,
# then feed it to fixtures/sparse_selection.py's consumers instead of a generator.
```
