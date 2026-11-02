# Phase 3 — experimental integration and end-to-end validation of A3 and B3

Branch `kernel-lab-phase3`, derived from `kernel-lab`. Nothing pushed, nothing merged.
The serving pair was stopped and restarted under operator intent for this phase and is
left on the shipped baseline, verified healthy (§10).

Every number below is read from a committed record under `kernel-lab/phase3/records/`
produced by the repository's own bench harness. Where something was not measured, it says
so.

These are **experimental records, not `bench/records/` entries.** They do not make any
profile `PROFILE_MEASURED=1`, and no number from this phase is quoted in `README.md` or
`PERFORMANCE.md`. The lane's official records stay exactly as they were; a `bench/records/`
entry is listed in §8 as part of landing A3, not as something already done.

---

## 1. Two corrections to the Phase 2 record, found before any integration

Both change what "the baseline" means, so they come first.

### 1.1 B3 is not a new optimization — it has been shipped since v0.1.0

`m31.sh:408`, present since the initial release commit `52802f0`:

```
    # Wide (128-column, 4-way k-split) coop tile on both MoE decode stages: 8-15% faster coop
    # launches on this pack's per-rank shapes (exllamav3 picks narrow for K=1536 by default).
    -e EXL3_MOE_COOP_WIDE=1
```

Confirmed live: `docker exec m31-flash-tp2 env | grep EXL3_MOE_COOP_WIDE` →
`EXL3_MOE_COOP_WIDE=1`.

The Phase 2 lab used the **library default** (env unset) as its "auto" baseline. That is
not the served configuration. Consequences:

* The Phase 2 B-table understates production decode speed by the 5-11% it attributed to
  "B3".
* **B3 as an optimization is already banked.** AGENTS.md's rule — do not re-propose an
  optimization that is already implemented — applies, and Phase 2 broke it.
* What remains is (a) validating the shipped choice end-to-end, which this phase does as
  the ON-vs-OFF axis, and (b) an upstream-hygiene change to `moe_coop_pick_wide` so the
  library picks it without the env var. (b) has **zero** delta against today's server and
  is therefore not a performance candidate. No CUDA patch is shipped in this phase.

### 1.2 The Phase 2 A3 measurement used the wrong `nb`

`plan_prefill_slices_fast` iterates the **physical q4 pool**, not the prompt length. At
`KVB=20 GB` per rank over 60 layers:

```
page bytes/rank 36,864   bytes/layer 333,333,333   pool pages per layer nb = 9,042
reference planner [rows, nb] int64 at rows=16384, nb=9042 -> 1.10 GiB per slice call
```

Phase 2's fixtures used `nb = ceil(seq/128)` (128-1,664), understating the reference
planner by up to 70× in bytes. Re-measured at the served geometry
(`kernel-lab/results/a3_planner.json`, `a1b_q4_prefill_op.json`, regenerated with
`LAB_POOL_PAGES=9042`):

| 16K chunk | slices | plan ref µs | plan v2 µs | plan speedup | op ref µs | op v2 µs | **op speedup** |
|---|---|---|---|---|---|---|---|
| prefix 0 | 1 | 443 | 438 | 1.0× | 10,066 | 10,046 | 1.00× |
| prefix 16K | 1 | 457 | 455 | 1.0× | 10,628 | 10,557 | 1.01× |
| prefix 48K | 6 | 43,475 | 2,229 | 19.5× | 59,752 | 16,980 | **3.52×** |
| prefix 112K | 10 | 68,275 | 2,726 | 25.1× | 88,334 | 20,742 | **4.26×** |
| prefix 192K | 10 | 69,838 | 2,569 | 27.2× | 90,174 | 21,143 | **4.26×** |
| prefix 192K, drift | 40 | 265,660 | 8,856 | 30.0× | 318,437 | 55,036 | **5.79×** |

All plans identical; all attention outputs bit-identical (`rel_l2 = 0.0`,
`max_abs = 0.0`). At prefix ≤ 16K the request's page union fits the 384-page staging cap,
the per-query split path is never entered, and the change is neutral.

The planner is **74-77% of the whole prefill sparse-attention operation** at prefixes
≥ 48K (`a1b_q4_prefill_op.json`: 16K rows / 192K prefix → op 91.8 ms, of which plan
69.7 ms, stage-fill 2.9 ms, attention 10.4 ms).

---

## 2. What was integrated

| Item | Where | Reversible how |
|---|---|---|
| A3 planner | `patches/0020-minimax-exl3-q4-slice-planner-v2.patch` — new `src/vllm_exl3/slice_planner.py` + one call site in `minimax_q4kv.py` | `VLLM_EXL3_Q4KV_PLANNER_V2`, default **off** = reference planner byte for byte. The patch is **not** in `PATCH_FILES`, so the reference build, its `PATCH_SHA256` and its pins are unchanged |
| A3 switch | `m31.sh` serve env | `M31_EXPERIMENT_A3`, default **0** |
| B3 switch | `m31.sh` serve env | `M31_EXPERIMENT_B3`, default **1** = the shipped `EXL3_MOE_COOP_WIDE=1`; `0` unsets it and lets exllamav3 choose |

Both switches fail closed: a value outside `0/1/true/false/on/off/yes/no` aborts
`./m31.sh start` and raises inside the plugin (verified, §2.2). Both land in the serve
env, which `weight_snapshot.py` already hashes into the snapshot key, so a mis-set switch
can never silently reuse another configuration's snapshot.

**Untouched:** the drafter (DFlash2 k=3), the quantization, the weights, the MoE routing,
the pins, the profiles, the tests and their tolerances. `git diff` against `kernel-lab`
touches only `m31.sh` (the two flags), `patches/0020-…` (new) and `kernel-lab/`.

### 2.1 One experimental image, four cells

`kernel-lab/phase3/build_overlay.sh` builds
`m31-flash-exl3-tp2:053c634d-8d9c23f6-a3` as `FROM` the serving image plus the two python
files produced by applying patch 0020 to a pristine copy of the pinned plugin tree
(`git apply` into a scratch repo, so the overlay is exactly what the patch produces).
Nothing is recompiled: the compiled extensions, the base image and the image **LABELS are
inherited**, so `./m31.sh start`'s pin/label gate still passes. Built on both ranks from
the same context (the base image already exists on both), so no 29 GiB image ship. The
functional image tag is never overwritten.

| Cell | Image | A3 | B3 | Meaning |
|---|---|---|---|---|
| V0 | base | off | on | **the shipped baseline** |
| V1 | overlay | on | on | A3 effect |
| V2 | base | off | off | B3 effect, measured by removing the shipped setting |
| V3 | overlay | on | off | both, interaction check |

### 2.2 Pre-flight: do the switches actually change executed code?

`kernel-lab/phase3/verify_flags.py`, run inside the overlay image both with and without
the flag, through the **real served wrapper** `minimax_q4kv._make_prefill_wrapper`:

```
planners agree: True  slices 40
plan: reference 265,611 us  v2 9,180 us  28.9x
wrapper op with this env: 320,420 us
wrapper op with VLLM_EXL3_Q4KV_PLANNER_V2=1: 56,482 us
attention output bit-identical across the flag: True
malformed flag values fail closed: OK
```

On the live server, `docker exec m31-flash-tp2 env` confirms
`VLLM_EXL3_Q4KV_PLANNER_V2=1` and `EXL3_MOE_COOP_WIDE=1` reach the container in the right
cells.

### 2.3 Correctness before performance

`kernel-lab/tests/test_planner_extreme.py` — **25/25 plan-equality cases pass** against
the reference planner: served pool width (nb 9,042) with 611-, 567-, 306- and 174-slice
splits; empty selections (every entry `-1`); single row; single row at a 262K prefix;
`cap=1`; `cap=2`; `cap == topk`; `cap == topk-1` (a row alone exceeds the cap: the hard
floor); `cap >= nb` and `cap > nb` (the no-split branch); `nb < topk` (partial blocks at
the causal edge); `topk=1`; `topk=64`; 64 tiny requests; one long plus one short request;
prefixes that are not block multiples; duplicate pages inside and across rows;
structurally identical rows. Each asserts identical slice count, `reqs`, `q` ranges, page
counts and element-wise identical page id lists.

Repo gates on this branch: `tests/test_recipe.sh` OK, `tests/test_pack_layout.py` OK,
`python3 -m py_compile` clean.

---

## 3. Method

* Driver `kernel-lab/phase3/run_matrix.sh <round>`: one boot per cell — `./m31.sh stop`,
  then `IMAGE=… M31_EXPERIMENT_A3=… M31_EXPERIMENT_B3=… ./m31.sh start`, which blocks
  until `/v1/models` answers and runs the recipe's own smoke tests and warmup.
* Round 1 order **V0 → V1 → V2 → V3**; round 2 **V3 → V2 → V1 → V0**; round 3
  **V1 → V2 → V0 → V3**; final pass **V0 → V1 → V2**. No cell is always first or last.
* Every step is the repo's own harness (`bench decode`, `bench prefill`, `bench spec`,
  `bench check`, `quality`, `tools`, `api-test`, `bench needle`) with `--out` pointed at
  `kernel-lab/phase3/records/<round>-<cell>/<step>/`. Raw `results.json` is committed;
  nothing is transcribed by hand. `kernel-lab/phase3/analyze.py` regenerates every table
  in §4 from those files.
* Decode: 30 s windows × 3 rounds, plus a 90 s window in the final pass. Temperature 0,
  `max_tokens 512`, tokens from `usage.completion_tokens`. Prefill: one cold request per
  length with a fresh `uuid` nonce, so vLLM's automatic prefix cache cannot apply.
* Telemetry (`nvidia-smi` clocks / temperature / utilisation / power, compute apps,
  `MemAvailable`, `SwapFree`) captured before and after every step into each cell's
  `telemetry.log`.
* Boot time is recorded separately from inference time.
* The pair is always left on the shipped baseline.

Boot cost per cell: 95-224 s. The plugin source digest is part of the weight-snapshot
key, so the base and overlay images evict each other's snapshot and an overlay cell
usually pays the ~220 s first boot while a base cell can boot in ~100 s. That is a fixed,
per-cell cost, never inside a measured window, but it is not symmetric between images.

---

## 4. End-to-end results

### 4.1 Prefill (synthetic prompts, the repo's own `bench prefill`)

| prompt tokens | V0 tok/s | V1 (A3) | V2 (B3 off) | V3 | V0 TTFT s | V1 TTFT s | **A3 speedup** | **A3 TTFT Δ** |
|---|---|---|---|---|---|---|---|---|
| 8,951-8,972 | 1,555 | 1,561 | 1,411 | 1,550 | 5.76 | 5.75 | 1.004 | −0.2% |
| 17,638-17,829 | 1,539 | 1,537 | 1,535 | 1,523 | 11.46 | 11.60 | 0.999 | +1.2% |
| 52,561-52,654 | 1,518 | 1,520 | 1,518 | 1,523 | 34.67 | 34.59 | 1.001 | −0.2% |
| **139,438-139,963** | **578** | **1,226** | 570 | 1,214 | **241.70** | **114.13** | **2.12×** | **−52.8%** |

128K was measured twice per cell in different rounds: V0 `580 / 576`, V1 `1224 / 1228`,
V2 `576 / 563`, V3 `1214 / 1214` tok/s — within-cell spread under 2.4%.
The 8K/16K/48K rows are one sample per cell; the V2 8K value (1,411) is a single-sample
outlier and cannot be a B3 effect, since B3 only changes the decode MoE tile.

### 4.2 Prefill on natural text (`kernel-lab/phase3/natural_prefill.py`)

The repo flags this itself in `PERFORMANCE.md`: *"Prefill tuning used synthetic prompts.
In a 16K chunk of those prompts one expert receives about half the tokens. The tuning was
not repeated on natural-text routing."* That caveat matters more for A3 than for anything
else, because the planner's cost scales with how **diverse** the indexer's block
selections are, and a wall of random words maximizes that diversity.

Real prose taken from this repository's own markdown, seed-fixed, fresh document per rep
so prefix caching cannot apply, sized against the server's tokenizer through `/tokenize`:

| cell | prompt tokens | TTFT s | tok/s |
|---|---|---|---|
| V0 | 146,443 | 623.38 | 234.9 |
| V0 | 146,474 | 619.45 | 236.5 |
| V1 | 146,441 | 166.44 | 879.8 |
| V1 | 146,474 | 164.60 | 889.9 |

**A3 on natural text: 3.75× throughput, TTFT 621.4 s → 165.5 s (−73.4%).**
10 minutes 21 seconds of time-to-first-token becomes 2 minutes 46 seconds.

Natural prose is also far slower than synthetic filler at baseline (236 vs 578 tok/s at a
similar token count), which is the routing-skew effect `PERFORMANCE.md` already warns
about. The A3 gain is larger there, not smaller.

### 4.3 Decode (`bench decode`, aggregate tok/s; medians over 3×30 s + 1×90 s)

| cell | c=1 code | c=1 prose | c=4 code | c=4 prose |
|---|---|---|---|---|
| V0 (shipped) | **32.2** | 27.9 | **76.4** | 57.6 |
| V1 (A3 on) | 32.5 | 27.9 | 76.9 | 55.9 |
| V2 (B3 off) | 30.4 | 27.0 | 69.1 | 60.9 |
| V3 (A3 on, B3 off) | 30.3 | 29.1 | 69.8 | 57.5 |

B3 effect (on vs off), two independent pairings:

| pairing | c=1 code | c=1 prose | c=4 code | c=4 prose |
|---|---|---|---|---|
| V0 vs V2 (A3 off) | **+5.9%** | +3.3% | **+10.6%** | −5.4% |
| V1 vs V3 (A3 on) | **+7.3%** | −4.1% | **+10.2%** | −2.8% |

A3 effect on decode: +0.3% to +0.6% on `code`, ±1% on `prose` — i.e. nothing, as
expected, since A3 only changes prefill metadata.

**Decode noise, quantified.** Per-cell spread across rounds reaches 25% (V0 c=4 prose:
62.7 / 52.5 / 52.5 / 65.4). The cause is speculative acceptance, not the kernels —
`bench spec` at c=1 over the same four cells:

| cell | type | drafts | accepted | accept rate | tokens/step |
|---|---|---|---|---|---|
| V0 | prose | 477 | 547 | 0.382 | 1.147 |
| V1 | prose | 495 | 528 | 0.356 | 1.067 |
| V2 | prose | 468 | 556 | 0.396 | 1.188 |
| V3 | prose | 380 | 646 | 0.567 | 1.700 |
| V0 | code | 387 | 635 | 0.547 | 1.641 |
| V1 | code | 392 | 631 | 0.537 | 1.610 |
| V2 | code | 406 | 617 | 0.507 | 1.520 |
| V3 | code | 404 | 620 | 0.512 | 1.535 |

Prose acceptance ranges 0.356-0.567 across cells that differ only in a decode tile
heuristic. `code` acceptance is much tighter (0.507-0.547), and it is exactly the `code`
column where the B3 effect is consistent and above the noise floor in **both** pairings
(+5.9%/+7.3% at c=1, +10.6%/+10.2% at c=4). The prose column is not resolvable at this
sample size and no prose claim is made.

### 4.4 Mixed load (§5.4)

`kernel-lab/phase3/mixed_scenario.py`: four decode streams running, one ~136K-token
prefill arriving at t = 12 s.

| cell | long-prefill TTFT s | decode agg tok/s in the window | ITL before ms | ITL during the prefill ms | stall factor |
|---|---|---|---|---|---|
| V0 | 240.1 | 5.25 | 119.6 | 15,803 | 132× |
| V1 | 112.0 | 5.89 | 105.2 | 12,590 | 120× |
| V1 (repeat) | 112.2 | 5.20 | 122.1 | 12,623 | 103× |
| V2 | 238.9 | 5.15 | 116.2 | 16,175 | 139× |

A3 cuts the stall that a long prefill imposes on already-running decode streams by
**20.3%** (15,803 ms → 12,590 ms inter-token latency) and raises concurrent decode
throughput during the prefill by **12.2%** (5.25 → 5.89 tok/s), while cutting the
prefill's own TTFT by 53%. It does not fix the stall: with 16K-token chunks the profile
already documents an ~11 s chunk step, and 12.6 s is that step. A3 makes it shorter, not
absent.

### 4.5 NLL (`bench check`, reference `bench/reference.json`, tolerance unchanged at 10%)

| passage (drift %, round 1 / round 2) | V0 | V1 (A3) | V2 (B3 off) | V3 |
|---|---|---|---|---|
| prose | 0.92 / 1.12 | 0.15 / 0.60 | 0.15 / 0.55 | 1.28 / 0.55 |
| code | 2.19 / 2.19 | 1.85 / 4.58 | 2.97 / 3.58 | 2.19 / 1.63 |
| math | 0.47 / 0.83 | 0.83 / 0.83 | 0.47 / 0.83 | 0.44 / 0.44 |
| json | 5.60 / 6.34 | 6.46 / **8.75** | 6.34 / 6.34 | 5.94 / 5.60 |
| reasoning | 7.86 / 7.86 | 7.86 / 7.86 | 7.86 / 7.86 | 8.00 / 8.00 |
| dialogue | 1.36 / 0.10 | 1.33 / 1.36 | 5.43 / 1.33 | 5.43 / 1.33 |

All MATCH in all twelve matrix cells, tolerance left at 10%. The largest drift anywhere is
8.75% (V1, json, round 2); the **unmodified baseline V0 reaches 7.86% on the same
passage set**, and `reasoning` sits at 7.86-8.00% in every cell including V0. The spread
within a single cell across rounds (V0 dialogue 1.36 → 0.10; V2 dialogue 5.43 → 1.33)
is the same order as the spread between cells, which is the point: these are the stack's
own run-to-run fluctuations, already documented in `PERFORMANCE.md` (up to 2.26% per
passage across repeats, plus one bimodal dialogue passage that flips between 0% and
6.88%). A3's own prediction is *exactly zero* change — the plan and the attention output
are bit-identical offline (§2.3) — and nothing in the NLL table can be attributed to it
at this resolution.

---

## 5. Quality and stability gates

Run against V1 (overlay, A3 on) by `kernel-lab/phase3/run_gates.sh`, unmodified tests and
unmodified tolerances:

| gate | result |
|---|---|
| `bench check` (NLL) | 6/6 MATCH, largest drift 6.46% |
| `bench needle --tokens 65536 --depths 10,50,90` | **3/3 PASS** (prompts 69,988 / 69,952 / 70,257) |
| `quality` | **50/50** |
| `tools` | **25/27** (the published baseline is 24/27; the three misses are the documented single-turn `run_sql`-then-`create_order` behaviour) |
| `api-test` | **17 pass, 0 warn, 0 fail**, including `temperature-0 determinism: identical` |
| CUDA graph capture | clean start in all 12 matrix cells and 3 final-pass cells; `err 0` in every decode cell |
| memory guard | never fired; `ready_min_kib = 22,144,132 KiB` (21.1 GiB) against a 1 GiB steady floor |

Gates were run on V1 only, not re-run on V0 in this session: A3 is proven bit-identical
offline (§2.3), V0's NLL check ran in all matrix rounds, and the published baseline
numbers for these same gates are in `PERFORMANCE.md`.

### 5.1 A stability finding, on the baseline, not on a candidate

While sizing the natural-text probe, a **195,790-token** natural prompt (under
`max_model_len = 300,000`) killed the EngineCore on the **shipped baseline** (base image,
A3 off):

```
NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051)
      returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:1359
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

The host memory guard did **not** fire and was not the cause: `MemAvailable` stayed at
21.1 GiB against a 1 GiB floor. This is a GPU-side allocation failure. The synthetic
139,654-token prompt works; the failure threshold on the `q4-300k` profile on this pair
is somewhere between ~140K and ~196K tokens. `PERFORMANCE.md`'s 251,900- and 502,251-token
full-context runs used a different configuration (a 32 GiB pool, 8 streams), so they do
not cover this profile.

This is a pre-existing limit of the lane, discovered by a Phase 3 probe; it is not
attributable to A3 or B3. It is why the long-context ladder in §4.1 stops at 139K and the
natural-text ladder at 146K.

---

## 6. Amdahl cross-check: does the end-to-end gain match the profile?

Natural-text 146,443-token prompt, `MAX_NUM_BATCHED_TOKENS = 16,384` → 8.94 chunks.

* Measured saving: 621.4 s − 165.5 s = **455.9 s**, i.e. **51.0 s per 16K chunk**, i.e.
  **850 ms per attention layer** (60 layers).
* Lab, per layer at a 112K prefix (`a3_planner.json`): saves 65.6 ms (`real_like`),
  221.9 ms (`drift`), 3,091 ms (`uniform`).

The observed 850 ms/layer sits inside the lab's bracket, between `drift` and `uniform` —
so the lab **brackets** the server result rather than predicting it, and the fixture that
comes closest is the high-diversity end. The synthetic random-word bench prompt lands
lower (126 s saved over 8.5 chunks = 250 ms/layer), which is why the synthetic and
natural gains differ (2.12× vs 3.75×).

Implied share at V0 on natural text: a 16K chunk takes 16,384 / 235.7 = 69.5 s, of which
the planner is ~51 s — **~73% of the entire chunk step**. After A3 the chunk is 18.5 s.
That is consistent with the lab's separate measurement that the planner is 74-77% of the
sparse-attention operation, and it says the sparse-attention operation is nearly all of
the chunk at this context length.

Decode, B3: the lab measured +5.4% to +11.0% on the coop launch at 1-4 rows and neutral
at 8-24 rows. End-to-end, `code` decode gained +5.9%/+7.3% at c=1 (4 rows — inside the
predicted range) and +10.6%/+10.2% at c=4. The c=4 result exceeding "neutral at 16 rows"
is explained by partial batches: at c=4 the verify batch is 16 rows only when all four
streams decode together, and the capture set is `1,2,3,4,8,12,16,24`, so much of the run
is at row counts where the wide tile does apply.

---

## 7. Variability and uncertainty

* **Prefill 128K and natural 146K: high confidence.** Two independent samples per cell,
  different round orders, within-cell spread under 2.4%, effect size 2.1-3.8×.
* **Prefill 8K/16K/48K: one sample per cell.** Reported as "no change"; the spread there
  is a few percent and the claim is only that A3 does not regress short context.
* **Decode: moderate confidence for `code`, none for `prose`.** Prose acceptance ranges
  0.356-0.567 across cells differing only in a decode tile heuristic, which swamps a 5%
  effect. The B3 claim rests on the `code` column, where the effect is the same sign and
  similar size in both independent pairings.
* **Process-level drift is real.** Phase 2 measured 851 / 683 / 692 µs for one identical
  microbenchmark configuration across three processes. That is why every server
  comparison here is median-of-repeats across rounds with different cell orders, and why
  no claim rests on a single sample.
* Boot times differ between images (snapshot eviction); never inside a measured window.

---

## 8. Decisions

### A3 — q4 slice planner: **APROVADO**

Repeatable, large, and only where it should be:

| workload | A3 effect |
|---|---|
| prefill 8K / 16K / 48K | 1.00× (no change, no regression) |
| prefill 128K synthetic | **2.12× throughput, TTFT −52.8%** |
| prefill 146K natural prose | **3.75× throughput, TTFT −73.4%** |
| mixed load, concurrent decode stall | **−20.3%** inter-token latency during a long prefill |
| decode c=1 / c=4 | no change |

Correctness is not a risk: identical plan and page lists in 25/25 randomized cases
including 611-slice splits, bit-identical attention output through the real served
wrapper, and all quality gates pass. Memory goes **down** — the 1.1 GiB-per-slice
`[rows, nb]` int64 tensor disappears.

**Recommendation: land it.** Add `patches/0020-minimax-exl3-q4-slice-planner-v2.patch` to
`PATCH_FILES` in `config/pins.sh` (this changes `PATCH_SHA256` and the image tag, which
is the intended release ritual), add
`test -f /opt/vllm-exl3-local/src/vllm_exl3/slice_planner.py` to the Dockerfile's
self-checks, default `VLLM_EXL3_Q4KV_PLANNER_V2=1` in `m31.sh`, carry
`tests/test_planner_extreme.py` into the plugin's test suite, and open a
`bench/records/` entry from the natural-text and 128K numbers before quoting them in
`README.md` or `PERFORMANCE.md`.

### B3 — wide down tile on `sm_121`: **REJEITADO as a new candidate** (already shipped)

It has been the served configuration since v0.1.0. There is no gain to add. Measured as
ON-vs-OFF it is worth **+5.9% to +7.3% at c=1 and +10.2% to +10.6% at c=4 on `code`
decode**, and is not resolvable on `prose`. Removing it would be a regression, so the
shipped setting stays.

The only remaining work is hygiene, not performance: teach `moe_coop_pick_wide` in
exllamav3 to pick the wide tile for this shape class on `sm_121` without needing the env
var. That has zero delta against today's server, was not built or tested here, and is not
shipped as a patch in this phase. A follow-up worth running: the current rule keys on
`slots >= 32`, and the c=4 prose samples lean the other way, so a row-count-aware rule
should be A/B'd on prose decode with 90-second windows before any change.

### Combined (V3 vs V0)

Prefill 128K 2.10× — the A3 effect, undiminished. Decode: no combined effect beyond the
two individual ones; no interaction observed in either direction (A3 on does not change
what B3 does, and vice versa). All twelve matrix cells and three final-pass cells started
clean and reported `err 0`.

---

## 9. Limitations and measurements not taken

* **No prefill at 192K, 256K or 512K tokens.** A 195,790-token prompt killed the EngineCore
  on the baseline (§5.1). The long-context ladder therefore stops at ~146K. Chasing that
  limit is a separate piece of work and is arguably more valuable than any kernel change
  at long context.
* **No Nsight Compute / Systems profile of the live server.** The pair was in use for the
  matrix; profiling it would have violated the "no long runs competing with the active
  service" rule. All fractions in §6 are arithmetic on measured per-operation times and
  measured end-to-end times, labelled as such.
* **No real captured `topk_idx` or routing histogram.** The server was not started with
  `VLLM_EXL3_ROUTE_STATS=1`, and restarting it to capture was out of scope. The natural-
  text probe in §4.2 is a partial substitute: it is real prose through the real indexer,
  but it does not give per-layer selection sets.
* **Gates run on V1 only**, not re-run on V0 in this session (§5).
* **Prefill 8K/16K/48K: one sample per cell.**
* **The 128K synthetic prefill number is not comparable to `PERFORMANCE.md`'s 1,561
  tok/s**, which is the 8K/16K mean. At 139K tokens the shipped lane runs 578 tok/s.
  That gap — attention and the planner, not the MoE — is the real long-context story.
* **B3's prose-decode column is unresolved** and no claim is made from it.
* The natural-text corpus is this repository's own markdown, including tables and fenced
  code. It is natural language but not a typical user document.

---

## 10. Rollback and the state of the pair

Rollback is by construction, at three levels:

1. **Runtime:** `M31_EXPERIMENT_A3=0` (default) and `M31_EXPERIMENT_B3=1` (default)
   reproduce the shipped configuration exactly. No rebuild needed.
2. **Image:** the experimental lane is a separate tag
   (`m31-flash-exl3-tp2:053c634d-8d9c23f6-a3`). The functional image
   `m31-flash-exl3-tp2:053c634d-8d9c23f6` was never overwritten; deleting the `-a3` tag
   removes the lane.
3. **Source:** patch 0020 is not in `PATCH_FILES`, so `./m31.sh build` still produces the
   reference image with the reference pins. Reverting the `m31.sh` flag block restores the
   previous control surface.

Final state, verified after the last measurement:

```
READY: minimax-m3.1-flash at http://10.100.24.2:8000 in 224.4s
EXL3_MOE_COOP_WIDE=1        (shipped setting, present)
VLLM_EXL3_Q4KV_PLANNER_V2   (absent -> reference planner)
VLLM_EXL3_Q4KV=1
bench check: 6/6 MATCH (largest drift 6.46%)
bench decode c=1 code: 32.3 tok/s agg, per-stream 33.62, err 0
```

That matches the V0 baseline measured in all three rounds (32.1-32.3 tok/s at c=1 code),
so the pair is back where it started.

One operational note for whoever runs this next: the natural-text probe and the mixed-load
probe both push single prompts well past the sizes the shipped profiles exercise, and one
of them took the engine down (§5.1). A `./m31.sh start` recovers it in ~110-225 s; the
weight snapshot and the pack are unaffected.
