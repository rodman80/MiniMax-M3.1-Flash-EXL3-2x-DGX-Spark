# Phase 3 — experimental integration and end-to-end validation of A3 and B3

Branch `kernel-lab-phase3`, derived from `kernel-lab`. Nothing was pushed or merged. The
serving pair was stopped and restarted under operator intent for this phase and is left on
the shipped baseline (§10).

---

## 1. Two corrections to the Phase 2 record, found before any integration

Both change what "the baseline" means. They are stated first because they reframe the
whole phase.

### 1.1 B3 is not a new optimization — it has been shipped since v0.1.0

`m31.sh:408` has set, unconditionally, since the initial release:

```
    # Wide (128-column, 4-way k-split) coop tile on both MoE decode stages: 8-15% faster coop
    # launches on this pack's per-rank shapes (exllamav3 picks narrow for K=1536 by default).
    -e EXL3_MOE_COOP_WIDE=1
```

Confirmed live: `docker exec m31-flash-tp2 env | grep EXL3_MOE_COOP_WIDE` →
`EXL3_MOE_COOP_WIDE=1`.

The Phase 2 lab measured the **library default** (env unset) as its "auto" baseline. That
is not the served configuration. So:

* The Phase 2 B-table (`b_coop_decode.json`) understates production decode speed by the
  5-11% it attributed to "B3".
* **B3 as an optimization is already banked.** AGENTS.md's rule — do not re-propose an
  optimization that is already implemented — applies, and Phase 2 violated it.
* What is left is (a) validating that the shipped choice is right on this node, which
  Phase 3 measures as the ON vs OFF axis, and (b) an upstream-hygiene change to
  `moe_coop_pick_wide` so the library picks it without the env var. (b) has **zero**
  performance delta against today's server and is therefore not a performance candidate.

### 1.2 The Phase 2 A3 measurement used the wrong `nb`

`plan_prefill_slices_fast` iterates the **physical q4 pool**, not the prompt length. At
`KVB=20 GB` per rank over 60 layers, one layer's container is:

```
page bytes/rank 36,864   bytes/layer 333,333,333   pool pages per layer nb = 9,042
reference planner [rows, nb] int64 at rows=16384, nb=9042 -> 1.10 GiB per slice call
```

Phase 2's fixtures used `nb = ceil(seq/128)` (128-1,664), understating the reference
planner by up to 70× in bytes. Re-measured at the served geometry
(`results/a3_planner.json`, `results/a1b_q4_prefill_op.json`, both regenerated with
`LAB_POOL_PAGES=9042`):

| 16K chunk | slices | plan ref µs | plan v2 µs | plan speedup | op ref µs | op v2 µs | **op speedup** |
|---|---|---|---|---|---|---|---|
| prefix 0 | 1 | 443 | 438 | 1.0 | 10,066 | 10,046 | 1.00 |
| prefix 16K | 1 | 457 | 455 | 1.0 | 10,628 | 10,557 | 1.01 |
| prefix 48K | 6 | 43,475 | 2,229 | 19.5× | 59,752 | 16,980 | **3.52×** |
| prefix 112K | 10 | 68,275 | 2,726 | 25.1× | 88,334 | 20,742 | **4.26×** |
| prefix 192K | 10 | 69,838 | 2,569 | 27.2× | 90,174 | 21,143 | **4.26×** |
| 192K, drift | 40 | 265,660 | 8,856 | 30.0× | 318,437 | 55,036 | **5.79×** |

All plans identical, all attention outputs bit-identical (`rel_l2 = 0.0`,
`max_abs = 0.0`). At prefix ≤ 16K the request's page union fits the 384-page cap, the
per-query split path is never entered, and the change is neutral.

The planner is **74-77% of the whole prefill sparse-attention operation** at prefixes
≥ 48K (`a1b_q4_prefill_op.json`: at 16K rows / 192K prefix, op 91.8 ms of which plan
69.7 ms, stage-fill 2.9 ms, attention 10.4 ms).

---

## 2. What was integrated

| Item | Where | Reversible how |
|---|---|---|
| A3 planner | `patches/0020-minimax-exl3-q4-slice-planner-v2.patch` — new `src/vllm_exl3/slice_planner.py` + one call site in `minimax_q4kv.py` | `VLLM_EXL3_Q4KV_PLANNER_V2` (default **off** = reference planner byte for byte); patch is **not** in `PATCH_FILES`, so the reference build and its pins are unchanged |
| B3 switch | `m31.sh` serve env | `M31_EXPERIMENT_B3` (default **1** = the shipped `EXL3_MOE_COOP_WIDE=1`; `0` unsets it and lets exllamav3 choose) |
| A3 switch | `m31.sh` serve env | `M31_EXPERIMENT_A3` (default **0**) |

Both switches fail closed: a value outside `0/1/true/false/on/off/yes/no` aborts `start`
in `m31.sh` and raises inside the plugin. Both land in the serve env, which
`weight_snapshot.py` already hashes into the snapshot key, so a mis-set switch can never
silently reuse another configuration's snapshot.

**The drafter, the quantization, the weights, the MoE routing, the pins and the serving
profiles are untouched.** `git diff` against `kernel-lab` touches only `m31.sh` (the two
flags), `patches/0020-…` (new), and `kernel-lab/`.

### 2.1 One experimental image, four cells

`kernel-lab/phase3/build_overlay.sh` builds
`m31-flash-exl3-tp2:053c634d-8d9c23f6-a3` as `FROM` the serving image plus the two python
files produced by applying patch 0020 to a pristine copy of the pinned plugin tree.
Nothing is recompiled; the compiled extensions, the base image and the image **LABELS are
inherited**, so `./m31.sh start`'s pin/label gate still passes. Built on both ranks from
the same context (the base image already exists on both), so no 29 GiB image ship. The
functional image tag is never overwritten.

| Cell | Image | A3 | B3 | Meaning |
|---|---|---|---|---|
| V0 | base | off | on | **the shipped baseline** |
| V1 | overlay | on | on | A3 effect |
| V2 | base | off | off | B3 effect (shipped setting removed) |
| V3 | overlay | on | off | both, interaction check |

### 2.2 Pre-flight: do the switches actually change the executed code?

`kernel-lab/phase3/verify_flags.py`, run inside the overlay image, both with and without
the flag, through the **real served wrapper** `minimax_q4kv._make_prefill_wrapper`:

```
planners agree: True  slices 40
plan: reference 265,611 us  v2 9,180 us  28.9x
wrapper op with this env: 320,420 us
wrapper op with VLLM_EXL3_Q4KV_PLANNER_V2=1: 56,482 us
attention output bit-identical across the flag: True
malformed flag values fail closed: OK
```

And on the live server, `docker exec m31-flash-tp2 env` confirms
`VLLM_EXL3_Q4KV_PLANNER_V2=1` / `EXL3_MOE_COOP_WIDE=1` reach the container.

### 2.3 Correctness before performance

`kernel-lab/tests/test_planner_extreme.py` — 25 plan-equality cases against the reference
planner, all passing: served pool width (nb 9,042) with 611-, 567-, 306- and 174-slice
splits; empty selections (every entry `-1`); single row; single row at a 262K prefix;
`cap=1`; `cap=2`; `cap == topk`; `cap == topk-1` (a row alone exceeds the cap: the hard
floor); `cap >= nb` and `cap > nb` (the no-split branch); `nb < topk` (partial blocks at
the causal edge); `topk=1`; `topk=64`; 64 tiny requests; mixed long/short requests;
prefixes that are not block multiples; duplicate pages inside and across rows;
structurally identical rows. Every case asserts identical slice count, `reqs`, `q`
ranges, page counts and element-wise identical page id lists.

Repo gates on the integration branch: `tests/test_recipe.sh` OK,
`tests/test_pack_layout.py` OK, `py_compile` clean.

---

## 3. Method

* Driver: `kernel-lab/phase3/run_matrix.sh <round>`. One boot per cell; `./m31.sh stop`,
  then `IMAGE=… M31_EXPERIMENT_A3=… M31_EXPERIMENT_B3=… ./m31.sh start`, which blocks
  until `/v1/models` answers and runs the recipe's own smoke + warmup.
* Round 1 order **V0 → V1 → V2 → V3**; round 2 **V3 → V2 → V1 → V0**; round 3
  **V1 → V2 → V0 → V3**. No cell is always first or last.
* Every step is the repo's own harness (`bench decode`, `bench prefill`, `bench spec`,
  `bench check`, `quality`, `tools`, `api-test`, `bench needle`) with `--out` pointed at
  `kernel-lab/phase3/records/<round>-<cell>/<step>/`. Raw `results.json` + `summary.md`
  are committed; nothing is transcribed by hand.
* Telemetry (`nvidia-smi` clocks/temp/util/power, compute apps, `MemAvailable`,
  `SwapFree`) is captured before and after every step into each cell's `telemetry.log`.
* Prefill prompts carry a fresh `uuid` nonce per request, so vLLM's automatic prefix
  cache cannot apply. Decode is fixed 30 s windows at temperature 0, `max_tokens 512`,
  tokens from `usage.completion_tokens`.
* Boot time is recorded separately from inference time (`meta` per cell).
* The pair is always left on the shipped baseline.

Cost per boot: 218-222 s in every cell. The plugin source digest is part of the weight
snapshot key, so the base and overlay images evict each other's snapshot and **every**
cell pays the slow first boot. That is a fixed, symmetric cost and does not bias the
comparison, but it is why no cell boots in ~100 s.

---

## 4. End-to-end results

(Tables are generated by `python3 kernel-lab/phase3/analyze.py` from the committed
records; they are pasted verbatim below.)

### 4.1 Prefill

### 4.2 Decode

### 4.3 Speculative acceptance

### 4.4 NLL

### 4.5 Mixed load

### 4.6 Natural-text long prefill

---

## 5. Quality and stability gates

---

## 6. Amdahl cross-check

---

## 7. Variability and uncertainty

---

## 8. Decisions

---

## 9. Limitations and measurements not taken

---

## 10. Rollback and the state of the pair
