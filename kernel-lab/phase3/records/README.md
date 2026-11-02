# Phase 3 experimental records

Raw output of the repository's own bench harness against the live TP=2 pair, one directory
per (round, cell). **Not** `bench/records/`: these do not make any profile
`PROFILE_MEASURED=1` and none of these numbers is quoted in `README.md` or
`PERFORMANCE.md`.

Layout:

```
r<round>-<cell>/            matrix cells, rounds 1-3
final-<cell>/               final pass: 90 s decode windows + natural prefill + mixed load
gates-V1/                   quality, tools, api-test, needle 64K, NLL check on V1
natural-V<cell>/            natural-text long prefill
mixed-V1/                   mixed-load probe
b3focus-*/                  (not run; superseded by the 90 s windows in final-*)
```

Inside a cell directory:

| File | What it is |
|---|---|
| `cell.json` | cell name, image id, A3/B3 switch values, per-step return codes |
| `env` | the `VLLM_EXL3_*` / `EXL3_*` switches actually present in the container |
| `start.log` | the full `./m31.sh start` transcript, including the READY line and boot time |
| `meta` | per-step return code and wall time |
| `telemetry.log` | `nvidia-smi` clocks / temperature / utilisation / power, compute apps, `MemAvailable`, `SwapFree`, before and after each step |
| `<step>/results.json`, `<step>/summary.md` | the harness's own output for that step |
| `<step>.log` | the harness's stdout for that step |

Cell definitions:

| Cell | Image | `M31_EXPERIMENT_A3` | `M31_EXPERIMENT_B3` | Meaning |
|---|---|---|---|---|
| V0 | `m31-flash-exl3-tp2:053c634d-8d9c23f6` | 0 | 1 | the shipped baseline |
| V1 | `…-a3` (overlay, patch 0020) | 1 | 1 | A3 effect |
| V2 | base | 0 | 0 | B3 effect, by removing the shipped setting |
| V3 | `…-a3` | 1 | 0 | both, interaction check |

Round orders, so no cell is always measured first or last:

* round 1: V0, V1, V2, V3
* round 2: V3, V2, V1, V0
* round 3: V1, V2, V0, V3 (decode only)
* final pass: V0, V1, V2

Regenerate every table in `kernel-lab/PHASE3_REPORT.md` with
`python3 kernel-lab/phase3/analyze.py`.

Known incident, recorded here because it is part of the evidence: `natural-V0` contains a
failed rep. A 195,790-token natural prompt on the **baseline** (base image, A3 off) killed
the EngineCore with an NVRM `NV_ERR_NO_MEMORY`; the host memory guard did not fire. See
PHASE3_REPORT.md §5.1.
