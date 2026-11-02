# TensorFold lane — what creating it takes

Status: **design only**. Nothing here is implemented, pinned or measured on our hardware. It records what we
verified about [ashhart/TensorFold](https://github.com/ashhart/TensorFold) v1.2.1 and what has to change before a
`tensorfold` lane can exist in this repository.

## Why this is a second engine, not another knob

TensorFold serves the model from a Zig binary (`tensorfold-native serve`): it reads the checkpoint, tokenizes and runs
its own CUDA kernels, with no Python and no vLLM. The lane we run today is vLLM-shaped end to end — a Dockerfile that
stacks the `vllm-exl3` plugin and 19 patches on a vLLM base image, `vllm serve` flags, the `VLLM_EXL3_*` environment,
the merged `serve-config.json`, per-rank weight snapshots, `--kv-cache-memory-bytes`, CUDA-graph capture sizes, and a
benchmark harness that reads vLLM's `/metrics` counters. A TensorFold lane shares the two Sparks, the fabric plan, the
memory-guard *requirement* and the fetch-and-verify discipline with that lane. It shares the build and the launch path
with none of it.

So "lane" here means: a new pin set, a new fetch target, a new launch path, and a new set of profiles — with the
existing engine's path left exactly as it is.

## What we verified

| Claim | Where it came from |
|---|---|
| MiniMax-M3.1-Flash on CUDA at TP=2 over two GB10s is in the v1.2.1 release | `CHANGELOG.md` under `## 1.2.1` (PR #1388); release published 2026-11-01 |
| The path is admitted **only** on sm_121 and only at two ranks | `zig/src/families/minimax_m3/cuda_native.zig:74-77` — `MiniMaxM3RequiresSm121`, `MiniMaxM3RequiresTwoRanks` |
| The engine binary is a 40 MB prebuilt archive, no build step | release asset `tensorfold-1.2.1-linux-aarch64.tar.gz` (GitHub API), sha256 `74a1047d…f0b0` |
| Our host qualifies | driver 580.178.04 (needs R580+), glibc 2.39 (needs 2.28+), aarch64, two GB10 |
| The qualified pack is `Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw` @ `c33e7a6619f0f12ac99098e07ec5e56acb7eeda9` | `docs/recipes/minimax-m3.1-flash-cuda.md`; HF tree: 49 files, 171,035,902,206 bytes = 159.3 GiB |
| Its KV cache is bf16 and exact; 75 KiB a token a rank at TP=2 | same doc, "Run" section |
| Its fidelity against the release: mean KL 0.0788, top-1 agreement 89.90% (2,406 teacher-forced tokens) | same doc, "Checks" — the author's measurement, not ours |
| Its decode at one stream: prose 24.3-24.4, code 38.1-39.4 tok/s; two streams 31.0 / 44.9 in total | same doc — the author's measurement, not ours |

For reference, this pair's own vLLM-lane baseline (`bench/records/2026-11-01-q4-300k/`): prefill 1,408.7 / 1,551.1
tok/s at 8K / 16K; decode at c=1 28.9 prose / 33.4 code, at c=4 65.1 / 79.9 aggregate; KV pool 726,578 tokens a rank.

## Blocker 1 — the pack: TensorFold refuses the one this lane pins

`zig/src/families/minimax_m3/cuda_exl3.zig:23-28` gates the checkpoint on `quantization_config`:

```zig
const book = q.object.get("codebook") orelse return error.UnsupportedExl3;
if (book != .string or !std.mem.eql(u8, book.string, "mul1")) return error.UnsupportedExl3;
const version = q.object.get("version") orelse return error.UnsupportedExl3;
if (version != .string or !std.mem.startsWith(u8, version.string, "1.")) return error.UnsupportedExl3;
```

Ours (`vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`) is
`{"quant_method":"exl3","version":"0.0.2"}` — **no `codebook` key at all**, and a `0.0.2` version. It fails both
checks. The qualified pack is `{"quant_method":"exl3","version":"1.6.1","codebook":"mul1","avg_bits":3.06}`.

Do not "fix" this by editing the pack's `config.json`. The loader also checks every projection's trellis, codebook
marker and scales as it reads them, so a relabelled SAGE pack fails at tensor level, not at the config gate. A lane
that wants TensorFold pins a different checkpoint. The format gate is format-level, so other `mul1` + `1.x` packs would
load (`Mia-AiLab/…-EAQ-v1-3.5-bpw` is `mul1` 1.6.1 at 3.55 bpw, `jayleaton/…-3.55bpw-mm31` is `mul1` 1.6.4), but only
the 3.06 bpw revision is qualified by the engine's own recipe, and pinning an unqualified pack would leave us without
anyone's fidelity measurement.

The two packs are not interchangeable in size either: 159.3 GiB against 110.81 GiB. Keeping both on disk costs
270.1 GiB a node (we have 1.9 TB and 2.0 TB free), and 85.3 GiB a rank of it is resident at TP=2 against our present
~55 GiB.

## Blocker 2 — the serving shape: the q4-300k profile does not exist there

Our profile's cost model and TensorFold's are different models:

| | this lane, `q4-300k` | TensorFold MiniMax CUDA |
|---|---|---|
| KV cache | nvfp4 main KV + fp8 indexer + fp8 draft (lossy) | bf16, exact |
| KV bytes a token a rank | 27,520 measured | 76,800 (75 KiB, from the engine's doc) |
| Weights a rank | ~55 GiB | 85.3 GiB |
| Pool control | `--kv-cache-memory-bytes` (`KVB`), pinned by the profile | no byte budget; `--context` and `--parallel` size it, `TENSORFOLD_MEMORY_RESERVE_GIB` floors the host |
| Qualified shape | 300,000 × 6 streams | 32,768 × 2 streams (`--prompt-cache-gib 3 --vision`) |

Rough room on a 121.7 GiB GB10: 121.7 − 85.3 (weights) − 3 (reserve) − ~8 (engine and caches) ≈ 25 GiB a rank of KV,
which at 75 KiB a token is ~360K tokens — enough for 6 streams at 60K or 2 at 180K, nowhere near 300K × 6 (that would
need 135 GiB a rank). This is arithmetic from published numbers, **not a measurement**: the first TensorFold boot has
to report the real ceiling.

The honest framing of the lane, if we build it: **less context, exact KV, plus vision** (the SAGE pack we serve has no
vision-tower weights; the EAQ pack carries the tower verbatim, and `--vision` serves it). It is a quality-and-simplicity
lane, not a context lane. Anyone expecting it to beat `q4-300k` on context is reading the wrong column.

## Blocker 3 — the harness reads vLLM's surface

| Harness part | Today | Under TensorFold |
|---|---|---|
| `/v1/models`, chat, streaming, `usage`, `/tokenize`, `/v1/completions`, `/v1/detokenize`, `/health` | used | present on the native server; should port as is |
| `bench spec` | deltas of `vllm:spec_decode_num_drafts_total`, `…_accepted_tokens_total` | those counters do not exist; the native `/metrics` reports its own draft/acceptance counters, and `--no-drafts` is the off switch instead of a spec config |
| `smoke`'s mixed-K dispatch line, `probe`'s reasoning-parser checks | greps the plugin's log and relies on `--reasoning-parser minimax_m3` / `--tool-call-parser minimax_m3` | the family has its own chat template and tool markup; there is no parser to select, so those checks need replacing, not re-flagging |
| `preflight`'s image-label gate (base digest, exl3 rev, plugin rev, patch hash) | refuses an image that is not the pinned stack | there is no image; the gate becomes archive sha256 + `tensorfold-native capabilities --json` |
| `make_serve_config.py` merge | keeps the pack's model definition, swaps `quantization_config` | not used: the pack is served as published |
| weight snapshots (`patches/0004`) | first boot writes them, later boots restore | not applicable; TensorFold loads its own way |

`bench decode`/`prefill`/`needle`, `quality`, `tools` and `api-test` are HTTP-only and should carry over; the DSpark
acceptance-rate reporting and the boot-timing record need new sources.

## Blocker 4 — the memory guard has no container to kill

The guard is mandatory on GB10 and today it watches and kills a container name. The TensorFold quick start runs the
release archive as a host binary, which is the path the engine qualifies. A lane that follows it needs a process-level
variant of `scripts/memguard.sh` (kill the rank's process group), or it has to accept an unqualified container wrapper.
Deciding this is part of the design, not an implementation detail: a host OOM on GB10 takes the node down.

## Pins a `tensorfold` lane would carry

```
ENGINE=tensorfold
TF_VERSION=1.2.1
TF_ASSET=tensorfold-1.2.1-linux-aarch64.tar.gz
TF_SHA256=74a1047df80ba87a1cb3204dce2b4ba3c02ab3db0f6e517378c97b322c5ef0b0
MODEL_REPO=Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw
MODEL_REV=c33e7a6619f0f12ac99098e07ec5e56acb7eeda9
MODEL_BYTES=171035902206            # 49 files, 159.3 GiB
```

The link keys, `SSH_USER`, `BASE_DIR`/`MODELS_DIR`/`CACHE_DIR`, the ssh-per-rank discipline and the fail-closed preflight
carry over unchanged. `MASTER_PORT`/`NCCL_COMM_ID`, `TF_MINIMAX_RANK`/`TF_MINIMAX_WORLD`, `TF_ROCE`,
`TF_ROCE_HCA`/`TF_ROCE_GID_INDEX`, `NCCL_RUNTIME_CONNECT=0`, `NCCL_CUMEM_ENABLE=0`, `NCCL_CUMEM_HOST_ENABLE=0` and
`TENSORFOLD_MEMORY_RESERVE_GIB` are new.

## Staged plan, with abort criteria

1. **Binary only, one rank, one short prompt.** Fetch the archive, check the sha256, run `capabilities --json` on both
   nodes, and confirm `admit()` accepts our GPUs. Aborts if the binary refuses our driver or the capability list has no
   `minimax_m3` CUDA entry. Nothing is downloaded at this point.
2. **Pack fetch at the pinned revision, verified**, same split-and-fan-out over the cable as today. Aborts on any
   sha256 mismatch. This is the 159.3 GiB step and the one to approve explicitly.
3. **Launch at the engine's own qualified shape** (32,768 × 2, `--prompt-cache-gib 3 --vision`), with the
   process-level guard, and read the real KV ceiling off the boot log. Aborts on `MiniMaxM3Requires*`, on any guard
   kill, or on a boot that leaves less than `GUARD_MIB` free.
4. **Port the harness**: decode/prefill/needle/quality/tools/api-test unchanged, a new spec-metrics source, and the
   smoke/probe checks replaced by equivalents rather than deleted.
5. **Only then** a profile family and a record under `bench/records/`, with the pack's own fidelity numbers quoted as
   the author's, never as ours.

## Open decision

Where the lane lives. This repo keeps one controller with an engine dispatch, which is a real change to `m31.sh` and
the Docker-centric build path but keeps one set of benches and records. A sibling recipe (the shape of
`GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` and `DeepSeek-V4.1-Flash-1017`) keeps each engine's controller honest and
lets each pin its own guard, at the cost of two places to keep in sync. The deciding factor is the guard: if the lane
runs the binary on the host, `m31.sh`'s container-shaped `start`/`stop`/`status`/`preflight` all need an engine branch,
and a separate controller is likely the cleaner home.

Not answered here, and the reason this is a design doc: the pack download has not been approved, and nothing about
TensorFold's decode speed on *our* pair has been measured, so no number in this file is ours except the ones marked
from `bench/records/`.
