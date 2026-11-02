#!/usr/bin/env bash
# Experiment B2 sweep: one container per coop knob configuration (the knobs are latched
# once per process by moe_coop_env(), so they cannot be changed inside a run).
#
#   kernel-lab/runner/sweep_coop_knobs.sh
#
# Nothing here touches the serving container; each run is --rm and memory-capped.
set -uo pipefail
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$LAB_DIR/.." && pwd)"
IMAGE="${LAB_IMAGE:-m31-flash-exl3-tp2:053c634d-8d9c23f6}"
SRC="${LAB_SRC:-/home/rodman/build/m31/src}"
MODELS="${LAB_MODELS:-/home/rodman/models}"
CACHE="${LAB_CACHE:-/home/rodman/.cache/m31/lab}"
MEM="${LAB_MEM:-10g}"
mkdir -p "$CACHE/triton"

CONFIGS=(
  ""
  "EXL3_MOE_COOP_DSPLIT=1"
  "EXL3_MOE_COOP_DSPLIT=4"
  "EXL3_MOE_COOP_DSPLIT=16"
  "EXL3_MOE_COOP_KSPLIT=2"
  "EXL3_MOE_COOP_KSPLIT=4"
  "EXL3_MOE_COOP_WIDE=1"
  "EXL3_MOE_COOP_WIDE=0"
)

for cfg in "${CONFIGS[@]}"; do
  envargs=()
  [[ -n "$cfg" ]] && envargs+=(-e "$cfg")
  echo "=== config: ${cfg:-defaults} ==="
  docker run --rm -i \
    --name "m31-lab-b2-$$" \
    --gpus all --ipc host --shm-size 2g \
    --memory "$MEM" --memory-swap "$MEM" --oom-score-adj 1000 \
    --security-opt seccomp=unconfined \
    -e TRITON_CACHE_DIR=/cache/triton -e LAB_BUDGET="${LAB_BUDGET:-6000}" \
    -e LAB_ROWS="${LAB_ROWS:-1,4,16,24}" -e PYTHONUNBUFFERED=1 \
    "${envargs[@]}" \
    -v "$LAB_DIR":/lab:rw \
    -v "$SRC/vllm-exl3/src":/src/vllm-exl3/src:ro \
    -v "$MODELS":/models:ro \
    -v "$CACHE":/cache:rw \
    --entrypoint python3 "$IMAGE" /lab/benchmarks/b2_coop_knobs.py 2>&1 | grep -v "^W1[0-9]"
done
