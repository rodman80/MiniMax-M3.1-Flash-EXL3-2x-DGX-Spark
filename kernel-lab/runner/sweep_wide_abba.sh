#!/usr/bin/env bash
# Experiment B3 driver: A/B/A/B over EXL3_MOE_COOP_WIDE, one process per pass because the
# knob is latched once per process. A real effect must appear in both wide passes and the
# two auto passes must agree with each other.
set -uo pipefail
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LAB_IMAGE:-m31-flash-exl3-tp2:053c634d-8d9c23f6}"
SRC="${LAB_SRC:-/home/rodman/build/m31/src}"
MODELS="${LAB_MODELS:-/home/rodman/models}"
CACHE="${LAB_CACHE:-/home/rodman/.cache/m31/lab}"
MEM="${LAB_MEM:-10g}"
mkdir -p "$CACHE/triton"

PASSES=("auto:1" "wide1:2" "auto:3" "wide1:4")
for p in "${PASSES[@]}"; do
  wide="${p%%:*}"; pass="${p##*:}"
  envargs=()
  [[ "$wide" == "wide1" ]] && envargs+=(-e EXL3_MOE_COOP_WIDE=1)
  echo "=== pass $pass: EXL3_MOE_COOP_WIDE=${wide/auto/<unset>} ==="
  docker run --rm -i \
    --name "m31-lab-b3-$$" \
    --gpus all --ipc host --shm-size 2g \
    --memory "$MEM" --memory-swap "$MEM" --oom-score-adj 1000 \
    --security-opt seccomp=unconfined \
    -e TRITON_CACHE_DIR=/cache/triton -e LAB_BUDGET="${LAB_BUDGET:-6000}" \
    -e LAB_ROWS="${LAB_ROWS:-1,4,8,16,24}" -e LAB_PASS="$pass" -e PYTHONUNBUFFERED=1 \
    "${envargs[@]}" \
    -v "$LAB_DIR":/lab:rw \
    -v "$SRC/vllm-exl3/src":/src/vllm-exl3/src:ro \
    -v "$MODELS":/models:ro \
    -v "$CACHE":/cache:rw \
    --entrypoint python3 "$IMAGE" /lab/benchmarks/b3_wide_abba.py 2>&1 | grep -v "^W1[0-9]"
done
