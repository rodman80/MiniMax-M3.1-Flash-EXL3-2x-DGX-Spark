#!/usr/bin/env bash
# Run one kernel-lab script inside the serving image, on the local GB10, without touching
# the running m31-flash-tp2 container.
#
#   kernel-lab/runner/run.sh benchmarks/bench_sparse_attn.py [--args...]
#
# The container is memory-capped: on GB10 the GPU and the host share one pool, so a lab
# process that allocates freely can take the node down next to the serving container
# (AGENTS.md: the memory guard is mandatory). LAB_MEM caps the cgroup; the lab scripts
# themselves keep their working set under LAB_BUDGET (MiB) and refuse to exceed it.
#
# Nothing here writes to the serving container, the pack, or the weight snapshots.
set -uo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$LAB_DIR/.." && pwd)"
IMAGE="${LAB_IMAGE:-m31-flash-exl3-tp2:053c634d-8d9c23f6}"
SRC="${LAB_SRC:-/home/rodman/build/m31/src}"
MODELS="${LAB_MODELS:-/home/rodman/models}"
CACHE="${LAB_CACHE:-/home/rodman/.cache/m31/lab}"
MEM="${LAB_MEM:-10g}"
LAB="${1:?usage: run.sh <script.py> [args...]}"; shift

mkdir -p "$CACHE/triton"

avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if (( avail_kb < 8 * 1024 * 1024 )); then
  echo "run.sh: MemAvailable $((avail_kb/1024)) MiB < 8 GiB floor; refusing to start a GPU process" >&2
  exit 1
fi

exec docker run --rm -i \
  --name "m31-kernel-lab-$$" \
  --gpus all --ipc host --shm-size 2g \
  --memory "$MEM" --memory-swap "$MEM" --oom-score-adj 1000 \
  --security-opt seccomp=unconfined \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e LAB_BUDGET="${LAB_BUDGET:-6000}" \
  -e PYTHONUNBUFFERED=1 \
  -v "$LAB_DIR":/lab:rw \
  -v "$SRC/vllm-exl3/src":/src/vllm-exl3/src:ro \
  -v "$SRC/exllamav3/exllamav3":/src/exllamav3:ro \
  -v "$MODELS":/models:ro \
  -v "$CACHE":/cache:rw \
  --entrypoint python3 \
  "$IMAGE" /lab/"$LAB" "$@"
