#!/usr/bin/env bash
# Phase 3 end-to-end matrix on the real TP=2 pair. One boot per cell, benches against the
# live OpenAI endpoint, raw records under kernel-lab/phase3/records/.
#
#   kernel-lab/phase3/run_matrix.sh [round]
#
# Cells (the 2x2 over the two switches):
#   V0  base image,    A3 off, B3 on   <- the shipped baseline (B3 has been on since v0.1.0)
#   V1  overlay image, A3 on,  B3 on
#   V2  base image,    A3 off, B3 off  <- library auto tile
#   V3  overlay image, A3 on,  B3 off
#
# Round 1 runs the cells in order V0 V1 V2 V3; round 2 runs them reversed, so no cell is
# always measured first or last. The stack is always left on the shipped baseline.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_IMG="m31-flash-exl3-tp2:053c634d-8d9c23f6"
A3_IMG="m31-flash-exl3-tp2:053c634d-8d9c23f6-a3"
REC="$ROOT/kernel-lab/phase3/records"
ROUND="${1:-1}"
mkdir -p "$REC"

tele() {
  { echo "=== $1 $(date -Is) ==="
    nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu,power.draw --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    awk '/MemAvailable|SwapFree/{print}' /proc/meminfo
  } >> "$TELE" 2>&1
}

cell() {
  local name="$1" img="$2" a3="$3" b3="$4" steps="$5"
  local dir="$REC/r${ROUND}-${name}"
  mkdir -p "$dir"
  OUT_DIR="$dir" TELE="$dir/telemetry.log"
  echo "### round $ROUND cell $name (image=$img A3=$a3 B3=$b3) $(date -Is)" | tee -a "$REC/session.log"

  ./m31.sh stop >>"$dir/stop.log" 2>&1
  sleep 20
  tele "pre-start"
  local t0; t0=$(date +%s)
  IMAGE="$img" M31_EXPERIMENT_A3="$a3" M31_EXPERIMENT_B3="$b3" ./m31.sh start >"$dir/start.log" 2>&1
  local rc=$? t1; t1=$(date +%s)
  echo "start rc=$rc wall=$((t1-t0))s" >> "$dir/meta"
  tele "post-start"
  if (( rc != 0 )); then
    echo "CELL $name FAILED TO START" | tee -a "$REC/session.log"
    tail -25 "$dir/start.log" | tee -a "$REC/session.log"
    return 1
  fi
  docker exec m31-flash-tp2 env 2>/dev/null | grep -E "PLANNER_V2|MOE_COOP_WIDE" | sort > "$dir/env"

  for step in $steps; do
    case "$step" in
      decode)  args=(bench decode --conc 1,4 --types prose,code --seconds 30) ;;
      prefill) args=(bench prefill --lengths 8k,16k,48k,128k) ;;
      p128)    args=(bench prefill --lengths 128k) ;;
      spec)    args=(bench spec --spec-types prose,code) ;;
      check)   args=(bench check) ;;
      *) continue ;;
    esac
    tele "pre-${step}"
    local s0; s0=$(date +%s)
    ./m31.sh "${args[@]}" --out "$dir/$step" >"$dir/$step.log" 2>&1
    echo "$step rc=$? wall=$(( $(date +%s) - s0 ))s" >> "$dir/meta"
    tele "post-${step}"
  done
  { echo "cell=$name image=$img A3=$a3 B3=$b3 round=$ROUND"
    docker image inspect "$img" --format 'image_id={{.Id}} patch_sha={{index .Config.Labels "local.m31.patch-sha256"}}'
    cat "$dir/meta"; } > "$dir/cell.json"
  echo "### cell $name done $(date -Is)" | tee -a "$REC/session.log"
}

R1_STEPS="decode prefill spec check"
R2_STEPS="decode p128 check"
R3_STEPS="decode"

if [[ "$ROUND" == "1" ]]; then
  cell V0 "$BASE_IMG" 0 1 "$R1_STEPS"
  cell V1 "$A3_IMG"   1 1 "$R1_STEPS"
  cell V2 "$BASE_IMG" 0 0 "$R1_STEPS"
  cell V3 "$A3_IMG"   1 0 "$R1_STEPS"
elif [[ "$ROUND" == "2" ]]; then
  cell V3 "$A3_IMG"   1 0 "$R2_STEPS"
  cell V2 "$BASE_IMG" 0 0 "$R2_STEPS"
  cell V1 "$A3_IMG"   1 1 "$R2_STEPS"
  cell V0 "$BASE_IMG" 0 1 "$R2_STEPS"
else
  # round 3: decode only, a third order, because round 1 showed decode spread of up to
  # 28% between two cells that share a B3 setting
  cell V1 "$A3_IMG"   1 1 "$R3_STEPS"
  cell V2 "$BASE_IMG" 0 0 "$R3_STEPS"
  cell V0 "$BASE_IMG" 0 1 "$R3_STEPS"
  cell V3 "$A3_IMG"   1 0 "$R3_STEPS"
fi

# Always leave the pair on the shipped baseline.
echo "### restoring shipped baseline $(date -Is)" | tee -a "$REC/session.log"
./m31.sh stop >>"$REC/restore.log" 2>&1
sleep 20
./m31.sh start >"$REC/restore.log" 2>&1
echo "restore rc=$?" | tee -a "$REC/session.log"
