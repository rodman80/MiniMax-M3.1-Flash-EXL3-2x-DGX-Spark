#!/usr/bin/env bash
# Phase 3: a decode-only, longer-window comparison aimed squarely at the B3 axis.
#
# The 30-second windows in the matrix showed decode spread of up to 28% between two cells
# that share a B3 setting, which is larger than the 5-11% the microbenchmark predicts.
# This run uses 90-second windows and repeats each cell, so the B3 effect (if any) has to
# clear the noise floor.
#
#   kernel-lab/phase3/run_b3_focus.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_IMG="m31-flash-exl3-tp2:053c634d-8d9c23f6"
REC="$ROOT/kernel-lab/phase3/records"
cd "$ROOT"

cell() {
  local name="$1" b3="$2" rep="$3"
  local dir="$REC/b3focus-${name}r${rep}"
  mkdir -p "$dir"
  echo "### b3focus $name rep $rep B3=$b3 $(date -Is)" | tee -a "$REC/session.log"
  ./m31.sh stop >>"$dir/stop.log" 2>&1; sleep 20
  IMAGE="$BASE_IMG" M31_EXPERIMENT_A3=0 M31_EXPERIMENT_B3="$b3" ./m31.sh start >"$dir/start.log" 2>&1
  local rc=$?
  echo "start rc=$rc" >> "$dir/meta"
  (( rc == 0 )) || { echo "b3focus $name failed to start" | tee -a "$REC/session.log"; return 1; }
  docker exec m31-flash-tp2 env 2>/dev/null | grep -E "MOE_COOP_WIDE|PLANNER_V2" | sort > "$dir/env"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader >> "$dir/telemetry.log"
  ./m31.sh bench decode --conc 1,4 --types code,prose --seconds 90 --out "$dir/decode" >"$dir/decode.log" 2>&1
  echo "decode rc=$?" >> "$dir/meta"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader >> "$dir/telemetry.log"
  { echo "cell=$name image=$BASE_IMG A3=0 B3=$b3 kind=b3focus rep=$rep"
    cat "$dir/meta"; } > "$dir/cell.json"
}

cell V0 1 1
cell V2 0 1
cell V0 1 2
cell V2 0 2

echo "### restoring shipped baseline $(date -Is)" | tee -a "$REC/session.log"
./m31.sh stop >>"$REC/restore.log" 2>&1; sleep 20
./m31.sh start >"$REC/restore.log" 2>&1
echo "restore rc=$?" | tee -a "$REC/session.log"
