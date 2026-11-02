#!/usr/bin/env bash
# Phase 3 final pass: the B3 decode axis with 90-second windows, and the natural-text
# long prefill, measured in the SAME boot of each of two cells to keep the boot count down.
#
#   kernel-lab/phase3/run_final_pass.sh
#
# Cells: V0 (shipped baseline) and V1 (A3 on). B3 is toggled by M31_EXPERIMENT_B3; A3 by
# the image plus M31_EXPERIMENT_A3. Order alternates.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_IMG="m31-flash-exl3-tp2:053c634d-8d9c23f6"
A3_IMG="m31-flash-exl3-tp2:053c634d-8d9c23f6-a3"
REC="$ROOT/kernel-lab/phase3/records"
cd "$ROOT"

cell() {
  local name="$1" img="$2" a3="$3" b3="$4"
  local dir="$REC/final-${name}"
  mkdir -p "$dir"
  echo "### final $name image=$img A3=$a3 B3=$b3 $(date -Is)" | tee -a "$REC/session.log"
  ./m31.sh stop >>"$dir/stop.log" 2>&1; sleep 20
  IMAGE="$img" M31_EXPERIMENT_A3="$a3" M31_EXPERIMENT_B3="$b3" ./m31.sh start >"$dir/start.log" 2>&1
  local rc=$?
  echo "start rc=$rc wall=$(date +%s)" >> "$dir/meta"
  (( rc == 0 )) || { echo "final $name FAILED TO START" | tee -a "$REC/session.log"; return 1; }
  docker exec m31-flash-tp2 env 2>/dev/null | grep -E "MOE_COOP_WIDE|PLANNER_V2" | sort > "$dir/env"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader >>"$dir/telemetry.log"
  ./m31.sh bench decode --conc 1,4 --types code,prose --seconds 90 --out "$dir/decode90" >"$dir/decode90.log" 2>&1
  echo "decode90 rc=$?" >> "$dir/meta"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader >>"$dir/telemetry.log"
  python3 kernel-lab/phase3/natural_prefill.py --tokens 131072 --reps 2 \
      --out "$dir/natural" >"$dir/natural.log" 2>&1
  echo "natural rc=$?" >> "$dir/meta"
  python3 kernel-lab/phase3/mixed_scenario.py --streams 4 --seconds 75 --prefill-words 98304 \
      --out "$dir/mixed" >"$dir/mixed.log" 2>&1
  echo "mixed rc=$?" >> "$dir/meta"
  { echo "cell=$name image=$img A3=$a3 B3=$b3 kind=final"
    docker image inspect "$img" --format 'image_id={{.Id}}'
    cat "$dir/meta"; } > "$dir/cell.json"
  echo "### final $name done $(date -Is)" | tee -a "$REC/session.log"
}

cell V0 "$BASE_IMG" 0 1
cell V1 "$A3_IMG"   1 1
cell V2 "$BASE_IMG" 0 0

echo "### restoring shipped baseline $(date -Is)" | tee -a "$REC/session.log"
./m31.sh stop >>"$REC/restore.log" 2>&1; sleep 20
./m31.sh start >"$REC/restore.log" 2>&1
echo "restore rc=$?" | tee -a "$REC/session.log"
