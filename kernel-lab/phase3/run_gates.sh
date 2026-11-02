#!/usr/bin/env bash
# Phase 3 quality and stability gates, run against the live pair for one cell.
#
#   kernel-lab/phase3/run_gates.sh <cell-name> <image> <A3> <B3>
#
# Nothing here is modified or re-toleranced: these are the repo's own gates, run as-is.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="$1"; IMG="$2"; A3="$3"; B3="$4"
REC="$ROOT/kernel-lab/phase3/records"; DIR="$REC/gates-${NAME}"
mkdir -p "$DIR"
cd "$ROOT"

echo "### gates $NAME image=$IMG A3=$A3 B3=$B3 $(date -Is)" | tee -a "$REC/session.log"
./m31.sh stop >>"$DIR/stop.log" 2>&1; sleep 20
IMAGE="$IMG" M31_EXPERIMENT_A3="$A3" M31_EXPERIMENT_B3="$B3" ./m31.sh start >"$DIR/start.log" 2>&1
rc=$?
echo "start rc=$rc" >> "$DIR/meta"
if (( rc != 0 )); then echo "GATES $NAME: start failed" | tee -a "$REC/session.log"; tail -20 "$DIR/start.log"; exit 1; fi
docker exec m31-flash-tp2 env 2>/dev/null | grep -E "PLANNER_V2|MOE_COOP_WIDE" | sort > "$DIR/env"

for step in check needle quality tools api-test; do
  case "$step" in
    check)    args=(bench check) ;;
    needle)   args=(bench needle --tokens 65536 --depths 10,50,90) ;;
    quality)  args=(quality --out "$DIR/quality.json") ;;
    tools)    args=(tools --out "$DIR/tools.json") ;;
    api-test) args=(api-test) ;;
  esac
  echo "--- $step $(date -Is)" >> "$DIR/meta"
  ./m31.sh "${args[@]}" >"$DIR/$step.log" 2>&1
  echo "$step rc=$?" >> "$DIR/meta"
done
{ echo "cell=$NAME image=$IMG A3=$A3 B3=$B3 kind=gates"
  docker image inspect "$IMG" --format 'image_id={{.Id}} patch_sha={{index .Config.Labels "local.m31.patch-sha256"}}'
  cat "$DIR/meta"; } > "$DIR/cell.json"
echo "### gates $NAME done $(date -Is)" | tee -a "$REC/session.log"
