#!/usr/bin/env bash
# Confirmacao dos candidatos de drafter: janelas de 30s, duas amostras por config.
set -uo pipefail
cd /home/rodman/llm/MiniMax-M3.1-Flash-EXL3-2x-DGX-Spark
OUT=${OUT:-/tmp/dflash-confirm}; mkdir -p "$OUT"

derive_captures() {
  python3 - "$1" <<'PY'
import sys
k = int(sys.argv[1]); rows = 6 * (k + 1)
base = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96]
sizes = [s for s in base if s <= rows]
if not sizes or sizes[-1] != rows:
    sizes.append(rows)
print(",".join(str(s) for s in sorted(set(sizes))))
PY
}

parse() {
  python3 - "$1" "$2" <<'PY'
import re, sys
dec, spec = open(sys.argv[1]).read(), open(sys.argv[2]).read()
cells = {}
for line in dec.splitlines():
    m = re.match(r"\s*(c=\d+)\s+(\w+)\s+([\d.]+) tok/s agg\s+per-stream\s+([\d.]+)", line)
    if m: cells[(m.group(1), m.group(2))] = m.group(3)
acc = {}
for line in spec.splitlines():
    m = re.match(r"\s*(\w+)\s+accepted/step ([\d.]+)\s+accept ([\d.]+)", line)
    if m: acc[m.group(1)] = m.group(2)
print("c1p=%s c1c=%s c4p=%s c4c=%s accp=%s accc=%s" % (
    cells.get(("c=1", "prose"), "-"), cells.get(("c=1", "code"), "-"),
    cells.get(("c=4", "prose"), "-"), cells.get(("c=4", "code"), "-"),
    acc.get("prose", "-"), acc.get("code", "-")))
PY
}

run() {   # rotulo  SPEC_METHOD  k  [captures]
  local tag=$1 method=$2 k=$3 caps=${4:-}
  [[ -n "$caps" ]] || caps=$(derive_captures "$k")
  echo "=== $tag (method=$method k=$k caps=$caps)"
  SPEC_METHOD="$method" NUM_SPECULATIVE_TOKENS="$k" CAPTURE_SIZES="$caps" ./m31.sh restart > "$OUT/restart-$tag.log" 2>&1
  if ! grep -q READY "$OUT/restart-$tag.log"; then echo "$tag BOOT FAIL"; tail -3 "$OUT/restart-$tag.log"; return 1; fi
  SPEC_METHOD="$method" ./m31.sh bench decode --conc 1,4 --types prose,code --seconds 30 > "$OUT/decode-$tag.log" 2>&1
  SPEC_METHOD="$method" ./m31.sh bench spec > "$OUT/spec-$tag.log" 2>&1
  printf '%-22s %s\n' "$tag" "$(parse "$OUT/decode-$tag.log" "$OUT/spec-$tag.log")" | tee -a "$OUT/summary.txt"
}

run dflash-k2 dflash 2
run dflash-k3 dflash 3
run dspark-k3 dspark 3 "1,2,3,4,8,12,16,24"
run dflash-k2b dflash 2
run dflash-k4x dflash 4 "1,2,3,4,6,8,12,16,20,24,30"
echo "CONFIRM DONE"
