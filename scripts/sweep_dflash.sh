#!/usr/bin/env bash
# Sweep do drafter DFlash2: K e capture sizes derivados, decode c=1/c=4 + aceite.
# Caller env vence .env, entao cada config e passada por variavel, sem editar .env.
set -uo pipefail
cd /home/rodman/llm/MiniMax-M3.1-Flash-EXL3-2x-DGX-Spark
OUT=${OUT:-/tmp/dflash-sweep}; KS=${KS:-"1 2 4 5 7"}
mkdir -p "$OUT"
TSV="$OUT/results.tsv"
[[ -f "$TSV" ]] || printf 'k\tcaptures\tc1_prose\tc1_code\tc4_prose\tc4_code\tacc_prose\tacc_code\tlen_prose\tlen_code\tready_s\trc\n' > "$TSV"

derive_captures() {   # mesma derivacao de m31.sh: rows = MAX_NUM_SEQS x (k+1)
  python3 - "$1" "${SEQS:-6}" <<'PY'
import sys
k, seqs = int(sys.argv[1]), int(sys.argv[2])
rows = seqs * (k + 1)
base = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96]
sizes = [s for s in base if s <= rows]
if not sizes or sizes[-1] != rows:
    sizes.append(rows)
print(",".join(str(s) for s in sorted(set(sizes))))
PY
}

parse_cell() {        # decode log -> 4 numeros; spec log -> aceite/comprimento
  python3 - "$1" "$2" <<'PY'
import re, sys
dec, spec = open(sys.argv[1]).read(), open(sys.argv[2]).read()
def grab(text, pat):
    m = re.search(pat, text)
    return m.group(1) if m else "-"
cells = {}
for line in dec.splitlines():
    m = re.match(r"\s*(c=\d+)\s+(\w+)\s+([\d.]+) tok/s agg\s+per-stream\s+([\d.]+)", line)
    if m:
        cells[(m.group(1), m.group(2))] = m.group(3)
acc, ln = {}, {}
for line in spec.splitlines():
    m = re.match(r"\s*(\w+)\s+accepted/step ([\d.]+)\s+accept ([\d.]+)", line)
    if m:
        acc[m.group(1)] = round(float(m.group(2)), 3)
        ln[m.group(1)] = round(float(m.group(2)) + 1, 2)
print("\t".join([
    cells.get(("c=1", "prose"), "-"), cells.get(("c=1", "code"), "-"),
    cells.get(("c=4", "prose"), "-"), cells.get(("c=4", "code"), "-"),
    acc.get("prose", "-"), acc.get("code", "-"), ln.get("prose", "-"), ln.get("code", "-"),
]))
PY
}

for k in $KS; do
  caps=$(derive_captures "$k")
  echo "=== k=$k captures=$caps"
  NUM_SPECULATIVE_TOKENS="$k" CAPTURE_SIZES="$caps" ./m31.sh restart > "$OUT/restart-k$k.log" 2>&1
  rc=$?
  ready=$(sed -n 's/.*READY.* in \([0-9.]*\)s.*/\1/p' "$OUT/restart-k$k.log" | tail -1)
  if (( rc != 0 )); then
    printf '%s\t%s\tBOOT_FAIL\t-\t-\t-\t-\t-\t-\t-\t%s\t%s\n' "$k" "$caps" "${ready:--}" "$rc" >> "$TSV"
    echo "k=$k FALHOU (rc=$rc)"; tail -3 "$OUT/restart-k$k.log"; continue
  fi
  ./m31.sh bench decode --conc 1,4 --types prose,code --seconds 20 > "$OUT/decode-k$k.log" 2>&1
  ./m31.sh bench spec > "$OUT/spec-k$k.log" 2>&1
  printf '%s\t%s\t%s\t%s\n' "$k" "$caps" "$(parse_cell "$OUT/decode-k$k.log" "$OUT/spec-k$k.log")" "${ready:-?}" "$rc" >> "$TSV"
  echo "k=$k ok"
done
echo "SWEEP DONE"
column -t "$TSV"
