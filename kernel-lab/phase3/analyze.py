#!/usr/bin/env python3
"""Build the Phase 3 tables from the raw records under kernel-lab/phase3/records/.

Every number printed here is read from a results.json written by the repo's own bench
harness. Nothing is estimated. Where a cell is missing (a step that did not run, or a
cell that failed to start) the table says so instead of filling a gap.
"""
import glob
import json
import os
import statistics
import sys

REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records")
CELLS = ["V0", "V1", "V2", "V3"]
LABEL = {
    "V0": "V0 baseline (A3 off, B3 on = shipped)",
    "V1": "V1 (A3 on, B3 on)",
    "V2": "V2 (A3 off, B3 off)",
    "V3": "V3 (A3 on, B3 off)",
}


def load(cell, step):
    out = []
    for d in sorted(glob.glob(os.path.join(REC, f"r*-{cell}", step, "results.json"))) + \
                  sorted(glob.glob(os.path.join(REC, f"final-{cell}", step, "results.json"))):
        try:
            out.append(json.load(open(d)))
        except Exception:
            pass
    return out


def decode_table():
    print("## Decode (tok/s; higher is better). 'w' = measurement window in seconds\n")
    print("| cell | conc | type | agg tok/s per sample (w) | median | median (90 s only) | per-stream med | TTFT med s | errors |")
    print("|---|---|---|---|---|---|---|---|---|")
    agg = {}
    for cell in CELLS:
        for d in sorted(glob.glob(os.path.join(REC, f"r*-{cell}", "decode", "results.json"))) + \
                  sorted(glob.glob(os.path.join(REC, f"final-{cell}", "decode90", "results.json"))):
            res = json.load(open(d))
            w = 90 if "decode90" in d else 30
            for c in res["result"]["cells"]:
                agg.setdefault((cell, c["concurrency"], c["prompt_type"]), []).append(
                    (c["agg_tok_s"], w, c))
    for k in sorted(agg):
        cell, conc, typ = k
        cs = agg[k]
        vals = [v for v, w, c in cs]
        long90 = [v for v, w, c in cs if w == 90]
        agg[k] = statistics.median(vals)
        print(f"| {cell} | {conc} | {typ} | {['%.1f/%ds' % (v, w) for v, w, c in cs]} "
              f"| **{statistics.median(vals):.1f}** "
              f"| {('%.1f' % statistics.median(long90)) if long90 else '-'} "
              f"| {statistics.median([c['per_stream_tok_s_median'] for v, w, c in cs]):.1f} "
              f"| {statistics.median([c['ttft_s_median'] for v, w, c in cs]):.2f} "
              f"| {sum(c['errors'] for v, w, c in cs)} |")
    return agg


def prefill_table():
    print("\n## Prefill (tok/s and TTFT s)\n")
    print("| cell | prompt tok (median) | tok/s (rounds) | median tok/s | TTFT s median |")
    print("|---|---|---|---|---|")
    agg = {}
    ttft = {}
    for cell in CELLS:
        for res in load(cell, "prefill") + load(cell, "p128"):
            for r in res["result"]["rows"]:
                b = 8192 if r["prompt_tokens"] < 13000 else (16384 if r["prompt_tokens"] < 30000
                                                             else 49152 if r["prompt_tokens"] < 90000
                                                             else 131072 if r["prompt_tokens"] < 190000
                                                             else 196608)
                agg.setdefault((cell, b), []).append(r)
    for k in sorted(agg, key=lambda x: (x[1], x[0])):
        cell, b = k
        rs = agg[k]
        vals = [r["tok_s"] for r in rs]
        agg[k] = statistics.median(vals)
        ttft[k] = statistics.median([r["ttft_s"] for r in rs])
        print(f"| {cell} | {b} | {['%.0f' % v for v in vals]} | **{statistics.median(vals):.0f}** "
              f"| {ttft[k]:.2f} |")
    return agg, ttft


def ratios(dec, pre, ttft):
    print("\n## Isolated and combined effects\n")
    print("| metric | V0 | V1 (A3) | V2 (B3) | V3 (both) | A3 | B3 | both |")
    print("|---|---|---|---|---|---|---|---|")

    def row(label, get):
        v = {c: get(c) for c in CELLS}
        if any(v[c] is None for c in CELLS):
            cells = " | ".join("n/a" if v[c] is None else f"{v[c]:.1f}" for c in CELLS)
            print(f"| {label} | {cells} | " + " | ".join(
                "n/a" if v[c] is None or v["V0"] is None else f"{v[c] / v['V0']:.3f}"
                for c in ("V1", "V2", "V3")) + " |")
        else:
            print(f"| {label} | {v['V0']:.1f} | {v['V1']:.1f} | {v['V2']:.1f} | {v['V3']:.1f} "
                  f"| **{v['V1'] / v['V0']:.3f}** | **{v['V2'] / v['V0']:.3f}** "
                  f"| **{v['V3'] / v['V0']:.3f}** |")

    for conc in (1, 4):
        for typ in ("prose", "code"):
            if any((c, conc, typ) in dec for c in CELLS):
                row(f"decode c={conc} {typ} tok/s",
                    lambda c, conc=conc, typ=typ: dec.get((c, conc, typ)))
    for b in sorted({k[1] for k in pre}):
        if any((c, b) in pre for c in CELLS):
            row(f"prefill {b} tok/s", lambda c, b=b: pre.get((c, b)))
            row(f"prefill {b} TTFT s (lower is better)", lambda c, b=b: ttft.get((c, b)))


def nll_table():
    print("\n## NLL check (bench check)\n")
    for cell in CELLS:
        for d in sorted(glob.glob(os.path.join(REC, f"r*-{cell}", "check", "*.log"))):
            txt = open(d).read()
            keep = [ln for ln in txt.splitlines() if "NLL" in ln or "drift" in ln or "PASS" in ln or "FAIL" in ln]
            if keep:
                print(f"### {os.path.basename(os.path.dirname(os.path.dirname(d)))}")
                for ln in keep[:14]:
                    print("   ", ln.strip())


def spec_table():
    print("\n## Speculative acceptance (bench spec)\n")
    for cell in CELLS:
        rs = load(cell, "spec")
        if not rs:
            continue
        print(f"### {cell}")
        print(json.dumps(rs[-1]["result"], indent=1)[:700])


def extras():
    print("\n## Natural-text long prefill (real prose, not the bench's random-word filler)\n")
    print("| cell | prompt tokens | TTFT s | tok/s |")
    print("|---|---|---|---|")
    for cell in ("V0", "V1"):
        for d in sorted(glob.glob(os.path.join(REC, f"natural-{cell}", "results.json"))):
            for r in json.load(open(d))["rows"]:
                print(f"| {cell} | {r.get('prompt_tokens')} | {r.get('ttft_s')} | {r.get('tok_s')} |")
    print("\n## Mixed load: 4 decode streams with a ~136K-token prefill arriving mid-window\n")
    print("| cell | long-prefill TTFT s | decode agg tok/s in window | ITL before ms | ITL during ms | stall factor |")
    print("|---|---|---|---|---|---|")
    for cell in ("V0", "V1", "V2"):
        for d in sorted(glob.glob(os.path.join(REC, f"final-{cell}", "mixed", "results.json")) +
                        glob.glob(os.path.join(REC, "mixed-V1", "results.json"))):
            r = json.load(open(d))
            b, a = r["itl_before_median_ms"], r["itl_after_median_ms"]
            print(f"| {cell} | {r['prefill'].get('ttft_s')} | {r['decode_agg_tok_s']} | {b} | {a} "
                  f"| {a / b:.1f}x |")


def main():
    print(f"records: {REC}")
    for d in sorted(glob.glob(os.path.join(REC, "r*-*"))):
        cj = os.path.join(d, "cell.json")
        print(" ", os.path.basename(d), "->", open(cj).read().splitlines()[0] if os.path.exists(cj) else "incomplete")
    print()
    dec = decode_table()
    pre, ttft = prefill_table()
    ratios(dec, pre, ttft)
    nll_table()
    spec_table()
    extras()


if __name__ == "__main__":
    sys.exit(main())
