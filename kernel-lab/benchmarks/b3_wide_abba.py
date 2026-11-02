#!/usr/bin/env python3
"""Experiment B3: A/B/A on EXL3_MOE_COOP_WIDE, with a numerics check.

The B2 sweep showed the wide (128-column, 4-way k-split) tile beating the auto choice at
the C1 decode shapes. One sweep is not evidence: the node drifts, and the serving
container shares its memory bandwidth. This script runs one configuration per process
and the driver runs it in the order auto, wide=1, auto, wide=1, so a real effect has to
show up twice and the drift has to show up in between.

Numerics: the wide tile changes the k-split, so the accumulation order changes. The
check is not "wide equals auto bit for bit" (it will not), it is "wide is as close to
the reconstruct+GEMM reference as auto is", which is the standard the repo already uses
for the coop path (patches/0011: coop vs fp32 6.4-6.5e-4).
"""
import os
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/fixtures")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from labutil import timeit, rel_err, write_result, env_record
import pack_weights as pw
import _coop_harness as ch

DEV = "cuda"
ROWS = [int(x) for x in os.environ.get("LAB_ROWS", "1,4,8,16,24").split(",")]
FAMILIES = ("indep", "corr0.5")


def main() -> int:
    torch.manual_seed(0)
    lay = pw.load_layer(ch.LAYER, DEV)
    n_exp = len(lay["experts"])
    lay_k, inv = ch.k_sorted_layer(lay, n_exp)
    tp = ch.temps(DEV)
    bw = ch.stream_bandwidth()
    out_rows = []
    for rows in ROWS:
        for family in FAMILIES:
            sel_nat = ch.routing(rows, family, n_exp, 1000 + rows)
            rw = (torch.rand((rows, ch.TOPK), generator=torch.Generator().manual_seed(7))
                  .to(DEV).half() * 0.5 + 0.25)
            x = (torch.randn((rows, ch.HIDDEN), device=DEV) * 0.05).half().contiguous()
            sel_k = torch.tensor([[inv[int(e)] for e in r] for r in sel_nat.tolist()],
                                 dtype=torch.int64, device=DEV).contiguous()
            ctr = ch.ctr_for(DEV, rows)
            out = torch.zeros((rows, ch.HIDDEN), dtype=torch.float32, device=DEV)
            t = timeit(lambda: ch.call_mk(lay_k, x, sel_k, rw, tp, ctr, out, n_exp),
                       reps=60, warmup=30)
            distinct = sorted({int(e) for e in sel_nat.reshape(-1).tolist()})
            wb = pw.weight_bytes(lay, distinct)
            rec = {
                "rows": rows, "family": family, "distinct": len(distinct),
                "weight_MiB": round(wb / 2**20, 1), "us": t["median_us"],
                "p10": t["p10_us"], "p90": t["p90_us"], "spread_pct": t["spread_pct"],
                "GBps": round(wb / 1e9 / (t["median_us"] / 1e6), 1),
                "pct_of_ceiling": round(
                    wb / 1e9 / bw["read_GBps"] * 1e6 / t["median_us"] * 100, 1),
            }
            if rows <= 8:
                ref = ch.reference(lay, x, sel_nat, rw)
                out.zero_()
                ch.call_mk(lay_k, x, sel_k, rw, tp, ctr, out, n_exp)
                torch.cuda.synchronize()
                e = rel_err(out, ref)
                rec["ref_rel_l2"] = e["rel_l2"]
                rec["ref_max_abs"] = e["max_abs"]
            out_rows.append(rec)
            print(f"rows {rows:3d} {family:9} {rec['us']:8.1f} us  "
                  f"{rec['GBps']:6.1f} GB/s  {rec['pct_of_ceiling']:5.1f}% ceiling  "
                  f"spread {rec['spread_pct']:4.1f}%"
                  + (f"  rel {e['rel_l2']:.2e}" if rows <= 8 else ""), flush=True)
    wide = os.environ.get("EXL3_MOE_COOP_WIDE")
    tag = f"wide{wide}" if wide is not None else "auto"
    path = write_result(f"b3_wide_{tag}_{os.environ.get('LAB_PASS', 'p1')}", {
        "env": env_record(), "wide": wide, "bandwidth_probe": bw, "rows": out_rows})
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
