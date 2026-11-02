#!/usr/bin/env python3
"""Experiment B2: the coop kernel's own tuning surface on GB10.

exl3_moe_coop latches three environment knobs once per process, so every configuration
needs its own process (kernel-lab/runner/sweep_coop_knobs.sh drives the sweep):

  EXL3_MOE_COOP_KSPLIT  forced split-k blocks per item (0 = the kernel picks per launch)
  EXL3_MOE_COOP_DSPLIT  cap on the device-picked split (default 8; 1 disables splitting)
  EXL3_MOE_COOP_WIDE    tile geometry: unset auto, 0 narrow, 1 wide (128-col, 4-way k-split)

The question this answers with numbers: is any of the remaining tuning surface worth
anything on this node, or is the kernel already pinned to the memory wall?
"""
import os
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/fixtures")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from labutil import timeit, write_result, env_record
import pack_weights as pw
import _coop_harness as ch

DEV = "cuda"
ROWS = [int(x) for x in os.environ.get("LAB_ROWS", "1,4,16,24").split(",")]
FAMILIES = ("indep", "corr0.5")


def main() -> int:
    torch.manual_seed(0)
    lay = pw.load_layer(ch.LAYER, DEV)
    n_exp = len(lay["experts"])
    lay_k, inv = ch.k_sorted_layer(lay, n_exp)
    tp = ch.temps(DEV)
    bw = ch.stream_bandwidth()
    rows_out = []
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
                       reps=30, warmup=20)
            distinct = sorted({int(e) for e in sel_nat.reshape(-1).tolist()})
            wb = pw.weight_bytes(lay, distinct)
            rows_out.append({
                "rows": rows, "family": family, "distinct": len(distinct),
                "weight_MiB": round(wb / 2**20, 1), "us": t["median_us"],
                "p10": t["p10_us"], "p90": t["p90_us"], "spread_pct": t["spread_pct"],
                "GBps": round(wb / 1e9 / (t["median_us"] / 1e6), 1),
                "pct_of_ceiling": round(
                    wb / 1e9 / bw["read_GBps"] * 1e6 / t["median_us"] * 100, 1),
            })
            print(f"rows {rows:3d} {family:9} {rows_out[-1]['us']:8.1f} us  "
                  f"{rows_out[-1]['GBps']:6.1f} GB/s  "
                  f"{rows_out[-1]['pct_of_ceiling']:5.1f}% of ceiling", flush=True)
    cfg = {k: os.environ.get(k) for k in
           ("EXL3_MOE_COOP_KSPLIT", "EXL3_MOE_COOP_DSPLIT", "EXL3_MOE_COOP_WIDE")}
    tag = "_".join(f"{k.rsplit('_', 1)[-1].lower()}{v}" if v is not None else "dflt"
                   for k, v in cfg.items())
    path = write_result(f"b2_knobs_{tag}", {"env": env_record(), "knobs": cfg,
                                            "bandwidth_probe": bw, "rows": rows_out})
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
