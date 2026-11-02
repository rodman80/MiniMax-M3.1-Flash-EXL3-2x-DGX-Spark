#!/usr/bin/env python3
"""Experiment B: EXL3 mixed-K cooperative MoE decode on the pack's real weights.

Baseline is the production path: one exl3_moe_coop_mk launch per stage, with the pack's
own trellis widths (K 2 and 3), mul1 codebook, fp16 scales and all 128 experts of one
MiniMax-M3.1 MoE layer (hidden 6144, intermediate 3072, topk 4).

Per (rows, routing family) the table reports the launch time, the number of DISTINCT
experts the step touches, the weight bytes those experts cost, the effective GB/s and
how close that is to this node's measured streaming read bandwidth. That last column is
the one that decides whether a kernel-level speedup is even possible: at 90% of the
memory wall, a 20% faster kernel cannot exist without reading fewer bytes.

Rows are the spec-verify batch shapes DFlash2 k=3 can produce: 1 (no spec / single
token), 4 (one stream at k=3), 8, 16, 24 (the coop eligibility cap, 6 streams at k=3)
and 32 (beyond the cap, where production falls back to exl3_moe_mixedk).
"""
import os
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/fixtures")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import labutil
from labutil import timeit, rel_err, write_result, emit, env_record
import pack_weights as pw
import _coop_harness as ch

DEV = "cuda"
ROWS = [1, 2, 4, 8, 16, 24, 32]
FAMILIES = ("indep", "corr0.5", "zipf", "hot4")


def main() -> int:
    torch.manual_seed(0)
    lay = pw.load_layer(ch.LAYER, DEV)
    n_exp = len(lay["experts"])
    tot = sum(p["bytes"] for p in lay["proj"].values())
    print(f"layer {ch.LAYER}: {n_exp} experts, {tot / 2**20:.0f} MiB trellis (+scales); "
          f"K gate {sorted(set(lay['proj']['gate']['ks']))} "
          f"up {sorted(set(lay['proj']['up']['ks']))} "
          f"down {sorted(set(lay['proj']['down']['ks']))}; "
          f"coop K range {ch.k_range(lay)}", flush=True)
    bw = ch.stream_bandwidth()
    print(f"measured streaming read bandwidth: {bw['read_GBps']} GB/s "
          f"(spread {bw['spread_pct']}%)", flush=True)

    tp = ch.temps(DEV)
    lay_k, inv = ch.k_sorted_layer(lay, n_exp)

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

            ch.call_mk(lay_k, x, sel_k, rw, tp, ctr, out, n_exp)
            torch.cuda.synchronize()
            t_mk = timeit(lambda: ch.call_mk(lay_k, x, sel_k, rw, tp, ctr, out, n_exp),
                          reps=30, warmup=20)
            t_nat = timeit(lambda: ch.call_mk(lay, x, sel_nat, rw, tp, ctr, out, n_exp),
                           reps=30, warmup=20)

            distinct = sorted({int(e) for e in sel_nat.reshape(-1).tolist()})
            wb = pw.weight_bytes(lay, distinct)
            ceil_us = wb / 1e9 / bw["read_GBps"] * 1e6
            rec = {
                "rows": rows, "family": family, "distinct_experts": len(distinct),
                "weight_MiB": round(wb / 2**20, 1),
                "coop_mk_us": t_mk["median_us"], "p10": t_mk["p10_us"], "p90": t_mk["p90_us"],
                "spread_pct": t_mk["spread_pct"],
                "coop_natural_us": t_nat["median_us"],
                "k_order_gain": round(t_nat["median_us"] / t_mk["median_us"], 3),
                "effective_GBps": round(wb / 1e9 / (t_mk["median_us"] / 1e6), 1),
                "ceiling_us": round(ceil_us, 1),
                "pct_of_ceiling": round(ceil_us / t_mk["median_us"] * 100, 1),
                "us_per_expert": round(t_mk["median_us"] / max(len(distinct), 1), 1),
            }
            if rows <= 4:
                ref = ch.reference(lay, x, sel_nat, rw)
                out.zero_()
                ch.call_mk(lay_k, x, sel_k, rw, tp, ctr, out, n_exp)
                torch.cuda.synchronize()
                e = rel_err(out, ref)
                rec["ref_rel_l2"] = e["rel_l2"]
                rec["ref_max_abs"] = e["max_abs"]
            rows_out.append(rec)
            print(f"rows {rows:3d} {family:9} distinct {len(distinct):3d} "
                  f"{wb / 2**20:8.1f} MiB  coop {t_mk['median_us']:8.1f} us  "
                  f"{rec['effective_GBps']:6.1f} GB/s  ceiling {ceil_us:7.1f} us  "
                  f"{rec['pct_of_ceiling']:5.1f}%  K-order {rec['k_order_gain']:.3f}"
                  + (f"  rel {e['rel_l2']:.1e}" if rows <= 4 else ""), flush=True)
            del ctr, out
            torch.cuda.empty_cache()

    path = write_result("b_coop_decode", {
        "env": env_record(), "layer": ch.LAYER, "hidden": ch.HIDDEN, "inter": ch.INTER,
        "topk": ch.TOPK, "bandwidth_probe": bw, "rows": rows_out,
    })
    emit({"rows": rows_out})
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
