"""Shared harness for the Experiment B decode-MoE benches: real pack weights, the
production exl3_moe_coop_mk call, routing families, a reconstruct+GEMM reference and a
measured streaming-bandwidth probe.

Kept separate from the bench scripts so b_coop_decode.py (the baseline table) and
b2_coop_knobs.py (the tuning-surface sweep) drive exactly the same fixture.
"""
from __future__ import annotations

import os

import torch
import exllamav3_ext

DEV = "cuda"
HIDDEN, INTER = 6144, 3072
TOPK = 4
ACT = 3              # MOE_COOP_ACT_SILU_OAI
LIMIT = 7.0          # swiglu_limit from the pack config
SLOTS_MAX = 256      # _COOP_SLOTS_MAX in the plugin
LAYER = int(os.environ.get("LAB_LAYER", "10"))


def temps(dev=DEV):
    """The plugin's cached coop scratch (_get_coop_temps in vllm_exl3/exl3.py)."""
    had_g = torch.empty((SLOTS_MAX, HIDDEN), dtype=torch.float16, device=dev)
    had_u = torch.empty_like(had_g)
    gu_g = torch.empty((SLOTS_MAX, 1, INTER), dtype=torch.float16, device=dev)
    gu_u = torch.empty_like(gu_g)
    act_out = torch.empty_like(gu_g)
    d_out = torch.empty((SLOTS_MAX, 1, HIDDEN), dtype=torch.float32, device=dev)
    return had_g, had_u, gu_g, gu_u, act_out, d_out


def ctr_for(dev=DEV, rows=1):
    return torch.zeros(
        SLOTS_MAX * (INTER // 128) + rows * (HIDDEN // 128) + 2 * SLOTS_MAX + 3,
        dtype=torch.int32, device=dev)


def routing(rows, family, n_exp, seed):
    """(rows, topk) expert ids. Families mirror what a TP=2 rank really sees.

    indep    every row routes independently (the pessimistic bound on distinct experts)
    corr0.5  a speculative block whose rows keep about half of row 0's experts, which is
             what a DFlash2 block of k+1 near-duplicate tokens actually looks like
    zipf     skewed router marginals, with duplicate picks inside a row
    hot4     every row on the same 4 experts (concentration adversary)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if family == "hot4":
        base = torch.randperm(n_exp, generator=g)[:4]
        sel = base[None, :].repeat(rows, 1)
    elif family == "indep":
        sel = torch.stack([torch.randperm(n_exp, generator=g)[:TOPK] for _ in range(rows)])
    elif family == "corr0.5":
        base = torch.randperm(n_exp, generator=g)[:TOPK]
        out = [base]
        for _ in range(rows - 1):
            keep = (torch.rand(TOPK, generator=g) < 0.5)
            rep = torch.randperm(n_exp, generator=g)[:TOPK]
            out.append(torch.where(keep, base, rep))
        sel = torch.stack(out)
    elif family == "zipf":
        w = 1.0 / torch.pow(torch.arange(1, n_exp + 1, dtype=torch.float64), 0.9)
        idx = torch.multinomial(w, rows * TOPK, replacement=True, generator=g)
        sel = idx.reshape(rows, TOPK).to(torch.int64)
    else:
        raise ValueError(family)
    return sel.to(DEV).contiguous()


def k_order(layer, n_exp):
    """The plugin's renumbering: experts in ascending (gate, up, down) K-tuple order, so
    the kernel -- which takes runs in descending expert order -- starts the heavy experts
    first (vllm_exl3/exl3.py, layer._exl3_coop_mk)."""
    ks = list(zip(layer["proj"]["gate"]["ks"], layer["proj"]["up"]["ks"],
                  layer["proj"]["down"]["ks"]))
    return sorted(range(n_exp), key=lambda i: ks[i])


def permute_proj(p, order):
    tr = [p["trellis"][i] for i in order]
    su = [p["suh"][i] for i in order]
    sv = [p["svh"][i] for i in order]
    ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=DEV)
    return {"trellis": tr, "suh": su, "svh": sv, "ptrs": (ptr(tr), ptr(su), ptr(sv)),
            "k": p["k"][order].contiguous(), "ks": [p["ks"][i] for i in order],
            "k_lo": p["k_lo"], "k_hi": p["k_hi"], "mul1": p["mul1"], "mcg": p["mcg"],
            "bytes": p["bytes"]}


def k_range(layer):
    """Production passes min/max of the K values over ALL THREE projections: the mk
    instance is chosen once per stage and reads each expert's K from the tables."""
    vals = [p["k_lo"] for p in layer["proj"].values()] + [p["k_hi"] for p in layer["proj"].values()]
    return min(vals), max(vals)


def call_mk(layer, x, sel, rw, tp, ctr, out, n_exp):
    g, u, d = (layer["proj"][k] for k in ("gate", "up", "down"))
    k_lo, k_hi = k_range(layer)
    exllamav3_ext.exl3_moe_coop_mk(
        x, sel, rw, 0, int(n_exp), HIDDEN,
        *g["ptrs"], *u["ptrs"], *d["ptrs"],
        None, None, None,
        g["k"], u["k"], d["k"], int(k_lo), int(k_hi),
        g["mcg"], g["mul1"], ACT, LIMIT, True,
        tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], ctr, out,
        None, None,
    )


def reference(layer, x, sel, rw):
    """The same linear the dense path evaluates: had_r_128(x*suh) @ W, then
    had_r_128(y*svh), with W reconstructed from the trellis (ext.reconstruct), SwiGLU-OAI
    clamped at 7.0, and the down projection fed fp16 like the kernel feeds it."""
    rows = x.shape[0]
    acc = torch.zeros((rows, HIDDEN), dtype=torch.float32, device=DEV)
    cache: dict[int, dict] = {}
    for r in range(rows):
        for j in range(TOPK):
            e = int(sel[r, j].item())
            if e not in cache:
                cache[e] = {}
                for name in ("gate", "up", "down"):
                    p = layer["proj"][name]
                    t = p["trellis"][e]
                    W = torch.empty((t.shape[0] * 16, t.shape[1] * 16), dtype=torch.half,
                                    device=DEV)
                    exllamav3_ext.reconstruct(W, t, p["ks"][e], p["mcg"], p["mul1"])
                    cache[e][name] = W
            xs = x[r:r + 1]

            def lin(xx, name, W):
                p = layer["proj"][name]
                tmp = torch.empty_like(xx)
                exllamav3_ext.had_r_128(xx, tmp, p["suh"][e], None, 1.0)
                y = torch.empty((xx.shape[0], W.shape[1]), dtype=torch.half, device=DEV)
                exllamav3_ext.hgemm(tmp, W, y)
                exllamav3_ext.had_r_128(y, y, None, p["svh"][e], 1.0)
                return y.float()

            gv = lin(xs, "gate", cache[e]["gate"])
            uv = lin(xs, "up", cache[e]["up"])
            gcl = gv.clamp(max=LIMIT)
            ucl = uv.clamp(-LIMIT, LIMIT)
            act = ((ucl + 1.0) * gcl * torch.sigmoid(1.702 * gcl)).half()
            dn = lin(act, "down", cache[e]["down"])
            acc[r] += (float(rw[r, j].item()) * dn).reshape(-1)
    return acc


def stream_bandwidth():
    """The node's practical streaming read bandwidth, measured on this run, not quoted
    from a spec sheet. torch.sum over 512 MiB is a pure read."""
    from labutil import timeit
    n = 512 * 1024 * 1024 // 2
    a = torch.empty(n, dtype=torch.float16, device=DEV)
    a.normal_()
    t = timeit(lambda: torch.sum(a), reps=20, warmup=10)
    gb = n * 2 / 1e9
    out = {"read_GBps": round(gb / (t["median_us"] / 1e6), 1),
           "bytes_MiB": round(n * 2 / 2**20), "median_us": t["median_us"],
           "spread_pct": t["spread_pct"]}
    del a
    torch.cuda.empty_cache()
    return out


def k_sorted_layer(lay, n_exp):
    order = k_order(lay, n_exp)
    inv = [0] * n_exp
    for new, old in enumerate(order):
        inv[old] = new
    lay_k = dict(lay)
    lay_k["proj"] = {k: permute_proj(lay["proj"][k], order) for k in lay["proj"]}
    return lay_k, inv
