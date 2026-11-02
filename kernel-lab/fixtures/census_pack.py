#!/usr/bin/env python3
"""Read the pack's safetensors HEADERS only and record the real per-expert EXL3 geometry.

No tensor bytes are read: the header of each shard is a JSON map of dtype/shape/offsets.
This is what pins the fixtures to the pack instead of a generic GEMM: trellis shape
[k//16, n//16, 16*K] gives the trellis width K per expert, and the suh/svh dtypes give
the codebook scales. Output: fixtures/pack_shapes.json.
"""
import json
import os
import struct
import sys
import collections

MODELS = os.environ.get("LAB_MODELS_DIR", "/home/rodman/models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw")
OUT = os.path.join(os.path.dirname(__file__), "pack_shapes.json")


def header(path: str) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def main() -> int:
    if not os.path.isdir(MODELS):
        print(f"missing pack at {MODELS}", file=sys.stderr)
        return 1
    h = {}
    for f in sorted(os.listdir(MODELS)):
        if f.endswith(".safetensors"):
            h.update({k: v for k, v in header(os.path.join(MODELS, f)).items() if k != "__metadata__"})
    out = {"pack_dir": MODELS, "n_tensors": len(h), "layers": {}, "dense": {}, "other": {}}
    k_hist = collections.Counter()
    for name, meta in sorted(h.items()):
        shp = meta.get("shape")
        if not shp:
            continue
        if ".experts." in name and name.endswith(".trellis"):
            parts = name.split(".")
            # language_model.model.layers.<L>.block_sparse_moe.experts.<E>.<proj>.trellis
            layer, exp, proj = parts[3], parts[6], parts[7]
            k, n, w = shp
            rec = out["layers"].setdefault(layer, {}).setdefault(exp, {})
            rec[proj] = {"trellis": shp, "K": w // 16, "k_in": k * 16, "n_out": n * 16,
                         "bytes": (shp[0] * shp[1] * shp[2]) * (2 if meta["dtype"] == "I16" else 4),
                         "dtype": meta["dtype"]}
            k_hist[(proj, w // 16)] += 1
        elif name.endswith(".trellis"):
            k, n, w = shp if len(shp) == 3 else (0, 0, 0)
            out["dense"][name] = {"trellis": shp, "K": w // 16, "k_in": k * 16, "n_out": n * 16,
                                  "dtype": meta["dtype"]}
            k_hist[("dense:" + name.split(".")[-3], w // 16)] += 1
        elif name.endswith((".suh", ".svh", ".mul1")):
            out.setdefault("scales", {})[name] = {"shape": shp, "dtype": meta["dtype"]}
    out["k_histogram"] = {f"{p}:K{k}": c for (p, k), c in sorted(k_hist.items())}
    # per-layer K multiset for the routed experts (what exl3_moe_coop_mk reads)
    per_layer = {}
    per_layer_k = {}
    for layer, exps in out["layers"].items():
        c = collections.Counter()
        # compact per-expert widths: the full tensor shapes are re-derivable from K,
        # k_in and n_out, which are identical across experts of one projection
        kl = {}
        for exp, projs in sorted(exps.items(), key=lambda kv: int(kv[0])):
            row = {}
            for proj, r in projs.items():
                c[(proj, r["K"])] += 1
                row[proj] = r["K"]
            kl[exp] = row
        per_layer[layer] = {f"{p}:K{k}": v for (p, k), v in sorted(c.items())}
        per_layer_k[layer] = kl
    out["per_layer_k"] = per_layer
    out["per_layer_expert_K"] = per_layer_k
    # the full per-expert tensor dump is a debug artifact, not a fixture: keep it out of git
    full = out.pop("layers")
    out["projection_geometry"] = {
        proj: {
            "K": sorted({e[proj]["K"] for ex in full.values() for e in ex.values() if proj in e}),
            "k_in": sorted({e[proj]["k_in"] for ex in full.values() for e in ex.values() if proj in e}),
            "n_out": sorted({e[proj]["n_out"] for ex in full.values() for e in ex.values() if proj in e}),
            "dtype": sorted({e[proj]["dtype"] for ex in full.values() for e in ex.values() if proj in e}),
        }
        for proj in ("w1", "w2", "w3")
    }
    with open(os.path.join(os.path.dirname(__file__), "pack_shapes_full.json"), "w") as fh:
        json.dump(full, fh)
    out.pop("scales", None)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT}: {out['n_tensors']} tensors, {len(per_layer)} MoE layers "
          f"(full per-expert dump in pack_shapes_full.json, git-ignored)")
    print("K histogram (routed + dense):")
    for k, v in out["k_histogram"].items():
        print(f"  {k}: {v}")
    l10 = per_layer_k.get("3") or {}
    print("sample layer 3 experts 0-2:", {e: l10[e] for e in list(l10)[:3]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
