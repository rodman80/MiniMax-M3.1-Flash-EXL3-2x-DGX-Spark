"""Load the REAL routed-expert EXL3 weights of one MiniMax-M3.1 MoE layer.

The pack stores each expert as trellis [k/16, n/16, 16K] int16 plus suh [k] and svh [n]
fp16 scales and a mul1 codebook flag. Reading only the tensors we ask for keeps the cost
at ~1.4 GiB for a 128-expert layer, and it means the lab benches the pack's actual
trellis widths (K 2 and 3), codebook and scales, not a synthetic GEMM.
"""
from __future__ import annotations

import json
import os

import torch
import safetensors.torch as st

PACK = os.environ.get("LAB_PACK", "/models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw")
PROJ = {"w1": "gate", "w3": "up", "w2": "down"}


def moe_layers() -> list[int]:
    idx = json.load(open(os.path.join(PACK, "model.safetensors.index.json")))
    ls = {int(k.split(".")[3]) for k in idx["weight_map"] if ".block_sparse_moe.experts." in k}
    return sorted(ls)


def load_layer(layer: int, dev: str = "cuda", experts: list[int] | None = None) -> dict:
    idx = json.load(open(os.path.join(PACK, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    pre = f"language_model.model.layers.{layer}.block_sparse_moe.experts."
    if experts is None:
        experts = sorted({int(k[len(pre):].split(".")[0]) for k in wm
                          if k.startswith(pre) and k.endswith(".trellis")})
    files = {}
    out = {"layer": layer, "experts": experts, "proj": {}}
    for pk, name in PROJ.items():
        trellis, suh, svh, ks, mul1 = [], [], [], [], []
        for e in experts:
            base = f"{pre}{e}.{pk}."
            for suffix in ("trellis", "suh", "svh", "mul1"):
                fn = wm[base + suffix]
                if fn not in files:
                    files[fn] = st.safe_open(os.path.join(PACK, fn), framework="pt")
                t = files[base and fn].get_tensor(base + suffix)
                if suffix == "trellis":
                    ks.append(t.shape[2] // 16)
                    trellis.append(t.to(dev).to(torch.int16).contiguous())
                elif suffix == "mul1":
                    mul1.append(int(t.reshape(-1)[0].item()))
                else:
                    (suh if suffix == "suh" else svh).append(t.to(dev).to(torch.float16).contiguous())
        assert len(set(mul1)) == 1, f"mixed mul1 within layer {layer} {pk}: {set(mul1)}"
        assert len(set(ks)) <= 2, f"more than two K values in layer {layer} {pk}: {set(ks)}"
        ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        out["proj"][name] = {
            "trellis": trellis, "suh": suh, "svh": svh,
            "ptrs": (ptr(trellis), ptr(suh), ptr(svh)),
            "k": torch.tensor(ks, dtype=torch.int32, device=dev),
            "ks": ks, "k_lo": min(ks), "k_hi": max(ks),
            "mul1": bool(mul1[0]), "mcg": False,
            "bytes": sum(t.numel() * t.element_size() for t in trellis),
            "keep": None,
        }
    return out


def weight_bytes(layer: dict, expert_ids) -> int:
    """Bytes the kernel must read for a set of distinct experts (trellis + both scales)."""
    tot = 0
    for name, p in layer["proj"].items():
        for e in expert_ids:
            t = p["trellis"][e]
            tot += t.numel() * t.element_size()
            tot += p["suh"][e].numel() * 2 + p["svh"][e].numel() * 2
    return tot
