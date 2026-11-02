#!/usr/bin/env python3
"""Correctness: the fast slice planner must reproduce the reference partition exactly.

Reference: vllm_exl3.minimax_q4kv_reference._greedy_row_slices_fast over the
[rows, nb] page bitmap, plus its row_union() page lists -- the exact objects
plan_prefill_slices_fast hands to q4_stage_pages and slice_fake_block_table.
"""
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/candidates")

import torch

from vllm_exl3 import minimax_q4kv_reference as ref
from fast_slices import greedy_row_slices

DEV = "cuda"
fails = 0


def bitmap_of(phys, nb):
    nkv, length, topk = phys.shape
    bm = torch.zeros((length, nb), dtype=torch.bool, device=phys.device)
    rows = torch.arange(length, device=phys.device, dtype=torch.int64)[None, :, None] \
        .expand_as(phys).reshape(-1)
    p = phys.reshape(-1)
    m = p >= 0
    bm[rows[m], p[m]] = True
    return bm


def case(length, nb, topk, nkv, max_pages, seed, density):
    global fails
    g = torch.Generator(device="cpu").manual_seed(seed)
    # a page pool where hot pages are picked far more often than cold ones, so the
    # union curve has both flat stretches and sharp jumps
    w = torch.ones(nb)
    w[: max(1, nb // 8)] = density
    idx = torch.multinomial(w, length * topk * nkv, replacement=True, generator=g)
    phys = idx.reshape(nkv, length, topk).to(torch.int64).to(DEV)
    if seed % 3 == 0:      # sprinkle -1 sentinels like the indexer emits
        mask = torch.rand((nkv, length, topk), generator=g) < 0.25
        phys = torch.where(mask.to(DEV), torch.full_like(phys, -1), phys)

    bm = bitmap_of(phys, nb)
    ref_ranges = ref._greedy_row_slices_fast(bm, max_pages)
    ref_pages = [bm[lo:hi].any(0).nonzero().reshape(-1).to(torch.int64)
                 for lo, hi in ref_ranges]

    mine = greedy_row_slices(phys, max_pages, nb)
    ok = len(mine) == len(ref_ranges)
    bad = []
    if ok:
        for (lo, hi, pl), (rlo, rhi), rp in zip(mine, ref_ranges, ref_pages):
            if (lo, hi) != (rlo, rhi):
                bad.append(f"range ({lo},{hi}) != ({rlo},{rhi})")
            elif int(pl.numel()) != int(rp.numel()) or bool((pl != rp).any()):
                bad.append(f"pages differ for ({lo},{hi}): {int(pl.numel())} vs {int(rp.numel())}")
    else:
        bad.append(f"n_slices {len(mine)} != {len(ref_ranges)}")
    fails += bool(bad)
    print(f"len={length:6d} nb={nb:5d} cap={max_pages:4d} density={density:5.1f} "
          f"slices={len(mine):4d} {'OK' if not bad else 'FAIL ' + bad[0]}")


for seed, (length, nb, cap, dens) in enumerate([
    (2048, 512, 384, 1.0),
    (2048, 512, 64, 1.0),
    (16384, 1664, 384, 4.0),
    (16384, 1664, 384, 1.0),
    (16384, 2344, 384, 8.0),
    (4096, 128, 384, 1.0),      # cap above the union: one slice
    (512, 4096, 16, 1.0),       # cap below one row's own picks
    (1000, 300, 32, 2.0),
    (64, 20, 4, 1.0),
    (1, 20, 4, 1.0),
]):
    case(length, nb, 16, 2, cap, seed + 1, dens)

sys.exit(1 if fails else 0)
