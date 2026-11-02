#!/usr/bin/env python3
"""A3 correctness: the v2 slice planner must reproduce the reference plan exactly.

Reference: vllm_exl3.minimax_q4kv_reference.plan_prefill_slices_fast -- the exact object
the served prefill path hands to q4_stage_pages and slice_fake_block_table.

Cases cover the extremes the brief asks for: empty selections (every entry -1), single
rows, single pages, duplicate pages inside a row and across rows, structurally identical
rows (the tie case for the greedy), caps at 1 / at the union size / one below it, caps at
or above nb (the branch where no split is needed), partial blocks at the causal edge,
many requests, and the served pool width (nb = 9,042 pages).

Every case asserts: same number of slices, same `reqs`, same `q` ranges, same page count,
and element-wise identical page id lists (same order). A plan difference is a correctness
failure, not a performance result.
"""
import os
import sys

sys.path.insert(0, "/lab")
sys.path.insert(0, "/lab/candidates")

import torch

from vllm_exl3 import minimax_q4kv_reference as ref
import fast_slices

DEV = "cuda"
KVH = 2
TOPK = 16
BLOCK = 128
fails = 0


def make_case(nb, lens, prefixes, seed, mode="normal", topk=TOPK):
    """A q4 pool, a block table and a topk_idx in the shape the served path passes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    total_q = sum(lens)
    pool = torch.randint(0, 256, (nb, 2 * KVH, 128, ref.ROW_BYTES),
                         dtype=torch.uint8, generator=g).to(DEV)
    seqs = [p + l for p, l in zip(prefixes, lens)]
    max_blk = max((s + BLOCK - 1) // BLOCK for s in seqs)
    bt = torch.zeros(len(lens), max_blk, dtype=torch.int32)
    for r, s in enumerate(seqs):
        n = (s + BLOCK - 1) // BLOCK
        # physical pages are a permutation of the logical ones: production never maps
        # identity, and the planner must not care
        perm = torch.randperm(nb, generator=g)[:n]
        bt[r, :n] = perm.to(torch.int32)
    topk_idx = torch.zeros(KVH, total_q, topk, dtype=torch.int32)
    row = 0
    for r, (l, p) in enumerate(zip(lens, prefixes)):
        for i in range(l):
            v = (p + i + BLOCK) // BLOCK            # causal: blocks this row may select
            if mode == "empty" or v <= 0:
                picks = [0] * topk                   # all sentinels below
            elif mode == "same":                     # every row selects the same blocks
                picks = list(range(min(v, topk)))
            elif mode == "dup":                      # duplicates inside a row
                picks = [0] * topk
                for j in range(topk):
                    picks[j] = (i // 3 + j // 4) % max(v, 1)
            else:
                k = min(topk, v)
                picks = torch.randperm(v, generator=g)[:k].tolist()
                picks += [0] * (topk - k)
            topk_idx[:, row, :len(picks)] = torch.tensor(picks, dtype=torch.int32)[None, :]
            row += 1
        if mode == "empty":
            topk_idx[:, cu_range(lens, r)[0]:cu_range(lens, r)[1], :] = -1
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()),
                      dtype=torch.int32, device=DEV)
    sl = torch.tensor(seqs, dtype=torch.int32, device=DEV)
    pl = torch.tensor(prefixes, dtype=torch.int32, device=DEV)
    return pool, bt.to(DEV), topk_idx.to(DEV), cu, sl, pl


def cu_range(lens, r):
    return sum(lens[:r]), sum(lens[:r + 1])


def check(name, nb, lens, prefixes, cap, seed=1, mode="normal", topk=TOPK):
    global fails
    pool, bt, topk_idx, cu, sl, pl = make_case(nb, lens, prefixes, seed, mode, topk)
    a = ref.plan_prefill_slices_fast(pool, bt, topk_idx, cu, KVH, cap)
    b = fast_slices.plan_prefill_slices_fast_v2(pool, bt, topk_idx, cu, KVH, cap)
    problems = []
    if len(a) != len(b):
        problems.append(f"n_slices {len(a)} != {len(b)}")
    else:
        for i, (x, y) in enumerate(zip(a, b)):
            if x["reqs"] != y["reqs"]:
                problems.append(f"slice {i}: reqs {x['reqs']} != {y['reqs']}")
            elif x["q"] != y["q"]:
                problems.append(f"slice {i}: q {x['q']} != {y['q']}")
            elif x["n"] != y["n"]:
                problems.append(f"slice {i}: n {x['n']} != {y['n']}")
            elif not bool((x["pages"] == y["pages"]).all()):
                problems.append(f"slice {i}: page lists differ ({x['n']} vs {y['n']} ids)")
    fails += bool(problems)
    print(f"{name:44} nb={nb:6d} cap={cap:5d} slices={len(a):5d} "
          f"rows={sum(lens):6d} {'OK' if not problems else 'FAIL ' + problems[0]}")


# --- the served geometry -----------------------------------------------------
check("served pool, 16K chunk, prefix 0", 9042, [16384], [0], 384)
check("served pool, 16K chunk, 192K prefix", 9042, [16384], [196608], 384, mode="same")
check("served pool, 16K chunk, 192K prefix, random", 9042, [16384], [196608], 384)
check("served pool, 16K chunk, 128K prefix, random", 9042, [16384], [131072], 384)
check("served pool, 8K chunk, 56K prefix, random", 9042, [8192], [57344], 384)
check("served pool, 16K chunk, 192K prefix, dup", 9042, [16384], [196608], 384, mode="dup")
check("served pool, 6 requests at 48K prefix", 9042, [4096] * 6, [49152] * 6, 384)
check("served pool, 6 requests, identical rows", 9042, [2048] * 6, [98304] * 6, 384,
      mode="same")

# --- extremes ----------------------------------------------------------------
check("empty selection (all -1)", 512, [256], [0], 384, mode="empty")
check("empty selection, cap 1", 512, [64], [0], 1, mode="empty")
check("single row", 512, [1], [0], 384)
check("single row, long prefix", 9042, [1], [262144 - 128], 384)
check("cap 1 (every row its own slice)", 512, [128], [0], 1)
check("cap 2", 512, [256], [0], 2)
check("cap == topk (one row exactly fills a slice)", 512, [256], [0], 16)
check("cap == topk - 1 (row alone exceeds the cap: hard floor)", 512, [256], [0], 15)
check("cap >= nb (no split possible)", 128, [4096], [0], 512)
check("cap > nb, long prefix", 9042, [8192], [98304], 20000)
check("nb smaller than topk (partial blocks at the causal edge)", 8, [64], [0], 4)
check("topk 1", 512, [512], [0], 8, topk=1)
check("topk 64 (union wider than the cap)", 2048, [1024], [0], 32, topk=64)
check("many tiny requests", 4096, [7] * 64, [65536] * 64, 384)
check("one long request, one short", 4096, [8000, 33], [32768, 0], 384)
check("prefix not a block multiple", 4096, [1000], [12345], 384)
check("rows == 2, cap forces a split mid-row-pair", 4096, [2], [262144], 8)

sys.exit(1 if fails else 0)
