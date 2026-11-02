"""Selection fixtures for MiniMax-M3 block-sparse attention.

topk_idx layout as vLLM keeps it: [num_kv_heads, total_q, topk] int32 LOGICAL block ids,
where only the first min(topk, (q_abs + 128) // 128) entries of each row are read by the
kernel (the tail is garbage). The indexer always forces the init blocks (score +1e30) and
the local block (score +1e29), so every row contains its own last block.

Three families, because the tiled kernel's whole premise is that neighbouring queries
share blocks, and that premise has to be tested at both ends:

  uniform      15 random valid blocks + the local block. This is what the plugin's own
               bench_sparse_attn.py generates. It is the WORST case for tiling (an 8-token
               tile's union is close to 8 x 16) and it is NOT what the model does.
  real_like    calibrated to the one live statistic the recipe records (patches/0012: an
               8-token tile's union averages 18.9 blocks against 128 unshared).
  disjoint     every token picks different blocks: the adversarial upper bound on union.

The real_like generator is a two-component model: a per-window "hot" set that consecutive
queries mostly share, plus a small per-token tail. Its parameters are chosen by
calibrate() so the measured mean union at T=8 matches the recorded 18.9; the calibration
is recorded in the fixture so a reader can see what was assumed.
"""

from __future__ import annotations

import torch

SPARSE_BLOCK = 128
TOPK = 16
INIT_BLOCKS = 0      # sparse_init_block = 0 in the pack config
LOCAL_BLOCKS = 1     # sparse_local_block = 1


def _valid_blocks(q_abs: torch.Tensor) -> torch.Tensor:
    return (q_abs + SPARSE_BLOCK) // SPARSE_BLOCK


def make_selection(
    lens: list[int], prefixes: list[int], kvh: int, seed: int, dev,
    family: str = "real_like", topk: int = TOPK, swap: int = 2,
    window: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (topk_idx, cu_seqlens_q, seq_lens, prefix_lens) for one prefill chunk.

    lens/prefixes are per request: `lens[i]` query rows, `prefixes[i]` cached context
    before them, so seq_len = prefix + len.

    Model: every `window` consecutive queries share a hot set of `topk - 1 - swap` blocks
    plus its own local block; each query then replaces `swap` of the hot picks with fresh
    random valid blocks. `swap` is the knob that sets the tile union size, and it is
    calibrated so mean union at T=8 matches the live figure the recipe records (18.9).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    total_q = sum(lens)
    seqs = [p + l for p, l in zip(prefixes, lens)]
    max_seq = max(seqs)
    topk_idx = torch.zeros(kvh, total_q, topk, dtype=torch.int32)
    for b, (l, p) in enumerate(zip(lens, prefixes)):
        n_win = (l + window - 1) // window
        hot_n = max(1, topk - LOCAL_BLOCKS - swap)
        for w in range(n_win):
            lo, hi = w * window, min((w + 1) * window, l)
            v_min = (p + lo + SPARSE_BLOCK) // SPARSE_BLOCK
            hot = torch.randperm(v_min, generator=g)[:hot_n].tolist()
            for i in range(lo, hi):
                v = (p + i + SPARSE_BLOCK) // SPARSE_BLOCK
                local = v - 1
                picks: list[int] = [local]
                for h in hot:
                    if h < v and h != local:
                        picks.append(h)
                    if len(picks) == topk - swap:
                        break
                if len(picks) < topk:
                    for e in torch.randperm(v, generator=g).tolist():
                        if e not in picks:
                            picks.append(e)
                        if len(picks) == topk:
                            break
                picks = (picks + [0] * topk)[:topk]
                topk_idx[:, i, :len(picks)] = torch.tensor(picks, dtype=torch.int32)[None, :]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    sl = torch.tensor(seqs, dtype=torch.int32, device=dev)
    pl = torch.tensor(prefixes, dtype=torch.int32, device=dev)
    return topk_idx.to(dev), cu, sl, pl


def make_uniform(lens, prefixes, kvh, seed, dev, topk: int = TOPK):
    """The plugin bench's generator: local block + (topk-1) uniform random valid blocks."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    total_q = sum(lens)
    seqs = [p + l for p, l in zip(prefixes, lens)]
    nblk = (max(seqs) + SPARSE_BLOCK - 1) // SPARSE_BLOCK
    topk_idx = torch.zeros(kvh, total_q, topk, dtype=torch.int32)
    row = 0
    for b, (l, p) in enumerate(zip(lens, prefixes)):
        for i in range(l):
            v = (p + i + SPARSE_BLOCK) // SPARSE_BLOCK
            r = torch.rand(v, generator=g)
            order = torch.argsort(r, descending=True)
            picks = [v - 1] + [int(x) for x in order.tolist() if int(x) != v - 1][: topk - 1]
            picks = (picks + [0] * topk)[:topk]
            topk_idx[:, row, :len(picks)] = torch.tensor(picks, dtype=torch.int32)[None, :]
            row += 1
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    sl = torch.tensor(seqs, dtype=torch.int32, device=dev)
    pl = torch.tensor(prefixes, dtype=torch.int32, device=dev)
    return topk_idx.to(dev), cu, sl, pl


def make_disjoint(lens, prefixes, kvh, seed, dev, topk: int = TOPK):
    """Adversarial: consecutive tokens pick disjoint blocks (union = T * topk)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    total_q = sum(lens)
    seqs = [p + l for p, l in zip(prefixes, lens)]
    topk_idx = torch.zeros(kvh, total_q, topk, dtype=torch.int32)
    row = 0
    for b, (l, p) in enumerate(zip(lens, prefixes)):
        for i in range(l):
            v = (p + i + SPARSE_BLOCK) // SPARSE_BLOCK
            # stride so neighbours never collide: block = (i * topk + j) % v
            picks = [(i * topk + j) % v for j in range(topk)]
            if v - 1 not in picks:
                picks[0] = v - 1
            topk_idx[:, row, :topk] = torch.tensor(picks, dtype=torch.int32)[None, :]
            row += 1
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    sl = torch.tensor(seqs, dtype=torch.int32, device=dev)
    pl = torch.tensor(prefixes, dtype=torch.int32, device=dev)
    return topk_idx.to(dev), cu, sl, pl


def make_drift(lens, prefixes, kvh, seed, dev, topk: int = TOPK, drift: int = 1):
    """Hot set that drifts one block per token, instead of resetting every `window` rows.

    This is the more plausible model of a real indexer: consecutive queries agree almost
    everywhere, and the agreement decays with distance. It keeps the local union at the
    recorded ~19 blocks while making the CHUNK union large, which is what drives the
    number of staging slices in the q4 prefill path.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    seqs = [p + l for p, l in zip(prefixes, lens)]
    total_q = sum(lens)
    topk_idx = torch.zeros(kvh, total_q, topk, dtype=torch.int32)
    row = 0
    for b, (l, p) in enumerate(zip(lens, prefixes)):
        hot = None
        for i in range(l):
            v = (p + i + SPARSE_BLOCK) // SPARSE_BLOCK
            if hot is None or len(hot) == 0:
                hot = list(torch.randperm(max(v, 1), generator=g)[: topk - LOCAL_BLOCKS].tolist())
            else:
                for _ in range(drift):
                    hot[torch.randint(len(hot), (1,), generator=g).item()] = \
                        int(torch.randint(max(v, 1), (1,), generator=g).item())
            local = v - 1
            picks = [local] + [h for h in hot if h < v and h != local][: topk - 1]
            if len(picks) < topk:
                for e in torch.randperm(max(v, 1), generator=g).tolist():
                    if e not in picks:
                        picks.append(e)
                    if len(picks) == topk:
                        break
            picks = (picks + [0] * topk)[:topk]
            topk_idx[:, row, :topk] = torch.tensor(picks, dtype=torch.int32)[None, :]
            row += 1
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    sl = torch.tensor(seqs, dtype=torch.int32, device=dev)
    pl = torch.tensor(prefixes, dtype=torch.int32, device=dev)
    return topk_idx.to(dev), cu, sl, pl


def union_stats(topk_idx, cu, pl, tile: int) -> dict:
    """Union size per tile, using the plugin's own metadata builder (so it is the same
    quantity the kernel iterates)."""
    from vllm_exl3.sparse_attn_tiled import build_tiles, build_union
    total_q = topk_idx.shape[1]
    q0, n, req, qabs0 = build_tiles(cu, pl, total_q, tile)
    _, ucount, _ = build_union(topk_idx, q0, n, qabs0, tile)
    u = ucount.float().reshape(-1)
    live = (n > 0).reshape(-1)
    if bool(live.any()):
        u = u.reshape(ucount.shape[0], -1)[:, live].reshape(-1)
    return {
        "tile": tile,
        "n_tiles": int(u.numel()),
        "mean_union": round(float(u.mean().item()), 2),
        "max_union": int(u.max().item()),
        "p90_union": float(u.quantile(0.9).item()),
        "unshared_ceiling": tile * topk_idx.shape[-1],
    }


def calibrate(lens, prefixes, kvh, dev, target_union: float, tile: int = 8) -> dict:
    """Pick `swap` so the measured mean union at T=tile matches target_union."""
    best = None
    for swap in range(0, 12):
        ti, cu, sl, pl = make_selection(lens, prefixes, kvh, 7, dev, "real_like",
                                        swap=swap, window=64)
        u = union_stats(ti, cu, pl, tile)["mean_union"]
        if best is None or abs(u - target_union) < abs(best[1] - target_union):
            best = (swap, u)
    return {"swap": best[0], "mean_union_at_T8": best[1], "target": target_union}
