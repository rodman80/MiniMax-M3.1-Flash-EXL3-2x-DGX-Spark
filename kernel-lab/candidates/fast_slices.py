"""Candidate A3: an exact, O(rows x topk) replacement for the q4 prefill slice planner.

What production does (vllm_exl3/minimax_q4kv_reference.py, plan_prefill_slices_fast):
when one request's own page union exceeds the staging cap, it builds a
`q_bitmap [rows, nb]` bool map and calls `_greedy_row_slices_fast`, which for EVERY slice
start `a` calls `_union_curve(a)` -> `_first_row_at_or_after(a)`, an
`[(rows - a), nb]` int64 where/amin. Cost per slice is O(rows x nb) x 8 bytes; with
rows = 16384 and nb = 1664 that is 218 MB per slice, and a long-context chunk needs
6-100+ slices. Measured: 13.7 ms per layer at 16K rows / 192K prefix, 42% of the whole
sparse-attention operation (results/a1b_q4_prefill_op.json).

The identity that makes it cheap: for a slice starting at row `a`,

    cov(a + k) = |union of pages of rows [a, a+k)| = #{page : first[page] < a + k}

where first[page] is the first row >= a that selects the page. cov is nondecreasing in k,
so the greedy's answer -- the largest k with cov(a+k) <= cap -- is just

    k* = first_sorted[cap] - a

one ascending sort of nb elements per slice, no [rows, nb] tensor at all. And the slice's
page list is the set of pages with first[page] < a + k*, which the same sort already
found, so `bitmap[lo:hi].any(0).nonzero()` disappears too.

Semantics preserved exactly: same greedy, same maximal ranges, same hard floor of one row
when a row alone exceeds the cap, same ascending page order, same -1 handling. The test
tests/test_planner.py pins the partition and the page lists against the reference
planner over randomized cases.
"""

from __future__ import annotations

import torch


def _row_pairs(phys: torch.Tensor, lo: int, hi: int, nb: int):
    """(rows, pages) pair lists for rows [lo, hi) of one request, row-major and sorted
    by row. Invalid (-1) entries go to a dummy column nb, never to a real page."""
    nkv, length, topk = phys.shape
    # permute before reshape: phys is [nkv, length, topk] and the pair list must be
    # grouped by ROW, which is axis 1, not axis 0.
    p = phys[:, lo:hi, :].permute(1, 0, 2).reshape(length, nkv * topk)
    pages = torch.where(p >= 0, p, torch.full_like(p, nb))
    rows = torch.arange(length, device=p.device, dtype=torch.int64)[:, None] \
        .expand(-1, nkv * topk)
    return rows.reshape(-1), pages.reshape(-1)


def greedy_row_slices(
    phys: torch.Tensor,          # [nkv, length, topk] physical page ids of ONE request
    max_pages: int,
    nb: int,
):
    """Same partition as _greedy_row_slices_fast on the [length, nb] bitmap of `phys`.

    Returns [(start, end, pages)] with pages the ascending unique page ids of the range.
    """
    length = phys.shape[1]
    if length == 0:
        return []
    dev = phys.device
    per_row = phys.shape[0] * phys.shape[2]
    rows, pages = _row_pairs(phys, 0, length, nb)
    sentinel = length
    out = []
    a = 0
    while a < length:
        # rows is [0]*per_row, [1]*per_row, ... so the suffix at row a starts at a*per_row.
        r_suf = rows[a * per_row:]
        p_suf = pages[a * per_row:]
        first = torch.full((nb + 1,), sentinel, dtype=torch.int64, device=dev)
        first.scatter_reduce_(0, p_suf, r_suf, reduce="amin")
        f = first[:nb]
        if max_pages < nb:
            val, order = torch.sort(f)          # ascending; order is the page id
            k = int(val[max_pages].item()) - a
            if k <= 0:
                k = 1                            # hard floor: one row alone
            k = min(k, length - a)
            npages = int((val < a + k).sum().item())
            pl = order[:npages]
        else:
            k = length - a
            keep = f < length
            npages = int(keep.sum().item())
            pl = torch.nonzero(keep, as_tuple=False).reshape(-1)
        pl = pl.sort().values.to(torch.int64)
        out.append((a, a + k, pl))
        a += k
    return out


def plan_prefill_slices_fast_v2(pool, block_table, topk_idx, cu_seqlens_q, num_kv_heads,
                                max_pages):
    """Drop-in for minimax_q4kv_reference.plan_prefill_slices_fast.

    Same slice plan (same greedy, same tie-breaks, same hard floor, same page lists and
    their ascending order), but the per-request split costs O(rows x nkv x topk) instead
    of O(slices x rows x nb). The request-level greedy stays on the reference code path:
    with <= max_num_seqs requests it is already trivial.
    """
    from vllm_exl3 import minimax_q4kv_reference as ref

    nb, slots, tokens, width = pool.shape
    if (slots != 2 * num_kv_heads or tokens != ref.POOL_BLOCK_SIZE
            or width != ref.ROW_BYTES):
        raise ValueError("pool must be [nb, 2*num_kv_heads, 128, 72] uint8")
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    dev = pool.device
    req_bitmap, phys, _ = ref._selected_page_bitmap(pool, block_table, topk_idx, cu_seqlens_q)
    cu = cu_seqlens_q.to(torch.int64).tolist()
    total_q = topk_idx.shape[1]
    if (cu[0] != 0 or cu[-1] != total_q
            or any(b < a for a, b in zip(cu, cu[1:]))):
        raise ValueError("cu_seqlens_q must start at 0, be non-decreasing, and cover "
                         f"the {total_q}-query axis of topk_idx")

    def row_union(bitmap, lo, hi):
        return bitmap[lo:hi].any(0).nonzero().reshape(-1).to(torch.int64)

    slices = []
    for rs, re_ in ref._greedy_row_slices_fast(req_bitmap, max_pages):
        pages = row_union(req_bitmap, rs, re_)
        if re_ - rs == 1 and int(pages.numel()) > max_pages:
            r = rs
            lo, hi = cu[r], cu[r + 1]
            for qs, qe, pl in greedy_row_slices(phys[:, lo:hi, :], max_pages, nb):
                slices.append({"reqs": [r], "q": (lo + qs, lo + qe), "pages": pl})
        else:
            slices.append({"reqs": list(range(rs, re_)), "q": (cu[rs], cu[re_]),
                           "pages": pages})
    for s in slices:
        s["n"] = int(s["pages"].numel())
    return slices
