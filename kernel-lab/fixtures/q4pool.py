"""A q4 (nvfp4-layout) KV pool container for timing, with dequantizable bytes.

The pool is [nb, 2*num_kv_heads, 128, 72] uint8: 64 bytes of packed e2m1 pairs and 8
bytes of e4m3 scale per 128-token row side. Random bytes are fine for TIMING, but 0x7f
and 0xff are NaN in e4m3fn, so a fully random pool makes the staged bf16 pool NaN and
any numeric comparison meaningless. The scale columns are therefore pinned to 0x38
(e4m3fn 1.0) and the data columns stay random.
"""
from __future__ import annotations

import torch

ROW_BYTES = 72
DATA_BYTES = 64
SCALE_BYTES = 8
BLOCK = 128
SLOT_REGION = BLOCK * ROW_BYTES          # 9216 bytes per (page, side)
DATA_REGION = BLOCK * DATA_BYTES         # 8192
E4M3_ONE = 0x38


def make_q4_pool(nb: int, num_kv_heads: int, seed: int = 3, dev: str = "cuda") -> torch.Tensor:
    """[nb, 2*num_kv_heads, 128, 72] uint8, laid out as the pack's container is:
    each 9216-byte slot region is 8192 bytes of packed e2m1 pairs followed by 1024 bytes
    of e4m3 scales. The scale bytes are 0x38 (e4m3fn 1.0); the data bytes are random."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    flat = torch.randint(0, 256, (nb, 2 * num_kv_heads, SLOT_REGION),
                         dtype=torch.uint8, generator=g)
    flat[:, :, DATA_REGION:] = E4M3_ONE
    return flat.reshape(nb, 2 * num_kv_heads, BLOCK, ROW_BYTES).to(dev).contiguous()
