"""Shared harness for the MiniMax M3.1 Flash kernel lab.

Timing rules (the whole point of the lab):
  - CUDA events, never wall clock, for kernel cost; the callable is warmed up and
    compiled OUTSIDE the measured region.
  - Every measurement is a list of independent reps; we report median / p10 / p90 /
    spread, never a single number.
  - "kernel cost" and "operation cost" are measured separately: an operation is the
    Python entry point including launches, metadata building, allocations and copies.
    A kernel speedup that the operation cost does not show is not a speedup.
  - Nothing is cached across reps that would make a later rep cheaper than a real
    call (no output reuse as an input, no pre-computed answers, no fixed seed reuse
    that lets the compiler fold work away).

Memory rules: LAB_BUDGET (MiB) is a hard ceiling on torch's CUDA allocation. On GB10
the GPU and the host share one pool and a host OOM takes the node down, so a lab
process that allocates freely is a bug, not a warning.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time

import torch

BUDGET_MIB = int(os.environ.get("LAB_BUDGET", "6000"))
RESULTS = os.environ.get("LAB_RESULTS", os.path.join(os.path.dirname(__file__), "results"))


class BudgetError(RuntimeError):
    pass


def budget_check(where: str = "") -> None:
    alloc = torch.cuda.memory_allocated() / 2**20
    resv = torch.cuda.memory_reserved() / 2**20
    if resv > BUDGET_MIB:
        raise BudgetError(
            f"lab CUDA reserved {resv:.0f} MiB > LAB_BUDGET {BUDGET_MIB} MiB {where}"
        )


def env_record() -> dict:
    """Everything a result needs to be attributable."""
    import triton

    p = torch.cuda.get_device_properties(0)
    nvcc = ""
    try:
        nvcc = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=20
        ).stdout.strip().splitlines()[-1]
    except Exception:
        pass
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": p.name,
        "cc": f"sm_{p.major}{p.minor}",
        "sms": p.multi_processor_count,
        "l2_mib": round(p.L2_cache_size / 2**20, 2),
        "total_mem_mib": round(p.total_memory / 2**20),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "nvcc": nvcc,
        "python": platform.python_version(),
        "driver": torch.cuda.get_device_capability(),
        "image": os.environ.get("LAB_IMAGE_NAME", ""),
        "clock_mhz": _clocks(),
        "neighbours": _neighbours(),
    }


def _clocks() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return {"raw": out}
    except Exception as exc:
        return {"error": str(exc)}


def _neighbours() -> dict:
    """Other GPU processes on the node: contention has to be on the record."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        return {"compute_apps": out}
    except Exception as exc:
        return {"error": str(exc)}


def timeit(fn, reps: int = 30, warmup: int = 5, inner: int = 1) -> dict:
    """CUDA-event timing of `fn`, `reps` independent samples.

    `inner` calls per sample amortize launch overhead only when the caller asks for
    it (used for sub-50 us kernels where the event pair itself is a large fraction);
    the default of 1 measures one call per sample, launch included.
    """
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0 / inner)  # us
    return stats(samples)


def timeit_wall(fn, reps: int = 30, warmup: int = 5) -> dict:
    """Wall-clock timing including Python and launch overhead (host-bound paths)."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return stats(samples)


def stats(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    med = statistics.median(s)
    return {
        "n": n,
        "median_us": round(med, 2),
        "mean_us": round(statistics.fmean(s), 2),
        "p10_us": round(s[max(0, int(0.10 * n) - 1) if n > 1 else 0], 2),
        "p90_us": round(s[min(n - 1, int(0.90 * n))], 2),
        "min_us": round(s[0], 2),
        "max_us": round(s[-1], 2),
        "spread_pct": round((s[-1] - s[0]) / med * 100.0, 1) if med else None,
        "samples_us": [round(x, 2) for x in s],
    }


def rel_err(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Relative L2 error plus the max absolute difference, in fp32/fp64."""
    a = a.detach().float()
    b = b.detach().float()
    denom = b.norm().clamp(min=1e-12)
    d = (a - b)
    return {
        "rel_l2": float((d.norm() / denom).item()),
        "max_abs": float(d.abs().max().item()),
        "mean_abs": float(d.abs().mean().item()),
        "ref_norm": float(b.norm().item()),
        "finite": bool(torch.isfinite(a).all().item()),
    }


def row_rel_err(a: torch.Tensor, b: torch.Tensor, dim=(1, 2)) -> dict:
    a = a.detach().float()
    b = b.detach().float()
    d = (a - b).norm(dim=dim)
    n = b.norm(dim=dim).clamp(min=1e-6)
    r = d / n
    return {
        "worst_row_rel": float(r.max().item()),
        "median_row_rel": float(r.median().item()),
        "worst_row": int(r.argmax().item()),
    }


def write_result(name: str, payload: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    payload.setdefault("env", env_record())
    path = os.path.join(RESULTS, name if name.endswith(".json") else name + ".json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
    return path


def emit(payload: dict) -> None:
    """Print a compact table line per row so runs are readable in the transcript."""
    rows = payload.get("rows") or []
    if not rows:
        return
    cols = [c for c in rows[0] if c not in ("samples_us",)]
    print("\n" + " | ".join(f"{c:>18}" for c in cols))
    print("-" * (21 * len(cols)))
    for r in rows:
        print(" | ".join(f"{str(r.get(c))[:18]:>18}" for c in cols))


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(2)
