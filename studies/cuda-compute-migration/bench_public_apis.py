"""
Benchmark public-facing Awkward Array APIs that exercise kernels migrated
from CuPy/.cu implementations to cuda.compute.

The script is intentionally self-contained and only uses the *public* Awkward
API so it can run on both pre-migration and post-migration commits.

It measures wall-clock time (with CUDA-device synchronization) for each API
on a few representative jagged shapes/dtypes, and writes the timings to a
JSON file. Use `compare.py` to diff two such JSON files.

Run:
    python bench_public_apis.py --output bench_results.json
    python bench_public_apis.py --apis sum argmin sort --warmup 3 --iters 10

Apis benchmarked (each maps to a migrated kernel family):
    sum       -> awkward_reduce_sum*           (ak.sum)
    max       -> awkward_reduce_max*           (ak.max)
    min       -> awkward_reduce_min*           (ak.min)
    prod      -> awkward_reduce_prod*          (ak.prod)
    count     -> awkward_reduce_count_64       (ak.count)
    countnz   -> awkward_reduce_countnonzero   (ak.count_nonzero)
    argmin    -> awkward_reduce_argmin*        (ak.argmin)
    argmax    -> awkward_reduce_argmax*        (ak.argmax)
    sort      -> awkward_sort  (cuda.compute segmented_sort; no pre-migration GPU support)
    pad_none  -> awkward_index_rpad_and_clip_axis0/1  (ak.pad_none(.., clip=True))
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field

def _require_cupy_for_cuda(backends):
    if "cuda" in backends:
        try:
            import cupy  # noqa: F401
        except ImportError:
            sys.exit("This benchmark requires cupy when --backends includes cuda. "
                     "Install cupy or pass --backends cpu.")


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #

def make_jagged_dataset(
    n_events: int,
    avg_items: int,
    dtype: str,
    seed: int = 42,
):
    """Return (cpu_array, n_total_items)."""
    import numpy as np
    import awkward as ak
    rng = np.random.default_rng(seed)

    # Truncated-normal-ish distribution centered on avg_items, never < 1
    lengths = np.clip(
        rng.integers(max(1, avg_items - avg_items // 2),
                     avg_items + avg_items // 2 + 1,
                     size=n_events),
        1, None,
    ).astype(np.int64)
    n_total = int(lengths.sum())

    if dtype == "float32":
        flat = rng.standard_normal(n_total).astype(np.float32)
    elif dtype == "float64":
        flat = rng.standard_normal(n_total).astype(np.float64)
    elif dtype == "int32":
        flat = rng.integers(-100, 100, size=n_total, dtype=np.int32)
    elif dtype == "int64":
        flat = rng.integers(-100, 100, size=n_total, dtype=np.int64)
    elif dtype == "bool":
        flat = (rng.random(n_total) > 0.5).astype(np.bool_)
    elif dtype == "complex64":
        re = rng.standard_normal(n_total).astype(np.float32)
        im = rng.standard_normal(n_total).astype(np.float32)
        flat = re + 1j * im
    elif dtype == "complex128":
        re = rng.standard_normal(n_total).astype(np.float64)
        im = rng.standard_normal(n_total).astype(np.float64)
        flat = re + 1j * im
    else:
        raise ValueError(f"unknown dtype {dtype}")

    arr = ak.unflatten(flat, lengths)
    return arr, n_total


# --------------------------------------------------------------------------- #
# Timing harness
# --------------------------------------------------------------------------- #

def sync(backend):
    if backend == "cuda":
        import cupy as cp
        cp.cuda.Device().synchronize()


def time_one(fn, warmup, iters, backend):
    """Run `fn` `warmup` times, then `iters` times, returning per-iter ms."""
    for _ in range(warmup):
        fn()
    sync(backend)
    times_ms = []
    for _ in range(iters):
        sync(backend)
        t0 = time.perf_counter()
        fn()
        sync(backend)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


# --------------------------------------------------------------------------- #
# API benchmarks
# --------------------------------------------------------------------------- #

@dataclass
class Case:
    api: str
    dtype: str
    n_events: int
    avg_items: int
    backend: str   # "cpu" or "cuda"
    n_total: int = 0
    times_ms: list = field(default_factory=list)
    min_ms: float = 0.0
    mean_ms: float = 0.0
    error: str = ""


def bench_api(
    api: str,
    dtype: str,
    n_events: int,
    avg_items: int,
    backend: str,
    warmup: int,
    iters: int,
) -> Case:
    import awkward as ak
    case = Case(api=api, dtype=dtype, n_events=n_events,
                avg_items=avg_items, backend=backend)
    try:
        arr_cpu, n_total = make_jagged_dataset(n_events, avg_items, dtype)
        case.n_total = n_total
        arr = ak.to_backend(arr_cpu, backend)

        if api == "sum":
            fn = lambda: ak.sum(arr, axis=-1)        # noqa: E731
        elif api == "max":
            fn = lambda: ak.max(arr, axis=-1)        # noqa: E731
        elif api == "min":
            fn = lambda: ak.min(arr, axis=-1)        # noqa: E731
        elif api == "prod":
            fn = lambda: ak.prod(arr, axis=-1)       # noqa: E731
        elif api == "count":
            fn = lambda: ak.count(arr, axis=-1)      # noqa: E731
        elif api == "countnz":
            fn = lambda: ak.count_nonzero(arr, axis=-1)  # noqa: E731
        elif api == "argmin":
            fn = lambda: ak.argmin(arr, axis=-1)     # noqa: E731
        elif api == "argmax":
            fn = lambda: ak.argmax(arr, axis=-1)     # noqa: E731
        elif api == "sort":
            fn = lambda: ak.sort(arr, axis=-1)       # noqa: E731
        elif api == "pad_none":
            # clip=True triggers awkward_index_rpad_and_clip_*
            target = max(2, avg_items // 2)
            fn = lambda: ak.pad_none(arr, target, axis=1, clip=True)  # noqa: E731
        else:
            raise ValueError(f"unknown api {api}")

        case.times_ms = time_one(fn, warmup, iters, backend)
        case.min_ms = min(case.times_ms)
        case.mean_ms = sum(case.times_ms) / len(case.times_ms)
    except Exception as e:
        case.error = f"{type(e).__name__}: {e}"
        # Truncate traceback to keep JSON small
        case.error += " | " + traceback.format_exc().splitlines()[-1]
    return case


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

ALL_APIS = ["sum", "max", "min", "prod", "count", "countnz",
            "argmin", "argmax", "sort", "pad_none"]

# Default shapes: kept modest so this runs in a few minutes total.
DEFAULT_SHAPES = [
    # (n_events, avg_items_per_event)
    (1_000_000, 10),    # many short lists  (~10M elements)
    (100_000, 1_000),   # fewer long lists  (~100M elements)
]

DEFAULT_DTYPES = ["float64", "int64"]
# Complex dtypes only exercise the *_complex kernel paths
COMPLEX_DTYPES = ["complex128"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="bench_results.json")
    p.add_argument("--apis", nargs="+", default=ALL_APIS, choices=ALL_APIS)
    p.add_argument("--dtypes", nargs="+", default=DEFAULT_DTYPES)
    p.add_argument("--include-complex", action="store_true",
                   help="Also benchmark complex dtypes for the reducers that support them.")
    p.add_argument("--shapes", nargs="+", default=None,
                   help='Override shapes, e.g. "1000000,10" "100000,1000"')
    p.add_argument("--backends", nargs="+", default=["cuda"], choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--label", default="",
                   help="Free-form label stored in the JSON (e.g. commit sha).")
    args = p.parse_args()

    _require_cupy_for_cuda(args.backends)

    import awkward as ak

    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes]
    else:
        shapes = DEFAULT_SHAPES

    # Capture environment metadata
    dev_name = "n/a"
    cupy_version = "n/a"
    if "cuda" in args.backends:
        import cupy as cp
        cupy_version = cp.__version__
        try:
            dev = cp.cuda.Device()
            dev_name = dev.attributes.get("Name", "")
            if not dev_name:
                dev_name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
        except Exception:
            dev_name = "unknown"

    try:
        import cuda.compute as _cc
        cccl_version = getattr(_cc, "__version__", "unknown")
    except Exception:
        cccl_version = "not-installed"

    meta = {
        "label": args.label,
        "ak_version": ak.__version__,
        "cupy_version": cupy_version,
        "cuda_compute_version": cccl_version,
        "device": dev_name,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "warmup": args.warmup,
        "iters": args.iters,
        "git_sha": _git_sha(),
    }

    print(f"Environment: {meta}")
    print(f"Benchmarking apis={args.apis}, dtypes={args.dtypes}, "
          f"shapes={shapes}, backends={args.backends}")

    cases = []
    for backend in args.backends:
        for n_events, avg_items in shapes:
            for dtype in args.dtypes:
                for api in args.apis:
                    # Some APIs don't accept bool input meaningfully
                    if dtype == "bool" and api in ("argmin", "argmax", "sort", "pad_none"):
                        continue
                    print(f"  {backend:4s} {api:9s} {dtype:10s} "
                          f"n_events={n_events:>8d} avg_items={avg_items:>6d} ... ",
                          end="", flush=True)
                    c = bench_api(api, dtype, n_events, avg_items,
                                  backend, args.warmup, args.iters)
                    if c.error:
                        print(f"ERROR: {c.error[:100]}")
                    else:
                        print(f"min={c.min_ms:.3f}ms mean={c.mean_ms:.3f}ms")
                    cases.append(c)

            if args.include_complex:
                for dtype in COMPLEX_DTYPES:
                    for api in ("sum", "max", "min", "prod", "argmin", "argmax"):
                        if api not in args.apis:
                            continue
                        print(f"  {backend:4s} {api:9s} {dtype:10s} "
                              f"n_events={n_events:>8d} avg_items={avg_items:>6d} ... ",
                              end="", flush=True)
                        c = bench_api(api, dtype, n_events, avg_items,
                                      backend, args.warmup, args.iters)
                        if c.error:
                            print(f"ERROR: {c.error[:100]}")
                        else:
                            print(f"min={c.min_ms:.3f}ms mean={c.mean_ms:.3f}ms")
                        cases.append(c)

    out = {"meta": meta, "cases": [asdict(c) for c in cases]}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output} ({len(cases)} cases)")


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
