from __future__ import annotations

import cupy
import numpy as np

import awkward as ak


def build_array(num_lists: int, mean_len: int, max_len: int):
    rng = np.random.default_rng(42)
    lengths = rng.poisson(lam=mean_len, size=num_lists).astype(np.int64)
    lengths = np.clip(lengths, 0, max_len)
    offsets = np.empty(num_lists + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    values = np.random.rand(int(offsets[-1]))
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(offsets),
        ak.contents.NumpyArray(values),
    )
    return ak.Array(layout)


def bench(num_lists=1_000_000, mean_len=10, max_len=40, repeat=20):
    array = build_array(num_lists, mean_len, max_len)
    gpu_array = ak.to_backend(array, "cuda")

    start_event = cupy.cuda.Event()
    end_event = cupy.cuda.Event()
    times = []

    # Warm up to trigger JIT compilation
    ak.sum(gpu_array, axis=-1)
    cupy.cuda.Device().synchronize()

    import cProfile
    import pstats

    profiler = cProfile.Profile()

    profiler.enable()
    for _ in range(repeat):
        start_event.record()
        _ = ak.sum(gpu_array, axis=-1)

        end_event.record()
        end_event.synchronize()
        times.append(cupy.cuda.get_elapsed_time(start_event, end_event))
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("profile.prof")
    return times


def main():
    times = bench()
    avg = sum(times) / len(times) if times else 0.0
    print(f"cuda.compute sum_offsets times (ms): {times}")
    print(f"avg (ms): {avg:.4f}")


if __name__ == "__main__":
    main()
