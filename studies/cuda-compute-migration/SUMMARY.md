# CUDA → cuda.compute migration: status, LOC, benchmarks

Snapshot taken on 2026-05-26 against `upstream/main` (`1455cfe7`) and
`upstream/awkward3` (`496e74f7`), which holds the still-in-flight bulk
migration work.

## 1. How many kernels are migrated?

The CUDA backend has 130 `awkward_*.cu` kernel files. A kernel counts as
"migrated" when (a) a Python implementation exists in
`src/awkward/_connect/cuda/_compute.py` (or `reducers.py`) using the
`cuda.compute` library, **and** (b) the dispatcher in
`src/awkward/_backends/cupy.py` routes the kernel name to that
implementation.

### Merged on `main` (in production today)

| Kernel(s) | PR | Status |
|---|---|---|
| `awkward_sort` (segmented_sort) | [#3750] | merged — first GPU sort |
| `awkward_reduce_argmax` | [#3777] | merged |
| `awkward_reduce_argmin` | [#3811] | merged |
| `awkward_reduce_sum`, `*_int32_bool_64`, `*_int64_bool_64`, `*_bool` | [#3840] | merged (kernels reimplemented via cccl) |
| `awkward_reduce_max`, `_min`, `_prod`, `_prod_bool`, `_count_64`, `_countnonzero` | [#3840] | merged |
| Custom-overload reducers (`ArgMin/ArgMax` axis=None path) | [#3894] | merged |
| `awkward_missing_repeat` | [#3922] | merged |
| `awkward_index_rpad_and_clip_axis1` | [#3923] | merged |
| `awkward_index_rpad_and_clip_axis0` | [#4012] | merged |
| `awkward_reduce_sum_complex` | [#4016] | merged |
| `awkward_reduce_max_complex` | [#4018] | merged |

**Total on `main`: 17 distinct kernel names migrated**, plus 2 axis-none
reducer helpers (`awkward_axis_none_reduce_argmax/argmin`) which are
cuda.compute-only paths with no `.cu` predecessor.

### Merged on `awkward3` branch (next release line, not yet on `main`)

Everything above, plus:

| Kernel(s) | PR | Notes |
|---|---|---|
| Bulk migration of dozens of structural kernels | [#4019] | indexed/byte-masked/list/regular array kernels (~+1959 LOC in `_compute.py`) |
| Initial batch of non-reducer kernels | [#3981] | |
| `awkward_reduce_max_complex`, `_min_complex` | [#4018], (PR #4017 open) | merged on awkward3 |

The functional count in `_compute.py` on `awkward3` is **100** vs **23**
on `main`, so the in-flight migration ~5× the surface that's live today.

### Open PRs (in progress)

| PR | Kernel(s) | +/- |
|---|---|---|
| [#4017] | `awkward_reduce_min_complex` | +71 / -2 |
| [#4044] | `awkward_IndexedArray_fill`, `awkward_IndexedArray_fill_count` | +63 / -2 |
| [#3977] | `awkward_IndexedArray_simplify` | +30 / -2 |
| [#3971] | `awkward_IndexedArray_flatten_nextcarry` | +35 / -2 |
| [#3960] | `awkward_IndexedArray_numnull` | +19 / -2 |
| [#3932] | `awkward_ListArray_getitem_jagged_{carrylen,descend,numvalid}` | +108 / -15 |
| [#3920] | `awkward_localindex` | +18 / -1 |
| [#3806] | perf-tuning + lazy `parents` allocation for axis=None reducers | +819 / -88 |

## 2. LOC change

Each migrated kernel replaces a `.cu` file (CUDA C++ template) with a
small Python function that drives `cuda.compute`. The CPU baseline
(`awkward-cpp/src/cpu-kernels/<name>.cpp`) is unchanged.

| Kernel | Old `.cu` LOC | New Python LOC | Δ |
|---|---:|---:|---:|
| `awkward_sort` (new on GPU) | 0 | 39 | +39 |
| `awkward_reduce_argmax` | 156 | 33 | **-123** |
| `awkward_reduce_argmin` | 157 | 33 | **-124** |
| `awkward_reduce_sum` | 76 | 35 | -41 |
| `awkward_reduce_sum_int32_bool_64` | 76 | 39 (shared) | -37 |
| `awkward_reduce_sum_int64_bool_64` | 76 | 39 (shared) | -37 |
| `awkward_reduce_sum_bool` | 88 | 37 | -51 |
| `awkward_reduce_sum_complex` | 84 | 52 | -32 |
| `awkward_reduce_max` | 78 | 40 | -38 |
| `awkward_reduce_max_complex` | 86 | 63 | -23 |
| `awkward_reduce_min` | 161 | 40 | **-121** |
| `awkward_reduce_prod` | 103 | 36 | -67 |
| `awkward_reduce_prod_bool` | 103 | 37 | -66 |
| `awkward_reduce_count_64` | 72 | 37 | -35 |
| `awkward_reduce_countnonzero` | 74 | 43 | -31 |
| `awkward_missing_repeat` | 22 | 22 | 0 |
| `awkward_index_rpad_and_clip_axis0` | 22 | 22 | 0 |
| `awkward_index_rpad_and_clip_axis1` | 19 | 7 | -12 |
| axis=None `argmax/argmin` (new) | 0 | 130 | +130 |

**Net LOC change on `main`: ~-668** (Python ranges aggressively reuse
shared helpers; the `.cu` files are pure-C++ template machinery).

(Same methodology applied to the `awkward3` branch would show ~+1959 net
LOC in `_compute.py` for PR #4019 alone — because the structural kernels
being migrated there were *not* trivial reductions, and several of them
do not have a corresponding short `.cu`.)

Note: at the time of writing the `.cu` files are still on disk in
`src/awkward/_connect/cuda/cuda_kernels/`. They're skipped at runtime via
`dev/generate-kernel-signatures.py`, which writes `out[...] = None` for
migrated names so the dispatcher falls through to the
`cuda.compute` implementation. Removing the files is a follow-up.

## 3. Performance: how to measure

I cannot run on a GPU; the scripts in this directory let you do that.

### Pick public-facing APIs (not kernels) so we can run on old commits

| API | Migrated kernel(s) it exercises |
|---|---|
| `ak.sum(arr, axis=-1)` | `awkward_reduce_sum*` |
| `ak.max(arr, axis=-1)` | `awkward_reduce_max*` |
| `ak.min(arr, axis=-1)` | `awkward_reduce_min*` |
| `ak.prod(arr, axis=-1)` | `awkward_reduce_prod*` |
| `ak.count(arr, axis=-1)` | `awkward_reduce_count_64` |
| `ak.count_nonzero(arr, axis=-1)` | `awkward_reduce_countnonzero` |
| `ak.argmin(arr, axis=-1)` | `awkward_reduce_argmin*` |
| `ak.argmax(arr, axis=-1)` | `awkward_reduce_argmax*` |
| `ak.sort(arr, axis=-1)` | `awkward_sort` (no pre-migration GPU path — expect error on old commit) |
| `ak.pad_none(arr, k, axis=1, clip=True)` | `awkward_index_rpad_and_clip_axis0/1` |

If you want to keep the benchmark suite to 3–4 APIs, pick
`ak.sum`, `ak.argmin`, `ak.sort`, `ak.pad_none(.., clip=True)` — they
cover the four code paths (segmented reduce, segmented arg-reduce,
segmented sort, transform).

### How to run

1. Install once on the target GPU machine:
   ```bash
   pip install -e .[test,gpu]
   pip install cuda-compute   # whichever package provides cuda.compute
   ```

2. Single-shot benchmark at the current commit:
   ```bash
   cd studies/cuda-compute-migration
   python bench_public_apis.py --output bench_main.json --label "main-HEAD"
   ```

3. Compare two commits:
   ```bash
   # On a separate worktree so this script can checkout safely
   git worktree add /tmp/awkward-bench main
   cd /tmp/awkward-bench
   ./studies/cuda-compute-migration/run_compare.sh c7e21ea0 main \
        -- --apis sum argmin sort pad_none --iters 20
   ```

   - `c7e21ea0` is the commit *just before* PR #3750 (the first
     cuda.compute integration). On `main`, that's
     "pre-everything" — every migrated reducer is still using its old
     `.cu` kernel, and `ak.sort` on GPU will error out (the script
     records the error in the JSON and continues).
   - The driver writes
     `studies/cuda-compute-migration/results/{bench_before,bench_after}.json`
     and a `report.md` with a side-by-side comparison.

4. Useful three-way comparison (today's `main` vs in-flight `awkward3`):
   ```bash
   ./studies/cuda-compute-migration/run_compare.sh main upstream/awkward3
   ```

### What to look for

- **Reducers**: cccl's `unary_transform` path has known overhead vs the
  hand-tuned `.cu` warp-reduce. Expect mixed results on small batches
  (`avg_items ~ 10`, `n_events ~ 1M`); large batches (`avg_items ~ 1000`)
  should be competitive.
- **`ak.argmin/argmax`**: Look for the per-call overhead — the
  cuda.compute version replaces a tight C++ template with a JIT'ed
  Python `op`, so first-call cost is much higher (handled by `--warmup`).
- **`ak.sort`**: No pre-migration baseline. Compare CUDA vs CPU instead:
  `python bench_public_apis.py --backends cpu cuda --apis sort` and look
  at the speedup column in `compare.py`.
- **`ak.pad_none(.., clip=True)`**: Tiny kernels — wall-clock will be
  dominated by Python+launch overhead.

### Sanity checks before trusting numbers

- `cp.cuda.Device().synchronize()` is called between iterations; if the
  workload is too short the JIT/launch overhead dominates. Scale up
  shapes (`--shapes 10000000,100`) if min/mean numbers look flat.
- The JSON includes `git_sha`, library versions and device name — paste
  the meta block into any report.
- If `cuda.compute` is not installed the script still runs (CPU paths +
  legacy CuPy paths), which is useful for pre-migration baselines.

[#3750]: https://github.com/scikit-hep/awkward/pull/3750
[#3777]: https://github.com/scikit-hep/awkward/pull/3777
[#3811]: https://github.com/scikit-hep/awkward/pull/3811
[#3840]: https://github.com/scikit-hep/awkward/pull/3840
[#3894]: https://github.com/scikit-hep/awkward/pull/3894
[#3922]: https://github.com/scikit-hep/awkward/pull/3922
[#3923]: https://github.com/scikit-hep/awkward/pull/3923
[#3981]: https://github.com/scikit-hep/awkward/pull/3981
[#4012]: https://github.com/scikit-hep/awkward/pull/4012
[#4016]: https://github.com/scikit-hep/awkward/pull/4016
[#4018]: https://github.com/scikit-hep/awkward/pull/4018
[#4019]: https://github.com/scikit-hep/awkward/pull/4019
[#3806]: https://github.com/scikit-hep/awkward/pull/3806
[#3920]: https://github.com/scikit-hep/awkward/pull/3920
[#3932]: https://github.com/scikit-hep/awkward/pull/3932
[#3960]: https://github.com/scikit-hep/awkward/pull/3960
[#3971]: https://github.com/scikit-hep/awkward/pull/3971
[#3977]: https://github.com/scikit-hep/awkward/pull/3977
[#4017]: https://github.com/scikit-hep/awkward/pull/4017
[#4044]: https://github.com/scikit-hep/awkward/pull/4044
