# CUDA Kernel Bugs Found and Fixed

## Bug 1: awkward_ListOffsetArray_rpad_axis1.cu
**Issue**: `scan_in_array` allocated with size `fromlength`, but kernel writes to `scan_in_array[thread_id + 1]` where `thread_id` ranges from 0 to `fromlength-1`, causing write to index `fromlength` (out of bounds).

**Fix**: Changed allocation from `cupy.zeros(fromlength, ...)` to `cupy.zeros(fromlength + 1, ...)` in Python comment (line 6).

**Reason**: Prefix-sum pattern needs N+1 elements for N items (element 0 is the starting offset).

---

## Bug 2: awkward_reduce_sum_complex.cu
**Issue**: Inside reduction loop, `temp[thread_id * 2]` accessed without bounds check. When `thread_id >= lenparents`, this writes beyond allocated `temp` array size `2 * lenparents`.

**Fix**: Added bounds check around temp writes (lines 67-68):
```cuda
if (thread_id < lenparents) {
    temp[thread_id * 2] += real;
    temp[thread_id * 2 + 1] += imag;
}
```

**Pattern**: Kernel launches more threads than data elements; threads beyond valid range must not access arrays.

---

## Bug 3: awkward_reduce_countnonzero_complex.cu
**Issue**: Same pattern as Bug 2 - `temp[thread_id] += val` accessed without bounds check inside reduction loop.

**Fix**: Added bounds check around temp write (line 63):
```cuda
if (thread_id < lenparents) {
    temp[thread_id] += val;
}
```

**Pattern**: Same as Bug 2.

---

## Bug 4: awkward_UnionArray_regular_index.cu
**Issue**: Kernel `_c` checks `thread_id < length` and accesses `atomicAdd_toptr[thread_id]` and `current[thread_id]`, but both arrays have size `size` (not `length`). When `length > size`, out-of-bounds access occurs.

**Fix**: Changed condition from `if (thread_id < length)` to `if (thread_id < size)` at line 67.

**Reason**: Both `atomicAdd_toptr` and `current` have size `size`. The kernel copies final counts from `atomicAdd_toptr` back to `current`, iterating over `size` elements, not `length`.

---

## Bug 5: awkward_reduce_argmin/argmax (_compute.py)
**Issue**: Used `segmented_reduce` from cuda.compute with global indices array sized `parents_length + 1`. For empty segments, result contained out-of-bounds indices that caused `awkward_NumpyArray_reduce_adjust_starts_64` to access `parents[-1]` or beyond array bounds.

**Fix**: Replaced `segmented_reduce` approach with `unary_transform` and explicit per-segment logic. Empty segments now correctly return -1, which the adjust_starts kernel already handles with `if (i >= 0)` check.

**Reason**: The `segmented_reduce` API had issues handling empty segments and struct unpacking. Direct per-segment computation with `unary_transform` is simpler and more reliable.

---

## Bug 6: awkward_ListArray_getitem_jagged_shrink.cu
**Issue**: `scan_in_array_k` allocated with size `len_array`, but kernel writes to `scan_in_array_k[thread_id + 1]` where `thread_id < length`, requiring size `length + 1`.

**Fix**: Changed allocation from `cupy.zeros(len_array, ...)` to `cupy.zeros(length + 1, ...)` at line 10.

**Reason**: All three scan arrays are written to with `[thread_id + 1]` pattern and need consistent size `length + 1`.

---

## Bug 7: awkward_reduce_prod_complex.cu & ListOffsetArray
**Issue**: Two related issues:
1. `temp` array allocated with size `2*lenparents` but needs `2*grid_size*block[0]` to hold all threads during tree reduction
2. ListOffsetArray._reduce_next computed `nextlen = offsets[-1] - offsets[0]` without clamping to actual content length, causing kernels to be called with lenparents > actual data size

**Fix**:
1. Changed temp allocation from `cupy.tile([1, 0], lenparents)` to `cupy.zeros(2 * grid_size * block[0], dtype=toptr.dtype)` in Python comment (line 10)
2. Added conditional clamping after computing nextlen in 3 places in listoffsetarray.py (lines 1344, 1622, 1690):
   ```python
   if nextlen is not unknown_length and self.content.length is not unknown_length:
       nextlen = min(nextlen, self.content.length)
   ```
   (The conditional check is needed to support TypeTracer mode where lengths can be unknown)

**Reason**: Tree reduction requires temp space for ALL threads in block, not just active threads with data. Offsets can reference indices beyond content length, causing invalid lenparents.

---

## Bug 8: awkward_ListOffsetArray_reduce_local_nextparents_64
**Issue**: Kernel writes to `nextparents[j]` where `j` ranges from `offsets[i] - initialoffset` to `offsets[i+1] - initialoffset`, but `nextparents` array was allocated with size clamped to content length. When offsets reference indices beyond content length, kernel writes beyond allocated array bounds.

**Fix**: Added `nextparents_length` parameter to kernel signature and added bounds check `j < nextparents_length` in the inner loop:
- Updated kernel-specification.yml to add 4th parameter `nextparents_length` to all 3 specializations
- Updated CUDA kernel (.cu file) to add parameter and bounds check: `for (...; j < stop && j < nextparents_length; j += blockDim.y)`
- Updated CPU kernel (.cpp file) to add parameter and bounds check: `j < offsets[i + 1] - initialoffset && j < nextparents_length`
- Updated all 5 call sites in listoffsetarray.py to pass `nextparents.length` as 4th argument
- Regenerated kernel signatures with `nox -s prepare`
- Copied updated `_kernel_signatures.py` to venv and rebuilt awkward-cpp library with `pip install -e .`

**Reason**: Kernel used offset values directly to compute array indices without knowing the actual allocated array size. The caller (Bug #7 fix) clamped nextlen but didn't communicate this bound to the kernel.
