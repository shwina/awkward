# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

from awkward._connect.cuda._compute._utils import (
    CudaComputeUnsupportedError,
    _array_from_ptr,
)
from awkward._nplikes.cupy import Cupy


def _hi_minus_lo(hi, lo):
    """Compute difference between two int64 values."""
    return hi - lo


def _count_pairs(n):
    """Compute C(n,2) = n*(n-1)/2 for n>=2, else 0."""
    return (n * (n - 1) // 2) if n >= 2 else np.int64(0)


def _unrank_k2(input_tuple):
    """
    Unrank a global output index to (out0, out1) gather indices.

    Input: (g, b, nn, out_base_val) where:
      g = global output index
      b = base offset for this segment (precomputed)
      nn = segment length (precomputed)
      out_base_val = output base offset for this segment (precomputed)
    Output: (out0, out1) gather indices into flattened values
    """
    g = input_tuple[0]
    b = input_tuple[1]
    nn = input_tuple[2]
    out_base_val = input_tuple[3]

    r = g - out_base_val

    # Initial guess via quadratic inversion (k=2).
    bf = np.float64(2 * nn - 1)
    disc = bf * bf - np.float64(8) * np.float64(r)
    ii = np.int64((bf - disc**0.5) * 0.5)  # trunc==floor since x>=0

    # Fix-up using S(i) = i*(2n - i - 1)/2 (inline)
    while ii > 0:
        Si = ii * (2 * nn - ii - 1) // 2
        if Si <= r:
            break
        ii -= 1

    while (ii + 1) < nn:
        i2 = ii + 1
        Si2 = i2 * (2 * nn - i2 - 1) // 2
        if Si2 > r:
            break
        ii = i2

    Si = ii * (2 * nn - ii - 1) // 2
    t = r - Si
    i = ii
    j = ii + 1 + t

    return (b + i, b + j)


def combinations_length(
    totallen,
    tooffsets,
    n,
    replacement,
    starts,
    stops,
    length,
):
    """
    Compute the output offsets and total length for combinations.

    Currently only supports n=2 without replacement using CCCL.

    Args:
        totallen: output array of shape (1,) to store total combinations count
        tooffsets: output array of shape (length+1,) for output offsets
        n: number of items to choose (must be 2 for CCCL path)
        replacement: whether to allow replacement (must be False for CCCL path)
        starts: input starts array
        stops: input stops array
        length: number of lists

    Raises:
        CudaComputeUnsupportedError: if n != 2 or replacement is True
    """
    # Only support n=2 without replacement for now
    if n != 2 or replacement:
        raise CudaComputeUnsupportedError(
            f"cuda.compute combinations only supports n=2 without replacement, "
            f"got n={n}, replacement={replacement}"
        )

    import cuda.compute as cc
    from cuda.compute import OpKind

    cupy_nplike = Cupy.instance()
    cp = cupy_nplike._module

    # Ensure int64
    starts64 = starts.astype(cp.int64, copy=False)
    stops64 = stops.astype(cp.int64, copy=False)

    # Compute lengths: lengths[i] = stops[i] - starts[i]
    lengths = cp.empty(length, dtype=cp.int64)
    cc.binary_transform(stops64, starts64, lengths, _hi_minus_lo, length)

    # Compute counts: counts[i] = C(lengths[i], 2)
    counts = cp.empty(length, dtype=cp.int64)
    cc.unary_transform(lengths, counts, _count_pairs, length)

    # Compute output offsets via exclusive scan
    init = np.array([0], dtype=np.int64)
    cc.exclusive_scan(counts, tooffsets[:length], OpKind.PLUS, init, length)

    # Set total and final offset
    if length > 0:
        total_pairs = int((tooffsets[length - 1] + counts[length - 1]).item())
    else:
        total_pairs = 0
    tooffsets[length] = np.int64(total_pairs)
    totallen[0] = np.int64(total_pairs)


def combinations(
    tocarry,
    toindex,
    fromindex,
    n,
    replacement,
    starts,
    stops,
    length,
):
    """
    Compute the gather indices for combinations.

    Currently only supports n=2 without replacement using CCCL.

    Args:
        tocarry: array of pointers to n output arrays (each of size totallen)
        toindex: work array of size n
        fromindex: work array of size n
        n: number of items to choose (must be 2 for CCCL path)
        replacement: whether to allow replacement (must be False for CCCL path)
        starts: input starts array
        stops: input stops array
        length: number of lists

    Raises:
        CudaComputeUnsupportedError: if n != 2 or replacement is True
    """
    # Only support n=2 without replacement for now
    if n != 2 or replacement:
        raise CudaComputeUnsupportedError(
            f"cuda.compute combinations only supports n=2 without replacement, "
            f"got n={n}, replacement={replacement}"
        )

    import cuda.compute as cc
    from cuda.compute import OpKind, ZipIterator

    cupy_nplike = Cupy.instance()
    cp = cupy_nplike._module

    # Ensure int64
    starts64 = starts.astype(cp.int64, copy=False)
    stops64 = stops.astype(cp.int64, copy=False)

    # Compute lengths
    lengths = cp.empty(length, dtype=cp.int64)
    cc.binary_transform(stops64, starts64, lengths, _hi_minus_lo, length)

    # Compute counts
    counts = cp.empty(length, dtype=cp.int64)
    cc.unary_transform(lengths, counts, _count_pairs, length)

    # Compute output offsets
    out_offsets = cp.empty(length + 1, dtype=cp.int64)
    out_offsets[0] = np.int64(0)
    init = np.array([0], dtype=np.int64)
    cc.exclusive_scan(counts, out_offsets[:length], OpKind.PLUS, init, length)

    if length > 0:
        total_pairs = int((out_offsets[length - 1] + counts[length - 1]).item())
    else:
        total_pairs = 0
    out_offsets[length] = np.int64(total_pairs)

    if total_pairs == 0:
        return

    # Get the output arrays from tocarry (array of pointers)
    # tocarry is an ndarray containing pointers; we need to interpret it
    # For CuPy arrays passed from awkward, tocarry[i] contains the data pointer
    # We need to get the actual arrays - they're passed via the kernel interface

    # The tocarry is passed as an array of pointers (intp type)
    # We need to work with the actual destination arrays
    # In the awkward kernel interface, tocarry contains raw pointers
    # We reconstruct cupy arrays from these pointers

    out0 = _array_from_ptr(tocarry[0], total_pairs, cp.int64, cp)
    out1 = _array_from_ptr(tocarry[1], total_pairs, cp.int64, cp)

    # Precompute lookup arrays
    base_arr = starts64
    n_arr = lengths
    out_base_arr = out_offsets[:-1]

    # Process in chunks to avoid memory issues with large arrays
    chunk_size = 4_000_000
    start = 0
    while start < total_pairs:
        end = min(total_pairs, start + chunk_size)
        m = end - start

        g_chunk = cp.arange(start, end, dtype=cp.int64)
        seg_chunk = cp.searchsorted(out_offsets[1:], g_chunk, side="right").astype(
            cp.int64, copy=False
        )

        # Gather per-element values from lookup arrays
        b_vals = base_arr[seg_chunk]
        nn_vals = n_arr[seg_chunk]
        out_base_vals = out_base_arr[seg_chunk]

        zip_in = ZipIterator(g_chunk, b_vals, nn_vals, out_base_vals)
        zip_out = ZipIterator(out0[start:end], out1[start:end])

        cc.unary_transform(zip_in, zip_out, _unrank_k2, m)

        start = end

