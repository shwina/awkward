# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np


class CudaComputeUnsupportedError(Exception):
    """
    Raised when cuda.compute cannot handle the operation.

    This signals to the backend that it should fall back to CuPy kernels.
    """

    pass


# Cache for cuda.compute availability
_cuda_compute_available: bool | None = None


def is_available() -> bool:
    global _cuda_compute_available

    if _cuda_compute_available is not None:
        return _cuda_compute_available

    try:
        import cuda.compute  # noqa: F401

        _cuda_compute_available = True
    except ImportError:
        _cuda_compute_available = False

    return _cuda_compute_available


def _array_from_ptr(ptr, size, dtype, cp):
    """
    Create a CuPy array from a raw device pointer.

    Args:
        ptr: Raw device pointer (as integer)
        size: Number of elements
        dtype: NumPy/CuPy dtype
        cp: CuPy module

    Returns:
        CuPy array viewing the memory at ptr
    """
    mem = cp.cuda.UnownedMemory(int(ptr), size * np.dtype(dtype).itemsize, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(size, dtype=dtype, memptr=memptr)

