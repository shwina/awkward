# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._backends.backend import Backend, KernelKeyType
from awkward._backends.dispatch import register_backend
from awkward._kernels import (
    CudaComputeKernel,
    CudaComputeKernelWithFallback,
    CupyKernel,
    NumpyKernel,
)
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Final

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@register_backend(Cupy)  # type: ignore[type-abstract]
class CupyBackend(Backend):
    name: Final[str] = "cuda"

    _cupy: Cupy

    @property
    def nplike(self) -> Cupy:
        return self._cupy

    def __init__(self):
        self._cupy = Cupy.instance()

    def __getitem__(
        self, index: KernelKeyType
    ) -> CudaComputeKernel | CupyKernel | NumpyKernel:
        from awkward._connect import cuda
        from awkward._connect.cuda import _compute as cuda_compute

        kernel_name = index[0] if index else ""

        # For operations where CCCL is preferred (faster), try CCCL first
        if self._prefers_cuda_compute(kernel_name):
            if cuda_compute.is_available():
                compute_impl = self._get_cuda_compute_impl(kernel_name)
                if compute_impl is not None:
                    return CudaComputeKernelWithFallback(
                        compute_impl, index, self._get_cupy_kernel
                    )
            # Fall through to CuPy if CCCL not available

        # Try CuPy kernels
        cupy = cuda.import_cupy("Awkward Arrays with CUDA")
        _cuda_kernels = cuda.initialize_cuda_kernels(cupy)
        func = _cuda_kernels[index]

        if func is not None:
            # CuPy kernel exists, use it
            return CupyKernel(func, index)

        # CuPy kernel not found, try cuda.compute as fallback
        if self._supports_cuda_compute(kernel_name):
            if cuda_compute.is_available():
                # Return CudaComputeKernel for supported operations
                compute_impl = self._get_cuda_compute_impl(kernel_name)
                if compute_impl is not None:
                    return CudaComputeKernel(compute_impl, index)
            else:
                # cuda.compute is needed but not available
                raise NotImplementedError(
                    f"Operation '{kernel_name}' on CUDA backend requires cuda.compute library "
                    f"(no CuPy kernel available). "
                    f"Please install cuda.compute or use the CPU backend: "
                    f"ak.to_backend(array, 'cpu')"
                )

        # Neither CuPy kernel nor cuda.compute implementation found
        raise AssertionError(
            f"Operation '{kernel_name}' is not supported on CUDA backend. "
            f"CuPy kernel not found: {index!r}"
        )

    def _get_cupy_kernel(self, index: KernelKeyType) -> CupyKernel:
        """Get the CuPy kernel for the given index."""
        from awkward._connect import cuda

        cupy = cuda.import_cupy("Awkward Arrays with CUDA")
        _cuda_kernels = cuda.initialize_cuda_kernels(cupy)
        func = _cuda_kernels[index]
        if func is None:
            raise AssertionError(f"CuPy kernel not found: {index!r}")
        return CupyKernel(func, index)

    def _prefers_cuda_compute(self, kernel_name: str) -> bool:
        """
        Check if the given kernel operation should prefer cuda.compute over CuPy.

        These are operations where the cuda.compute implementation is significantly
        faster than the CuPy kernel, but the CuPy kernel still works as a fallback.

        Currently prefers cuda.compute for:
        - awkward_ListArray_combinations_length (k=2 only)
        - awkward_ListArray_combinations (k=2 only)
        """
        return kernel_name in (
            "awkward_ListArray_combinations_length",
            "awkward_ListArray_combinations",
        )

    def _supports_cuda_compute(self, kernel_name: str) -> bool:
        """
        Check if the given kernel operation is supported by cuda.compute.

        These are operations where cuda.compute is required (no CuPy kernel exists).

        Currently supports:
        - awkward_sort
        """
        return kernel_name in ("awkward_sort",)

    def _get_cuda_compute_impl(self, kernel_name: str):
        """
        Get the cuda.compute implementation for a kernel operation.

        Args:
            kernel_name: Name of the kernel operation (e.g., "awkward_sort")

        Returns:
            Callable implementing the operation, or None if not supported
        """
        from awkward._connect.cuda import _compute as cuda_compute

        if kernel_name == "awkward_sort":
            return cuda_compute.segmented_sort
        elif kernel_name == "awkward_ListArray_combinations_length":
            return cuda_compute.combinations_length
        elif kernel_name == "awkward_ListArray_combinations":
            return cuda_compute.combinations

        return None
