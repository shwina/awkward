# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list

# Filter out deprecation warning from cuda.core.experimental (used by numba-cuda)
pytestmark = pytest.mark.filterwarnings(
    "ignore:The cuda.core.experimental namespace is deprecated:DeprecationWarning"
)


def test_combinations_cuda_basic():
    """Test basic k=2 combinations on CUDA."""
    data = ak.Array([[1, 2, 3], [4, 5], [6]])
    gpu_data = ak.to_backend(data, "cuda")
    gpu_result = ak.combinations(gpu_data, 2)
    result = ak.to_backend(gpu_result, "cpu")

    expected = [[(1, 2), (1, 3), (2, 3)], [(4, 5)], []]
    assert to_list(result) == expected


def test_combinations_cuda_empty_lists():
    """Test k=2 combinations with empty lists."""
    data = ak.Array([[], [1, 2, 3], [], [4, 5]])
    gpu_data = ak.to_backend(data, "cuda")
    gpu_result = ak.combinations(gpu_data, 2)
    result = ak.to_backend(gpu_result, "cpu")

    expected = [[], [(1, 2), (1, 3), (2, 3)], [], [(4, 5)]]
    assert to_list(result) == expected


def test_combinations_cuda_single_element_lists():
    """Test k=2 combinations with single-element lists (should produce empty)."""
    data = ak.Array([[1], [2], [3, 4]])
    gpu_data = ak.to_backend(data, "cuda")
    gpu_result = ak.combinations(gpu_data, 2)
    result = ak.to_backend(gpu_result, "cpu")

    expected = [[], [], [(3, 4)]]
    assert to_list(result) == expected


def test_combinations_cuda_large():
    """Test k=2 combinations with larger data."""
    np.random.seed(42)
    data_list = []
    for _ in range(100):
        size = np.random.randint(0, 20)
        if size > 0:
            data_list.append(np.random.randint(0, 100, size).tolist())
        else:
            data_list.append([])

    data = ak.Array(data_list)
    cpu_result = ak.combinations(data, 2)

    gpu_data = ak.to_backend(data, "cuda")
    gpu_result = ak.combinations(gpu_data, 2)
    result = ak.to_backend(gpu_result, "cpu")

    assert to_list(result) == to_list(cpu_result)


def test_combinations_cuda_fields():
    """Test k=2 combinations with field names."""
    data = ak.Array([[1, 2, 3], [4, 5]])
    gpu_data = ak.to_backend(data, "cuda")
    gpu_result = ak.combinations(gpu_data, 2, fields=["x", "y"])
    result = ak.to_backend(gpu_result, "cpu")

    expected = [
        [{"x": 1, "y": 2}, {"x": 1, "y": 3}, {"x": 2, "y": 3}],
        [{"x": 4, "y": 5}],
    ]
    assert to_list(result) == expected


def test_combinations_cuda_nested():
    """Test k=2 combinations with nested arrays."""
    data = ak.Array([[[1, 2, 3], [4, 5]], [[6, 7]]])
    cpu_result = ak.combinations(data, 2, axis=-1)

    gpu_data = ak.to_backend(data, "cuda")
    gpu_result = ak.combinations(gpu_data, 2, axis=-1)
    result = ak.to_backend(gpu_result, "cpu")

    assert to_list(result) == to_list(cpu_result)


def test_combinations_cuda_n3_fallback():
    """Test that n=3 combinations falls back to CuPy kernels."""
    data = ak.Array([[1, 2, 3, 4], [5, 6, 7]])
    cpu_result = ak.combinations(data, 3)

    gpu_data = ak.to_backend(data, "cuda")
    gpu_result = ak.combinations(gpu_data, 3)
    result = ak.to_backend(gpu_result, "cpu")

    assert to_list(result) == to_list(cpu_result)


def test_combinations_cuda_replacement_fallback():
    """Test that combinations with replacement falls back to CuPy kernels."""
    data = ak.Array([[1, 2, 3], [4, 5]])
    cpu_result = ak.combinations(data, 2, replacement=True)

    gpu_data = ak.to_backend(data, "cuda")
    gpu_result = ak.combinations(gpu_data, 2, replacement=True)
    result = ak.to_backend(gpu_result, "cpu")

    assert to_list(result) == to_list(cpu_result)


def test_combinations_cuda_no_compute():
    """Test that combinations work when cuda.compute is not available (uses CuPy)."""
    from awkward._connect.cuda import _compute as cuda_compute

    original_available = cuda_compute._cuda_compute_available

    try:
        # Temporarily make cuda.compute unavailable
        cuda_compute._cuda_compute_available = False

        data = ak.Array([[1, 2, 3], [4, 5], [6]])
        gpu_data = ak.to_backend(data, "cuda")
        gpu_result = ak.combinations(gpu_data, 2)
        result = ak.to_backend(gpu_result, "cpu")

        expected = [[(1, 2), (1, 3), (2, 3)], [(4, 5)], []]
        assert to_list(result) == expected

    finally:
        # Restore original state
        cuda_compute._cuda_compute_available = original_available


def test_combinations_cuda_matches_cpu():
    """Comprehensive test that CUDA combinations matches CPU for various inputs."""
    test_cases = [
        [[1, 2, 3, 4], [], [5], [6, 7, 8]],
        [[], [], []],
        [[1, 2]],
        [[i for i in range(10)], [i for i in range(5)]],
    ]

    for data_list in test_cases:
        data = ak.Array(data_list)
        cpu_result = ak.combinations(data, 2)

        gpu_data = ak.to_backend(data, "cuda")
        gpu_result = ak.combinations(gpu_data, 2)
        result = ak.to_backend(gpu_result, "cpu")

        assert to_list(result) == to_list(cpu_result), f"Mismatch for input: {data_list}"

