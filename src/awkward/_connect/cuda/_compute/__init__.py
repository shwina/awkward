# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._connect.cuda._compute._combinations import (
    combinations,
    combinations_length,
)
from awkward._connect.cuda._compute._sort import segmented_sort
from awkward._connect.cuda._compute._utils import (
    CudaComputeUnsupportedError,
    is_available,
)

__all__ = [
    "CudaComputeUnsupportedError",
    "combinations",
    "combinations_length",
    "is_available",
    "segmented_sort",
]

