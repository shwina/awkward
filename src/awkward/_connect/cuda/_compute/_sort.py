# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._nplikes.cupy import Cupy


def segmented_sort(
    toptr,
    fromptr,
    length,
    offsets,
    offsetslength,
    parentslength,
    ascending,
    stable,
):
    from cuda.compute import SortOrder, segmented_sort

    cupy_nplike = Cupy.instance()
    cp = cupy_nplike._module

    # Ensure offsets are int64 as expected by segmented_sort
    if offsets.dtype != cp.int64:
        offsets = offsets.astype(cp.int64)

    num_segments = offsetslength - 1
    num_items = int(offsets[-1]) if len(offsets) > 0 else 0

    start_offsets = offsets[:-1]
    end_offsets = offsets[1:]

    order = SortOrder.ASCENDING if ascending else SortOrder.DESCENDING

    segmented_sort(
        fromptr,  # d_in_keys
        toptr,  # d_out_keys
        None,  # d_in_values (not sorting values, just keys)
        None,  # d_out_values
        num_items,  # num_items
        num_segments,  # num_segments
        start_offsets,  # start_offsets_in
        end_offsets,  # end_offsets_in
        order,  # order (ASCENDING or DESCENDING)
        None,  # stream (use default stream)
    )

