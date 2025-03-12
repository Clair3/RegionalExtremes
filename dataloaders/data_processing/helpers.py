import xarray as xr

import dask.array as da
import numpy as np


def _compute_shifted_sums(data, window_size, direction=1):
    """Helper function to compute shifted sums and counts."""
    data = _ensure_xarray(data)
    sum_vals = xr.zeros_like(data)
    count_vals = xr.zeros_like(data, dtype=int)

    for i in range(1, window_size + 1):
        shifted = data.shift(time=i * direction)
        mask = ~np.isnan(shifted)
        sum_vals += xr.where(mask, shifted, 0)
        count_vals += mask.astype(int)

    return sum_vals, count_vals


def _ensure_xarray(data):
    """Ensure input is an xarray DataArray."""
    return data if isinstance(data, xr.DataArray) else xr.DataArray(data, dims=["time"])


def _ensure_time_chunks(data):
    if data.chunks[0][0] != 50:
        data = data.chunk({"time": 50, "location": -1})
    return data
