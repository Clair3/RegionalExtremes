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


def circular_rolling_mean(arr, window_size=4, min_periods=1):
    """Apply a rolling mean to a numpy array with cyclic handling.
    Args:
        arr (numpy.ndarray): Input array for which the rolling mean is calculated.
        window_size (int): Number of elements in the rolling window. Default is 4.
        min_periods (int): Minimum valid (non-NaN) values required to compute the mean. Default is 1.
    Returns:
        numpy.ndarray: Array with rolling mean applied, handling NaN values gracefully.
    """
    n = arr.shape[0]  # Get the length of the input array
    result = np.full_like(arr, np.nan)  # Initialize result array with NaNs
    half_window = window_size // 2  # Determine half the window size for indexing
    for i in range(n):
        # Compute cyclic indices for the rolling window
        indices = [(i + j - half_window) % n for j in range(window_size)]
        # Extract valid (non-NaN) values from the array within the computed window
        valid_values = arr[indices][~np.isnan(arr[indices])]
        # Compute the mean only if enough valid values exist
        if len(valid_values) >= min_periods:
            result[i] = np.mean(valid_values)
    # Replace NaN values in the result where original array had NaNs
    return np.nan_to_num(np.where(np.isnan(arr), result, arr), nan=0.0)


def _ensure_xarray(data):
    """Ensure input is an xarray DataArray."""
    return data if isinstance(data, xr.DataArray) else xr.DataArray(data, dims=["time"])


def _ensure_time_chunks(data):
    if data.chunks[0][0] != 50:
        data = data.chunk({"time": 50, "location": -1})
    return data
