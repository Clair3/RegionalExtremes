from typing import Union, Optional
import xarray as xr

import dask.array as da
import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod
from .helpers import _compute_shifted_sums, _ensure_xarray, _compute_shifted_max


class NoiseRemovalBase(ABC):
    """Class for detecting and removing cloud-related noise in vegetation index time series."""

    def remove_cloud_noise(self, data, half_window=2, gapfill=True):
        """Detect and remove cloud noise using shifted mean values."""
        before_max = _compute_shifted_max(data, half_window, direction=1)
        after_max = _compute_shifted_max(data, half_window, direction=-1)

        # mean_before = xr.where(
        #     before_max > 0, before_sum / (before_count + 1e-10), np.nan
        # )
        # mean_after = xr.where(
        #     after_count > 0, after_sum / (after_count + 1e-10), np.nan
        # )

        is_cloud = (data + 0.01 < before_max) & (data + 0.01 < after_max)

        if gapfill:
            before_sum, before_count = _compute_shifted_sums(
                data, half_window, direction=1
            )
            after_sum, after_count = _compute_shifted_sums(
                data, half_window, direction=-1
            )

            mean_before = xr.where(
                before_count > 0, before_sum / (before_count + 1e-10), np.nan
            )
            mean_after = xr.where(
                after_count > 0, after_sum / (after_count + 1e-10), np.nan
            )
            replacement_values = (mean_before + mean_after) / 2
            return xr.where(is_cloud, replacement_values, data)

        return xr.where(is_cloud, np.nan, data)

    def fill_nans(self, series: xr.DataArray, window: int) -> xr.DataArray:
        """Fill NaN values using rolling mean."""
        rolling_mean = series.rolling(time=window, center=True, min_periods=1).mean()
        return series.where(~np.isnan(series), other=rolling_mean)

    def clean_and_gapfill_timeseries(
        self,
        data: xr.DataArray,
        nan_fill_windows: list = [5, 7],
        noise_half_windows: list = [1, 3],
    ) -> xr.DataArray:
        """
        Cleans and smooths a time series using outlier removal, cloud noise correction,
        and NaN gap-filling with rolling windows.

        Parameters
        ----------
        data : xr.DataArray
            Input time series data.
        nan_fill_windows : list of int, optional
            List of window sizes for gap-filling NaN values, applied sequentially.
        cloud_noise_half_windows : list of int, optional
            List of window sizes for removing local minima due to cloud noise.

        Returns
        -------
        xr.DataArray
            Cleaned and gap-filled time series.
        """
        # data = _ensure_xarray(data):
        if not isinstance(data, xr.DataArray):
            raise TypeError("Input must be an xarray DataArray")

        # Step 1: Remove outliers (values should be between 0 and 1)
        data = data.where((data >= 0) & (data <= 1), np.nan)

        # Step 2: Remove cloud noise using specified window sizes
        for window in noise_half_windows:
            data = self.remove_cloud_noise(data, half_window=window, gapfill=False)

        # Step 3: Fill NaN values using specified window sizes
        for window in nan_fill_windows:
            data = self.fill_nans(data, window)

        # Step 4: Final cloud noise removal after NaN gap-filling
        data = self.remove_cloud_noise(data, half_window=noise_half_windows[0])

        return data

    def cloudfree_timeseries(self, data, noise_half_windows=[1, 3]):
        data = data.where((data >= 0) & (data <= 1), np.nan)
        for half_window in noise_half_windows:
            data = self.remove_cloud_noise(data, half_window=half_window, gapfill=False)
        return data
