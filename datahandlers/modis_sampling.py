from .common_imports import *
from .base import DatasetHandler
from .sentinel2 import Sentinel2DatasetHandler
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from pyproj import Transformer
import cf_xarray as cfxr


class ModisSamplingDatasetHandler(Sentinel2DatasetHandler):
    def _calculate_evi(self, ds):
        """Calculates the Enhanced Vegetation Index (EVI)."""
        return (ds["250m_16_days_EVI"] + 2000) / 12000

    def load_minicube(self, minicube_path, process_entire_minicube=False):
        filepath = Path(minicube_path)  # EARTHNET_FILEPATH + minicube_path
        with xr.open_zarr(filepath, chunks="auto") as ds:

            # Transform UTM to lat/lon
            ds = self._transform_utm_to_latlon(ds)

            if not process_entire_minicube:
                # Select a random vegetation location
                ds = self._get_random_vegetation_pixel_series(ds)
                if ds is None:
                    return None

            self.variable_name = "evi"  # ds.attrs["data_id"]
            # Filter based on vegetation occurrence

            # Calculate EVI and apply cloud/vegetation mask
            evi = self._calculate_evi(ds)
            evi = self._apply_masks(ds, evi)
            if "start_range" in evi.coords:
                evi = evi.rename({"start_range": "time"})
            else:
                evi = evi.rename({"time_modis-13Q1-061": "time"})
            data = xr.Dataset(
                data_vars={
                    f"{self.variable_name}": evi,  # Adding 'evi' as a variable
                    # "landcover": ds[
                    #    "esa_worldcover_2021"
                    # ],  # Adding 'landcover' as another variable
                },
                coords={
                    "source_path": minicube_path,  # Add the path as a coordinate
                },
            )
            if process_entire_minicube:
                self.saver.update_saving_path(filepath.stem)
            return data

    def remove_cloud_noise(self, data, window_size=2, gapfill=True):

        # Ensure input is xarray DataArray
        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(data, dims=["time"])

        # Create arrays to store sum and count for before/after
        before_sum = 0
        before_count = 0
        after_sum = 0
        after_count = 0

        # Calculate sums using shifts instead of rolling
        for i in range(1, window_size + 1):
            # Before window
            shifted_before = data.shift(time=i)
            mask_before = ~np.isnan(shifted_before)
            before_sum += xr.where(mask_before, shifted_before, 0)
            before_count += mask_before.astype(int)

            # After window
            shifted_after = data.shift(time=-i)
            mask_after = ~np.isnan(shifted_after)
            after_sum += xr.where(mask_after, shifted_after, 0)
            after_count += mask_after.astype(int)

        # Calculate means, avoiding division by zero
        mean_before = xr.where(before_count > 0, before_sum / before_count, data)
        mean_after = xr.where(after_count > 0, after_sum / after_count, data)

        # Detect cloud points: current value is LOWER than both mean before and after
        is_cloud = (data < mean_before) & (data < mean_after)

        # For cleaning, use the mean of before and after values
        replacement_values = (mean_before + mean_after) / 2

        # Replace detected cloud points
        gapfilled_data = xr.where(is_cloud, replacement_values, data)
        clean_data = xr.where(is_cloud, np.nan, data)
        if gapfill:
            return gapfilled_data
        else:
            return clean_data

    def cumulative_evi(self, deseaonalized, window_size=2):

        # Ensure input is xarray DataArray
        if not isinstance(deseaonalized, xr.DataArray):
            deseaonalized = xr.DataArray(deseaonalized, dims=["time"])

        # Create arrays to store sum and count for before/after
        sum = 0
        count = 0

        # Calculate sums using shifts instead of rolling
        for i in range(1, window_size + 1):
            # Before window
            shifted_before = deseaonalized.shift(time=i)
            mask_before = ~np.isnan(shifted_before)
            sum += xr.where(mask_before, shifted_before, 0)
            count += mask_before.astype(int)

        # Calculate means, avoiding division by zero
        mean = xr.where(count > 1, sum / count, np.nan)
        return mean

    def clean_timeseries(
        self,
        data: xr.DataArray,
        rolling_window: int = 1,
    ) -> xr.DataArray:
        """
        Clean and smooth a time series using rolling windows, neighbor comparison.

        Parameters
        ----------
        data : xr.DataArray
            Input time series data
        rolling_window : int, optional
            Size of rolling window for initial smoothing, default 3

        Returns
        -------
        xr.DataArray
            Cleaned and smoothed time series
        """

        try:
            # Input validation
            if not isinstance(data, xr.DataArray):
                raise TypeError("Input must be an xarray DataArray")

            # Step 1: Remove outliers (values below 0 or values above 1)
            data = data.where((data >= 0) & (data <= 1), np.nan)

            # Step 2: Remove local minima
            clean_data = self.remove_cloud_noise(data, window_size=2)
            clean_data = self.remove_cloud_noise(clean_data, window_size=3)

            # Step 3: Handle NaN values
            clean_data = self.fill_nans(data, rolling_window)
            clean_data = self.fill_nans(
                clean_data, rolling_window
            )  # Second pass for remaining NaNs

            # Step 4: Remove local minima again after NaN filling
            clean_data = self.remove_cloud_noise(clean_data, window_size=1)
            return clean_data
        except Exception as e:
            raise RuntimeError(f"Error processing time series: {str(e)}") from e

    def fill_nans(self, series: xr.DataArray, window: int) -> xr.DataArray:
        """Fill NaN values using rolling mean."""
        rolling_mean = series.rolling(time=window, center=True, min_periods=1).mean()
        return series.where(~np.isnan(series), other=rolling_mean)

    def clean_timeseries_fixed_stats(self, deseasonalized):

        # Calculate GLOBAL median and std for each location (not rolling)
        global_mean = deseasonalized.mean(dim="time", skipna=True)
        global_std = deseasonalized.std(dim="time", skipna=True).clip(min=0.01)
        # # Broadcast to match dimensions for computation
        mean_broadcast = global_mean.broadcast_like(deseasonalized)
        std_broadcast = global_std.broadcast_like(deseasonalized)

        # Identify and replace outliers using global statistics
        is_negative_outlier = deseasonalized < (mean_broadcast - 2 * std_broadcast)
        cleaned_data = xr.where(is_negative_outlier, np.nan, deseasonalized)
        return cleaned_data

    def compute_msc(
        self,
        clean_data: xr.DataArray,
        smoothing_window: int = 5,
        poly_order: int = 2,
    ):
        mean_seasonal_cycle = clean_data.groupby("time.dayofyear").mean(
            "time", skipna=True
        )
        rolling_mean = mean_seasonal_cycle.rolling(
            dayofyear=4, center=True, min_periods=1
        ).mean()
        mean_seasonal_cycle = mean_seasonal_cycle.where(
            ~np.isnan(mean_seasonal_cycle), other=rolling_mean
        )
        mean_seasonal_cycle = mean_seasonal_cycle.fillna(0)

        # Step 6: Apply Savitzky-Golay smoothing
        mean_seasonal_cycle_values = savgol_filter(
            mean_seasonal_cycle.values, smoothing_window, poly_order, axis=0
        )
        print("here23")
        mean_seasonal_cycle = mean_seasonal_cycle.copy(data=mean_seasonal_cycle_values)
        # Ensure all values are non-negative
        mean_seasonal_cycle = mean_seasonal_cycle.where(mean_seasonal_cycle > 0, 0)
        return mean_seasonal_cycle

    def _apply_masks(self, ds, evi):
        """Applies cloud and vegetation masks to the EVI data."""
        mask = xr.ones_like(evi)  # Default mask (all ones, meaning no masking)

        # Apply cloud mask if available
        mask = ds["250m_16_days_pixel_reliability"].where(
            ds["250m_16_days_pixel_reliability"] == 0, np.nan
        )

        mask = mask.where(
            mask != 0, 1
        )  # Convert to binary mask (1 = valid, NaN = masked)

        return evi * mask

    def _deseasonalize(self, data, msc):
        # Align subset_msc with subset_data
        aligned_msc = msc.sel(dayofyear=data["time.dayofyear"])
        # Subtract the seasonal cycle

        deseasonalized = data - aligned_msc
        deseasonalized = deseasonalized.reset_coords("dayofyear", drop=True)
        return deseasonalized

    def growing_season(self, msc, evi):
        """Removes winter periods (outside EOS-SOS) from an EVI time series.

        Parameters:
        msc (xarray.DataArray): Mean Seasonal Cycle of EVI (1D or 3D: time, lat, lon)
        evi (xarray.DataArray): Multi-year EVI time series (time, lat, lon)

        Returns:
        xarray.DataArray: EVI with winter periods masked (NaN)
        """
        if "dayofyear" not in msc.dims:
            raise ValueError(
                "Mean Seasonal Cycle (msc) must have a 'dayofyear' dimension."
            )
        # Compute the first derivative (rate of change)
        first_derivative = np.gradient(
            msc.values, axis=msc.dims.index("dayofyear")
        )  # Ensure time axis is last (-1)

        # Determine SOS (max increase) and EOS (max decrease)
        sos_doy = np.argmax(
            first_derivative, axis=msc.dims.index("dayofyear")
        )  # SOS for each pixel
        eos_doy = np.argmin(
            first_derivative, axis=msc.dims.index("dayofyear")
        )  # EOS for each pixel

        # Get Day of Year (DOY) for each time step in the EVI dataset
        doy = evi["time"].dt.dayofyear  # Shape: (time,)
        doy_expanded = doy.values[:, np.newaxis]  # Shape: (time, 1, 1)
        growing_season_mask = (doy_expanded >= sos_doy) & (doy_expanded <= eos_doy)
        evi_growing_season = evi.where(growing_season_mask)
        return evi_growing_season

    def preprocess_data(
        self,
        scale=True,
        reduce_temporal_resolution=True,
        return_time_serie=False,
        remove_nan=True,
        minicube_path=None,
    ):
        """
        Preprocess data based on the index.
        """
        printt("start of the preprocess")

        if minicube_path:
            self._minicube_specific_loading(minicube_path=minicube_path)
        else:
            self._dataset_specific_loading()
        self.data = self.data[self.variable_name]
        # Randomly select n indices from the location dimension

        printt(
            f"Computation on the entire dataset. {self.data.sizes['location']} samples"
        )
        self.data = self.data.chunk({"time": 50, "location": -1})
        self.saver._save_data(self.data, "evi")
        gapfilled_data = self.clean_timeseries(self.data)

        self.msc = self.compute_msc(gapfilled_data)
        self.msc = self.msc.transpose("location", "dayofyear", ...)
        self.saver._save_data(self.msc, "msc")

        clean_data = self.remove_cloud_noise(self.data, window_size=2, gapfill=False)
        clean_data = self.remove_cloud_noise(clean_data, window_size=3, gapfill=False)
        deseasonalized = self._deseasonalize(clean_data, self.msc)
        self.saver._save_data(deseasonalized, "deseasonalized")
        # Align the seasonal cycle with the deseasonalized data
        aligned_msc = self.msc.sel(dayofyear=deseasonalized.time.dt.dayofyear)

        deseasonalized = self.cumulative_evi(deseasonalized, window_size=4)
        self.saver._save_data(deseasonalized, "cumulative_evi")
        self.data = deseasonalized + aligned_msc
        self.data = self.data.reset_coords("dayofyear", drop=True)
        self.saver._save_data(self.data, "clean_data")

        # Add the seasonal cycle back to reconstruct the original data

        if return_time_serie:
            self.data = self.data.transpose("location", "time", ...).compute()
            return self.msc, self.data
        else:
            return self.msc
