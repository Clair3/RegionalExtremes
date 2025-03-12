from .common_imports import *
from .sentinel2 import Sentinel2Dataloader
from .data_processing.helpers import _ensure_time_chunks


class ModisDataloader(Sentinel2Dataloader):
    def _calculate_evi(self, ds):
        """Calculates the Enhanced Vegetation Index (EVI)."""
        return (ds["250m_16_days_EVI"] + 2000) / 12000

    def load_file(self, minicube_path, process_entire_minicube=False):
        filepath = Path(minicube_path)
        with xr.open_zarr(filepath) as ds:

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
            elif "time_modis-13Q1-061" in evi.coords:
                evi = evi.rename({"time_modis-13Q1-061": "time"})
            elif "time" not in evi.coords:
                raise IndexError(f"No time coordinates available in {filepath}")

            data = xr.Dataset(
                data_vars={
                    f"{self.variable_name}": evi,
                },
                coords={
                    "source_path": minicube_path,  # Add the path as a coordinate
                },
            )
            if process_entire_minicube:
                self.saver.update_saving_path(filepath.stem)

            return data

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
        mean = xr.where(count > 1, sum / (count + 1e-10), np.nan)
        return mean

    def compute_msc(
        self,
        clean_data: xr.DataArray,
        smoothing_window: int = 5,
        poly_order: int = 2,
    ):
        def rolling_window_mean(arr, window_size=4, min_periods=1):
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
            half_window = (
                window_size // 2
            )  # Determine half the window size for indexing

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

        # Step 1: Compute mean seasonal cycle
        mean_seasonal_cycle = clean_data.groupby("time.dayofyear").mean(
            "time", skipna=True
        )

        # Step 2: Apply circular padding along the dayofyear axis before rolling
        # # edge case growing season during the change of year
        padded_values = np.pad(
            mean_seasonal_cycle.values,
            (
                (smoothing_window, smoothing_window),
                (0, 0),
            ),  # Pad along the dayofyear axis
            mode="wrap",  # Wrap-around to maintain continuity
        )
        #
        rolled_values = rolling_window_mean(padded_values, window_size=4, min_periods=1)

        # Step 5: Apply Savitzky-Golay smoothing
        smoothed_values = savgol_filter(
            rolled_values, smoothing_window, poly_order, axis=0
        )
        mean_seasonal_cycle = mean_seasonal_cycle.copy(
            data=smoothed_values[smoothing_window:-smoothing_window]
        )
        # Step 6: Ensure all values are non-negative
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
        # scale=True,
        # reduce_temporal_resolution=True,
        return_time_series=False,  # Renamed for clarity
        minicube_path=None,
    ):
        """
        Preprocess dataset by applying noise removal, gap-filling, cloud removal,
        deseasonalization, and cumulative EVI computation.

        Parameters
        ----------
        scale : bool, optional
            Whether to scale the data (not yet implemented in this function).
        reduce_temporal_resolution : bool, optional
            Whether to reduce the temporal resolution of the dataset.
        return_time_series : bool, optional
            If True, return both the mean seasonal cycle (MSC) and processed time series.
        minicube_path : str, optional
            Path to a precomputed minicube dataset for loading.

        Returns
        -------
        xr.DataArray
            Mean seasonal cycle (MSC), and optionally, the processed time series.
        """

        printt("Starting preprocessing...")

        # Define window sizes for gap-filling and cloud noise removal
        nan_fill_windows = [5, 7]
        noise_half_windows = [1, 3]

        # Load data either from a minicube or from the default dataset
        if minicube_path:
            data = self.load_minicube(minicube_path=minicube_path)
        else:
            data = self.load_dataset()

        printt(f"Processing entire dataset: {data.sizes['location']} locations.")
        self.saver._save_data(data, "evi")

        # Step 1: Gap-filling & noise removal
        gapfilled_data = self.noise_removal.clean_and_gapfill_timeseries(
            data,
            nan_fill_windows=nan_fill_windows,
            noise_half_windows=noise_half_windows,
        )

        # Compute Mean Seasonal Cycle (MSC)
        msc = self.compute_msc(gapfilled_data)
        msc = msc.transpose("location", "dayofyear", ...)
        self.saver._save_data(msc, "msc")

        if not return_time_series:
            return msc

        # Step 2: Cloud removal
        data = self.noise_removal.cloudfree_timeseries(
            data, noise_half_windows=noise_half_windows
        )
        self.saver._save_data(data, "clean_data")

        # Step 3: Deseasonalization
        data = self._deseasonalize(data, msc)  # needed?
        self.saver._save_data(data, "deseasonalized")

        # Step 5: Cumulative EVI computation
        data = self.cumulative_evi(data, window_size=4)
        self.saver._save_data(data, "cumulative_evi")

        data = _ensure_time_chunks(data)
        data = data.transpose("location", "time", ...).compute()
        return msc, data
