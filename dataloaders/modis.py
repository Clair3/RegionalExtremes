from .common_imports import *
from .sentinel2 import Sentinel2Dataloader
from .data_processing.helpers import _ensure_time_chunks, circular_rolling_mean


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

    def compute_msc(
        self,
        clean_data: xr.DataArray,
        smoothing_window: int = 5,
        poly_order: int = 2,
    ):

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
        padded_values = np.nan_to_num(padded_values, nan=0)
        #
        # rolled_values = circular_rolling_mean(
        #    padded_values, window_size=4, min_periods=1
        # )

        # Step 5: Apply Savitzky-Golay smoothing
        smoothed_values = savgol_filter(
            padded_values, smoothing_window, poly_order, axis=0
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
