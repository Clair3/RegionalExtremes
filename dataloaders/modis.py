from .common_imports import *
from .sentinel2 import Sentinel2Dataloader
from .data_processing.helpers import _ensure_time_chunks, circular_rolling_mean
from pyproj import Transformer


class ModisDataloader(Sentinel2Dataloader):
    def _calculate_evi(self, ds):
        """Calculates the Enhanced Vegetation Index (EVI)."""
        return (ds["250m_16_days_EVI"] + 2000) / 12000

    def load_file(self, minicube_path, process_entire_minicube=False):
        print(minicube_path)
        filepath = Path(minicube_path)
        with xr.open_zarr(filepath) as ds:

            # Transform UTM to lat/lon
            # ds = self._transform_utm_to_latlon(ds)

            if not process_entire_minicube:
                # Select a random vegetation location
                ds = self._get_random_vegetation_pixel_series(ds)
                if ds is None:
                    return None

            self.variable_name = "evi"  # ds.attrs["data_id"]
            # Filter based on vegetation occurrence
            epsg = (
                ds.attrs.get("spatial_ref")
                or ds.attrs.get("EPSG")
                or ds.attrs.get("CRS")
            )

            if epsg is None:

                raise ValueError("EPSG code not found in dataset attributes.")

            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)

            lon, lat = transformer.transform(ds.x.values, ds.y.values)

            ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)}).rename(
                {"x": "longitude", "y": "latitude"}
            )
            # ds = ds.rename({"x": "longitude", "y": "latitude"})
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
        if "250m_16_days_pixel_reliability" not in ds:
            return self._apply_mask_VI_Quality(ds, evi)

        mask = xr.ones_like(evi)  # Default mask (all ones, meaning no masking)

        # Apply cloud mask if available
        mask = ds["250m_16_days_pixel_reliability"].where(
            ds["250m_16_days_pixel_reliability"] == 0, np.nan
        )

        mask = mask.where(
            mask != 0, 1
        )  # Convert to binary mask (1 = valid, NaN = masked)

        return evi * mask

    def _apply_mask_VI_Quality(self, ds, evi):
        """Applies cloud and vegetation masks to the EVI data using VI_Quality."""
        vi_quality = ds["250m_16_days_VI_Quality"].astype(np.uint16)

        # Define bitwise extract function
        def extract_bit(arr, bit):
            return (arr >> bit) & 1

        # Use apply_ufunc to apply across xarray
        bit_10 = xr.apply_ufunc(extract_bit, vi_quality.load(), 10, vectorize=True)
        bit_12 = xr.apply_ufunc(extract_bit, vi_quality.load(), 12, vectorize=True)
        modland_qa = xr.apply_ufunc(
            lambda x: x & 0b11, vi_quality.load(), vectorize=True
        )

        # Create masks
        cloud_mask = (bit_10 == 0) & (bit_12 == 0)
        valid_qa_mask = (modland_qa == 0) | (modland_qa == 1)

        final_mask = cloud_mask & valid_qa_mask

        # Mask EVI with NaNs where data is invalid
        final_mask = final_mask.where(final_mask, np.nan)

        return evi * final_mask

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

    def get_config(self):
        # Define window sizes for gap-filling and cloud noise removal
        config = dict()
        config["nan_fill_windows"] = [3, 5]
        config["noise_half_windows"] = [1, 2]
        config["smoothing_window_msc"] = 7
        config["poly_msc"] = 2
        config["period_size"] = 16

        return config

    def preprocess_data(
        self,
        return_time_series=False,
        minicube_path=None,
    ):
        """
        Preprocess dataset by applying noise removal, gap-filling, cloud removal,
        deseasonalization, and cumulative EVI computation.

        Parameters
        ----------
        return_time_series : bool, optional
            If True, return both the mean seasonal cycle (MSC) and processed time series.
        minicube_path : str, optional
            Path to a precomputed minicube dataset for loading.

        Returns
        -------
        xr.DataArray
            Mean seasonal cycle (MSC), and optionally, the processed time series.
        """
        msc = self.loader._load_data("msc")
        if msc is not None and not return_time_series:
            return msc.msc

        printt("Starting preprocessing...")
        self.config_dict = self.get_config()
        # Load data either from a minicube or from the default dataset
        if minicube_path:
            printt(f"Loading minicube from {minicube_path}")
            data = self.load_minicube(minicube_path=minicube_path)
            if data is None:
                return None, None
        else:
            data = self.load_dataset()

        printt(f"Processing entire dataset: {data.sizes['location']} locations.")
        data = self.compute_max_per_period(data, self.config_dict["period_size"])
        data = self.noise_removal.cloudfree_timeseries(
            data, noise_half_windows=self.config_dict["noise_half_windows"]
        )
        data = self._remove_low_vegetation_location(data, threshold=0.2)
        self.saver._save_data(data, "evi")

        # Compute Mean Seasonal Cycle (MSC)
        msc = self.compute_msc(data)
        msc = msc.transpose("location", "dayofyear", ...)
        self.saver._save_data(msc, "msc")

        if not return_time_series:
            return msc

        # Step 3: Deseasonalization
        data = self._deseasonalize(data, msc)  # needed?
        self.saver._save_data(data, "deseasonalized")

        # Step 5: Cumulative EVI computation
        # data = self.cumulative_evi(
        #     data, window_size=dict_config["cumulative_evi_window"]
        # )
        # self.saver._save_data(data, "cumulative_evi")

        data = _ensure_time_chunks(data)
        data = data.transpose("location", "time", ...).compute()
        return msc, data
