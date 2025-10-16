from .common_imports import *
from .sentinel2 import Sentinel2Dataloader
from .data_processing.helpers import _ensure_time_chunks, circular_rolling_mean
from pyproj import Transformer


class ModisDataloader(Sentinel2Dataloader):
    def _calculate_vegetation_index(self, ds):
        """Calculates the Vegetation Index (EVI)."""
        if self.config.index == "EVI_MODIS" or "EVI":
            ds = (ds["250m_16_days_EVI"] + 2000) / 12000
        elif self.config.index == "NDVI":
            ds = (ds["250m_16_days_NDVI"] + 2000) / 12000
        else:
            raise ValueError(f"Unknown vegetation index: {self.config.index}")
        if "start_range" in ds.coords:
            ds = ds.rename({"start_range": "time"})
        return ds

    def _ensure_coordinates(self, ds):
        """Transforms UTM coordinates to latitude and longitude."""

        epsg = (
            ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
        )

        # Transform UTM coordinates to latitude and longitude if EPSG is provided
        if epsg is not None:
            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
            lon, lat = transformer.transform(ds.x.values, ds.y.values)
            # if "spatial_ref" in ds:
            #    ds = ds.drop_vars("spatial_ref")
            ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)})

        elif "time_modis-13Q1-061" in ds.coords:
            ds = ds.rename({"time_modis-13Q1-061": "time"})
        elif "y231" in ds.coords:
            ds = ds.rename({"x231": "x", "y231": "y"})
        ds["time"] = ds["time"].dt.floor("D")

        return ds.rename({"x": "longitude", "y": "latitude"})

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
        mean_seasonal_cycle = mean_seasonal_cycle.where(mean_seasonal_cycle > 0, 0)
        return mean_seasonal_cycle

    def _compute_masks(self, ds, evi):
        """Applies cloud and vegetation masks to the EVI data."""

        if "250m_16_days_pixel_reliability" not in ds:
            return self._compute_mask_VI_Quality(ds, evi)

        # mask = xr.ones_like(evi)  # Default mask (all ones, meaning no masking)

        # Apply cloud mask if available
        mask = ds["250m_16_days_pixel_reliability"].where(
            ds["250m_16_days_pixel_reliability"] == 0, np.nan
        )
        if "start_range" in mask.coords:
            mask = mask.rename({"start_range": "time"})
        mask = mask.where(
            mask != 0, 1
        )  # Convert to binary mask (1 = valid, NaN = masked)

        return evi * mask

    def _compute_mask_VI_Quality(self, ds, evi):
        """Applies cloud and vegetation masks to the EVI data using VI_Quality."""
        vi_quality = ds["250m_16_days_VI_Quality"].astype(np.uint16)
        if "start_range" in vi_quality.coords:
            vi_quality = vi_quality.rename({"start_range": "time"})

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

    def _choose_random_pixel(self, ds):
        """Selects a random time serie vegetation pixel location in the minicube based on SCL classification."""
        return (
            np.random.randint(ds.sizes["longitude"]),
            np.random.randint(ds.sizes["latitude"]),
        )

    def _deseasonalize(self, data, msc):
        # Align subset_msc with subset_data
        aligned_msc = msc.sel(dayofyear=data["time.dayofyear"])
        # Subtract the seasonal cycle

        deseasonalized = data - aligned_msc
        deseasonalized = deseasonalized.reset_coords("dayofyear", drop=True)
        return deseasonalized

    def get_config(self):
        # Define window sizes for gap-filling and cloud noise removal
        config = dict()
        config["nan_fill_windows"] = [3, 5]
        config["noise_half_windows"] = [1]
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

        data = self.noise_removal.cloudfree_timeseries(
            data, noise_half_windows=self.config_dict["noise_half_windows"]
        )
        # data = self._remove_low_vegetation_location(data, threshold=0.2)
        self.saver._save_data(data, "vegetation_index")
        # Compute Mean Seasonal Cycle (MSC)
        msc = self.compute_msc(data)
        msc = msc.transpose("location", "dayofyear", ...)
        self.saver._save_data(msc, "msc")

        if not return_time_series:
            return msc

        # Step 3: Deseasonalization
        data = self._deseasonalize(data, msc)  # needed?
        self.saver._save_data(data, "deseasonalized")

        data = _ensure_time_chunks(data)
        data = data.transpose("location", "time", ...).compute()
        return msc, data
