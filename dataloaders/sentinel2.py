from .common_imports import *
from .base import Dataloader
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from pyproj import Transformer
import cf_xarray as cfxr
from .data_processing.helpers import _ensure_time_chunks, circular_rolling_mean
from datetime import datetime, timedelta, date
import os


class Sentinel2Dataloader(Dataloader):
    def load_dataset(self):
        """
        Load and preprocess the dataset. If a cached version exists, use it.
        Otherwise, sample and process new data, then save for future use.
        """
        # Attempt to load cached training data
        # training_data = self.loader._load_data("temp_file")
        path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/comparison/S2LR_regional_20_lowcloud/EVI/temp_file.zarr"  # "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-09-20_12:06:49_S2_low_res/EVI_EN/temp_file.zarr"
        training_data = xr.open_zarr(path)
        training_data = cfxr.decode_compress_to_multi_index(training_data, "location")
        if training_data is not None:
            if "evi" in training_data:
                training_data = training_data.rename({"evi": self.config.index.lower()})
            # Identify duplicates and remove them
            # location_index = training_data.location.to_index()
            # duplicates = location_index.duplicated()
            # training_data = training_data.sel(location=~duplicates)
            self.data = training_data[self.config.index.lower()]
            # Convert location MultiIndex to a proper index
            if self.n_samples is not None and self.n_samples < len(self.data.location):
                random_indices = np.random.choice(
                    len(self.data.location), size=self.n_samples, replace=False
                )
                self.data = self.data.isel(location=random_indices)

            # self.data = _ensure_time_chunks(self.data)
            return self.data
        sample_paths = []
        for path in self.config.data_source_path:
            if path == "/Net/Groups/BGI/work_2/scratch/DeepExtremes/dx-minicubes/full/":
                deepextremes = pd.read_csv(
                    "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/africa_samples.csv"
                )
                paths = deepextremes["path"].tolist()
                sample_paths += [Path(path) for path in paths]
            else:
                paths = [Path(path + folder) for folder in os.listdir(path)]
                sample_paths += paths

        printt(
            f"Number of samples for the training set: {len(sample_paths)} \n Number of locations for the training set: {self.n_samples}"
        )

        # Randomly sample paths for processing
        sample_paths = np.random.choice(sample_paths, size=self.n_samples, replace=True)

        data = [delayed(self.load_file)(path) for path in sample_paths]
        with ProgressBar():
            data = compute(*data, scheduler="processes")
        ds = [d for d in data if isinstance(d, xr.Dataset)]
        ds = xr.concat(ds, dim="location", combine_attrs="override")
        location_index = ds.location.to_index()
        # Identify duplicates
        duplicates = location_index.duplicated()
        # Keep only the first occurrence of each location
        ds = ds.sel(location=~duplicates)
        printt("Chunking dataset...")
        ds = ds.chunk({"time": 100, "location": 500})
        ds = cfxr.encode_multi_index_as_compress(ds, "location")
        path = self.config.saving_path / "temp_file.zarr"
        printt("Saving dataset to cache...")
        encoding = {var: {"compressor": None} for var in ds.data_vars}
        ds.to_zarr(path, mode="w", encoding=encoding)
        printt(f"Data saved to {path}")
        # Explicitly close dataset to release memory and file handles
        ds.close()
        del ds, data
        # Force garbage collection (optional but helpful for huge datasets)
        ds = self.loader._load_data("temp_file")[self.config.index.lower()]
        ds = _ensure_time_chunks(ds)
        printt("Dataset loaded.")
        return ds

    def load_minicube(self, minicube_path):
        filepath = Path(minicube_path)
        data = self.load_file(filepath, process_entire_minicube=True)
        if data is None:
            return None
        data = data[self.config.index.lower()]
        data = _ensure_time_chunks(data)
        return data

    def load_file(self, filepath, process_entire_minicube=False):
        try:
            ds = xr.open_zarr(
                filepath,
            ).astype(np.float32)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

        ds = self._ensure_coordinates(ds)

        if not process_entire_minicube:
            # Select a random vegetation location
            ds = self._get_random_vegetation_pixel_series(ds)
            if ds is None:
                return None
        else:
            ds = ds.stack(location=("longitude", "latitude"))
            self.saver.update_saving_path(filepath.stem)

        ds = self.generate_masked_vegetation_index(ds, filename=filepath.stem)
        if ds is None:
            return None
        ds = self.compute_max_per_period(ds, period_size=self.config.time_resolution)
        return ds

    def load_file(self, filepath, process_entire_minicube=False):
        try:
            ds = xr.open_zarr(
                filepath,
            ).astype(np.float32)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

        ds = self._ensure_coordinates(ds)
        if not process_entire_minicube:
            # Select a random vegetation location
            ds = self._get_random_vegetation_pixel_series(ds)
            if ds is None:
                return None
        else:
            ds = ds.stack(location=("longitude", "latitude"))
            self.saver.update_saving_path(filepath.stem)

        ds = self.generate_masked_vegetation_index(ds, filename=filepath.stem)
        if ds is None:
            return None
        ds = self.compute_max_per_period(ds, period_size=self.config.time_resolution)
        return ds

    def generate_masked_vegetation_index(self, ds, filename=None):
        # High-resolution computation
        evi = self._calculate_vegetation_index(ds)
        mask = self._compute_masks(ds, evi)
        masked_evi = evi * mask
        data = xr.Dataset(
            data_vars={
                f"{self.config.index.lower()}": masked_evi,
            },
        )
        if self._has_excessive_nan(masked_evi):
            return None
        return data

    def _ensure_coordinates(self, ds):
        """Transforms UTM coordinates to latitude and longitude."""
        if "time" not in ds.dims:
            ds = ds.rename({"time_sentinel-2-l2a": "time"})

        if "x20" in ds.dims:
            # Coarsen 10m bands to 20m resolution
            ds_10m = (
                ds[["B02", "B03", "B04", "B08"]]
                .coarsen(x=2, y=2, boundary="trim")
                .mean()
            )

            # Rename to match 20m grid
            ds_10m = ds_10m.rename({"x": "x20", "y": "y20"})

            # Select 20m bands
            ds_20m = ds[["B05", "B06", "B07", "B11", "B12", "B8A", "SCL"]]

            # Merge all bands
            ds = xr.merge([ds_10m, ds_20m])
            ds = ds.rename({"x20": "x", "y20": "y"})
        epsg = (
            ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
        )

        # Transform UTM coordinates to latitude and longitude if EPSG is provided
        if epsg is not None:
            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
            lon, lat = transformer.transform(ds.x.values, ds.y.values)
            ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)})

        if "spatial_ref" in ds:
            ds = ds.drop_vars("spatial_ref")

        ds["time"] = ds["time"].dt.floor("D")
        ds = ds.sel(time=slice(date(2017, 3, 1), None))

        return ds.rename({"x": "longitude", "y": "latitude"})

    def _get_random_vegetation_pixel_series(self, ds):
        """
        Select a random vegetation pixel time series from the minicube based on SCL classification.
        Returns None if no eligible vegetation pixels exist.
        """
        lon_idx, lat_idx = self._choose_random_pixel(ds)
        if (lon_idx == None) and (lat_idx == None):
            return None
        selected_data = self._select_random_pixel(ds, lon_idx, lat_idx)

        # Stack spatial dimensions for simpler downstream processing
        return selected_data.stack(location=("longitude", "latitude"))

    def _find_eligible_vegetation_pixels(self, ds):
        """Return array of (lon, lat) indices with sufficient vegetation coverage."""
        vegetation_count = (ds.SCL == 4).sum(dim="time")
        years = np.unique(ds.time.dt.year)
        threshold = 0.25 * (366 / self.config.time_resolution) * len(years)
        mask = vegetation_count > threshold
        return np.argwhere(mask.values)

    def _choose_random_pixel(self, ds):
        """Randomly select one pixel index from the eligible vegetation pixels."""
        eligible_indices = self._find_eligible_vegetation_pixels(ds)
        if eligible_indices.size == 0:
            return (None, None)
        random_index = eligible_indices[np.random.choice(eligible_indices.shape[0])]
        return tuple(random_index)  # (lon_idx, lat_idx)

    def _select_random_pixel(self, ds, lon_idx, lat_idx):
        """
        Select a single pixel from the dataset.
        """
        selected = ds.isel(longitude=lon_idx, latitude=lat_idx)
        return selected.expand_dims(
            longitude=[selected.longitude.values.item()],
            latitude=[selected.latitude.values.item()],
        )

    def _calculate_vegetation_index(self, ds):
        """Calculates the Vegetation Index."""
        if self.config.index in ("EVI_EN", "EVI"):
            return (2.5 * (ds.B8A - ds.B04)) / (
                ds.B8A + 6 * ds.B04 - 7.5 * ds.B02 + 1 + 10e-8
            )
        elif self.config.index == "NDVI":
            return (ds.B8A - ds.B04) / (ds.B8A + ds.B04 + 10e-8)
        else:
            raise ValueError(f"Unknown vegetation index: {self.config.index}")

    def _compute_masks(self, ds, evi=None):
        """
        Applies cloud and vegetation masks.
        If use_coarsen=True, masks are coarsened before being returned.
        """

        # Start with all valid
        mask = xr.ones_like(ds.B04)  # .stack(location=("latitude", "longitude"))

        # Apply vegetation mask if SCL exists
        if "SCL" in ds.data_vars:
            valid_scl = ds.SCL.isin([4, 5, 6, 7])
            valid_scl = valid_scl  # .stack(location=("latitude", "longitude"))
            mask = mask.where(valid_scl, np.nan)

            # drop timesteps where less than 90% valid pixels
            valid_ratio = valid_scl.sum(dim=["location"]) / valid_scl.count(
                dim=["location"]
            )
            invalid_time_steps = valid_ratio < 0.9
            mask = mask.where(~invalid_time_steps, np.nan)

        # Apply cloud mask if available
        if "cloudmask_en" in ds.data_vars:
            mask = mask.where(ds.cloudmask_en == 0, np.nan)

        return mask

    def _has_excessive_nan(self, data):
        """Checks if the masked data contains excessive NaN values."""
        nan_percentage = data.isnull().mean().values * 100
        return nan_percentage > 95

    def _deseasonalize(self, data, msc):
        # Align subset_msc with subset_data
        aligned_msc = msc.sel(dayofyear=data["time.dayofyear"])
        # Subtract the seasonal cycle

        deseasonalized = data - aligned_msc
        deseasonalized = deseasonalized.reset_coords("dayofyear", drop=True)
        return deseasonalized

    def compute_msc(
        self,
        clean_data: xr.DataArray,
        smoothing_window: int = 7,  # 9,
        poly_order: int = 2,
    ):

        # Step 1: Compute mean seasonal cycle
        mean_seasonal_cycle = clean_data.groupby("time.dayofyear").mean(
            "time", skipna=True
        )
        # Apply circular padding along the dayofyear axis before rolling
        # # edge case growing season during the change of year
        padded_values = np.pad(
            mean_seasonal_cycle.values,
            (
                (smoothing_window, smoothing_window),
                (0, 0),
            ),  # Pad along the dayofyear axis
            mode="wrap",  # Wrap-around to maintain continuity
        )

        padded_values = circular_rolling_mean(
            padded_values, window_size=2, min_periods=1
        )
        padded_values = circular_rolling_mean(
            padded_values, window_size=4, min_periods=1
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

    def compute_max_per_period(self, data, period_size=10):
        # Function to generate valid dates (time bins) for all years at once
        def get_time_periods(bin_size, years):
            periods = []
            for year in years:
                bins = np.arange(1, 367, bin_size)
                base_date = datetime(year, 1, 1).date()
                dates = [base_date + timedelta(days=int(d - 1)) for d in bins]
                periods.extend(dates)
            return periods

        # Define a function to map the timestamp to the corresponding period using searchsorted
        def get_period_from_timestamp(timestamp, periods):
            period_dates = pd.to_datetime(periods)  # Convert periods to datetime
            # Efficient way to find the period index using searchsorted
            period_index = np.searchsorted(period_dates, timestamp, side="right") - 1
            return period_index

        # Remove unrealistic values
        data = data.where((data >= 0) & (data <= 1), np.nan)

        # Prepare the periods list (one time-period list for all years in the dataset)
        years = pd.to_datetime(data.time).year.unique()
        periods = get_time_periods(period_size, years)
        # Map each timestamp to a period
        periods_assigned = [
            get_period_from_timestamp(t, periods) for t in pd.to_datetime(data.time)
        ]

        # Add period as a new dimension to the DataArray
        data.coords["period"] = ("time", periods_assigned)

        # Group by the 'period' and compute max per period
        data_grouped = data.groupby("period")

        # Compute max for each period
        max_per_period = data_grouped.max(dim="time")
        # max_per_period = data_grouped.max(dim="time")

        # Apply the transformation to convert periods back to midpoints in time
        start_period_times = [
            pd.to_datetime(periods[p]) for p in max_per_period.coords["period"].values
        ]

        # Update max_per_period with the transformed 'time' coordinates (midpoints)
        max_per_period.coords["time"] = ("period", start_period_times)
        max_per_period = max_per_period.swap_dims({"period": "time"}).drop_vars(
            "period"
        )
        max_per_period = max_per_period.set_index(location=["longitude", "latitude"])
        return max_per_period

    def get_config(self):
        # Define window sizes for gap-filling and cloud noise removal
        config = dict()
        # config["nan_fill_windows"] = [3, 4]
        config["noise_half_windows"] = [1]
        config["smoothing_window_msc"] = 7
        config["poly_msc"] = 2
        return config

    def _remove_low_vegetation_location(self, vegetation_index, threshold=0.1):
        mean_vi = vegetation_index.mean("time", skipna=True)
        valid_locations = (mean_vi > threshold) & ~np.isnan(mean_vi)

        # remove low vegetation locations
        filtered_vegetation_index = vegetation_index.sel(location=valid_locations)
        return filtered_vegetation_index

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
        self.config_dict = self.get_config()
        # msc = self.loader._load_data("msc")
        # if msc is not None and not return_time_series:
        #     return msc.msc
        # if dict_config["deseasonalization"]:
        #     data = self.loader._load_data("deseasonalized")
        #     if msc is not None and data is not None:
        #         return msc.msc, data.deseasonalized
        # else:
        #     data = self.loader._load_data("evi")
        #     if msc is not None and data is not None:
        #         return msc.msc, data.evi
        printt("Starting preprocessing...")
        # Load data either from a minicube or from the default dataset
        if minicube_path:
            printt(f"Loading minicube from {minicube_path}")
            data = self.load_minicube(minicube_path=minicube_path)
            if data is None:
                return None, None
        else:
            data = self.load_dataset()
        data = self.noise_removal.cloudfree_timeseries(
            data, noise_half_windows=self.config_dict["noise_half_windows"]
        )
        # data = self._remove_low_vegetation_location(data, threshold=0.2)

        printt(f"Processing entire dataset: {data.sizes['location']} locations.")
        # data = data.compute()
        self.saver._save_data(data, self.config.index.lower())
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
