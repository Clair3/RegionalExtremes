from .common_imports import *
from .base import Dataloader
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from pyproj import Transformer
import cf_xarray as cfxr
from .data_processing.helpers import _ensure_time_chunks, circular_rolling_mean
from datetime import datetime, timedelta, date
import os
from tqdm import tqdm


class Sentinel2Dataloader(Dataloader):
    def load_dataset(self):
        """
        Load and preprocess the dataset. If a cached version exists, use it.
        Otherwise, sample and process new data, then save for future use.
        """
        self.variable_name = "evi"
        sample_paths = [folder for folder in os.listdir(EARTHNET_FILEPATH)]
        sample_count = self.n_samples or 50000

        # Attempt to load cached training data
        training_data = self.loader._load_data("temp_file")
        # path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_11:57:24_full_fluxnet_therightone_highveg/EVI_EN/temp_file.zarr"
        # training_data = None  # xr.open_zarr(path)
        # training_data = cfxr.decode_compress_to_multi_index(training_data, "location")

        if training_data is not None:
            self.data = training_data[self.variable_name]
            # if self.n_samples:
            #    random_indices = np.random.choice(
            #        len(self.data.location), size=self.n_samples, replace=False
            #    )
            #    self.data = self.data.isel(location=random_indices)

            self.data = _ensure_time_chunks(self.data)
            return self.data

        printt(f"Number of samples for the training set: {sample_count}")

        # Randomly sample paths for processing
        sample_paths = np.random.choice(sample_paths, size=sample_count, replace=True)

        # Load and process each sample in parallel
        data = [
            delayed(self.load_file)(Path(EARTHNET_FILEPATH + path))
            for path in sample_paths
        ]
        with ProgressBar():
            data = compute(*data, scheduler="threads")

        # Filter valid datasets
        data = [d for d in data if isinstance(d, xr.Dataset)]
        if not data:
            raise ValueError("Dataset is empty")

        # Combine datasets
        ds = xr.combine_by_coords(data, combine_attrs="override").compute()
        print("Dataset loaded and concatenated.")
        ds = ds.sel(location=~ds.get_index("location").duplicated())
        # ds = self.compute_max_per_period(ds, self.config_dict["period_size"])

        # ds = _ensure_time_chunks(ds)
        #
        # # Cache the dataset
        self.saver._save_data(ds, "temp_file")
        ds = self.loader._load_data("temp_file")[self.variable_name]
        ds = _ensure_time_chunks(ds)
        #
        # printt("Dataset loaded.")
        return ds

    def load_minicube(self, minicube_path):
        filepath = Path(minicube_path)
        data = self.load_file(filepath, process_entire_minicube=True)
        if data is None:
            return None
        if self.n_samples:
            random_indices = np.random.choice(
                len(data.location), size=self.n_samples, replace=False
            )
            data = data.isel(location=random_indices)
        # self.data = self.data  # .isel(location=slice(0, 100))
        self.variable_name = "evi"
        data = data[self.variable_name]
        data = _ensure_time_chunks(data)
        return data

    def sample_locations(self, n_samples):
        """
            Randomly sample locations from a DataFrame with replacement
        .

            Parameters:
            -----------
            df : pandas.DataFrame
                DataFrame containing latitude and longitude columns
            n_samples : int
                Number of samples to generate (with replacement)

            Returns:
            --------
            pandas.DataFrame
                DataFrame containing original lat/lon plus random grid coordinates
        """

        def _is_in_europe(self, lon, lat):
            """
            Check if the given longitude and latitude are within the bounds of Europe.
            """
            # Define Europe boundaries (these are approximate)
            lon_min, lon_max = -31.266, 39.869  # Longitude boundaries
            lat_min, lat_max = 27.636, 81.008  # Latitude boundaries

            # Check if the point is within the defined boundaries
            in_europe = (
                (lon >= lon_min)
                & (lon <= lon_max)
                & (lat >= lat_min)
                & (lat <= lat_max)
            )
            return in_europe

        metadata_file = (
            "/Net/Groups/BGI/work_2/scratch/DeepExtremes/mc_earthnet_biome.csv"
        )
        df = pd.read_csv(metadata_file, delimiter=",", low_memory=False)[
            [
                "path",
                "group",
                "check",
                "start_date",
                "lat",
                "lon",
            ]
        ]
        df = df.loc[
            (df["check"] == 0)
            & df.apply(lambda row: _is_in_europe(row["lon"], row["lat"]), axis=1)
        ]
        # /Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/DeepExtremes_enough_vegetation_europe.csv
        # Random sampling with replacement
        sampled_indices = np.random.choice(df.index, size=n_samples, replace=True)
        return df.loc[
            sampled_indices
        ].values  # df.loc[sampled_indices, "path"].values  #

    def load_file(self, filepath, process_entire_minicube=False):
        ds = xr.open_zarr(filepath).astype(np.float32)

        ds = self._ensure_coordinates(ds)
        ds["time"] = ds["time"].dt.floor("D")
        ds = ds.sel(time=slice(date(2017, 3, 1), None))

        if not process_entire_minicube:
            # Select a random vegetation location
            ds = self._get_random_vegetation_pixel_series(ds)
            if ds is None:
                printt(f"No valid vegetation pixel found in {filepath}.")
                return None
        else:
            ds = ds.stack(location=("longitude", "latitude"))
            self.saver.update_saving_path(filepath.stem)

        data = self.compute_vegetation_index(ds)
        return data

    def compute_vegetation_index(self, ds):
        self.variable_name = "evi"

        # Calculate EVI and apply cloud/vegetation mask
        evi = self._calculate_evi(ds)
        mask = self._compute_masks(ds, evi)

        if self.config.modis_resolution:
            evi = evi.unstack("location")
            mask = mask.unstack("location")
            evi = evi.coarsen(latitude=12, longitude=12, boundary="trim").mean(
                skipna=True
            )
            mask_coarse_frac = mask.coarsen(
                latitude=12, longitude=12, boundary="trim"
            ).mean(skipna=True)

            mask = xr.where(mask_coarse_frac > 0.5, 1, np.nan)

            # mask = mask.where(mask, np.nan)

            evi = evi.stack(location=("latitude", "longitude"))
            mask = mask.stack(location=("latitude", "longitude"))

        masked_evi = evi * mask
        data = xr.Dataset(
            data_vars={
                f"{self.variable_name}": masked_evi,
            },
        )
        # Check for excessive missing data
        if self._has_excessive_nan(masked_evi):
            printt(f"Excessive NaN values for the selected location.")
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

        if epsg is None:
            raise ValueError("EPSG code not found in dataset attributes.")

        transformer = Transformer.from_crs(epsg, 4326, always_xy=True)

        lon, lat = transformer.transform(ds.x.values, ds.y.values)

        # ds = ds.drop_vars("spatial_ref")
        return ds.assign_coords({"x": ("x", lon), "y": ("y", lat)}).rename(
            {"x": "longitude", "y": "latitude"}
        )

    def _get_random_vegetation_pixel_series(self, ds):
        """Selects a random time serie vegetation pixel location in the minicube based on SCL classification."""

        def _has_sufficient_vegetation(self, ds):
            """Checks if sufficient vegetation exists across the dataset."""
            count_of_4 = (ds.SCL == 4).sum(dim="time")
            mask = count_of_4 > 1 / 4 * (366 / self.config_dict["period_size"]) * len(
                np.unique(ds.time.dt.year)
            )

            eligible_indices = np.argwhere(mask.values)
            return eligible_indices

        eligible_indices = _has_sufficient_vegetation(ds)

        if eligible_indices.size > 0:
            random_index = eligible_indices[np.random.choice(eligible_indices.shape[0])]
            selected_data = ds.isel(longitude=random_index[0], latitude=random_index[1])
            # Expand dimensions and rename for clarity
            selected_data = selected_data.expand_dims(
                longitude=[selected_data.longitude.values.item()],
                latitude=[selected_data.latitude.values.item()],
            )
            # Stack spatial dimensions for easier processing
            return selected_data.stack(location=("longitude", "latitude"))
        return None

    def _calculate_evi(self, ds):
        """Calculates the Enhanced Vegetation Index (EVI)."""
        # return (2.5 * (ds.B08 - ds.B04)) / (
        #    ds.B08 + 6 * ds.B04 - 7.5 * ds.B02 + 1 + 10e-8
        # )
        return (2.5 * (ds.B8A - ds.B04)) / (
            ds.B8A + 6 * ds.B04 - 7.5 * ds.B02 + 1 + 10e-8
        )

    def _compute_masks(self, ds, evi):
        """Applies cloud and vegetation masks to the EVI data."""
        mask = xr.ones_like(evi)  # Default mask (all ones, meaning no masking)

        # Apply vegetation mask if SCL exists
        if "SCL" in ds.data_vars:
            # keep only pixel valid accordingly to the SCL
            valid_scl = ds.SCL.isin([4, 5, 6, 7])

            mask = mask.where(valid_scl, np.nan)

            # keep only time steps with more than 50% of valid pixels
            valid_ratio = valid_scl.sum(
                dim=["location"]  # ["latitude", "longitude"]  # ["location"]  #
            ) / valid_scl.count(
                dim=["location"]  # ["latitude", "longitude"]  # ["location"]
            )  # ["latitude", "longitude"])
            invalid_time_steps = valid_ratio < 0.9
            mask = mask.where(~invalid_time_steps, np.nan)

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
        # Step 1: Compute mean seasonal cycle with at least 2 values
        # mean_seasonal_cycle = clean_data.groupby("time.dayofyear").apply(
        #    lambda x: (
        #        x.mean("time")
        #        if x.count("time") >= 2
        #        else x.isel(time=0) * float("nan")
        #    )
        # )

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
        config["nan_fill_windows"] = [3, 4]
        config["noise_half_windows"] = [1, 2]
        config["cumulative_evi_window"] = 5
        config["period_size"] = 16
        config["smoothing_window_msc"] = 5
        config["poly_msc"] = 2
        config["deseasonalization"] = True

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

        printt(f"Processing entire dataset: {data.sizes['location']} locations.")
        data = self.compute_max_per_period(data, self.config_dict["period_size"])
        data = self.noise_removal.cloudfree_timeseries(
            data, noise_half_windows=self.config_dict["noise_half_windows"]
        )
        data = self._remove_low_vegetation_location(data, threshold=0.2)
        # data = data.compute()
        printt("compute is not the issue")
        self.saver._save_data(data, "evi")

        # Compute Mean Seasonal Cycle (MSC)
        msc = self.compute_msc(data)
        msc = msc.transpose("location", "dayofyear", ...)
        self.saver._save_data(msc, "msc")

        if not return_time_series:
            return msc

        # Step 3: Deseasonalization
        if self.config_dict["deseasonalization"]:
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
