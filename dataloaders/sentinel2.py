from .common_imports import *
from .base import Dataloader
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from pyproj import Transformer
import cf_xarray as cfxr
from .data_processing.helpers import _ensure_time_chunks, circular_rolling_mean
from datetime import datetime, timedelta, date


class Sentinel2Dataloader(Dataloader):

    def load_dataset(self):
        """
        Preprocess data based on the index and load the dataset.
        """

        self.variable_name = "evi_earthnet"

        # Attempt to load preprocessed training data
        training_data = self.loader._load_data("temp_file")
        path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-12-21_13:11:46_30000_locations/EVI_EN/temp_file.zarr"
        training_data = xr.open_zarr(path)
        training_data = cfxr.decode_compress_to_multi_index(training_data, "location")
        if training_data is not None:
            self.data = training_data
            if self.n_samples:
                random_indices = np.random.choice(
                    len(self.data.location), size=self.n_samples, replace=False
                )
                self.data = self.data.isel(location=random_indices)
            self.data = self.data[self.variable_name]
            self.data = _ensure_time_chunks(self.data)
            return self.data

        # Determine the number of samples to process (default: 10,000)
        sample_count = self.n_samples or 10_000
        printt(f"count: {sample_count}")
        samples_paths = self.sample_locations(sample_count)

        printt("Loading dataset...")

        # Use dask.delayed to parallelize the loading and processing
        data = [delayed(self.load_file)(path) for path in samples_paths]

        with ProgressBar():
            data = compute(*data, scheduler="processes")

        data = list(filter(lambda x: isinstance(x, xr.Dataset), data))
        if not data:
            raise ValueError("Dataset is empty")

        # Combine the data into an xarray dataset and save it
        self.data = xr.concat(data, dim="location")
        self.data = self.data.sel(
            location=~self.data.get_index("location").duplicated()
        )
        self.data = self.data.drop("band")
        self.saver._save_data(self.data, "temp_file")
        self.data = self.loader._load_data("temp_file")
        printt("Dataset loaded.")
        self.data = self.data[self.variable_name]
        # Ensure dataset has time chunks before further processing
        self.data = _ensure_time_chunks(self.data)
        return self.data

    def load_minicube(self, minicube_path):
        data = self.load_file(minicube_path, process_entire_minicube=True)
        if data is None:
            return None
        self.data = data.stack(location=("longitude", "latitude"))
        if self.n_samples:
            random_indices = np.random.choice(
                len(self.data.location), size=self.n_samples, replace=False
            )
            self.data = self.data.isel(location=random_indices)
        # self.data = self.data  # .isel(location=slice(0, 100))
        self.data = self.data[self.variable_name]
        self.data = _ensure_time_chunks(self.data)
        return self.data

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
        metadata_file = (
            "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/DeepExtremes_enough_vegetation_europe.csv"
            # "/Net/Groups/BGI/work_2/scratch/DeepExtremes/mc_earthnet_biome.csv"
        )
        df = pd.read_csv(metadata_file, delimiter=",", low_memory=False)  # [
        #    [
        #        "path",
        #        "group",
        #        "check",
        #        "start_date",
        #        "lat",
        #        "lon",
        #    ]
        # ]
        # df = df.loc[
        #     (df["check"] == 0)
        #     & df.apply(lambda row: self._is_in_europe(row["lon"], row["lat"]), axis=1)
        # ]
        # Random sampling with replacement
        sampled_indices = np.random.choice(df.index, size=n_samples, replace=True)
        return df.loc[sampled_indices].values  # df.loc[sampled_indices, "path"].values

    def load_file(self, minicube_path, process_entire_minicube=False):
        filepath = Path(minicube_path)  # EARTHNET_FILEPATH + minicube_path
        ds = xr.open_zarr(filepath).astype(np.float32)

        ds = self._ensure_coordinates(ds)
        ds = ds.sel(time=slice(date(2017, 3, 1), None))

        # Add landcover
        if "esa_worldcover_2021" not in ds.data_vars:
            ds = self.loader._load_and_add_landcover(filepath, ds)

        if not process_entire_minicube:
            # Select a random vegetation location
            ds = self._get_random_vegetation_pixel_series(ds)
            if ds is None:
                printt(f"No valid vegetation pixel found in {minicube_path}.")
                return None
        else:
            self.saver.update_saving_path(filepath.stem)

        data = self.compute_vegetation_index(ds, minicube_path)
        return data

    def compute_vegetation_index(self, ds, filepath):
        self.variable_name = "evi"

        # Calculate EVI and apply cloud/vegetation mask
        evi = self._calculate_evi(ds)
        masked_evi = self._apply_masks(ds, evi)
        data = xr.Dataset(
            data_vars={
                f"{self.variable_name}": masked_evi,  # Adding 'evi' as a variable
                # "landcover": ds[
                #    "esa_worldcover_2021"
                # ],  # Adding 'landcover' as another variable
            },
            coords={
                "source_path": filepath,  # Add the path as a coordinate
            },
        )
        # Check for excessive missing data
        if self._has_excessive_nan(masked_evi):
            printt(f"Excessive NaN values in {filepath}.")
            return None
        return data

    def _ensure_coordinates(self, ds):
        """Transforms UTM coordinates to latitude and longitude."""
        if "time" not in ds.dims:
            ds = ds.rename({"time_sentinel-2-l2a": "time"})
        if "longitude" not in ds.dims:
            ds = ds.rename({"x": "longitude", "y": "latitude"})

        if "x_20" in ds.dims:
            for band in bands_20m:
                bands_20m = ["B05", "B06", "B07", "B8A", "SCL"]
                ds[band] = ds[band].interp(
                    x=xr.DataArray(ds.x, dims="x"),
                    y=xr.DataArray(ds.y, dims="y"),
                    method="linear",
                )
        return ds

    def _get_random_vegetation_pixel_series(self, ds):
        """Selects a random time serie vegetation pixel location in the minicube based on SCL classification."""
        eligible_indices = self._has_sufficient_vegetation(ds)

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

    def _has_sufficient_vegetation(self, ds):
        """Checks if sufficient vegetation exists across the dataset."""
        count_of_4 = (ds.SCL == 4).sum(dim="time")
        mask = count_of_4 > 1 / 4 * (366 / self.config_dict["period_size"]) * len(
            np.unique(ds.time.dt.year)
        )

        eligible_indices = np.argwhere(mask.values)
        return eligible_indices  # .size > 0

    def _calculate_evi(self, ds):
        """Calculates the Enhanced Vegetation Index (EVI)."""
        # return (2.5 * (ds.B08 - ds.B04)) / (
        #    ds.B08 + 6 * ds.B04 - 7.5 * ds.B02 + 1 + 10e-8
        # )
        return (2.5 * (ds.B8A - ds.B04)) / (
            ds.B8A + 6 * ds.B04 - 7.5 * ds.B02 + 1 + 10e-8
        )

    def _apply_masks(self, ds, evi):
        """Applies cloud and vegetation masks to the EVI data."""
        mask = xr.ones_like(evi)  # Default mask (all ones, meaning no masking)

        # Apply vegetation mask if SCL exists
        if "SCL" in ds.data_vars:
            # keep only pixel valid accordingly to the SCL
            valid_scl = ds.SCL.isin([4, 5, 6])
            mask = mask.where(valid_scl, np.nan)

            # keep only time steps with more than 50% of valid pixels
            valid_ratio = valid_scl.sum(
                dim=["latitude", "longitude"]
            ) / valid_scl.count(dim=["latitude", "longitude"])
            invalid_time_steps = valid_ratio < 0.97
            mask = mask.where(~invalid_time_steps, np.nan)

        if "cloudmask_en" in ds.data_vars:
            mask = mask.where(ds.cloudmask_en == 0, np.nan)

        return evi * mask

    def _has_excessive_nan(self, data):
        """Checks if the masked data contains excessive NaN values."""
        nan_percentage = data.isnull().mean().values * 100
        return nan_percentage > 90

    def filter_dataset_specific(self):
        """
        Standarize the ecological xarray. Remove irrelevant area, and reshape for the PCA.
        """
        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "location")
        ), f"Dimension missing. dimension are: {self.data.size}"

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
            padded_values, window_size=smoothing_window, min_periods=1
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

    def compute_mean_per_period(self, data, period_size=10):
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
        mean_per_period = data_grouped.mean(dim="time")
        # mean_per_period = data_grouped.max(dim="time")

        # Apply the transformation to convert periods back to midpoints in time
        start_period_times = [
            pd.to_datetime(periods[p]) for p in mean_per_period.coords["period"].values
        ]

        # Update mean_per_period with the transformed 'time' coordinates (midpoints)
        mean_per_period.coords["time"] = ("period", start_period_times)
        mean_per_period = mean_per_period.swap_dims({"period": "time"}).drop_vars(
            "period"
        )
        mean_per_period = mean_per_period.set_index(location=["longitude", "latitude"])
        return mean_per_period

    def get_config(self):
        # Define window sizes for gap-filling and cloud noise removal
        config = dict()
        config["nan_fill_windows"] = [3, 5]
        config["noise_half_windows"] = [1, 2]
        config["cumulative_evi_window"] = 5
        config["period_size"] = 16
        config["smoothing_window_msc"] = 7
        config["poly_msc"] = 2
        config["deseasonalization"] = True

        return config

    def _remove_low_vegetation_location(self, vegetation_index, threshold=0.1):
        mean_vi = vegetation_index.mean("time", skipna=True)
        valid_locations = (mean_vi > 0.2) & ~np.isnan(mean_vi)

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
        dict_config = self.get_config()
        # WARNING: maybe bug somewhere on threshold with the following line. Idk why...
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
        data = self.compute_mean_per_period(data, dict_config["period_size"])
        data = self.noise_removal.cloudfree_timeseries(
            data, noise_half_windows=dict_config["noise_half_windows"]
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
        if dict_config["deseasonalization"]:
            data = self._deseasonalize(data, msc)  # needed?
            self.saver._save_data(data, "deseasonalized")

        # Step 5: Cumulative EVI computation
        # data = self.cumulative_evi(
        #     data, window_size=dict_config["cumulative_evi_window"]
        # )
        # self.saver._save_data(data, "cumulative_evi")

        data = _ensure_time_chunks(data)
        data = data.transpose("location", "time", ...).compute()
        # data = data.chunk({"time": -1, "location": 100})
        # data = _ensure_time_chunks(data)
        return msc, data
