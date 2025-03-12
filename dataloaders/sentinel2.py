from .common_imports import *
from .base import Dataloader
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from pyproj import Transformer
import cf_xarray as cfxr
from .data_processing.helpers import _ensure_time_chunks


class Sentinel2Dataloader(Dataloader):

    def load_dataset(self):
        """
        Preprocess data based on the index and load the dataset.
        """

        self.variable_name = "evi_earthnet"

        # Attempt to load preprocessed training data
        # training_data = self.loader._load_data("temp_file")
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
        print(f"count: {sample_count}")
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
        # self.data = self.data.chunk({"time": -1, "latitude": 10, "longitude": 10})
        self.saver._save_data(self.data, "temp_file")
        self.data = self.loader._load_data("temp_file")
        printt("Dataset loaded.")
        self.data = self.data[self.variable_name]
        # Ensure dataset has time chunks before further processing
        self.data = _ensure_time_chunks(self.data)
        return self.data

    def load_minicube(self, minicube_path):
        data = self.load_file(minicube_path, process_entire_minicube=True)
        self.data = data.stack(location=("longitude", "latitude"))
        if self.n_samples:
            random_indices = np.random.choice(
                len(self.data.location), size=self.n_samples, replace=False
            )
            self.data = self.data.isel(location=random_indices)
        # self.data = self.data  # .isel(location=slice(0, 100))
        self.data = self.data[self.variable_name]
        # Ensure dataset has time chunks before further processing
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
        with xr.open_zarr(filepath) as ds:
            # Transform UTM to lat/lon
            ds = self._transform_utm_to_latlon(ds)
            # Add landcover
            if "esa_worldcover_2021" not in ds.data_vars:
                ds = self.loader._load_and_add_landcover(filepath, ds)

            if not process_entire_minicube:
                # Select a random vegetation location
                print("single pixel")
                ds = self._get_random_vegetation_pixel_series(ds)
                if ds is None:
                    return None
            else:
                if not self._has_sufficient_vegetation(ds):
                    print("Not enough vegetation")
                    return None
            self.variable_name = "evi"
            # Filter based on vegetation occurrence

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
                    "source_path": minicube_path,  # Add the path as a coordinate
                },
            )
            # Check for excessive missing data
            if self._has_excessive_nan(masked_evi):
                print("Excessive NaN values")
                return None
            if process_entire_minicube:
                self.saver.update_saving_path(filepath.stem)

            return data

    def _transform_utm_to_latlon(self, ds):
        """Transforms UTM coordinates to latitude and longitude."""
        epsg = (
            ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
        )
        if epsg is None:
            raise ValueError("EPSG code not found in dataset attributes.")

        transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
        lon, lat = transformer.transform(ds.x.values, ds.y.values)

        if "time" not in ds.dims:
            ds = ds.rename({"time_sentinel-2-l2a": "time"})
        return ds.assign_coords({"x": ("x", lon), "y": ("y", lat)}).rename(
            {"x": "longitude", "y": "latitude"}
        )

    def _get_random_vegetation_pixel_series(self, ds):
        """Selects a random time serie vegetation pixel location in the minicube based on SCL classification."""
        count_of_4 = (ds.SCL == 4).sum(dim="time")  # Count vegetation occurrences
        mask = count_of_4 > 24  # Threshold for sufficient vegetation

        eligible_indices = np.argwhere(mask.values)
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
        mask = count_of_4 > 24
        eligible_indices = np.argwhere(mask.values)
        return eligible_indices.size > 0

    def _calculate_evi(self, ds):
        """Calculates the Enhanced Vegetation Index (EVI)."""
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
            invalid_time_steps = valid_ratio < 0.50
            mask = mask.where(~invalid_time_steps, np.nan)

        if "cloudmask_en" in ds.data_vars:
            mask = mask.where(ds.cloudmask_en == 0, np.nan)

        return evi * mask

    def _has_excessive_nan(self, data):
        """Checks if the masked data contains excessive NaN values."""
        nan_percentage = data.isnull().mean().values * 100
        return nan_percentage > 80

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
        daily_msc = msc.interp(dayofyear=np.arange(1, 367, 1))
        # Align subset_msc with subset_data
        aligned_msc = daily_msc.sel(dayofyear=data["time.dayofyear"])
        # Subtract the seasonal cycle
        deseasonalized = data - aligned_msc
        deseasonalized = deseasonalized.reset_coords("dayofyear", drop=True)
        return deseasonalized

    def compute_msc(
        self,
        clean_data: xr.DataArray,
        bin_size: int = 5,
        smoothing_window: int = 12,
        poly_order: int = 2,
    ):
        # Step 5: Add day of year for deseasonalizing data
        clean_data = clean_data.assign_coords(dayofyear=clean_data["time"].dt.dayofyear)
        bins = np.arange(1, 367, bin_size)
        mean_seasonal_cycle = (
            clean_data.groupby_bins("time.dayofyear", bins=bins, right=False)
            .mean("time", skipna=True)
            .rename({"dayofyear_bins": "dayofyear"})
        )

        # Set dayofyear to bin midpoints
        mean_seasonal_cycle["dayofyear"] = [
            interval.mid for interval in mean_seasonal_cycle.dayofyear.values
        ]

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
        #     padded_values, window_size=4, min_periods=1
        # )

        # Fill any remaining NaNs with 0
        # mean_seasonal_cycle = mean_seasonal_cycle.fillna(0)
        # Step 6: Apply Savitzky-Golay smoothing
        smoothed_values = savgol_filter(
            padded_values, smoothing_window, poly_order, axis=0
        )
        mean_seasonal_cycle = mean_seasonal_cycle.copy(
            data=smoothed_values[smoothing_window:-smoothing_window]
        )
        # Ensure all values are non-negative
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

    # def clean_timeseries_fixed_stats(self, deseasonalized):
    #     # Calculate GLOBAL median and std for each location (not rolling)
    #     global_mean = deseasonalized.mean(dim="time", skipna=True)
    #     global_std = deseasonalized.std(dim="time", skipna=True).clip(min=0.01)
    #     # # Broadcast to match dimensions for computation
    #     mean_broadcast = global_mean.broadcast_like(deseasonalized)
    #     std_broadcast = global_std.broadcast_like(deseasonalized)
    #     # Identify and replace outliers using global statistics
    #     is_negative_outlier = deseasonalized < (mean_broadcast - 2 * std_broadcast)
    #     cleaned_data = xr.where(is_negative_outlier, np.nan, deseasonalized)
    #     return cleaned_data

    def get_config(self):
        # Define window sizes for gap-filling and cloud noise removal
        config = dict()
        config["nan_fill_windows"] = [5, 7]
        config["noise_half_windows"] = [10, 3, 1]
        config["cumulative_evi_window"] = 5
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
        dict_config = self.get_config()
        # Load data either from a minicube or from the default dataset
        if minicube_path:
            data = self.load_minicube(minicube_path=minicube_path)
        else:
            data = self.load_dataset()
        printt(f"Processing entire dataset: {data.sizes['location']} locations.")
        # self.saver._save_data(data, "evi")
        # Step 1: Gap-filling & noise removal
        gapfilled_data = self.noise_removal.clean_and_gapfill_timeseries(
            data,
            nan_fill_windows=dict_config["nan_fill_windows"],
            noise_half_windows=dict_config["noise_half_windows"],
        )
        # Compute Mean Seasonal Cycle (MSC)
        msc = self.compute_msc(gapfilled_data)
        msc = msc.transpose("location", "dayofyear", ...)
        self.saver._save_data(msc, "msc")

        if not return_time_series:
            return msc

        cumulative_evi = self.loader._load_data("cumulative_evi")
        if cumulative_evi is not None:
            return msc, cumulative_evi

        # Step 2: Cloud removal
        data = self.noise_removal.cloudfree_timeseries(
            data, noise_half_windows=dict_config["noise_half_windows"]
        )
        self.saver._save_data(data, "clean_data")

        # Step 3: Deseasonalization
        data = self._deseasonalize(data, msc)  # needed?
        self.saver._save_data(data, "deseasonalized")

        # Step 5: Cumulative EVI computation
        data = self.cumulative_evi(
            data, window_size=dict_config["cumulative_evi_window"]
        )
        self.saver._save_data(data, "cumulative_evi")

        data = _ensure_time_chunks(data)
        data = data.transpose("location", "time", ...).compute()
        return msc, data
