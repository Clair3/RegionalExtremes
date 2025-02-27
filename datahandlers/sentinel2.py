from .common_imports import *
from .base import DatasetHandler
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from pyproj import Transformer
import cf_xarray as cfxr

violent = False


class Sentinel2DatasetHandler(DatasetHandler):

    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index and load the dataset.
        """

        self.variable_name = "evi_earthnet"

        # Attempt to load preprocessed training data
        # training_data = self.loader._load_data("temp_file")
        path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-12-21_13:11:46_30000_locations/EVI_EN/temp_file.zarr"
        # "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-01-23_10:01:46_deep_extreme_global/EVI_EN/temp_file.zarr"
        training_data = xr.open_zarr(path)
        training_data = cfxr.decode_compress_to_multi_index(training_data, "location")
        if training_data is not None:
            self.data = training_data
            if self.n_samples:
                random_indices = np.random.choice(
                    len(self.data.location), size=self.n_samples, replace=False
                )
                self.data = self.data.isel(location=random_indices)
            return self.data

        # Determine the number of samples to process (default: 10,000)
        sample_count = self.n_samples or 10_000
        print(f"count: {sample_count}")
        samples_paths = self.sample_locations(sample_count)

        printt("Loading dataset...")

        # Use dask.delayed to parallelize the loading and processing
        data = [delayed(self.load_minicube)(path) for path in samples_paths]

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

        return self.data

    def _minicube_specific_loading(self, minicube_path):
        data = self.load_minicube(minicube_path, process_entire_minicube=True)
        self.data = data.stack(location=("longitude", "latitude"))
        if self.n_samples:
            random_indices = np.random.choice(
                len(self.data.location), size=self.n_samples, replace=False
            )
            self.data = self.data.isel(location=random_indices)
        # self.data = self.data  # .isel(location=slice(0, 100))
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

    def load_minicube(self, minicube_path, process_entire_minicube=False):
        filepath = Path(minicube_path)  # EARTHNET_FILEPATH + minicube_path
        with xr.open_zarr(filepath, chunks="auto") as ds:
            # Add landcover
            if "esa_worldcover_2021" not in ds.data_vars:
                ds = self.loader._load_and_add_landcover(filepath, ds)
            # Transform UTM to lat/lon
            ds = self._transform_utm_to_latlon(ds)

            if not process_entire_minicube:
                # Select a random vegetation location
                ds = self._get_random_vegetation_pixel_series(ds)
                if ds is None:
                    return None
            else:
                if not self._has_sufficient_vegetation(ds):
                    print("Not enough vegetation")
                    return None
            self.variable_name = "evi"  # ds.attrs["data_id"]
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
            # if self._has_excessive_nan(masked_evi):
            #     print("Excessive NaN values")
            #     return None
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

        # Apply cloud mask if available
        if "cloudmask_en" in ds.data_vars:
            mask = ds.cloudmask_en.where(ds.cloudmask_en == 0, np.nan)

        # Apply vegetation mask if SCL exists
        if "SCL" in ds.data_vars:
            valid_mask = ds.SCL.isin([4, 5, 6, 7])
            valid_ratio = valid_mask.sum(
                dim=["latitude", "longitude"]
            ) / valid_mask.count(dim=["latitude", "longitude"])
            invalid_time_steps = valid_ratio < 0.90
            mask = mask.where(~invalid_time_steps, np.nan)
            # else:
            #     mask = mask.where(ds.SCL.isin([4, 5]), np.nan)
        # mask = mask.where(
        #     mask != 0, 1
        # )  # Convert to binary mask (1 = valid, NaN = masked)

        return evi * mask

    def _has_excessive_nan(self, data):
        """Checks if the masked data contains excessive NaN values."""
        nan_percentage = data.isnull().mean().values * 100
        print(nan_percentage)
        return nan_percentage > 99

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
        return deseasonalized

    def clean_timeseries(
        self, data: xr.DataArray, window_size: list = [10, 3], gapfill: bool = True
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

        def remove_cloud_noise(data, window_size=5, gapfill=True):

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

        try:
            # Input validation
            if not isinstance(data, xr.DataArray):
                raise TypeError("Input must be an xarray DataArray")

            # Apply initial mask once, as a separate step
            data_masked = data.where((data >= 0) & (data <= 1), np.nan)

            cleaned_data = remove_cloud_noise(
                data_masked, window_size=10, gapfill=gapfill
            )

            # Fill NaNs with rolling mean (this is still needed)
            rolling_mean = cleaned_data.rolling(
                time=3, center=True, min_periods=1
            ).mean()
            cleaned_data = cleaned_data.fillna(rolling_mean)
            rolling_mean = cleaned_data.rolling(
                time=5, center=True, min_periods=1
            ).mean()
            cleaned_data = cleaned_data.fillna(rolling_mean)

            cleaned_data = remove_cloud_noise(
                cleaned_data, window_size=3, gapfill=gapfill
            )
            return cleaned_data

        except Exception as e:
            raise RuntimeError(f"Error processing time series: {str(e)}") from e

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
        # Fill any remaining NaNs with 0
        mean_seasonal_cycle = mean_seasonal_cycle.fillna(0)
        # Step 6: Apply Savitzky-Golay smoothing
        mean_seasonal_cycle_values = savgol_filter(
            mean_seasonal_cycle.values, smoothing_window, poly_order, axis=0
        )
        mean_seasonal_cycle = mean_seasonal_cycle.copy(data=mean_seasonal_cycle_values)
        # Ensure all values are non-negative
        mean_seasonal_cycle = mean_seasonal_cycle.where(mean_seasonal_cycle > 0, 0)
        return mean_seasonal_cycle

    def _remove_low_vegetation_location(self, threshold, msc):
        # Calculate mean data across the dayofyear dimension
        mean_msc = msc.mean("dayofyear", skipna=True)

        # Create a boolean mask for locations where mean is greater than or equal to the threshold
        mask = mean_msc >= threshold

        return mask

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

        if not return_time_serie:
            self.msc = self.loader._load_data("msc")
            if self.msc is not None:
                return self.msc.msc

        if minicube_path:
            self._minicube_specific_loading(minicube_path=minicube_path)
        else:
            self._dataset_specific_loading()
        # self.filter_dataset_specific()  # useless, legacy...
        self.data = self.data[self.variable_name]
        self.saver._save_data(self.data, "evi")
        # Randomly select n indices from the location dimension

        printt(
            f"Computation on the entire dataset. {self.data.sizes['location']} samples"
        )
        gapfilled_data = self.clean_timeseries(self.data, window_size=[10, 3])
        self.msc = self.compute_msc(gapfilled_data)
        self.saver._save_data(self.msc, "msc")

        self.msc = self.msc.transpose("location", "dayofyear", ...)
        if not return_time_serie:
            return self.msc
        else:
            self.data = self.data.transpose("location", "time", ...)
            return self.msc, self.data
