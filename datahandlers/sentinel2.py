from .common_imports import *
from .base import DatasetHandler
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from pyproj import Transformer


class EarthnetDatasetHandler(DatasetHandler):

    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index and load the dataset.
        """

        self.variable_name = "evi_earthnet"

        # Attempt to load preprocessed training data
        training_data = self.loader._load_data("temp_file")
        if training_data is not None:
            self.data = training_data
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

        data = list(filter(lambda x: isinstance(x, xr.DataArray), data))

        if not data:
            raise ValueError("Dataset is empty")

        # Combine the data into an xarray dataset and save it
        self.data = xr.concat(data, dim="location").to_dataset(name=self.variable_name)
        self.data = self.data.sel(
            location=~self.data.get_index("location").duplicated()
        )
        self.saver._save_data(self.data, "temp_file")
        self.data = self.loader._load_data("temp_file")
        printt("Dataset loaded.")

        return self.data

    def _minicube_specific_loading(self):
        data = None
        i = 0
        while (i < 5) and (data is None):
            i += 1
            samples_indice = self.sample_locations(1)
            data = self.load_minicube(samples_indice[0], process_entire_minicube=True)
        data = data.stack(location=("longitude", "latitude"))
        self.data = data.to_dataset(name=self.variable_name)
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
            & df.apply(lambda row: self._is_in_europe(row["lon"], row["lat"]), axis=1)
        ]

        # Random sampling with replacement
        sampled_indices = np.random.choice(df.index, size=n_samples, replace=True)
        return df.loc[sampled_indices, "path"].values

    def load_minicube(self, row, process_entire_minicube=False):
        # row = "/full/1.3/mc_25.61_44.32_1.3_20231018_0.zarr"
        # Load data
        with xr.open_zarr(EARTHNET_FILEPATH + row) as ds:
            transformer = Transformer.from_crs(
                ds.attrs["spatial_ref"], 4326, always_xy=True
            )
            lon, lat = transformer.transform(ds.x.values, ds.y.values)
            # Update the dataset with new coordinates
            ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)})
            ds = ds.rename({"x": "longitude", "y": "latitude"})

            if not process_entire_minicube:
                # Select a random vegetation location
                ds = self._get_random_vegetation_pixel_series(ds)
                if ds is None:
                    return None
            else:
                self.variable_name = ds.attrs["data_id"]
                # Filter based on vegetation occurrence
                if not self._has_sufficient_vegetation(ds):
                    return None
            # Calculate EVI and apply cloud/vegetation mask
            evi = self._calculate_evi(ds)
            masked_evi = self._apply_masks(ds, evi)

            # Check for excessive missing data
            if self._has_excessive_nan(masked_evi):
                return None
            if process_entire_minicube:
                self.saver.update_saving_path(ds.attrs["data_id"])
            return masked_evi

    def _load_data(self, row):
        """Loads the dataset and selects the relevant time slice."""
        try:
            path = self.df.at[row]
            ds = xr.open_zarr(EARTHNET_FILEPATH + path, chuncks={"time": -1})
            # ds = ds.sel(
            #     time=slice(datetime.date(2017, 3, 7), datetime.date(2022, 9, 30))
            # )
            transformer = Transformer.from_crs(
                ds.attrs["spatial_ref"], 4326, always_xy=True
            )
            print(ds.x.values, ds.y.values)
            lon, lat = transformer.transform(ds.x.values, ds.y.values)
            print(lon, lat)
            # Update the dataset with new coordinates
            ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)})
            ds = ds.rename({"x": "longitude", "y": "latitude"})
            print(ds.latitude.values, ds.longitude.values)

            return ds
        except Exception as e:
            printt(f"Error loading data: {e}")
            return None

    def _transform_utm_to_latlon(ds):
        transformer = Transformer.from_crs(
            ds.attrs["spatial_ref"], 4326, always_xy=True
        )
        lon, lat = transformer.transform(ds.x.values, ds.y.values)
        return lon, lat

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
        mask = ds.cloudmask_en.where(ds.cloudmask_en == 0, np.nan)
        mask = mask.where(ds.SCL.isin([4, 5]), np.nan)
        mask = mask.where(mask != 0, 1)
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

    def _remove_low_vegetation_location(self, threshold, msc):
        # Calculate mean data across the dayofyear dimension
        mean_msc = msc.mean("dayofyear", skipna=True)

        # Create a boolean mask for locations where mean is greater than or equal to the threshold
        mask = mean_msc >= threshold

        return mask

    def _deseasonalize(self, data, msc):
        daily_msc = msc.interp(dayofyear=np.arange(1, 366, 1))
        # Align subset_msc with subset_data
        aligned_msc = daily_msc.sel(dayofyear=data["time.dayofyear"])
        # Subtract the seasonal cycle
        deseasonalized = data - aligned_msc
        return deseasonalized

    def clean_and_smooth_timeseries(
        self,
        data: xr.DataArray,
        rolling_window: int = 3,
        bin_size: int = 5,
        smoothing_window: int = 12,
        poly_order: int = 2,
    ) -> xr.DataArray:
        """
        Clean and smooth a time series using rolling windows, neighbor comparison,
        and Savitzky-Golay filtering.

        Parameters
        ----------
        data : xr.DataArray
            Input time series data
        rolling_window : int, optional
            Size of rolling window for initial smoothing, default 3
        bin_size : int, optional
            Size of bins for grouping days, default 16
        smoothing_window : int, optional
            Window size for Savitzky-Golay filter, default 10
        poly_order : int, optional
            Polynomial order for Savitzky-Golay filter, default 2

        Returns
        -------
        xr.DataArray
            Cleaned and smoothed time series
        """

        def remove_local_minima(series: xr.DataArray) -> xr.DataArray:
            """Replace values that are smaller than both neighbors with neighbor mean."""
            # Shift the data to get neighbors
            left_neighbor = series.shift(time=1, fill_value=float("inf"))
            right_neighbor = series.shift(time=-1, fill_value=float("inf"))

            # Find where the value is smaller than both neighbors
            is_smaller = (series < left_neighbor) & (series < right_neighbor)
            masked_series = xr.where(is_smaller, float("nan"), series)
            mean_neighbors = masked_series.rolling(
                time=5, center=True, min_periods=1
            ).mean()
            return xr.where(is_smaller, mean_neighbors, series)

        def fill_nans(series: xr.DataArray, window: int) -> xr.DataArray:
            """Fill NaN values using rolling mean."""
            rolling_mean = series.rolling(
                time=window, center=True, min_periods=1
            ).mean()
            return series.where(~np.isnan(series), other=rolling_mean)

        try:
            # Input validation
            if not isinstance(data, xr.DataArray):
                raise TypeError("Input must be an xarray DataArray")

            # Step 1: Initial rolling window maximum
            # clean_data = (
            #     data.rolling(time=rolling_window, center=True, min_periods=1)
            #     .construct("window_dim")
            #     .max(dim="window_dim")
            # )

            # Step 2: Remove local minima
            clean_data = remove_local_minima(data)

            # Step 3: Handle NaN values
            clean_data = fill_nans(clean_data, rolling_window)
            clean_data = fill_nans(
                clean_data, rolling_window
            )  # Second pass for remaining NaNs

            # Step 4: Remove local minima again after NaN filling
            clean_data = remove_local_minima(clean_data)
            # Step 5: Compute mean seasonal cycle
            bins = np.arange(1, 367, bin_size)
            clean_data = clean_data.assign_coords(
                dayofyear=clean_data["time"].dt.dayofyear
            )
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
            mean_seasonal_cycle = mean_seasonal_cycle.copy(
                data=mean_seasonal_cycle_values
            )
            # Ensure all values are non-negative
            mean_seasonal_cycle = mean_seasonal_cycle.where(mean_seasonal_cycle > 0, 0)

            return clean_data, mean_seasonal_cycle

        except Exception as e:
            raise RuntimeError(f"Error processing time series: {str(e)}") from e

    def apply_cleaning_pipeline(
        self, data: xr.DataArray, config: Optional[dict] = None
    ) -> xr.DataArray:
        """
        Apply the complete cleaning pipeline with optional configuration.

        Parameters
        ----------
        data : xr.DataArray
            Input time series data
        config : dict, optional
            Configuration parameters for the cleaning process

        Returns
        -------
        xr.DataArray
            Cleaned and smoothed time series
        """
        default_config = {
            "rolling_window": 3,
            "bin_size": 5,
            "smoothing_window": 12,
            "poly_order": 2,
        }

        if config is None:
            config = default_config
        else:
            config = {**default_config, **config}  # Merge with defaults

        return self.clean_and_smooth_timeseries(
            data,
            rolling_window=config["rolling_window"],
            bin_size=config["bin_size"],
            smoothing_window=config["smoothing_window"],
            poly_order=config["poly_order"],
        )

    def preprocess_data(
        self,
        scale=True,
        reduce_temporal_resolution=True,
        return_time_serie=False,
        remove_nan=True,
        process_entire_minicube=False,
    ):
        """
        Preprocess data based on the index.
        """
        printt("start of the preprocess")
        if process_entire_minicube:
            self._minicube_specific_loading()
        else:
            self._dataset_specific_loading()
        self.filter_dataset_specific()  # useless, legacy...

        self.data = self.data[self.variable_name]
        printt(
            f"Computation on the entire dataset. {self.data.sizes['location']} samples"
        )

        self.data, self.msc = self.apply_cleaning_pipeline(self.data)

        self.msc = self.msc.transpose("location", "dayofyear", ...)
        self.saver._save_data(self.msc, "msc")
        # self.saver._save_data(self.data, "clean_data")
        # with ProgressBar():
        #     # Compute the tasks in parallel (this will trigger Dask's parallel computation)
        #     data = compute(*data, scheduler="processes")  # , scheduler="processes"

        if return_time_serie:
            self.data = self.data.transpose("location", "time", ...)
            return self.msc, self.data
        else:
            return self.msc
