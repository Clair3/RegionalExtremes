from abc import ABC, abstractmethod
from config import InitializationConfig
from typing import Union, Optional
from .common_imports import *


class DatasetHandler(ABC):
    def __init__(self, config: InitializationConfig, n_samples: Union[int, None]):
        """
        Initialize DatasetHandler.

        Parameters:
        n_samples (Union[int, None]): Number of samples to select.
        time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        # Config class to deal with loading and saving the model.
        self.config = config
        # Number of samples to load. If None, the full dataset is loaded.
        self.n_samples = n_samples
        # Loader class to load intermediate steps.
        self.loader = Loader(config)
        # Saver class to save intermediate steps.
        self.saver = Saver(config)

        self.start_year = self.config.start_year

        # data loaded from the dataset
        self.data = None
        # Mean seasonal cycle
        self.msc = None

        # minimum and maximum of the data for normalization
        self.max_data = None
        self.min_data = None

    def preprocess_data(
        self,
        scale=True,
        reduce_temporal_resolution=True,
        return_time_serie=False,
        remove_nan=True,
    ):
        """
        Preprocess data based on the index.
        """
        self._dataset_specific_loading()
        self.filter_dataset_specific()

        # Stack the dimensions
        self.data = self.data.stack(location=("longitude", "latitude"))

        # Select only a subset of the data if n_samples is specified
        if self.n_samples:
            self.randomly_select_n_samples()
        else:
            self.data = self.data[self.variable_name]
            printt(
                f"Computation on the entire dataset. {self.data.sizes['location']} samples"
            )

        self.compute_msc()

        if reduce_temporal_resolution:
            self._reduce_temporal_resolution()

        if remove_nan and not self.n_samples:
            self._remove_nans()

        if scale:
            self._scale_msc()

        self.msc = self.msc.transpose("location", "dayofyear", ...)

        if return_time_serie:
            self.data = self.data.transpose("location", "time", ...)
            return self.msc, self.data
        else:
            return self.msc

    @abstractmethod
    def _dataset_specific_loading(self, *args, **kwargs):
        pass

    @abstractmethod
    def filter_dataset_specific(self, *args, **kwargs):
        pass

    def _spatial_filtering(self, data):
        # Filter data from the polar regions
        remove_poles = np.abs(data.latitude) <= NORTH_POLE_THRESHOLD

        # Filter dataset to select Europe
        # Select European data
        in_europe = self._is_in_europe(data.longitude, data.latitude)
        printt("Data filtred to Europe.")

        in_land = self._is_in_land(data)

        self.data = data.where(remove_poles & in_europe & in_land, drop=True)

        return self.data

    def _is_in_land(self, data):
        # Create a land mask
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        mask = land.mask(data.longitude, data.latitude).astype(bool)

        # Mask the ocean
        mask = ~mask

        return mask

    @abstractmethod
    def _remove_low_vegetation_location(self, data):
        # This method will be implemented by subclasses
        pass

    def randomly_select_n_samples(self, factor=5):
        """
        Randomly select a subset of n_samples of data.
        """
        # Generate a large number of random coordinates
        n_candidates = self.n_samples * factor
        lons = np.random.choice(self.data.longitude, size=n_candidates, replace=True)
        lats = np.random.choice(self.data.latitude, size=n_candidates, replace=True)

        selected_locations = list(zip(lons, lats))
        self.data = self.data.chunk({"time": len(self.data.time), "location": 1})

        # Select the values at the specified coordinates
        selected_data = self.data[self.variable_name].sel(location=selected_locations)
        # Remove NaNs
        condition = ~selected_data.isnull().any(dim="time").compute()  #
        selected_data = selected_data.where(condition, drop=True)

        # Select randomly n_samples samples in selected_data
        self.data = selected_data.isel(
            location=np.random.choice(
                selected_data.location.size,
                size=min(self.n_samples, selected_data.location.size),
                replace=False,
            )
        )

        if self.data.sizes["location"] != self.n_samples:
            raise ValueError(
                f"Number of samples ({self.data.sizes['location']}) != n_samples ({self.n_samples}). The number of samples without NaNs is likely too low, increase the factor of n_candidates."
            )
        printt(f"Randomly selected {self.data.sizes['location']} samples for training.")

    def _is_in_europe(self, lon, lat):
        """
        Check if the given longitude and latitude are within the bounds of Europe.
        """
        # Define Europe boundaries (these are approximate)
        lon_min, lon_max = -31.266, 39.869  # Longitude boundaries
        lat_min, lat_max = 27.636, 81.008  # Latitude boundaries

        # Check if the point is within the defined boundaries
        in_europe = (
            (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        )
        return in_europe

    def compute_msc(self):
        """
        Compute the Mean Seasonal Cycle (MSC) and optionally the Variance Seasonal Cycle (VSC)
        of n samples and scale it between 0 and 1.

        Time resolution reduces the resolution of the MSC to decrease computation workload.
        Number of values = 366 / time_resolution.
        """
        msc = self._compute_msc()
        printt("MSC computed.")

        if self.config.compute_variance:
            vsc = self._compute_vsc()
            self.msc = self._combine_msc_vsc(msc, vsc)
            printt("Variance is computed.")
        else:
            self.msc = msc
        self._rechunk_data()

    def _compute_msc(self):
        return self.data.groupby("time.dayofyear").mean("time", skipna=True)

    def _compute_vsc(self):
        return (
            self.data.groupby("time.dayofyear")
            .var("time", skipna=True)
            .isel(dayofyear=slice(1, 365))
        )

    def _combine_msc_vsc(self, msc, vsc):
        msc_vsc = xr.concat([msc, vsc], dim="dayofyear")
        total_days = len(msc_vsc.dayofyear)
        return msc_vsc.assign_coords(dayofyear=("dayofyear", range(total_days)))

    def _rechunk_data(self):
        self.msc = self.msc.chunk({"dayofyear": len(self.msc.dayofyear), "location": 1})

    def _remove_nans(self):
        # If a NaN mask is precomputed
        mask = self.loader._load_spatial_masking()  # return None if no mask available
        if mask:
            printt("Mask loaded.")
            mask_broadcasted = mask.EVIgapfilled_QCdyn.broadcast_like(self.msc)
            mask_broadcasted = mask_broadcasted.sel(location=self.msc.location)
            mask_broadcasted = mask_broadcasted.chunk("auto")
            self.msc = self.msc.where(mask_broadcasted.compute(), drop=True)
            printt("Mask applied. NaNs removed")

        else:
            printt("Mask precomputed unavailable.")
            not_low_vegetation = self._remove_low_vegetation_location(0.1, self.msc)
            condition = ~self.msc.isnull().any(dim="dayofyear")
            self.msc = self.msc.where(
                (condition & not_low_vegetation).compute(), drop=True
            )
            # self.computespatial_masking = self.spatial_masking.broadcast_like(condition)
            self.saver._save_spatial_masking(condition & not_low_vegetation)
            printt("Mask Computed. NaNs removed")

    def _reduce_temporal_resolution(self):
        # first day and last day are removed due to error in the original data.
        self.msc = self.msc.isel(
            dayofyear=slice(2, len(self.msc.dayofyear) - 1, self.config.time_resolution)
        )

    def _get_min_max_data(self):
        minmax_data = self.loader._load_minmax_data()
        if isinstance(minmax_data, xr.Dataset):
            self.min_data = minmax_data.min_data
            self.max_data = minmax_data.max_data
        else:
            self._compute_and_save_min_max_data()

    def _compute_and_save_min_max_data(self):
        assert (
            self.max_data and self.min_data
        ) is None, "the min and max of the data are already defined."
        # assert self.config.path_load_experiment is None, "A model is already loaded."

        self.max_data = self.msc.max(dim=["location"])
        self.min_data = self.msc.min(dim=["location"])
        # Save min_data and max_data
        self.saver._save_minmax_data(
            self.max_data, self.min_data, self.msc.coords["dayofyear"].values
        )

    def _scale_msc(self):
        self._get_min_max_data()
        self.msc = (self.min_data - self.msc) / (self.max_data - self.min_data) + 1e-8
        self.msc = self.msc.chunk({"dayofyear": len(self.msc.dayofyear), "location": 1})
        printt("Data are scaled between 0 and 1.")

    def _deseasonalize(self, subset_data, subset_msc):
        # Align subset_msc with subset_data
        aligned_msc = subset_msc.sel(dayofyear=subset_data["time.dayofyear"])
        # Subtract the seasonal cycle
        deseasonalized = subset_data - aligned_msc
        deseasonalized = deseasonalized.isel(
            time=slice(2, len(deseasonalized.time) - 1)
        )
        return deseasonalized
