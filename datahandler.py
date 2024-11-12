import xarray as xr
import zarr
import dask
import numpy as np
import random
import datetime
import regionmask
import sys
from typing import Union
import pandas as pd
from abc import ABC, abstractmethod
from config import (
    InitializationConfig,
    CLIMATIC_INDICES,
    ECOLOGICAL_INDICES,
    EARTHNET_INDICES,
)
from loader_and_saver import Loader, Saver
from concurrent.futures import ThreadPoolExecutor

np.random.seed(2024)
np.set_printoptions(threshold=sys.maxsize)
from utils import printt

NORTH_POLE_THRESHOLD = 66.5
SOUTH_POLE_THRESHOLD = -66.5
MAX_NAN_PERCENTAGE = 0.7
CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
ECOLOGICAL_FILEPATH = (
    lambda index: f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{index}/Groups_{index}gapfilled_QCdyn.zarr"
)
VARIABLE_NAME = lambda index: f"{index}gapfilled_QCdyn"
EARTHNET_FILEPATH = "/Net/Groups/BGI/work_2/scratch/DeepExtremes/dx-minicubes"  # "/Net/Groups/BGI/tscratch/crobin/dx-minicubes_interpolated"


@staticmethod
def create_handler(config, n_samples):
    if config.is_generic_xarray_dataset:
        return GenericDatasetHandler(config=config, n_samples=n_samples)
    elif config.index in ECOLOGICAL_INDICES:
        return EcologicalDatasetHandler(config=config, n_samples=n_samples)
    elif config.index in CLIMATIC_INDICES:
        return ClimaticDatasetHandler(config=config, n_samples=n_samples)
    elif config.index in EARTHNET_INDICES:
        return EarthnetDatasetHandler(config=config, n_samples=n_samples)
    else:
        raise ValueError("Invalid index")


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


class ClimaticDatasetHandler(DatasetHandler):
    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index.
        """
        if self.config.index in ["pei_30", "pei_90", "pei_180"]:
            self.load_data(CLIMATIC_FILEPATH)
        else:
            raise ValueError(
                "Index unavailable. Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'."
            )
        return self.data

    def load_data(self, filepath):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        if not filepath:
            filepath = CLIMATIC_FILEPATH(self.config.index)
        # name of the variable in the xarray. self.variable_name
        self.variable_name = self.config.index
        self.data = xr.open_zarr(filepath)[[self.variable_name]]
        self._transform_longitude()
        printt("Data loaded from {}".format(filepath))

    def _transform_longitude(self):
        # Transform the longitude coordinates
        self.data = self.data.roll(
            longitude=180 * 4, roll_coords=True
        )  # Shifts the data of longitude of 180*4 elements, elements that roll past the end are re-introduced

        # Transform the longitude coordinates to -180 and 180
        self.data = self.data.assign_coords(
            longitude=self._coordstolongitude(self.data.longitude)
        )

    def _coordstolongitude(self, x):
        """Transform the longitude coordinates from between 0 and 360 to between -180 and 180."""
        return ((x + 180) % 360) - 180

    def filter_dataset_specific(self):
        """
        Apply climatic transformations using xarray.apply_ufunc.
        """
        assert (
            self.config.index in CLIMATIC_INDICES
        ), f"Index unavailable. Index available: {CLIMATIC_INDICES}."

        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"
        # Ensure longitude values are within the expected range
        assert (
            (self.data.longitude >= -180) & (self.data.longitude <= 180)
        ).all(), "Longitude values should be in the range -180 to 180"

        # Remove the years before 1970 due to quality
        self.data = self.data.sel(
            time=slice(datetime.date(1970, 1, 1), datetime.date(2022, 12, 31))
        )

        self.data = self._spatial_filtering(self.data)

        printt(f"Climatic data loaded with dimensions: {self.data.sizes}")

    @abstractmethod
    def _remove_low_vegetation_location(self, data):
        # not applicable to this dataset
        return data


class EcologicalDatasetHandler(DatasetHandler):

    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index.
        """
        if self.config.index in ECOLOGICAL_INDICES:
            filepath = ECOLOGICAL_FILEPATH(self.config.index)
            self.load_data(filepath)
            # self.reduce_resolution()
        else:
            raise NotImplementedError(
                f"Index {self.config.index} unavailable. Ecological Index available: {ECOLOGICAL_INDICES}."
            )
        return self.data

    def load_data(self, filepath=None):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        if not filepath:
            filepath = ECOLOGICAL_FILEPATH(self.config.index)

        self.variable_name = VARIABLE_NAME(self.config.index)
        self.data = xr.open_zarr(filepath, consolidated=False)[[self.variable_name]]
        printt("Data loaded from {}".format(filepath))
        self._stackdims()

    def _stackdims(self):
        self.data = self.data.stack(
            {
                "latitude": ["latchunk", "latstep_modis"],
                "longitude": ["lonchunk", "lonstep_modis"],
            }
        )
        self.data = self.data.reset_index(["latitude", "longitude"])
        self.data["latitude"] = self.data.latchunk + self.data.latstep_modis
        self.data["longitude"] = self.data.lonchunk + self.data.lonstep_modis

        self.data = self.data.set_index(latitude="latitude", longitude="longitude")
        self.data = self.data.drop(
            ["latchunk", "latstep_modis", "lonchunk", "lonstep_modis"]
        )

    def reduce_resolution(self):
        res_lat, res_lon = len(self.data.latitude), len(self.data.longitude)
        self.data = self.data.coarsen(latitude=5, longitude=5, boundary="trim").mean()
        printt(
            f"Reduce the resolution from ({res_lat}, {res_lon}) to ({len(self.data.latitude)}, {len(self.data.longitude)})."
        )

    def filter_dataset_specific(self):
        """
        Standarize the ecological xarray. Remove irrelevant area, and reshape for the PCA.
        """
        assert (
            self.config.index in ECOLOGICAL_INDICES
        ), f"Index {self.config.index} unavailable. Index available: {ECOLOGICAL_INDICES}."

        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"

        # Temporal filtering.
        self.data = self.data.sel(
            time=slice(
                datetime.date(self.start_year, 1, 1), datetime.date(2022, 12, 31)
            )
        )
        # Then remove specific years using boolean indexing
        years_to_exclude = [2003, 2018, 2019, 2020, 2022]
        self.data = self.data.sel(time=~self.data.time.dt.year.isin(years_to_exclude))

        self.data = self._spatial_filtering(self.data)

        printt(f"Ecological data loaded with dimensions: {self.data.sizes}")

    def _remove_low_vegetation_location(self, threshold, msc):
        # Calculate mean data across the dayofyear dimension
        mean_msc = msc.mean("dayofyear", skipna=True)

        # Create a boolean mask for locations where mean is greater than or equal to the threshold
        mask = mean_msc >= threshold

        return mask


class GenericDatasetHandler(DatasetHandler):
    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index.
        """
        self.data = self.config.data
        return self.data

    def filter_dataset_specific(self):
        """
        Standarize the ecological xarray. Remove irrelevant area, and reshape for the PCA.
        """
        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"

    def _remove_low_vegetation_location(self, threshold, msc):
        # Calculate mean data across the dayofyear dimension
        mean_msc = msc.mean("dayofyear", skipna=True)

        # Create a boolean mask for locations where mean is greater than or equal to the threshold
        mask = mean_msc >= threshold

        return mask


class EarthnetDatasetHandler(DatasetHandler):
    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index.
        """
        self.variable_name = "evi_earthnet"
        if self.n_samples:
            samples_indices, df = self.sample_locations(self.n_samples)
        else:
            samples_indices, df = self.sample_locations(10000)
        self.df = df
        with ThreadPoolExecutor() as executor:
            data = list(executor.map(self.load_minicube, samples_indices))
        filtered_data_arrays = [da for da in data if da is not None]
        ds = xr.concat(filtered_data_arrays, dim="location")
        self.data = ds.to_dataset(name=self.variable_name)
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
        df = pd.read_csv(metadata_file, delimiter=",")[
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

        return sampled_indices, df

    def load_minicube(self, row):

        ds = (
            xr.open_zarr(EARTHNET_FILEPATH + self.df.loc[row]["path"])
            # .isel(x=row["x"], y=row["y"])
            .sel(time=slice(datetime.date(2017, 3, 7), datetime.date(2022, 9, 30)))
        )
        ds = self._select_randomly_vegetation_location(ds)
        if ds is None:
            return None
        evi = (2.5 * (ds.B8A - ds.B04)) / (ds.B8A + 6 * ds.B04 - 7.5 * ds.B02 + 1)
        # evi = (ds.B8A - ds.B04) / (ds.B8A + ds.B04)
        # Mask
        # cloudmask_en > 1 where there is a cloud, 0 where data
        mask = ds.cloudmask_en.where(ds.cloudmask_en == 0, np.nan)
        mask = mask.where(ds.SCL == (4 or 5), np.nan)
        mask = mask.where(mask != 0, 1)

        nan_percentage = mask.isnull().mean().values * 100
        if nan_percentage > 80:
            return None
        else:
            return evi * mask

    def _select_randomly_vegetation_location(self, ds):
        # Count occurrences of 4 and 5 at each location across the 'time' dimension
        count_of_4 = (ds.SCL == 4).sum(dim="time")  # vegetation classification
        mask = count_of_4 > 24

        # Get indices of locations that satisfy the condition
        eligible_indices = np.argwhere(mask.values)
        if eligible_indices.size > 0:
            random_index = eligible_indices[np.random.choice(eligible_indices.shape[0])]
            selected_data = ds.isel(x=random_index[0], y=random_index[1])

            # Expand x and y as singleton dimensions to enable stacking
            selected_data = selected_data.expand_dims(
                x=[selected_data.x.values.item()],
                y=[selected_data.y.values.item()],
            )

            # Rename x and y to longitude and latitude
            selected_data = selected_data.rename({"x": "longitude", "y": "latitude"})

            # Stack x and y into a single location dimension
            selected_data = selected_data.stack(location=("longitude", "latitude"))
            return selected_data
        else:
            return None

    def filter_dataset_specific(self):
        """
        Standarize the ecological xarray. Remove irrelevant area, and reshape for the PCA.
        """
        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "location")
        ), "Dimension missing"

    def _remove_low_vegetation_location(self, threshold, msc):
        # Calculate mean data across the dayofyear dimension
        mean_msc = msc.mean("dayofyear", skipna=True)

        # Create a boolean mask for locations where mean is greater than or equal to the threshold
        mask = mean_msc >= threshold

        return mask

    # def _deseasonalize(self, subset_data, subset_msc):
    #    # Assuming `self.data` has a 'time' coordinate that includes multiple years
    #    subset_data = subset_data.assign_coords(
    #        dayofyear=subset_data["time"].dt.dayofyear
    #    )
    #    print(subset_data)
    #    print(subset_data.values.shape)
    #
    #    # Define 15-day bins for dayofyear
    #    bins = np.arange(1, 367, 15)  # Bin edges to create 15-day intervals
    #    midpoints = (
    #        bins[:-1] + 7.5
    #    )  # Midpoints for each bin, representing the center of each 15-day interval
    #
    #    # Group by year first, then apply groupby_bins on dayofyear within each year
    #    subset_data = subset_data.groupby(
    #        "time.year"
    #    ).apply(  # Group by each year first
    #        lambda x: x.groupby_bins("dayofyear", bins=bins, right=False).mean(
    #            "time", skipna=True
    #        )
    #    )  # Apply groupby_bins and mean on each year separately
    #    # Rename `dayofyear_bins` to `dayofyear` and set the midpoints
    #    subset_data = subset_data.rename({"dayofyear_bins": "dayofyear"})
    #    subset_data["dayofyear"] = midpoints  # Set dayofyear to the midpoints
    #    msc_expanded = subset_msc.expand_dims("year")
    #    print(msc_expanded)
    #    result = subset_data - msc_expanded
    #
    #    print(result)
    #
    #    # Optionally, reassign the year as a coordinate
    #    # Step 1: Convert `year` to a regular coordinate
    #    # subset_data = subset_data.reset_index("year")
    #
    #    # Step 2: Drop `year` if you don’t need it as a coordinate anymore
    #    # subset_data = subset_data.drop_vars("year")
    #    # subset_data = subset_data.reset_coords(["year"], drop=True)
    #
    #    # subset_data = subset_data.assign_coords(year=("year", subset_data["year"]))
    #    # Align subset_msc with subset_data
    #    # aligned_msc = subset_msc.sel(dayofyear=subset_data["time.dayofyear"])
    #    # Subtract the seasonal cycle
    #    # deseasonalized = subset_data - subset_msc
    #
    #    return deseasonalized
    def _deseasonalize(self, subset_data, subset_msc):
        # Align subset_msc with subset_data
        aligned_msc = subset_msc.sel(dayofyear=subset_data["time.dayofyear"])
        # Subtract the seasonal cycle
        deseasonalized = subset_data - aligned_msc
        return deseasonalized

    def compute_msc_15d_period(self):
        # Assign "day of year" as a coordinate based on the time dimension
        self.data = self.data.assign_coords(dayofyear=self.data["time"].dt.dayofyear)

        # Create 15-day bins by dividing day of year into 15-day intervals
        # (Adjust the range if you want it to exactly match the last days of the year)
        bins = np.arange(1, 367, 15)  # Adjusts for 366 days if leap year included

        # Use `groupby_bins` to group days into 15-day bins, then take the mean over each bin
        self.msc = (
            self.data.groupby_bins("dayofyear", bins=bins, right=False)
            .mean("time", skipna=True)
            .rename({"dayofyear_bins": "dayofyear"})
        )
        # Set the 'dayofyear' to the midpoint of each bin
        self.msc["dayofyear"] = [interval.mid for interval in self.msc.dayofyear.values]

        # Step 3: Extend the data to facilitate circular interpolation
        # To simulate circular continuity, set indices just outside the original range
        # Adjust 'dayofyear' for wrapping data
        prepend = self.msc.isel(dayofyear=-1).assign_coords(
            dayofyear=self.msc.dayofyear[0] - 15
        )
        append = self.msc.isel(dayofyear=0).assign_coords(
            dayofyear=self.msc.dayofyear[-1] + 15
        )

        # Concatenate with adjusted indices for circular continuity
        self.msc_ext = xr.concat([prepend, self.msc, append], dim="dayofyear")

        # Step 4: Perform linear interpolation along the extended 'dayofyear' dimension
        self.msc_ext = self.msc_ext.chunk(dict(dayofyear=-1)).interpolate_na(
            dim="dayofyear", method="linear"
        )

        # Step 5: Remove the extra bins to return to the original dataset shape
        self.msc = self.msc_ext.isel(dayofyear=slice(1, -1))

        self.msc = self.msc.fillna(0)

    def compute_msc(self):
        return self.data.groupby("time.dayofyear").mean("time", skipna=True)

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
        self.filter_dataset_specific()  # useless, legacy...

        self.data = self.data[self.variable_name]
        printt(
            f"Computation on the entire dataset. {self.data.sizes['location']} samples"
        )
        # if self.n_samples:
        # self.compute_msc_15d_period()
        # else:
        self.msc = self.compute_msc()

        if scale:
            self._scale_msc()

        self.msc = self.msc.transpose("location", "dayofyear", ...)

        if return_time_serie:
            self.data = self.data.transpose("location", "time", ...)
            return self.msc, self.data
        else:
            return self.msc
