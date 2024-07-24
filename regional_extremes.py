import xarray as xr
import dask.array as da
from argparse import Namespace
import numpy as np
import json
import random
import datetime
from sklearn.decomposition import PCA
import pandas as pd
import pickle as pk
from pathlib import Path
from typing import Union
import time
import sys
import os


from utils import printt, int_or_none

np.set_printoptions(threshold=sys.maxsize)

from global_land_mask import globe

import argparse  # import ArgumentParser

CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
CURRENT_DIRECTORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIRECTORY_PATH = os.path.abspath(os.path.join(CURRENT_DIRECTORY_PATH, os.pardir))
NORTH_POLE_THRESHOLD = 66.5
SOUTH_POLE_THRESHOLD = -66.5


# Argparser for all configuration needs
def parser_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="id of the experiment is time of the job launch and job_id",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="name_of_the_experiment",
    )

    parser.add_argument(
        "--index",
        type=str,
        default="pei_180",
        help=" The climatic or ecological index to be processed (default: pei_180). "
        "Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'. \n Ecological: 'None.",
    )

    parser.add_argument(
        "--compute_variance",
        type=bool,
        default=False,
        help="compute variance",
    )

    parser.add_argument(
        "--time_resolution",
        type=int,
        default=5,
        help="time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload (default: 5). ",
    )

    parser.add_argument(
        "--n_components", type=int, default=3, help="Number of component of the PCA."
    )

    parser.add_argument(
        "--n_samples",
        type=int_or_none,
        default=100,
        help="Select randomly n_samples**2. Use 'None' for no limit.",
    )

    parser.add_argument(
        "--n_bins",
        type=int,
        default=25,
        help="number of bins to define the regions of similar seasonal cycle.",
    )

    parser.add_argument(
        "--saving_path",
        type=str,
        default=None,
        help="Absolute path to save the experiments 'path/to/experiment'. "
        "If None, the experiment will be save in a folder /experiment in the parent folder.",
    )

    parser.add_argument(
        "--path_load_experiment",
        type=str,
        default=None,
        help="Path of the trained model folder.",
    )
    return parser


class SharedConfig:
    def __init__(self, args: Namespace):
        """
        Initialize SharedConfig with the provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        if args.path_load_experiment is None:
            self.path_load_experiment = None
            printt(
                "Initialisation of a new model, no path provided for an existing model."
            )
            self._initialize_new_experiment(args)
        else:
            self.path_load_experiment = Path(args.path_load_experiment)
            printt(f"Loading of the model path: {self.path_load_experiment}")
            self._load_existing_experiment()

    def _initialize_new_experiment(self, args: Namespace):
        """
        Initialize settings for a new model when no model is loaded.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        self.time_resolution = args.time_resolution
        self.index = args.index
        self.compute_variance = args.compute_variance

        self._set_saving_path(args)
        self._save_args(args)

    def _set_saving_path(self, args: Namespace):
        """
        Set the saving path for the new model.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        # Model launch with the command line. If model launch with sbatch, the id can be define using the id job + date
        if not args.id:
            args.id = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")

        if args.saving_path:
            self.saving_path = Path(args.saving_path) / {args.id} / self.index
        else:
            if args.name:
                self.saving_path = (
                    Path(PARENT_DIRECTORY_PATH)
                    / "experiments/"
                    / f"{args.id}_{args.name}"
                    / self.index
                )
            else:
                self.saving_path = (
                    Path(PARENT_DIRECTORY_PATH) / "experiments/" / args.id / self.index
                )
        printt(f"The saving path is: {self.saving_path}")
        self.saving_path.mkdir(parents=True, exist_ok=True)
        args.saving_path = str(self.saving_path)

    def _save_args(self, args: Namespace):
        """
        Save the arguments to a JSON file for future reference.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        assert self.path_load_experiment is None

        # Saving path
        args_path = self.saving_path / "args.json"

        # Convert to a dictionnary
        args_dict = vars(args)
        del args_dict["path_load_experiment"]

        if not args_path.exists():
            with open(args_path, "w") as f:
                json.dump(args_dict, f, indent=4)
        else:
            raise f"{args_path} already exist."
        printt(f"args saved, path: {args_path}")

    def _load_existing_experiment(self):
        """
        Load an existing model's PCA matrix and min-max data from files.
        """
        # Filter out 'slurm_files' in the path to load experiment to find the index used.
        self.index = [
            folder
            for folder in os.listdir(self.path_load_experiment)
            if folder != "slurm_files"
        ][0]
        self.saving_path = self.path_load_experiment / self.index
        self._load_args()

    def _load_args(self):
        """
        Load args data from the file.
        """
        args_path = self.saving_path / "args.json"
        if args_path.exists():
            with open(args_path, "r") as f:
                args = json.load(f)
                for key, value in args.items():
                    setattr(self, key, value)
            self.saving_path = Path(self.saving_path)
        else:
            raise FileNotFoundError(f"{args_path} does not exist.")


class RegionalExtremes(SharedConfig):
    def __init__(
        self,
        config: SharedConfig,
        n_components: int,
        n_bins: int,
    ):
        """
        Compute the regional extremes by defining boxes of similar region using a PCA computed on the mean seasonal cycle of the samples. Each values of the msc is considered
        as an independent component.

        Args:
            config (SharedConfig): Shared attributes across the classes.
            n_components (int): number of components of the PCA
            n_bins (int): Number of bins per component to define the boxes. Number of boxes = n_bins**n_components
        """
        self.config = config
        self.n_components = n_components
        self.n_bins = n_bins
        if self.config.path_load_experiment:
            self._load_pca_matrix()
        else:
            # Initialize a new PCA.
            self.pca = PCA(n_components=self.n_components)

    def _load_pca_matrix(self):
        """
        Load PCA matrix from the file.
        """
        pca_path = self.config.saving_path / "pca_matrix.pkl"
        with open(pca_path, "rb") as f:
            self.pca = pk.load(f)

    def compute_pca_and_transform(
        self,
        scaled_data,
    ):
        """compute the principal component analysis (PCA) on the mean seasonal cycle (MSC) of n samples and scale it between 0 and 1. Each time step of the msc is considered as an independent component. nb of time_step used for the PCA computation = 366 / time_resolution.

        Args:
            n_components (int, optional): number of components to compute the PCA. Defaults to 3.
            time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        assert not hasattr(
            self.pca, "explained_variance_"
        ), "A pca already have been fit."
        assert self.config.path_load_experiment is None, "A model is already loaded."
        assert scaled_data.dayofyear.shape[0] == round(
            366 / self.config.time_resolution
        )
        assert (self.n_components > 0) & (
            self.n_components <= 366
        ), "n_components have to be in the range of days of a years"
        # Fit the PCA. Each colomns give us the projection through 1 component.
        pca_components = self.pca.fit_transform(scaled_data)

        printt(
            f"PCA performed. sum explained variance: {sum(self.pca.explained_variance_ratio_)}. {self.pca.explained_variance_ratio_})"
        )

        # Save the PCA model
        pca_path = self.config.saving_path / "pca_matrix.pkl"
        with open(pca_path, "wb") as f:
            pk.dump(self.pca, f)
        printt(f"PCA saved: {pca_path}")

        return pca_components

    def apply_pca(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Compute the mean seasonal cycle (MSC) of the samples and scale it between 0 and 1.
        Then apply the PCA already fit on the new data. Each time step of the MSC is considered
        as an independent component. The number of time steps used for the PCA computation
        is 366 / time_resolution.

        Args:
            scaled_data (np.ndarray): Data to be transformed using PCA.

        Returns:
            np.ndarray: Transformed data after applying PCA.
        """
        self._validate_scaled_data(scaled_data)

        transformed_data = xr.apply_ufunc(
            self.pca.transform,
            scaled_data.compute(),
            input_core_dims=[["dayofyear"]],  # Apply PCA along 'dayofyear'
            output_core_dims=[["component"]],  # Resulting dimension is 'component'
        )
        printt("Data are projected in the feature space.")

        self._save_pca_projection(transformed_data)
        return transformed_data

    def _validate_scaled_data(self, scaled_data: np.ndarray) -> None:
        """Validates the scaled data to ensure it matches the expected shape."""
        expected_shape = round(366 / self.config.time_resolution)
        if scaled_data.shape[1] != expected_shape:
            raise ValueError(
                f"scaled_data should have {expected_shape} columns, but has {scaled_data.shape[1]} columns."
            )

    def _save_pca_projection(self, pca_projection) -> None:
        """Saves the limits bins to a file."""
        # Split the components into separate DataArrays
        # Create a new coordinate for the 'component' dimension
        component = np.arange(self.n_components)

        # Create the new DataArray
        pca_projection = xr.DataArray(
            data=pca_projection.values,
            dims=["lonlat", "component"],
            coords={
                "lonlat": pca_projection.lonlat,
                "component": component,
            },
            name="pca",
        )
        # Unstack lonlat for longitude and latitude as dimensions
        pca_projection = pca_projection.set_index(
            lonlat=["longitude", "latitude"]
        ).unstack("lonlat")

        # Explained variance for each component
        explained_variance = xr.DataArray(
            self.pca.explained_variance_ratio_,  # Example values for explained variance
            dims=["component"],
            coords={"component": component},
        )
        pca_projection["explained_variance"] = explained_variance

        # Saving path
        pca_projection_path = self.config.saving_path / "pca_projection.zarr"

        if os.path.exists(pca_projection_path):
            raise FileExistsError(
                f"The file {pca_projection_path} already exists. Rewriting is not allowed."
            )

        # Saving the data
        pca_projection.to_zarr(pca_projection_path)
        printt("Projection saved.")

    def define_limits_bins(self, projected_data: np.ndarray) -> list[np.ndarray]:
        """
        Define the bounds of each bin on the projected data for each component.
        Ideally applied on the largest possible amount of data to capture
        the distribution in the projected space (especially minimum and maximum).
        Fit the PCA with a subset of the data, then project the full dataset,
        then define the bins on the full dataset projected.
        n_bins is per component, so number of boxes = n_bins**n_components

        Args:
            projected_data (np.ndarray): Data projected after PCA.

        Returns:
            list of np.ndarray: List where each array contains the bin limits for each component.
        """
        self._validate_inputs(projected_data)
        limits_bins = self._calculate_limits_bins(projected_data)
        self._save_limits_bins(limits_bins)
        printt("Limits are computed and saved.")
        return limits_bins

    def _validate_inputs(self, projected_data: np.ndarray) -> None:
        """Validates the inputs for define_limits_bins."""
        if not hasattr(self.pca, "explained_variance_"):
            raise ValueError("PCA model has not been trained yet.")

        if projected_data.shape[1] != self.n_components:
            raise ValueError(
                "projected_data should have the same number of columns as n_components"
            )

        if self.n_bins <= 0:
            raise ValueError("n_bins should be greater than 0")

    def _calculate_limits_bins(self, projected_data: np.ndarray) -> list[np.ndarray]:
        """Calculates the limits bins for each component."""
        return [
            np.linspace(
                np.min(projected_data[:, component]),
                np.max(projected_data[:, component]),
                self.n_bins + 1,
            )[
                1:-1
            ]  # Remove first and last limits to avoid attributing new bins to extreme values
            for component in range(self.n_components)
        ]

    def _save_limits_bins(self, limits_bins: list[np.ndarray]) -> None:
        """Saves the limits bins to a file."""
        limits_bins_path = self.config.saving_path / "limits_bins.npy"
        if os.path.exists(limits_bins_path):
            raise FileExistsError(
                f"The file {limits_bins_path} already exists. Rewriting is not allowed."
            )
        np.save(limits_bins_path, limits_bins)

    # Function to find the box for multiple points
    def find_bins(self, projected_data, limits_bins):
        assert projected_data.shape[1] == len(limits_bins)
        assert (
            len(limits_bins) == self.n_components
        ), "the lenght of limits_bins list is not equal to the number of components"
        assert (
            limits_bins[0].shape[0] == self.n_bins - 1
        ), "the limits do not fit the number of bins"

        box_indices = np.zeros(
            (projected_data.shape[0], projected_data.shape[1]), dtype=int
        )
        for i, limits_bin in enumerate(limits_bins):
            box_indices[:, i] = np.digitize(projected_data[:, i], limits_bin)

        return box_indices

    def _save_bins(self, box_indices):
        """Saves the bins to a file."""
        boxes_path = self.config.saving_path / "boxes.ny"
        if os.path.exists(limits_bins_path):
            raise FileExistsError(
                f"The file {limits_bins_path} already exists. Rewriting is not allowed."
            )
        boxes.to_arr(boxes_path)

    def apply_threshold():
        raise NotImplementedError()


class DatasetHandler(SharedConfig):
    def __init__(
        self,
        config: SharedConfig,
        n_samples: Union[int, None],
    ):
        """
        Initialize DatasetHandler.

        Parameters:
        n_samples (Union[int, None]): Number of samples to select.
        load_data (bool): Flag to determine if data should be loaded during initialization.
        time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        self.config = config
        self.n_samples = n_samples

        self.max_data = None
        self.min_data = None
        self.data = None

    def preprocess_data(self):
        """
        Preprocess data based on the index.
        """
        if self.config.index in ["pei_30", "pei_90", "pei_180"]:
            filepath = CLIMATIC_FILEPATH
            self.load_data(filepath)
        else:
            raise NotImplementedError(
                "Index unavailable. Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'. \n Ecological: 'None."
            )

        # Select only a subset of the data if n_samples is specified
        if self.n_samples:
            self.randomly_select_data()
        else:
            printt(
                f"Computation on the entire dataset. {self.data.sizes['latitude'] * self.data.sizes['longitude']} samples"
            )

        self.standardize_climatic_dataset()
        self.compute_and_scale_the_msc()
        return self.data

    def load_data(self, filepath):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        # chunk_sizes = {"time": -1, "latitude": 100, "longitude": 100}
        self.data = xr.open_zarr(filepath)[[self.config.index]]
        printt("Data loaded from {}".format(filepath))

    def randomly_select_data(self):
        """
        Randomly select a subset of the data based on n_samples.
        """
        # select n_samples instead of n_samples**2 but computationnally expensive!
        # self.data = self.data.stack(lonlat=("longitude", "latitude")).transpose(
        #     "lonlat", "time", ...
        # )
        # lonlat_indices = random.choices(self.data.lonlat.values, k=self.n_samples)
        # self.data = self.data.sel(lonlat=lonlat_indices)
        lon_indices = []
        lat_indices = []

        while len(lon_indices) < self.n_samples:
            lon_index = random.randint(0, self.data.longitude.sizes["longitude"] - 1)
            lat_index = random.randint(0, self.data.latitude.sizes["latitude"] - 1)

            lon = self._coordstolongitude(self.data.longitude[lon_index].item())

            lat = self.data.latitude[lat_index].item()
            # if location is on a land and not in the polar regions.
            if (
                globe.is_land(lat, lon)
                and np.abs(lat) <= NORTH_POLE_THRESHOLD
                and self._is_in_europe(lon, lat)
            ):
                lon_indices.append(lon_index)
                lat_indices.append(lat_index)

        self.data = self.data.isel(longitude=lon_indices, latitude=lat_indices)
        printt(
            f"Randomly selected {self.data.sizes['latitude'] * self.data.sizes['longitude']} samples for training in Europe."
        )

    # Transform the longitude coordinates to -180 and 180
    def _coordstolongitude(self, x):
        return ((x + 180) % 360) - 180

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

    def standardize_climatic_dataset(self):
        """
        Apply climatic transformations using xarray.apply_ufunc.
        """
        assert self.config.index in [
            "pei_30",
            "pei_90",
            "pei_180",
        ], "Index unavailable. Index available: 'pei_30', 'pei_90', 'pei_180'."

        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"
        # Ensure longitude values are within the expected range
        assert (
            (self.data.longitude >= 0) & (self.data.longitude <= 360)
        ).all(), "Longitude values should be in the range 0 to 360"

        # Remove the years before 1970 due to quality
        self.data = self.data.sel(
            time=slice(datetime.date(1970, 1, 1), datetime.date(2030, 12, 31))
        )

        # Filter data from the polar regions
        self.data = self.data.where(
            np.abs(self.data.latitude) <= NORTH_POLE_THRESHOLD, drop=True
        )
        self.data = self.data.where(
            np.abs(self.data.latitude) >= SOUTH_POLE_THRESHOLD, drop=True
        )
        # Transform the longitude coordinates
        self.data = self.data.roll(
            longitude=180 * 4, roll_coords=True
        )  # Shifts the data of longitude of 180*4 elements, elements that roll past the end are re-introduced

        # Transform the longitude coordinates to -180 and 180
        self.data = self.data.assign_coords(
            longitude=self._coordstolongitude(self.data.longitude)
        )

        # Filter dataset to select Europe
        # Select European data
        in_europe = self._is_in_europe(self.data.longitude, self.data.latitude)
        self.data = self.data.where(in_europe, drop=True)
        printt("Data filtred to Europe.")

        # Stack the dimensions
        self.data = self.data.stack(lonlat=("longitude", "latitude")).transpose(
            "lonlat", "time", ...
        )

        printt(f"Climatic data loaded with dimensions: {self.data.sizes}")

    def compute_and_scale_the_msc(self):
        """
        compute the MSC of n samples and scale it between 0 and 1.
        Time_resolution reduce the resolution of the msc to reduce the computation workload during the computation. nb values = 366 / time_resolution.
        """
        # Compute the MSC
        self.data["msc"] = (
            self.data[self.config.index]
            .groupby("time.dayofyear")
            .mean("time")
            .drop_vars(["lonlat", "longitude", "latitude"])
        )
        if not self.config.compute_variance:
            # Reduce the temporal resolution
            self.data = self.data["msc"].isel(
                dayofyear=slice(1, 366, self.config.time_resolution)
            )
        else:
            # Compute the variance seasonal cycle
            self.data["vsc"] = (
                self.data[self.config.index]
                .groupby("time.dayofyear")
                .var("time")
                .drop_vars(["lonlat", "longitude", "latitude"])
            )

            # Instead of adding 370, create a new coordinate
            days_in_year = len(self.data["vsc"].dayofyear)
            new_dayofyear = pd.RangeIndex(days_in_year + 1, 2 * days_in_year + 1)

            self.data["vsc"] = self.data["vsc"].assign_coords(dayofyear=new_dayofyear)

            # Concatenate msc and vsc along the dayofyear dimension
            self.data["msc_vsc"] = xr.concat(
                [self.data["msc"], self.data["vsc"]], dim="dayofyear"
            )

            self.data = self.data["msc_vsc"].isel(
                dayofyear=slice(1, 366 + 370, self.config.time_resolution)
            )

            printt("Variance is computed")

        # Compute or load min and max of the data.
        min_max_data_path = self.config.saving_path / "min_max_data.zarr"
        if min_max_data_path.exists():
            self._load_min_max_data(min_max_data_path)
        else:
            self._compute_and_save_min_max_data(min_max_data_path)

        # Scale the data between 0 and 1

        self.data = (self.min_data.broadcast_like(self.data) - self.data) / (
            self.max_data.broadcast_like(self.data)
            - self.min_data.broadcast_like(self.data)
        )
        printt(f"Data are scaled between 0 and 1.")

    def _compute_and_save_min_max_data(self, min_max_data_path):
        assert (
            self.max_data and self.min_data
        ) is None, "the min and max of the data are already defined."
        assert self.config.path_load_experiment is None, "A model is already loaded."
        self.max_data = self.data.max(dim=["lonlat"])
        self.min_data = self.data.min(dim=["lonlat"])
        # Save min_data and max_data
        if not min_max_data_path.exists():
            min_max_data = xr.Dataset(
                {
                    "max_data": self.max_data,
                    "min_data": self.min_data,
                },
                coords={"dayofyear": self.data.coords["dayofyear"].values},
            )
            min_max_data.to_zarr(min_max_data_path)
            printt("Min and max data saved.")
        else:
            raise FileNotFoundError(f"{min_max_data_path} already exist.")

    def _load_min_max_data(self, min_max_data_path):
        """
        Load min-max data from the file.
        """
        min_max_data = xr.open_zarr(min_max_data_path)
        self.min_data = min_max_data.min_data
        self.max_data = min_max_data.max_data


# class EcologicalRegionalExtremes:
#     def __init__(
#         self,
#         vegetation_index,
#         # isclimatic: bool,
#     ):
#         self.vegetation_index = vegetation_index
#         self.filepath = f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{vegetation_index}/Groups_dyn_{vegetation_index}_MSC_snowfrac.zarr"
#         self.filename = f"Groups_dyn_{vegetation_index}_MSC_snowfrac"
#
#     def apply_transformations(self):
#         # Load the MSC of MODIS
#         ds = xr.open_zarr(self.filepath, consolidated=False)
#         ds_msc = ds[self.filename].stack(
#             {"lat": ["latchunk", "latstep_modis"], "lon": ["lonchunk", "lonstep_modis"]}
#         )
#
#         # Select k locations randomly to train the PCA:
#         lat_indices = random.choices(ds_msc.lat.values, k=3000)
#         lon_indices = random.choices(ds_msc.lon.values, k=3000)
#
#         # Select the MSC of those locations:
#         return


def main_train_pca(args):
    config = SharedConfig(args)
    dataset_processor = DatasetHandler(
        config=config,
        n_samples=5,  # args.n_samples,
    )
    data_subset = dataset_processor.preprocess_data()

    extremes_processor = RegionalExtremes(
        config=config,
        n_components=args.n_components,
        n_bins=5,  # args.n_bins,
    )
    projected_data = extremes_processor.compute_pca_and_transform(
        scaled_data=data_subset
    )

    dataset_processor = DatasetHandler(
        config=config,
        n_samples=5,  # None,  # None,  # all the dataset
    )
    data = dataset_processor.preprocess_data()

    projected_data = extremes_processor.apply_pca(scaled_data=data)
    extremes_processor.define_limits_bins(projected_data=projected_data)


def main_define_limits(args):
    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-07-24_11:17:05_Europe2"
    config = SharedConfig(args)

    dataset_processor = DatasetHandler(
        config=config, n_samples=None  # args.n_samples,  # all the dataset
    )
    data = dataset_processor.preprocess_data()

    extremes_processor = RegionalExtremes(
        config=config,
        n_components=args.n_components,
        n_bins=args.n_bins,
    )
    projected_data = extremes_processor.apply_pca(scaled_data=data)
    extremes_processor.define_limits_bins(projected_data=projected_data)


if __name__ == "__main__":
    args = parser_arguments().parse_args()
    args.compute_variance = True
    args.name = "data_normalized_per_day"

    # To train the PCA:
    main_train_pca(args)

    # To define the limits:
    # main_define_limits(args)
