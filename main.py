import xarray as xr

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import pickle as pk
import dask.array as da
import flox.xarray

import sys

from utils import initialize_logger, printt, int_or_none
from loader_and_saver import InMemorySaver, Loader, Saver
from datahandler import create_handler
from config import (
    InitializationConfig,
    CLIMATIC_INDICES,
    ECOLOGICAL_INDICES,
    EARTHNET_INDICES,
)

np.set_printoptions(threshold=sys.maxsize)

from global_land_mask import globe

import argparse

from dask.distributed import Client


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
        help="Compute variance of the seasonal cycle in addition of the mean seasonal cycle (default: False).",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="globe",
        help="Region of the globe to apply the regional extremes."
        "Region available: 'globe', 'europe'.",
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
        help="Select randomly n_samples. Use 'None' for no limit.",
    )

    parser.add_argument(
        "--n_bins",
        type=int,
        default=25,
        help="number of bins to define the regions of similar seasonal cycle. n_bins is proportional. ",
    )

    parser.add_argument(
        "--kernel_pca",
        type=str,
        default=False,
        help="Using a Kernel PCA instead of a PCA.",
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

    parser.add_argument(
        "--method",
        type=str,
        default="regional",
        help="Type of method to compute extremes. Either 'regional' or 'uniform'.",
    )
    return parser


class RegionalExtremes:
    def __init__(
        self,
        config: InitializationConfig,
        n_components: int,
        n_bins: int,
    ):
        """
        Compute the regional extremes by defining boxes of similar region using a PCA computed on the mean seasonal cycle of the samples.
        Each values of the msc is considered as an independent component.

        Args:
            config (InitializationConfig): Shared attributes across the classes.
            n_components (int): number of components of the PCA
            n_bins (int): Number of bins per component to define the boxes. Number of boxes = n_bins**n_components
        """
        self.config = config
        self.n_components = n_components
        self.n_bins = n_bins
        self.loader = Loader(config)
        self.saver = (
            InMemorySaver(config) if config.is_generic_xarray_dataset else Saver(config)
        )

        if self.config.path_load_experiment:
            # Load every variable if already available, otherwise return None.
            self.pca = self.loader._load_pca_matrix()
            self.projected_data = self.loader._load_pca_projection()
            self.limits_bins = self.loader._load_limits_bins()
            self.bins = self.loader._load_bins()
            self.thresholds = self.loader._load_thresholds()

        else:
            # Initialize a new PCA.
            if self.config.k_pca:
                self.pca = KernelPCA(n_components=self.n_components, kernel="rbf")
            else:
                self.pca = PCA(n_components=self.n_components)
            self.projected_data = None
            self.limits_bins = None
            self.bins = None
            self.thresholds = None

    def compute_pca_and_transform(
        self,
        scaled_data,
    ):
        """compute the principal component analysis (PCA) on the mean seasonal cycle (MSC) of n samples scaled between 0 and 1.
        Each time step of the msc is considered as an independent component. nb of time_step used for the PCA computation = 366 / time_resolution (defined in the dataloader).
        """
        assert not hasattr(
            self.pca, "explained_variance_"
        ), "A pca already have been fit."
        assert self.config.path_load_experiment is None, "A model is already loaded."

        assert (self.n_components > 0) & (
            self.n_components <= 366
        ), "n_components have to be in the range of days of a years"
        # Fit the PCA. Each colomns give us the projection through 1 component.
        pca_components = self.pca.fit_transform(scaled_data)

        if isinstance(self.pca, PCA):
            printt(
                f"PCA performed. Sum of explained variance: {sum(self.pca.explained_variance_ratio_)}."
                f"Explained variance ratio: {self.pca.explained_variance_ratio_}."
            )
        elif isinstance(self.pca, KernelPCA):
            printt(
                "KernelPCA performed. Explained variance ratio is not available for KernelPCA."
            )
        else:
            printt("Unknown PCA type.")

        # Save the PCA model
        self.saver._save_pca_model(self.pca)

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
        if self.projected_data is not None:
            raise ValueError(
                "self.projected_data is not None, projected_data already have been computed.",
            )

        self._validate_scaled_data(scaled_data)

        transformed_data = xr.apply_ufunc(
            self.pca.transform,
            scaled_data.compute(),
            input_core_dims=[["dayofyear"]],  # Apply PCA along 'dayofyear'
            output_core_dims=[["component"]],  # Resulting dimension is 'component'
        )
        printt("Data are projected in the feature space.")
        if isinstance(self.pca, PCA):
            self.saver._save_pca_projection(
                transformed_data, self.pca.explained_variance_ratio_
            )
        else:
            self.saver._save_pca_projection(transformed_data, None)
        self.projected_data = transformed_data
        return transformed_data

    def _validate_scaled_data(self, scaled_data: np.ndarray) -> None:
        """Validates the scaled data to ensure it matches the expected shape."""
        if self.config.index in EARTHNET_INDICES:
            expected_shape = 24
        else:
            if self.config.compute_variance:
                expected_shape = round(366 / self.config.time_resolution) * 2 + 1
            else:
                expected_shape = round(366 / self.config.time_resolution)
            if scaled_data.shape[1] != expected_shape:
                raise ValueError(
                    f"scaled_data should have {expected_shape} columns, but has {scaled_data.shape[1]} columns."
                )

    def define_limits_bins(self) -> list[np.ndarray]:
        """
        Define the bounds of each bin on the projected data for each component.
        Ideally applied on the largest possible amount of data to capture
        the distribution in the projected space (especially minimum and maximum).

        Returns:
            list of np.ndarray: List where each array contains the bin limits for each component.
        """
        self._validate_inputs(self.projected_data)

        self.limits_bins = self._calculate_limits_bins(self.projected_data)

        self.saver._save_limits_bins(self.limits_bins)
        printt("Limits are computed and saved.")
        return self.limits_bins

    def _validate_inputs(self, projected_data: np.ndarray) -> None:
        """Validates the inputs for define_limits_bins."""
        if isinstance(self.pca, PCA) and not hasattr(self.pca, "explained_variance_"):
            raise ValueError("PCA model has not been trained yet.")

        if projected_data.shape[1] != self.n_components:
            raise ValueError(
                "projected_data should have the same number of columns as n_components"
            )

        if self.n_bins <= 0:
            raise ValueError("n_bins should be greater than 0")

    def _calculate_limits_bins(self, projected_data: np.ndarray) -> list[np.ndarray]:
        """Calculates the limits bins for each component."""
        if isinstance(self.pca, PCA):
            return [
                np.linspace(
                    np.quantile(projected_data[:, component], 0.05),
                    np.quantile(projected_data[:, component], 0.95),
                    round(self.pca.explained_variance_ratio_[component] * self.n_bins)
                    + 1,
                )
                for component in range(self.n_components)
            ]
        # KPCA. Legacy, to remove?
        else:
            return [
                np.linspace(
                    np.quantile(projected_data[:, component], 0.05),
                    np.quantile(projected_data[:, component], 0.95),
                    self.n_bins + 1,
                )
                for component in range(self.n_components)
            ]

    def find_bins(self):
        """Function to attribute at every location the bin it belong to."""
        assert self.projected_data.shape[1] == len(self.limits_bins)
        assert (
            len(self.limits_bins) == self.n_components
        ), "the lenght of limits_bins list is not equal to the number of components"

        boxes_indices = np.zeros(
            (self.projected_data.shape[0], self.projected_data.shape[1]), dtype=int
        )
        # defines boxes
        for i, limits_bin in enumerate(self.limits_bins):
            # get the indices of the bins to which each value in input array belongs.
            boxes_indices[:, i] = np.digitize(self.projected_data[:, i], limits_bin)

        component = np.arange(boxes_indices.shape[1])

        self.bins = xr.DataArray(
            data=boxes_indices,
            dims=["location", "component"],
            coords={
                "location": self.projected_data.location,
                "component": component,
            },
            name="bins",
        )
        self.saver._save_bins(self.bins)
        return self.bins

    def apply_regional_threshold(self, deseasonalized, quantile_levels):
        """Compute and save a xarray (location, time) indicating the quantiles of extremes using the regional threshold definition."""
        assert self.config.method == "regional"
        LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL = quantile_levels
        quantile_levels = np.concatenate((LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL))

        # Intersection to select only the location where the bin is already attributed
        intersection = np.intersect1d(deseasonalized.location, self.bins.location)
        deseasonalized = deseasonalized.sel(location=intersection)
        self.bins = self.bins.sel(location=intersection)

        # Create a new DataArrays to store the quantile values (0.025 or 0.975) for extreme values
        extremes_array = xr.full_like(deseasonalized.astype(float), np.nan)

        # Create a new DataArray to store the threshold related to each quantiles.
        thresholds_array = xr.DataArray(
            data=np.full((len(deseasonalized.location), len(quantile_levels)), np.nan),
            dims=["location", "quantile"],
            coords={
                "location": deseasonalized.location,
                "quantile": quantile_levels,
            },
        )

        # Get unique id for each region
        unique_regions, counts = np.unique(self.bins.values, axis=0, return_counts=True)

        # Create a DataArray of region labels
        region_labels = xr.DataArray(
            np.argmax(
                np.all(
                    self.bins.values[:, :, None] == unique_regions.T[None, :, :], axis=1
                ),
                axis=1,
            ),
            dims=("location",),
            coords={"location": self.bins.location},
        )

        compute_only_thresholds = self.config.is_generic_xarray_dataset

        # Group the deseasonalized data by region labels
        grouped = deseasonalized.groupby(region_labels)
        # Apply the quantile calculation to each group
        results = grouped.map(
            lambda grp: self._compute_thresholds(
                grp,
                (LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL),
                return_only_thresholds=compute_only_thresholds,
            )
        )
        # Assign the results back to the quantile_array
        thresholds_array.values = results["thresholds"].values
        if not compute_only_thresholds:
            extremes_array.values = results["extremes"].values

        # save the array
        self.saver._save_thresholds(thresholds_array)
        if not compute_only_thresholds:
            self.saver._save_extremes(extremes_array)

    def apply_local_threshold(self, deseasonalized, quantile_levels):
        assert self.config.method == "local"
        """Compute and save a xarray (location, time) indicating the quantiles of extremes using a uniform threshold definition."""
        LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL = quantile_levels

        # Create a new DataArray to store the quantile values (0.025 or 0.975) for extreme values
        extremes_array = xr.full_like(deseasonalized.astype(float), np.nan)

        # Create a new DataArray to store the threshold related to each quantiles.
        quantile_levels = np.concatenate((LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL))
        thresholds_array = xr.DataArray(
            data=np.full((len(deseasonalized.location), len(quantile_levels)), np.nan),
            dims=["location", "quantile"],
            coords={
                "location": deseasonalized.location,
                "quantile": quantile_levels,
            },
        )

        compute_only_thresholds = self.config.is_generic_xarray_dataset

        # Apply the quantile calculation to each location
        results = self._compute_thresholds(
            deseasonalized=deseasonalized,
            quantile_levels=(LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL),
            method="local",
            return_only_thresholds=compute_only_thresholds,
        )
        printt("results computed")

        # Assign the results back to the quantile_array
        thresholds_array.values = results["thresholds"].values.T
        if not compute_only_thresholds:
            extremes_array.values = results["extremes"].values

        # save the array
        printt("Saving in progress")
        self.saver._save_thresholds(thresholds_array)
        if not compute_only_thresholds:
            self.saver._save_extremes(extremes_array)

    def _compute_thresholds(
        self,
        deseasonalized: xr.DataArray,
        quantile_levels,
        method="regional",
        return_only_thresholds=False,
    ):
        """
        Assign quantile levels to deseasonalized data.

        Args:
            deseasonalized (xarray.DataArray): Deseasonalized data.
            method (str): Method for computing quantiles. Either "regional" or "local".

        Returns:
            xarray.DataArray: Data with assigned quantile levels.
        """
        LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL = quantile_levels
        quantile_levels = np.concatenate((LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL))
        deseasonalized = deseasonalized.chunk("auto")

        if method == "regional":
            deseasonalized_dask = deseasonalized.data
            quantiles = da.percentile(
                deseasonalized_dask.flatten(),
                np.array([*LOWER_QUANTILES_LEVEL, *UPPER_QUANTILES_LEVEL]) * 100,
                method="linear",
            )
            all_levels = [*LOWER_QUANTILES_LEVEL, *UPPER_QUANTILES_LEVEL]
            quantiles_xr = xr.DataArray(
                quantiles, coords={"quantile": all_levels}, dims=["quantile"]
            )
        elif method == "local":
            quantiles_xr = deseasonalized.quantile(
                np.array([*LOWER_QUANTILES_LEVEL, *UPPER_QUANTILES_LEVEL]), dim=["time"]
            )
        else:
            raise NotImplementedError("Global threshold method is not yet implemented.")
        # print(dim)
        # lower_quantiles = deseasonalized.quantile(LOWER_QUANTILES_LEVEL)
        # upper_quantiles = deseasonalized.quantile(UPPER_QUANTILES_LEVEL)
        # all_quantiles = xr.concat([lower_quantiles, upper_quantiles], dim="quantile")

        masks = self._create_quantile_masks(deseasonalized, quantiles_xr)

        extremes = xr.full_like(deseasonalized.astype(float), np.nan)
        for i, mask in enumerate(masks):
            extremes = xr.where(mask, quantile_levels[i], extremes)

        results = xr.Dataset({"extremes": extremes, "thresholds": quantiles_xr})
        return results

    def _create_quantile_masks(self, data, quantiles):
        """
        Create masks for each quantile level.

        Args:
            data (xarray.DataArray): Input data.
            lower_quantiles (xarray.DataArray): Lower quantiles.
            upper_quantiles (xarray.DataArray): Upper quantiles.

        Returns:
            list: List of boolean masks for each quantile level.
        """
        lower_quantiles = quantiles.sel(quantile=LOWER_QUANTILES_LEVEL)
        upper_quantiles = quantiles.sel(quantile=UPPER_QUANTILES_LEVEL)
        masks = [
            data < lower_quantiles[0],
            *[
                (data >= lower_quantiles[i - 1]) & (data < lower_quantiles[i])
                for i in range(1, len(LOWER_QUANTILES_LEVEL))
            ],
            *[
                (data > upper_quantiles[i - 1]) & (data <= upper_quantiles[i])
                for i in range(1, len(UPPER_QUANTILES_LEVEL))
            ],
            data > upper_quantiles[-1],
        ]
        return masks


def regional_extremes_method(args, quantile_levels):
    """Fit the PCA with a subset of the data, then project the full dataset,
    then define the bins on the full dataset projected."""
    # Initialization of the configs, load and save paths, log.txt.
    config = InitializationConfig(args)
    assert config.method == "regional"
    # Initialization of RegionalExtremes, load data if already computed.
    extremes_processor = RegionalExtremes(
        config=config,
        n_components=config.n_components,
        n_bins=config.n_bins,
    )

    # Load a subset of the dataset and fit the PCA
    if not hasattr(extremes_processor.pca, "explained_variance_"):
        # Initialization of the climatic or ecological DatasetHandler
        dataset = create_handler(
            config=config,
            n_samples=config.n_samples,  # args.n_samples,  # all the dataset
        )
        # Load and preprocess the dataset
        data_subset = dataset.preprocess_data()
        # Fit the PCA on the data
        extremes_processor.compute_pca_and_transform(scaled_data=data_subset)

    # Apply the PCA to the entire dataset
    if extremes_processor.projected_data is None:
        dataset_processor = create_handler(
            config=config, n_samples=config.n_samples  # * 10  # None
        )  # all the dataset
        data = dataset_processor.preprocess_data(remove_nan=True)
        extremes_processor.apply_pca(scaled_data=data)

    # Define the boundaries of the bins
    if extremes_processor.limits_bins is None:
        extremes_processor.define_limits_bins()

    # Attribute a bins to each location
    if extremes_processor.bins is None:
        extremes_processor.find_bins()

    # Apply the regional threshold and compute the extremes
    # Load the data
    dataset_processor = create_handler(
        config=config, n_samples=100  # config.n_samples  # * 10
    )  # None)  # None)
    msc, data = dataset_processor.preprocess_data(
        scale=False,
        return_time_serie=True,
        reduce_temporal_resolution=False,
        remove_nan=False,
    )
    # Deseasonalize the data
    deseasonalized = dataset_processor._deseasonalize(data, msc)
    # Compute the quantiles per regions/biome (=bins)
    extremes_processor.apply_regional_threshold(
        deseasonalized, quantile_levels=quantile_levels
    )

    return extremes_processor


def local_extremes_method(args, quantile_levels):
    # Initialization of the configs, load and save paths, log.txt.
    config = InitializationConfig(args)
    assert config.method == "local"

    # Initialization of RegionalExtremes, load data if already computed.
    extremes_processor = RegionalExtremes(
        config=config,
        n_components=config.n_components,
        n_bins=config.n_bins,
    )

    dataset_processor = create_handler(config=config, n_samples=None)  # all the dataset

    msc, data = dataset_processor.preprocess_data(
        scale=False,
        return_time_serie=True,
        reduce_temporal_resolution=False,
        remove_nan=False,
    )
    # Deseasonalized data
    deseasonalized = dataset_processor._deseasonalize(data, msc)
    # Apply the local threshold
    extremes_processor.apply_local_threshold(deseasonalized, quantile_levels)

    return extremes_processor


def get_thresholds_from_generic_xarray_dataset(
    data: xr.Dataset, quantile_levels: tuple[float, float], method: str
) -> xr.Dataset:
    args = {
        "is_generic_xarray_dataset": True,
        "in_memory_results": True,
        "data": data,
        "index": None,
        "name": "test",
        "k_pca": False,
        "n_samples": 5,
        "n_components": 2,
        "n_bins": 50,
        "compute_variance": False,
        "path_load_experiment": None,
    }
    if method == "regional":
        p = regional_extremes_method(args, quantile_levels)
        return p.saver.thresholds
    elif method == "local":
        local_extremes_method(args, quantile_levels)
        return p.saver.thresholds


if __name__ == "__main__":
    args = parser_arguments().parse_args()
    args.name = "deep_extreme_HR"
    args.index = "EVI_EN"
    args.k_pca = False
    args.n_samples = 3000
    args.n_components = 3
    args.n_bins = 50
    args.compute_variance = False
    args.method = "regional"
    args.start_year = 2000
    args.is_generic_xarray_dataset = False

    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-11-11_13:11:58_deep_extreme_HR"

    LOWER_QUANTILES_LEVEL = np.array([0.01, 0.025, 0.05])
    UPPER_QUANTILES_LEVEL = np.array([0.95, 0.975, 0.99])

    if args.method == "regional":
        # Apply the regional extremes method
        regional_extremes_method(args, (LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL))
    elif args.method == "local":
        # Apply the uniform threshold method
        local_extremes_method(args, (LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL))
    elif args.method == "global":
        raise NotImplementedError("the global method is not yet implemented.")
