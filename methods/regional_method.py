import xarray as xr

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from RegionalExtremesPackage.utils.logger import printt
from RegionalExtremesPackage.utils import Loader, Saver
from RegionalExtremesPackage.datahandlers import create_handler
from RegionalExtremesPackage.utils.config import (
    InitializationConfig,
    CLIMATIC_INDICES,
    ECOLOGICAL_INDICES,
    EARTHNET_INDICES,
)

np.set_printoptions(threshold=sys.maxsize)


class RegionalExtremes:
    def __init__(
        self,
        config: InitializationConfig,
        loader: Loader,
        saver: Saver,
    ):
        """
        Compute the regional extremes by defining boxes of similar region using a PCA computed on the mean seasonal cycle of the samples.
        Each values of the msc is considered as an independent component.

        Args:
            config (InitializationConfig): Shared attributes across the classes.
            n_components (int): number of components of the PCA
            n_eco_clusters (int): Number of eco_clusters per component to define the boxes. Number of boxes = n_eco_clusters**n_components
        """
        self.config = config
        self.n_components = self.config.n_components
        self.n_eco_clusters = self.config.n_eco_clusters
        self.lower_quantiles = self.config.lower_quantiles
        self.upper_quantiles = self.config.upper_quantiles
        # Loader class to load intermediate steps.
        self.loader = Loader(config)
        # Saver class to save intermediate steps.
        self.saver = Saver(config)

        if self.config.path_load_experiment:
            # Load every variable if already available, otherwise return None.
            self.pca = self.loader._load_pca_matrix()
            self.projected_data = self.loader._load_pca_projection()
            self.limits_eco_clusters = self.loader._load_limits_eco_clusters()
            self.eco_clusters = self.loader._load_data("eco_clusters")
            self.thresholds = self.loader._load_data(
                "thresholds", location=False, cluster=True
            )

        else:
            # Initialize a new PCA.
            if self.config.k_pca:
                self.pca = KernelPCA(n_components=self.n_components, kernel="rbf")
            else:
                self.pca = PCA(n_components=self.n_components)
            self.projected_data = None
            self.limits_eco_clusters = None
            self.eco_clusters = None
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
        # assert self.config.path_load_experiment is None, "A model is already loaded."

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

    def define_limits_eco_clusters(self) -> list[np.ndarray]:
        """
        Define the bounds of each bin on the projected data for each component.
        Ideally applied on the largest possible amount of data to capture
        the distribution in the projected space (especially minimum and maximum).

        Returns:
            list of np.ndarray: List where each array contains the bin limits for each component.
        """
        self._validate_inputs(self.projected_data)

        self.limits_eco_clusters = self._calculate_limits_eco_clusters(
            self.projected_data
        )

        self.saver._save_limits_eco_clusters(self.limits_eco_clusters)
        printt("Limits are computed and saved.")
        return self.limits_eco_clusters

    def _validate_inputs(self, projected_data: np.ndarray) -> None:
        """Validates the inputs for define_limits_eco_clusters."""
        if isinstance(self.pca, PCA) and not hasattr(self.pca, "explained_variance_"):
            raise ValueError("PCA model has not been trained yet.")

        if projected_data.shape[1] != self.n_components:
            raise ValueError(
                "projected_data should have the same number of columns as n_components"
            )

        if self.n_eco_clusters <= 0:
            raise ValueError("n_eco_clusters should be greater than 0")

    def _calculate_limits_eco_clusters(
        self, projected_data: np.ndarray
    ) -> list[np.ndarray]:
        """Calculates the limits eco_clusters for each component."""
        if isinstance(self.pca, PCA):
            return [
                np.linspace(
                    np.quantile(projected_data[:, component], 0.05),
                    np.quantile(projected_data[:, component], 0.95),
                    round(
                        self.pca.explained_variance_ratio_[component]
                        * self.n_eco_clusters
                    )
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
                    self.n_eco_clusters + 1,
                )
                for component in range(self.n_components)
            ]

    def find_eco_clusters(self):
        """Function to attribute at every location the bin it belong to."""
        assert self.projected_data.shape[1] == len(self.limits_eco_clusters)
        assert (
            len(self.limits_eco_clusters) == self.n_components
        ), "the lenght of limits_eco_clusters list is not equal to the number of components"

        boxes_indices = np.zeros(
            (self.projected_data.shape[0], self.projected_data.shape[1]), dtype=int
        )
        # defines boxes
        for i, limits_bin in enumerate(self.limits_eco_clusters):
            # get the indices of the eco_clusters to which each value in input array belongs.
            boxes_indices[:, i] = np.digitize(self.projected_data[:, i], limits_bin)

        component = np.arange(boxes_indices.shape[1])

        self.eco_clusters = xr.DataArray(
            data=boxes_indices,
            dims=["location", "component"],
            coords={
                "location": self.projected_data.location,
                "component": component,
            },
            name="eco_clusters",
        )
        self.saver._save_data(self.eco_clusters, "eco_clusters")
        return self.eco_clusters

    def apply_regional_threshold(self, deseasonalized):
        # Load thresholds
        """Compute and save a xarray (location, time) indicating the quantiles of extremes using the regional threshold definition."""
        assert self.config.method == "regional"
        quantile_levels = np.concatenate((self.lower_quantiles, self.upper_quantiles))

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
        eco_cluster_labels = np.array(
            ["_".join(map(str, cluster)) for cluster in self.eco_clusters.values]
        )

        # Assign the eco_cluster labels as a coordinate
        deseasonalized = deseasonalized.assign_coords(
            eco_cluster=("location", eco_cluster_labels)
        )

        # Group data by the combined eco_cluster labels
        grouped = deseasonalized.groupby("eco_cluster")

        compute_only_thresholds = self.config.is_generic_xarray_dataset

        # Define the function to map thresholds to clusters
        def map_thresholds_to_clusters(grp):
            """
            Maps thresholds to clusters based on the string-based eco-cluster label.

            Parameters:
                grp: xarray DataArray group for a specific eco-cluster.
                eco_cluster_labels: Array of eco-cluster labels (string representation).
                thresholds: xarray DataArray indexed by the components of the eco-cluster.

            Returns:
                Thresholds corresponding to the current eco-cluster group.
            """
            # Get the string label for the current group
            eco_cluster_label = grp.eco_cluster.values[0]

            # Parse the label back into its components
            comp_values = list(map(int, eco_cluster_label.split("_")))
            print(comp_values)
            comp_values = [0, 10, 3]
            coords = self.thresholds.cluster.to_index()
            # [(0, 10, 3) (16, 4, 0) (30, 19, 2) (31, 0, 4)]

            # Check if the specific combination exists
            if (comp_values[0], comp_values[1], comp_values[2]) in coords:
                thresholds_grp = self.thresholds.sel(
                    cluster={
                        "component_1": comp_values[0],
                        "component_2": comp_values[1],
                        "component_3": comp_values[2],
                    }
                )
            else:
                thresholds_grp = np.nan
            return thresholds_grp

        print(self.thresholds.cluster.values)
        # Apply the quantile calculation to each group
        results = grouped.map(
            lambda grp: self._apply_thresholds(
                grp,
                map_thresholds_to_clusters(grp),
                (self.lower_quantiles, self.upper_quantiles),
                # return_only_thresholds=compute_only_thresholds,
            )
        )

        # Assign the results back to the quantile_array
        thresholds_array.values = results["thresholds"].values
        if not compute_only_thresholds:
            extremes_array.values = results["extremes"].values

        # save the array
        self.saver._save_data(thresholds_array, "thresholds")
        self.saver._save_data(thresholds_array, "thresholds_eco_clusters")
        if not compute_only_thresholds:
            self.saver._save_data(extremes_array, "extremes")

    def compute_regional_threshold(self, deseasonalized, quantile_levels):
        """
        Compute and save an xarray indicating the quantiles of extremes using the regional threshold definition.

        Parameters:
        - deseasonalized: xarray.DataArray
            The deseasonalized dataset.
        - quantile_levels: tuple
            A tuple containing lower and upper quantile levels (e.g., (0.025, 0.975)).
        """
        assert self.config.method == "regional"

        # Unpack and concatenate quantile levels
        lower_quantiles, upper_quantiles = quantile_levels
        quantile_levels = np.concatenate((lower_quantiles, upper_quantiles))

        # Initialize data arrays for results
        extremes_array = xr.full_like(deseasonalized, np.nan, dtype=float)
        thresholds_array = xr.DataArray(
            np.full((len(deseasonalized.location), len(quantile_levels)), np.nan),
            dims=["location", "quantile"],
            coords={
                "location": deseasonalized.location,
                "quantile": quantile_levels,
            },
        )

        # Generate unique regional cluster IDs
        unique_clusters, _ = np.unique(
            self.eco_clusters.values, axis=0, return_counts=True
        )
        eco_cluster_labels = xr.DataArray(
            data=np.argmax(
                np.all(
                    self.eco_clusters.values[:, :, None]
                    == unique_clusters.T[None, :, :],
                    axis=1,
                ),
                axis=1,
            ),
            dims=("location",),
            coords={"location": self.eco_clusters.location},
        )

        # Group deseasonalized data by cluster labels
        grouped_data = deseasonalized.groupby(eco_cluster_labels)

        # Compute thresholds and extremes for each group
        compute_only_thresholds = self.config.is_generic_xarray_dataset
        results = grouped_data.map(
            lambda grp: self._compute_thresholds(
                grp,
                (lower_quantiles, upper_quantiles),
                return_only_thresholds=compute_only_thresholds,
            )
        )

        # Calculate mean thresholds for each cluster
        group_thresholds = results["thresholds"].groupby(eco_cluster_labels).mean()

        multi_index = pd.MultiIndex.from_arrays(
            unique_clusters.T, names=["component_1", "component_2", "component_3"]
        )

        thresholds_by_cluster = xr.DataArray(
            data=group_thresholds.values,
            dims=("cluster", "quantile"),
            coords={
                "cluster": multi_index,
                "quantile": quantile_levels,
            },
            name="threshold",
        )

        # Save thresholds by cluster
        self.saver._save_data(
            thresholds_by_cluster, "thresholds", location=False, cluster=True
        )

        # Assign and save location-specific results if thresholds are not computed only
        if not compute_only_thresholds:
            thresholds_array.values = results["thresholds"].values
            self.saver._save_data(thresholds_array, "thresholds_locations")

            extremes_array.values = results["extremes"].values
            self.saver._save_data(extremes_array, "extremes")

    def _compute_thresholds(
        self,
        deseasonalized: xr.DataArray,
        quantile_levels,
        return_only_thresholds=False,
    ):
        """
        Assign quantile levels to deseasonalized data.

        Args:
            deseasonalized (xarray.DataArray): Deseasonalized data.
            quantile_levels (tuple): Tuple of lower and upper quantile levels.
            method (str): Method for computing quantiles. Either "regional" or "local".
            return_only_thresholds (bool): If True, only return quantile thresholds.

        Returns:
            xarray.Dataset: Dataset containing extremes and thresholds.
        """
        assert self.config.method == "regional"
        # Unpack quantile levels and prepare data
        lower_quantiles, upper_quantiles = quantile_levels
        quantile_levels_combined = np.concatenate((lower_quantiles, upper_quantiles))
        deseasonalized = deseasonalized.chunk("auto")

        # Compute quantiles based on the method
        quantiles_xr = self._compute_regional_quantiles(
            deseasonalized.data, quantile_levels_combined
        )

        # Return thresholds only if requested
        if return_only_thresholds:
            return xr.Dataset({"thresholds": quantiles_xr})

        extremes = self._apply_thresholds(deseasonalized, quantiles_xr, quantile_levels)
        results = xr.Dataset({"extremes": extremes, "thresholds": quantiles_xr})
        return results

    def _compute_regional_quantiles(self, data, quantile_levels):
        """
        Compute regional quantiles for the provided data.
        Args:
            data (dask.array.Array): Flattened data to compute quantiles for.
            quantile_levels (np.ndarray): Combined lower and upper quantile levels.
        Returns:
            xarray.DataArray: Regional quantiles as an xarray.DataArray.
        """
        import dask.array as da

        # Mask NaN values and flatten the data
        flattened_nonan = data.flatten()[~da.isnan(data.flatten())]
        # Compute quantiles using Dask
        quantiles = da.percentile(
            flattened_nonan,
            quantile_levels * 100,  # Convert to percentages
            method="linear",
        )
        # Wrap quantiles in an xarray.DataArray
        quantiles_xr = xr.DataArray(
            quantiles,
            coords={"quantile": quantile_levels},
            dims=["quantile"],
        )
        return quantiles_xr

    def _apply_thresholds(
        self,
        deseasonalized: xr.DataArray,
        thresholds,
        quantile_levels,
    ):
        extremes = xr.full_like(deseasonalized.astype(float), np.nan)
        if thresholds is np.nan:
            return extremes
        quantile_levels_combined = np.concatenate(
            (quantile_levels[0], quantile_levels[1])
        )
        masks = self._create_quantile_masks(
            deseasonalized,
            thresholds,
            quantile_levels=quantile_levels,
        )

        for i, mask in enumerate(masks):
            extremes = xr.where(mask, quantile_levels_combined[i], extremes)
        return extremes

    def _create_quantile_masks(self, data, quantiles, quantile_levels):
        """
        Create masks for each quantile level.

        Args:
            data (xarray.DataArray): Input data.
            lower_quantiles (xarray.DataArray): Lower quantiles.
            upper_quantiles (xarray.DataArray): Upper quantiles.

        Returns:
            list: List of boolean masks for each quantile level.
        """
        LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL = quantile_levels
        # Match the coordinate type if needed
        print(quantiles["quantile"])
        # if quantiles["quantile"].dtype != LOWER_QUANTILES_LEVEL.dtype:
        #    LOWER_QUANTILES_LEVEL = LOWER_QUANTILES_LEVEL.astype(
        #        quantiles["quantile"].dtype
        #    )
        print(quantiles)
        lower_quantiles = quantiles.sel(quantile=LOWER_QUANTILES_LEVEL)
        upper_quantiles = quantiles.sel(quantile=UPPER_QUANTILES_LEVEL)
        print(lower_quantiles)
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


def regional_extremes_method(args):
    """Fit the PCA with a subset of the data, then project the full dataset,
    then define the eco_clusters on the full dataset projected."""
    # Initialization of the configs, load and save paths, log.txt.
    config = InitializationConfig(args)
    # Loader class to load intermediate steps.
    loader = Loader(config)
    # Saver class to save intermediate steps.
    saver = Saver(config)

    assert config.method == "regional"
    # Initialization of RegionalExtremes, load data if already computed.
    extremes_processor = RegionalExtremes(
        config=config,
        loader=loader,
        saver=saver,
    )

    # Load a subset of the dataset and fit the PCA
    if not hasattr(extremes_processor.pca, "explained_variance_"):
        # Initialization of the climatic or ecological DatasetHandler
        dataset = create_handler(
            config=config,
            loader=loader,
            saver=saver,
            n_samples=config.n_samples,  # args.n_samples,  # all the dataset
        )
        # Load and preprocess the dataset
        data_subset = dataset.preprocess_data()
        # Fit the PCA on the data
        extremes_processor.compute_pca_and_transform(scaled_data=data_subset)

    # Apply the PCA to the entire dataset
    # if extremes_processor.projected_data is None:

    # Define the boundaries of the eco_clusters
    if extremes_processor.limits_eco_clusters is None:
        dataset_processor = create_handler(
            config=config,
            loader=loader,
            saver=saver,
            n_samples=1000,  # config.n_samples  # None
        )  # all the dataset
        data = dataset_processor.preprocess_data(remove_nan=True)
        extremes_processor.apply_pca(scaled_data=data)
        extremes_processor.define_limits_eco_clusters()

    # Apply the regional threshold and compute the extremes
    # Load the data
    dataset_processor = create_handler(
        config=config, loader=loader, saver=saver, n_samples=config.n_samples
    )
    msc, data = dataset_processor.preprocess_data(
        scale=False,
        return_time_serie=True,
        reduce_temporal_resolution=False,
        remove_nan=False,
    )
    extremes_processor.apply_pca(scaled_data=msc)
    extremes_processor.find_eco_clusters()
    # Deseasonalize the data
    deseasonalized = dataset_processor._deseasonalize(data, msc)
    # Compute the quantiles per regions/biome (=eco_clusters)
    extremes_processor.compute_regional_threshold(deseasonalized)

    return extremes_processor


def regional_extremes_minicube(args, minicube_path):
    """Fit the PCA with a subset of the data, then project the full dataset,
    then define the eco_clusters on the full dataset projected."""
    # Initialization of the configs, load and save paths, log.txt.
    config = InitializationConfig(args)
    assert config.method == "regional"

    # Loader class to load intermediate steps.
    loader = Loader(config)
    # Saver class to save intermediate steps.
    saver = Saver(config)

    # Initialization of RegionalExtremes, load data if already computed.
    extremes_processor = RegionalExtremes(
        config=config,
        loader=loader,
        saver=saver,
    )
    if extremes_processor.limits_eco_clusters is None:
        raise FileNotFoundError("limits_eco_clusters file unavailable.")
    if extremes_processor.thresholds is None:
        raise FileNotFoundError("thresholds file unavailable.")

    # Apply the regional threshold and compute the extremes
    # Load the data
    dataset_processor = create_handler(
        config=config, loader=loader, saver=saver, n_samples=1  # config.n_samples
    )
    msc, data = dataset_processor.preprocess_data(
        scale=False,
        return_time_serie=True,
        reduce_temporal_resolution=False,
        remove_nan=False,
        minicube_path=minicube_path,
    )
    extremes_processor.apply_pca(scaled_data=msc)
    extremes_processor.find_eco_clusters()
    # Deseasonalize the data
    deseasonalized = dataset_processor._deseasonalize(data, msc)
    # Compute the quantiles per regions/biome (=eco_clusters)
    extremes_processor.apply_regional_threshold(deseasonalized)
