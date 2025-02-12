import xarray as xr

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
import sys
from pathlib import Path
from scipy.spatial.distance import pdist, squareform


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
        """
        Compute and save a xarray indicating the quantiles of extremes using the regional threshold definition.

        Args:
            deseasonalized (xarray.DataArray): Input data that has been deseasonalized

        Returns:
            tuple: (thresholds, extremes_array) if not compute_only_thresholds, else thresholds

        Raises:
            AssertionError: If config method is not "regional"
            ValueError: If eco_clusters or thresholds are not properly configured
        """
        # Validate inputs
        if not isinstance(deseasonalized, xr.DataArray):
            raise TypeError("deseasonalized must be an xarray DataArray")

        assert self.config.method == "regional", "Method must be regional"

        # Initialize parameters
        compute_only_thresholds = self.config.is_generic_xarray_dataset

        def create_eco_cluster_labels():
            """Create standardized eco-cluster labels."""
            if not hasattr(self, "eco_clusters"):  # or not self.eco_clusters:
                raise ValueError("eco_clusters not properly initialized")
            return np.array(
                ["_".join(map(str, cluster)) for cluster in self.eco_clusters.values]
            )

        def prepare_deseasonalized_data(data, labels):
            """Prepare deseasonalized data with eco_cluster coordinates."""
            return data.assign_coords(eco_cluster=("location", labels))

        # Define the function to map thresholds to clusters
        def map_thresholds_to_clusters(grp):
            """
            Maps thresholds to clusters based on the eco-cluster label.

            Args:
                grp: xarray DataArray group for a specific eco-cluster

            Returns:
                xarray.DataArray: Thresholds for the current eco-cluster
            """
            # Get the string label for the current group
            eco_cluster_label = grp.eco_cluster.values[0]

            # Parse the label back into its components
            comp_values = list(map(int, eco_cluster_label.split("_")))
            # comp_values = [0, 10, 3]
            coords = self.thresholds.eco_cluster.to_index()
            # [(0, 10, 3) (16, 4, 0) (30, 19, 2) (31, 0, 4)]

            # Check if the specific combination exists
            cluster_key = tuple(comp_values)
            if cluster_key in coords:
                return self.thresholds.sel(
                    eco_cluster={
                        "component_1": comp_values[0],
                        "component_2": comp_values[1],
                        "component_3": comp_values[2],
                    }
                ).thresholds
            # Create a NaN array with the same structure as self.thresholds
            nan_array = xr.full_like(
                self.thresholds.isel(eco_cluster=0),  # Use first cluster as template
                fill_value=np.nan,
            )

            # Ensure the nan_array has the same coordinates as the original selection would have
            nan_array = nan_array.assign_coords(
                {
                    "component_1": comp_values[0],
                    "component_2": comp_values[1],
                    "component_3": comp_values[2],
                }
            )

            return nan_array.thresholds

        # Initialize output array
        extremes_array = xr.full_like(deseasonalized.astype(float), np.nan)

        # Process data
        eco_cluster_labels = create_eco_cluster_labels()
        deseasonalized = prepare_deseasonalized_data(deseasonalized, eco_cluster_labels)
        grouped = deseasonalized.groupby("eco_cluster")

        # Calculate thresholds
        thresholds = grouped.map(lambda grp: map_thresholds_to_clusters(grp))
        thresholds = thresholds.sel(eco_cluster=deseasonalized["eco_cluster"])
        thresholds = thresholds.drop_vars(["eco_cluster"])
        # Save thresholds
        self.saver._save_data(thresholds, "thresholds")

        # Calculate and save extremes if needed
        if not compute_only_thresholds:
            extremes = grouped.map(
                lambda grp: self._apply_thresholds(
                    grp,
                    map_thresholds_to_clusters(grp),
                    (self.lower_quantiles, self.upper_quantiles),
                )
            )
            extremes_array.values = extremes.values
            self.saver._save_data(extremes_array, "extremes")
            return thresholds, extremes_array

        return thresholds

    def compute_regional_threshold(self, deseasonalized):
        """
        Compute and save regional thresholds for extremes.

        Parameters:
        -----------
        deseasonalized : xarray.DataArray
            The deseasonalized dataset
        """
        assert self.config.method == "regional"

        def _process_group(grp):
            """
            Process a single group by filtering close locations and validating size.

            Parameters:
            -----------
            grp : xarray.DataArray
                Group to process

            Returns:
            --------
            xarray.DataArray
                Processed group with validation flag
            """
            # Filter close locations
            filtered_grp = _filter_close_locations(grp)

            # Add validation flag for minimum group size
            is_valid = filtered_grp.location.size >= 5
            valid_coord = xr.full_like(filtered_grp.location, is_valid, dtype=bool)
            return filtered_grp.assign_coords(valid_group=valid_coord)

        def _filter_close_locations(grp, min_distance=0.03):
            """
            Filter out locations that are too close to each other.

            Parameters:
            -----------
            grp : xarray.DataArray
                Group of locations to filter
            min_distance : float
                Minimum allowed distance between locations

            Returns:
            --------
            xarray.DataArray
                Filtered group with distant locations
            """
            locs = np.vstack([grp.longitude.values, grp.latitude.values]).T
            distances = squareform(pdist(locs))

            # Create distance matrix and mask close locations
            distances[np.tril_indices_from(distances)] = np.inf
            too_close_rows, _ = np.where(distances < min_distance)
            too_close_locations = np.unique(too_close_rows)

            # Remove close locations
            return grp.isel(
                location=np.setdiff1d(np.arange(len(grp.location)), too_close_locations)
            )

        def _create_cluster_labels(eco_clusters):
            """Create cluster labels for grouping."""
            unique_clusters, _ = np.unique(
                eco_clusters.values, axis=0, return_counts=True
            )
            labels = xr.DataArray(
                data=np.argmax(
                    np.all(
                        eco_clusters.values[:, :, None]
                        == unique_clusters.T[None, :, :],
                        axis=1,
                    ),
                    axis=1,
                ),
                dims=("location",),
                coords={"location": eco_clusters.location},
            )
            return unique_clusters, labels

        compute_only_thresholds = self.config.is_generic_xarray_dataset
        quantile_levels = np.concatenate((self.lower_quantiles, self.upper_quantiles))

        unique_clusters, eco_cluster_labels = _create_cluster_labels(self.eco_clusters)

        # Process and filter groups
        data = deseasonalized.groupby(eco_cluster_labels).map(_process_group)
        filtered_data = data.where(data.valid_group, drop=True)
        if filtered_data.size == 0:
            raise ValueError("No valid groups found.")

        aligned_labels = eco_cluster_labels.sel(location=filtered_data.location)
        results = filtered_data.groupby(aligned_labels).map(
            lambda grp: self._compute_thresholds(
                grp,
                (self.lower_quantiles, self.upper_quantiles),
                return_only_thresholds=compute_only_thresholds,
            )
        )

        # Compute cluster-level thresholds
        group_thresholds = results["thresholds"].groupby(aligned_labels).mean()
        unique_clusters, _ = _create_cluster_labels(
            self.eco_clusters.sel(location=filtered_data.location)
        )

        multi_index = pd.MultiIndex.from_arrays(
            unique_clusters.T, names=["component_1", "component_2", "component_3"]
        )

        thresholds_by_cluster = xr.DataArray(
            data=group_thresholds.values,
            dims=("eco_cluster", "quantile"),
            coords={
                "eco_cluster": multi_index,
                "quantile": quantile_levels,
            },
            name="thresholds",
        )

        self.saver._save_data(
            thresholds_by_cluster, "thresholds", location=False, eco_cluster=True
        )

        if not compute_only_thresholds:
            thresholds_array = xr.DataArray(
                np.full((len(deseasonalized.location), len(quantile_levels)), np.nan),
                dims=["location", "quantile"],
                coords={
                    "location": deseasonalized.location,
                    "quantile": quantile_levels,
                },
            )
            thresholds_array.loc[dict(location=results["thresholds"].location)] = (
                results["thresholds"].values
            )
            self.saver._save_data(thresholds_array, "thresholds_locations")

            extremes_array = xr.full_like(deseasonalized, np.nan, dtype=float)
            extremes_array.loc[dict(location=results["extremes"].location)] = results[
                "extremes"
            ].values
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
        # if quantiles["quantile"].dtype != LOWER_QUANTILES_LEVEL.dtype:
        #    LOWER_QUANTILES_LEVEL = LOWER_QUANTILES_LEVEL.astype(
        #        quantiles["quantile"].dtype
        #    )
        lower_quantiles = quantiles.sel(quantile=LOWER_QUANTILES_LEVEL).values
        upper_quantiles = quantiles.sel(quantile=UPPER_QUANTILES_LEVEL).values
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
    minicube_path = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_test/customcube_CO-MEL_1.95_-72.60_S2_v0.zarr.zip"
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
        dataset_processor = create_handler(
            config=config,
            loader=loader,
            saver=saver,
            n_samples=None,  # config.n_samples,  # all the dataset
        )
        # Load and preprocess the dataset
        data = dataset_processor.preprocess_data()
        # Fit the PCA on the data
        extremes_processor.compute_pca_and_transform(scaled_data=data)

    # Apply the PCA to the entire dataset
    # if extremes_processor.projected_data is None:

    # Define the boundaries of the eco_clusters
    if extremes_processor.limits_eco_clusters is None:
        dataset_processor = create_handler(
            config=config, loader=loader, saver=saver, n_samples=None
        )  # all the dataset
        data = dataset_processor.preprocess_data()
        extremes_processor.apply_pca(scaled_data=data)
        extremes_processor.define_limits_eco_clusters()

    # Apply the regional threshold and compute the extremes
    # Load the data
    dataset_processor = create_handler(
        config=config,
        loader=loader,
        saver=saver,
        n_samples=None,
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
        config=config, loader=loader, saver=saver, n_samples=None  # config.n_samples
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
