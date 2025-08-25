import xarray as xr

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from RegionalExtremesPackage.utils.logging_config import printt
from RegionalExtremesPackage.utils import Loader, Saver
from RegionalExtremesPackage.utils.config import (
    InitializationConfig,
    CLIMATIC_INDICES,
    ECOLOGICAL_INDICES,
    EARTHNET_INDICES,
)


class EcoCluster:
    """
    Class to handle the clustering of ecological data.
    """

    def __init__(
        self,
        config: InitializationConfig,
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
        # Loader class to load intermediate steps.
        self.loader = Loader(config)
        # Saver class to save intermediate steps.
        self.saver = Saver(config)

        #if self.config.load_existing_experiment:
        # Load every variable if already available, otherwise return None.
        self.pca = self.loader._load_pca_matrix()
        self.projected_data = self.loader._load_pca_projection()
        self.limits_eco_clusters = self.loader._load_limits_eco_clusters()
        self.eco_clusters = self.loader._load_data("eco_clusters")
        # else:
        #     # Initialize a new PCA.
        #     if self.config.k_pca:
        #         self.pca = KernelPCA(n_components=self.n_components, kernel="rbf")
        #     else:
        #         self.pca = PCA(n_components=self.n_components)
        #     self.projected_data = None
        #     self.limits_eco_clusters = None
        #     self.eco_clusters = None

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
        if self.config.k_pca:
                self.pca = KernelPCA(n_components=self.n_components, kernel="rbf")
        else:
            self.pca = PCA(n_components=self.n_components)
        #self.pca = PCA(n_components=self.n_components)
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
