import xarray as xr

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import dask.array as da
from abc import ABC


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from RegionalExtremesPackage.utils.logging_config import printt
from RegionalExtremesPackage.utils import Loader, Saver
from RegionalExtremesPackage.dataloaders import dataloader
from RegionalExtremesPackage.utils.config import (
    InitializationConfig,
)

np.set_printoptions(threshold=sys.maxsize)


@staticmethod
def quantiles(config: InitializationConfig):
    if config.dayofyear_extreme:
        return QuantilesPerDoy(config)
    else:
        return QuantilesBase(config)


class QuantilesBase(ABC):
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
        # self.eco_clusters = eco_clusters
        self.lower_quantiles = self.config.lower_quantiles
        self.upper_quantiles = self.config.upper_quantiles
        self.quantile_levels_combined = np.concatenate(
            (self.lower_quantiles, self.upper_quantiles)
        )
        # Loader class to load intermediate steps.
        self.loader = Loader(config)
        # Saver class to save intermediate steps.
        self.saver = Saver(config)
        if self.config.load_existing_experiment:
            # Load every variable if already available, otherwise return None.
            # self.eco_clusters = self.loader._load_data("eco_clusters")
            self.thresholds = self.loader._load_data(
                "thresholds", location=False, cluster=True
            )

        else:
            # self.eco_clusters = None
            self.thresholds = None

    def apply_regional_threshold(self, data, eco_clusters_load):
        """
        Compute and save a xarray indicating the quantiles of extremes using the regional threshold definition.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            tuple: (thresholds, extremes_array) if not compute_only_thresholds, else thresholds

        Raises:
            AssertionError: If config method is not "regional"
            ValueError: If eco_clusters or thresholds are not properly configured
        """
        # Validate inputs
        if not isinstance(data, xr.DataArray):
            raise TypeError("data must be an xarray DataArray")

        assert self.config.method == "regional", "Method must be regional"

        # Initialize parameters
        # compute_only_thresholds = self.config.is_generic_xarray_dataset

        def create_eco_cluster_labels(eco_clusters_load):
            """Create standardized eco-cluster labels."""
            # if not hasattr(self, "eco_clusters"):  # or not self.eco_clusters:
            #     raise ValueError("eco_clusters not properly initialized")
            return np.array(
                ["_".join(map(str, cluster)) for cluster in eco_clusters_load.values]
            )

        def prepare_data(data, labels):
            """Prepare data with eco_cluster coordinates."""
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
        extremes_array = xr.full_like(data.astype(float), np.nan)

        # Process data
        eco_cluster_labels = create_eco_cluster_labels(eco_clusters_load)
        data = prepare_data(data, eco_cluster_labels)
        grouped = data.groupby("eco_cluster")

        # Calculate thresholds
        thresholds = grouped.map(lambda grp: map_thresholds_to_clusters(grp))

        thresholds = thresholds.sel(eco_cluster=data["eco_cluster"])
        thresholds = thresholds.drop_vars(["eco_cluster"])
        # Save thresholds
        self.saver._save_data(thresholds, "thresholds")

        # Calculate and save extremes if needed
        extremes = grouped.map(
            lambda grp: self._apply_thresholds(
                grp,
                map_thresholds_to_clusters(grp),
            )
        )
        extremes_array.values = extremes.values
        self.saver._save_data(extremes_array, "extremes")
        return thresholds, extremes_array

    def generate_regional_threshold(self, data, eco_clusters_load):
        """
        Generate and save regional thresholds for extremes.

        Parameters:
        -----------
        data : xarray.DataArray (time, location) to compute the distribution
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

        unique_clusters, eco_cluster_labels = _create_cluster_labels(eco_clusters_load)
        # Process and filter groups
        data = data.groupby(eco_cluster_labels).map(_process_group)
        filtered_data = data.where(data.valid_group, drop=True)
        if filtered_data.sizes == 0:
            raise ValueError("No valid groups found.")

        aligned_labels = eco_cluster_labels.sel(location=filtered_data.location)

        unique_clusters, _ = _create_cluster_labels(
            eco_clusters_load.sel(location=filtered_data.location)
        )
        eco_cluster_index = pd.MultiIndex.from_arrays(
            unique_clusters.T, names=["component_1", "component_2", "component_3"]
        )

        thresholds_by_cluster = self._compute_thresholds_per_grp(
            filtered_data, aligned_labels, eco_cluster_index
        )
        self.saver._save_data(
            thresholds_by_cluster, "thresholds", location=False, eco_cluster=True
        )

    def _compute_thresholds_per_grp(self, data, labels, eco_cluster_index):
        """
        Compute thresholds for each group.

        Returns:
        --------
        xarray.DataArray
            Computed thresholds for the group
        """
        thresholds = data.groupby(labels).map(
            lambda grp: self._compute_thresholds(
                grp,
            )
        )
        thresholds_by_cluster = xr.DataArray(
            data=thresholds.values,
            dims=("eco_cluster", "quantile"),
            coords={
                "eco_cluster": eco_cluster_index,
                "quantile": self.quantile_levels_combined,
            },
            name="thresholds",
        )
        return thresholds_by_cluster

    def _compute_thresholds(
        self,
        data: xr.DataArray,
        # return_only_thresholds=False,
    ):
        """
        Assign quantile levels to data data.

        Args:
            data (xarray.DataArray): Data used to compute the quantiles.
            quantile_levels (tuple): Tuple of lower and upper quantile levels.
            return_only_thresholds (bool): If True, only return quantile thresholds.
        Returns:
            xarray.Dataset: Dataset containing extremes and thresholds.
        """
        assert self.config.method == "regional"
        data = data.chunk("auto")

        # Compute quantiles based on the method
        return self._compute_quantiles_per_grp(data)

    def _compute_quantiles_per_grp(self, data):
        """
        Compute regional quantiles for the provided data.
        Args:
            data (dask.array.Array): Flattened data to compute quantiles for.
            quantile_levels (np.ndarray): Combined lower and upper quantile levels.
        Returns:
            xarray.DataArray: Regional quantiles as an xarray.DataArray.
        """

        data = data.data
        non_nan_count = da.count_nonzero(~da.isnan(data))
        flattened_nonan = data.flatten()[~da.isnan(data.flatten())]
        if non_nan_count < 30:
            quantiles = da.full(
                self.quantile_levels_combined.shape, np.nan, dtype=float
            )  # Fill with NaNs
        else:
            # Compute quantiles using Dask
            quantiles = da.percentile(
                flattened_nonan,
                self.quantile_levels_combined * 100,  # Convert to percentages
                method="linear",
            )
        # Wrap quantiles in an xarray.DataArray
        quantiles_xr = xr.DataArray(
            quantiles,
            coords={"quantile": self.quantile_levels_combined},
            dims=["quantile"],
        )
        return quantiles_xr

    def _apply_thresholds(
        self,
        data: xr.DataArray,
        thresholds,
    ):
        extremes = xr.full_like(data.astype(float), np.nan)
        if thresholds is np.nan:
            return extremes

        masks = self._create_quantile_masks(
            data,
            thresholds,
        )

        for i, mask in enumerate(masks):
            extremes = xr.where(mask, self.quantile_levels_combined[i], extremes)
        return extremes

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

        lower_quantiles_thresholds = quantiles.sel(
            quantile=self.lower_quantiles
        )  # .values
        upper_quantiles_thresholds = quantiles.sel(
            quantile=self.upper_quantiles
        )  # .values

        masks = [
            data < lower_quantiles_thresholds[0],
            *[
                (data >= lower_quantiles_thresholds[i - 1])
                & (data < lower_quantiles_thresholds[i])
                for i in range(1, len(self.lower_quantiles))
            ],
            *[
                (data > upper_quantiles_thresholds[i - 1])
                & (data <= upper_quantiles_thresholds[i])
                for i in range(1, len(self.upper_quantiles))
            ],
            data > upper_quantiles_thresholds[-1],
        ]
        return masks


class QuantilesPerDoy(QuantilesBase):
    def _compute_thresholds_per_grp(self, data, labels, eco_cluster_index):
        """
        Compute thresholds for each group.

        Returns:
        --------
        xarray.DataArray
            Computed thresholds for the group
        """
        thresholds = data.groupby(labels).map(
            lambda grp: grp.groupby("time.dayofyear").map(
                lambda grp_time: self._compute_thresholds(
                    grp_time,
                )
            )
        )

        thresholds_by_cluster = xr.DataArray(
            data=thresholds.values,
            dims=("eco_cluster", "dayofyear", "quantile"),
            coords={
                "eco_cluster": eco_cluster_index,
                "dayofyear": thresholds.dayofyear,
                "quantile": self.quantile_levels_combined,
            },
            name="thresholds",
        )
        return thresholds_by_cluster
