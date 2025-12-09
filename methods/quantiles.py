import xarray as xr

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import dask.array as da
from abc import ABC

from typing import Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from RegionalExtremesPackage.utils import Loader, Saver
from RegionalExtremesPackage.utils.config import (
    InitializationConfig,
)


def quantiles(config: InitializationConfig):
    """Factory returning QuantilesPerDoy or QuantilesBase depending on config."""
    if config.dayofyear_extreme:
        return QuantilesPerDoy(config)
    else:
        return QuantilesBase(config)


class QuantilesBase(ABC):

    def __init__(self, config: InitializationConfig):
        """
        Base class for quantile threshold computation.

        Args:
            config: InitializationConfig with attributes used across methods.
        """
        self.config = config
        self.lower_quantiles = np.asarray(self.config.lower_quantiles)
        self.upper_quantiles = np.asarray(self.config.upper_quantiles)
        self.quantile_levels_combined = np.concatenate(
            (self.lower_quantiles, self.upper_quantiles)
        )

        # Loader / Saver (replace stubs above with your real classes)
        self.loader = Loader(config)
        self.saver = Saver(config)

        # If user requested loading existing experiment, try to load thresholds
        self.thresholds: Optional[xr.DataArray] = None
        if getattr(self.config, "load_existing_experiment", False):
            loaded = self.loader._load_data("thresholds", location=False, cluster=True)
            if loaded is not None:
                self.thresholds = loaded

    def apply_regional_threshold(
        self, data: xr.DataArray, eco_clusters_load: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute extremes and vci for the given data grouped by ECO clusters.

        Args:
            data: xarray DataArray with dims that include "location" and "time".
            eco_clusters_load: DataArray with shape (location, components...) describing cluster membership.
        """
        if not isinstance(data, xr.DataArray):
            raise TypeError("`data` must be an xarray.DataArray")

        if self.config.method != "regional":
            raise AssertionError("Method must be 'regional' for regional thresholds")

        # helper to create textual labels for eco clusters (e.g. "0_10_3")
        def create_eco_cluster_labels(ecocluster):
            return np.array(["_".join(map(str, row)) for row in ecocluster.values])

        def prepare_data_with_labels(data_arr, labels_arr):
            return data_arr.assign_coords(eco_cluster=("location", labels_arr))

        # Prepare labels and grouped object
        eco_labels = create_eco_cluster_labels(eco_clusters_load)
        data_with_labels = prepare_data_with_labels(data, eco_labels)
        grouped = data_with_labels.groupby("eco_cluster")

        # Map thresholds for each group's label
        thresholds = grouped.map(lambda grp: self.map_thresholds_to_clusters(grp))
        # align thresholds to existing eco_cluster coord and drop helper coords if present
        thresholds = thresholds.sel(eco_cluster=data_with_labels["eco_cluster"])
        if "eco_cluster" in thresholds.coords:
            thresholds = thresholds.drop_vars(["eco_cluster"], errors="ignore")

        # Save thresholds (backend-specific)
        self.saver._save_data(thresholds, "thresholds")

        # Compute extremes array per group
        extremes = self._compute_extremes_per_grp(grouped)
        extremes_array = xr.full_like(data.astype(float), np.nan)
        extremes_array.values = extremes.values
        self.saver._save_data(extremes_array, "extremes")

        if getattr(self.config, "vci", False):
            # Compute vci and save
            vci = self._compute_vci_per_grp(grouped)
            vci_array = xr.full_like(data.astype(float), np.nan)
            vci_array.values = vci.values
            self.saver._save_data(vci_array, "vci")

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
    ):
        """
        Assign quantile levels to data.

        Args:
            data (xarray.DataArray): Data used to compute the quantiles.
            quantile_levels (tuple): Tuple of lower and upper quantile levels.
        Returns:
            xarray.Dataset: Dataset containing extremes and thresholds.
        """
        assert self.config.method == "regional"
        data = data.chunk("auto")

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

    # Define the function to map thresholds to clusters
    def map_thresholds_to_clusters(self, grp):
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

    def _compute_extremes_per_grp(self, grouped):
        return grouped.map(
            lambda grp: self._apply_thresholds(
                grp,
                self.map_thresholds_to_clusters(grp),
            )
        )

    def _apply_vci(
        self,
        data: xr.DataArray,
        thresholds,
    ):

        extremes = xr.full_like(data.astype(float), np.nan)
        if thresholds is np.nan:
            return extremes

        min_data = thresholds.sel(quantile=0)
        max_data = thresholds.sel(quantile=1)
        return (data - min_data) / (max_data - min_data)

    def _compute_vci_per_grp(self, grouped):
        return grouped.map(
            lambda grp: self._apply_vci(grp, self.map_thresholds_to_clusters(grp))
        )


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

    def _compute_extremes_per_grp(self, grouped):
        return grouped.map(
            lambda grp: grp.groupby("time.dayofyear").map(
                lambda grp_day: self._apply_thresholds(
                    grp_day,
                    self.map_thresholds_to_clusters(grp).sel(
                        dayofyear=grp_day["time.dayofyear"][0]
                    ),
                )
            )
        )

    def _compute_vci_per_grp(self, grouped):
        return grouped.map(
            lambda grp: grp.groupby("time.dayofyear").map(
                lambda grp_day: self._apply_vci(
                    grp_day,
                    self.map_thresholds_to_clusters(grp).sel(
                        dayofyear=grp_day["time.dayofyear"][0]
                    ),
                )
            )
        )
