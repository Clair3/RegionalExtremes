import xarray as xr

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from RegionalExtremesPackage.utils.logger import printt
from RegionalExtremesPackage.utils import Loader, Saver
from RegionalExtremesPackage.datahandlers import create_handler
from RegionalExtremesPackage.utils.config import InitializationConfig


class LocalExtremes:
    def __init__(self, config: InitializationConfig):
        """
        Compute the local extremes by computing threshold for every location independently.
         Number of boxes = n_eco_clusters**n_components
        """
        self.config = config
        # Loader class to load intermediate steps.
        self.loader = Loader(config)
        # Saver class to save intermediate steps.
        self.saver = Saver(config)

        self.lower_quantiles = self.config.lower_quantiles
        self.upper_quantiles = self.config.upper_quantiles

    def compute_local_threshold(self, deseasonalized):
        assert self.config.method == "local"
        """Compute and save a xarray (location, time) indicating the quantiles of extremes using a uniform threshold definition."""

        # Create a new DataArray to store the quantile values (0.025 or 0.975) for extreme values
        extremes_array = xr.full_like(deseasonalized.astype(float), np.nan)

        # Create a new DataArray to store the threshold related to each quantiles.
        quantile_levels = np.concatenate((self.lower_quantiles, self.upper_quantiles))
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
            return_only_thresholds=compute_only_thresholds,
        )
        printt("results computed")

        # Assign the results back to the quantile_array
        thresholds_array.values = results["thresholds"].values.T
        if not compute_only_thresholds:
            extremes_array.values = results["extremes"].values

        # save the array
        printt("Saving in progress")
        self.saver._save_data(thresholds_array, "thresholds")
        if not compute_only_thresholds:
            self.saver._save_data(extremes_array, "extremes")

    def _compute_thresholds(
        self,
        deseasonalized: xr.DataArray,
        return_only_thresholds=False,
    ):
        """
        Assign quantile levels to deseasonalized data.

        Args:
            deseasonalized (xarray.DataArray): Deseasonalized data.
            quantile_levels (tuple): Tuple of lower and upper quantile levels.
            return_only_thresholds (bool): If True, only return quantile thresholds.

        Returns:
            xarray.Dataset: Dataset containing extremes and thresholds.
        """
        # Unpack quantile levels and prepare datas
        quantile_levels_combined = np.concatenate(
            (self.lower_quantiles, self.upper_quantiles)
        )
        deseasonalized = deseasonalized.chunk("auto")

        quantiles_xr = deseasonalized.quantile(
            quantile_levels_combined, dim="time"
        ).assign_coords(quantile=quantile_levels_combined)

        # Return thresholds only if requested
        if return_only_thresholds:
            return xr.Dataset({"thresholds": quantiles_xr})

        extremes = self._apply_thresholds(
            deseasonalized, quantiles_xr, (self.lower_quantiles, self.upper_quantiles)
        )
        results = xr.Dataset({"extremes": extremes, "thresholds": quantiles_xr})
        return results

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
        Create boolean masks for each quantile level.

        Args:
            data (xarray.DataArray): The input data.
            quantiles (xarray.DataArray): Quantile values.
            quantile_levels (tuple): Tuple of (lower_quantile_levels, upper_quantile_levels).

        Returns:
            list: List of boolean masks corresponding to each quantile level.
        """
        lower_levels, upper_levels = quantile_levels

        lower_quantiles = quantiles.sel(quantile=lower_levels).values.reshape(
            len(lower_levels), -1, 1
        )
        upper_quantiles = quantiles.sel(quantile=upper_levels).values.reshape(
            len(upper_levels), -1, 1
        )
        lower_mask_conditions = [
            data < lower_quantiles[0],
            *[
                (data >= lower_quantiles[i - 1]) & (data < lower_quantiles[i])
                for i in range(1, len(lower_levels))
            ],
        ]

        upper_mask_conditions = [
            *[
                (data > upper_quantiles[i - 1]) & (data <= upper_quantiles[i])
                for i in range(1, len(upper_levels))
            ],
            data > upper_quantiles[-1],
        ]

        return lower_mask_conditions + upper_mask_conditions


def local_extremes_method(args, minicube_path):
    # Initialization of the configs, load and save paths, log.txt.
    config = InitializationConfig(args)
    assert config.method == "local"

    # Initialization of RegionalExtremes, load data if already computed.
    extremes_processor = LocalExtremes(config=config)

    dataset_processor = create_handler(config=config, n_samples=None)  # all the dataset

    msc, data = dataset_processor.preprocess_data(
        scale=False,
        return_time_serie=True,
        reduce_temporal_resolution=False,
        remove_nan=False,
        minicube_path=minicube_path,
    )
    # Deseasonalized data
    deseasonalized = dataset_processor._deseasonalize(data, msc)
    # Apply the local threshold
    extremes_processor.compute_local_threshold(deseasonalized)
    printt("Local extremes computed")
