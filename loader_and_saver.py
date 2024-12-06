import os
import numpy as np
import xarray as xr
import pickle as pk
from pathlib import Path
import cf_xarray as cfxr

from config import InitializationConfig
from utils import printt

CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
ECOLOGICAL_FILEPATH = (
    lambda index: f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{index}/Groups_{index}gapfilled_QCdyn.zarr"
)
VARIABLE_NAME = lambda index: f"{index}gapfilled_QCdyn"


class Loader:
    def __init__(
        self,
        config: InitializationConfig,
    ):
        self.config = config

    def _load_pca_matrix(self):
        """
        Load PCA matrix from the file.
        """
        pca_path = self.config.saving_path / "pca_matrix.pkl"
        if not os.path.exists(pca_path):
            printt(f"PCA projection not found at {pca_path}")
            return None
        with open(pca_path, "rb") as f:
            pca = pk.load(f)
        return pca

    def _load_pca_projection(self, explained_variance=False):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        projection_path = self.config.saving_path / "pca_projection_3.zarr"
        if not os.path.exists(projection_path):
            printt(f"PCA projection not found at {projection_path}")
            return None

        data = xr.open_zarr(projection_path)
        data = cfxr.decode_compress_to_multi_index(data, "location")
        pca_projection = data.pca.transpose("location", "component", ...)
        # Remove NaNs
        condition = ~pca_projection.isnull().any(dim="component").compute()
        pca_projection = pca_projection.where(condition, drop=True)
        printt("Projection loaded from {}".format(projection_path))
        if explained_variance:
            return pca_projection, data.explained_variance
        else:
            return pca_projection

    def _load_spatial_masking(self):
        """Saves the extremes quantile to a file."""
        # mask_path = self.config.saving_path / "mask.zarr"
        # Overwrite to speed up the process
        mask_path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-18_10:05:59_eco_regional_2000_hr_50bins/EVI/mask.zarr"  # "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-01_14:52:57_eco_threshold_2000/EVI/mask.zarr"
        if not os.path.exists(mask_path):
            printt(f"The file {mask_path} not found.")
            return None
        mask = xr.open_zarr(mask_path).astype(np.int32)
        # Unstack location for longitude and latitude as dimensions
        mask = mask.stack(location=["longitude", "latitude"])
        printt("Mask loaded.")
        return mask

    def _load_limits_bins(self) -> list[np.ndarray]:
        """Loads the limits bins from a file."""
        limits_bins_path = self.config.saving_path / "limits_bins.npz"
        if not os.path.exists(limits_bins_path):
            printt(f"Limits bins not found at {limits_bins_path}")
            return None
        data = np.load(limits_bins_path)
        limits_bins = [data[f"arr_{i}"] for i in range(len(data.files))]
        printt("Limits bins loaded.")
        return limits_bins

    def _load_data(self, data, name):
        """Save the xarray in a file."""
        # Unstack location for longitude and latitude as dimensions
        data.name = name
        path = self.config.saving_path / f"{name}_{data.data_id}.zarr"
        if not os.path.exists(path):
            printt(f"Data not found at {path}")
            return None
        data = xr.open_zarr(path)
        data = cfxr.decode_compress_to_multi_index(data, "location")
        print(f"{name}.zarr loaded.")
        return data

    def _load_bins(self):
        bins_path = self.config.saving_path / "bins_2.zarr"
        if not os.path.exists(bins_path):
            bins_path = self.config.saving_path / "boxes.zarr"
            if not os.path.exists(bins_path):
                printt(f"The file {bins_path} not found.")
                return None
            Warning(
                'boxes.zarr is an inconsistent legacy name, change it for "bins.zarr"'
            )

        data = xr.open_zarr(bins_path)
        data = cfxr.decode_compress_to_multi_index(data, "location")
        data = data.bins.transpose("location", "component", ...)
        condition = ~data.isnull().any(dim="component").compute()
        data = data.where(condition, drop=True)
        printt("Bins loaded.")
        return data

    def _load_thresholds(self):
        """Saves the threshold quantiles to a file."""
        thresholds_path = self.config.saving_path / "thresholds.zarr"
        if not os.path.exists(thresholds_path):
            printt(f"The file {thresholds_path} not found.")
            return None
        thresholds = xr.open_zarr(thresholds_path)
        thresholds = cfxr.decode_compress_to_multi_index(thresholds, "location")
        # Unstack location for longitude and latitude as dimensions
        # extremes = extremes.stack(location=["longitude", "latitude"])
        printt("Thresholds loaded.")
        return thresholds

    def _load_extremes(self):
        """Saves the extremes quantile to a file."""
        extremes_path = self.config.saving_path / "extremes.zarr"
        if not os.path.exists(extremes_path):
            printt(f"The file {extremes_path} not found.")
            return None
        extremes = xr.open_zarr(extremes_path)
        extremes = cfxr.decode_compress_to_multi_index(extremes, "location")
        # Unstack location for longitude and latitude as dimensions
        # extremes = extremes.stack(location=["longitude", "latitude"])
        printt("Extremes loaded.")
        return extremes

    def _load_minmax_data(self):
        """
        Load min-max data from the file.
        """
        min_max_data_path = self.config.saving_path / "min_max_data.zarr"
        if not os.path.exists(min_max_data_path):
            printt(f"The file {min_max_data_path } not found.")
            return None
        min_max_data = xr.open_zarr(min_max_data_path)
        return min_max_data


class Saver:
    def __init__(
        self,
        config: InitializationConfig,
    ):
        self.config = config

    def _saving_path(self, base_name):
        # Initial path setup
        extension = ".zarr"

        # Start with index 0 and increment if the file already exists
        index = 0
        while True:
            path = self.config.saving_path / f"{base_name}_{index}{extension}"
            if not os.path.exists(path):
                break
            index += 1  # Increment index if file exists
        return path

    def _save_minmax_data(self, max_data, min_data, coords):
        min_max_data_path = self.config.saving_path / "min_max_data.zarr"
        # Save min_data and max_data
        if not min_max_data_path.exists():
            min_max_data = xr.Dataset(
                {
                    "max_data": max_data,
                    "min_data": min_data,
                },
                coords={"dayofyear": coords},
            )
            min_max_data.to_zarr(min_max_data_path)
            printt("Min and max data saved.")
        else:
            return None

    def _save_pca_model(self, pca):
        # Save the PCA model
        pca_path = self.config.saving_path / "pca_matrix.pkl"
        with open(pca_path, "wb") as f:
            pk.dump(pca, f)
        printt(f"PCA saved: {pca_path}")

    def _save_pca_projection(self, pca_projection, explained_variance_ratio_) -> None:
        """Saves the limits bins to a file."""
        # Split the components into separate DataArrays
        # Create a new coordinate for the 'component' dimension
        component = np.arange(pca_projection.shape[1])

        # Create the new DataArray
        pca_projection = xr.DataArray(
            data=pca_projection.values,
            dims=["location", "component"],
            coords={
                "location": pca_projection.location,
                "component": component,
            },
            name="pca",
        )

        # Explained variance for each component
        explained_variance = xr.DataArray(
            explained_variance_ratio_,
            dims=["component"],
            coords={"component": component},
        )
        pca_projection["explained_variance"] = explained_variance

        if isinstance(pca_projection, xr.DataArray):
            pca_projection = pca_projection.to_dataset()

        pca_projection = cfxr.encode_multi_index_as_compress(pca_projection, "location")

        path = self._saving_path("pca_projection")
        # Save the file with the unique path
        pca_projection.to_zarr(path)
        printt(f"PCA Projection computed and saved to {path}.")

    def _save_limits_bins(self, limits_bins: list[np.ndarray]) -> None:
        """Saves the limits bins to a file."""
        limits_bins_path = self.config.saving_path / "limits_bins.npz"
        if os.path.exists(limits_bins_path):
            raise FileExistsError(
                f"The file {limits_bins_path} already exists. Rewriting is not allowed."
            )
        np.savez(limits_bins_path, *limits_bins)
        printt(f"Limits bins saved to {limits_bins_path}")

    def _save_data(self, data, name):
        """Saves the data to a file."""
        # Unstack location for longitude and latitude as dimensions
        data.name = name
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        data = cfxr.encode_multi_index_as_compress(data, "location")
        data = data.chunk("auto")
        path = self._saving_path(name)

        data.to_zarr(path)
        printt(f"{name} computed and saved.")

    def _save_spatial_masking(self, mask):
        mask = mask.astype(np.int32)
        mask = (
            mask.drop_duplicates("location")
            .set_index(location=["longitude", "latitude"])
            .unstack("location")
        )

        mask_path = self.config.saving_path / "mask.zarr"
        if os.path.exists(mask_path):
            raise FileExistsError(
                f"The file {mask_path} already exists. Rewriting is not allowed."
            )
        mask.to_zarr(mask_path)
        printt("Mask computed and saved.")
