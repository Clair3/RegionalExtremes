from .common_imports import *


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
        projection_path = self.config.saving_path / "pca_projection_0.zarr"
        if not os.path.exists(projection_path):
            projection_path = self.config.saving_path / "pca_projection.zarr"
            if not os.path.exists(projection_path):
                printt(f"PCA projection not found at {projection_path}")
                return None

        data = xr.open_zarr(projection_path)
        # data = data.stack(location=["longitude", "latitude"])
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

    def _load_limits_eco_clusters(self) -> list[np.ndarray]:
        """Loads the limits eco_clusters from a file."""
        limits_eco_clusters_path = self.config.saving_path / "limits_eco_clusters.npz"
        if not os.path.exists(limits_eco_clusters_path):
            limits_eco_clusters_path = self.config.saving_path / "limits_bins.npz"
            if not os.path.exists(limits_eco_clusters_path):
                printt(f"Limits eco_clusters not found at {limits_eco_clusters_path}")
                return None

        data = np.load(limits_eco_clusters_path)
        limits_eco_clusters = [data[f"arr_{i}"] for i in range(len(data.files))]
        printt("Limits eco_clusters loaded.")
        return limits_eco_clusters

    def _load_data(self, name, basepath=None, location=True):
        """Save the xarray in a file."""
        # Unstack location for longitude and latitude as dimensions
        if basepath is None:
            basepath = self.config.saving_path
        path = basepath / f"{name}.zarr"  # f"{name}_{data.data_id}.zarr"
        print(path)
        if not os.path.exists(path):
            printt(f"Data not found at {path}")
            return None
        data = xr.open_zarr(path)
        if location:
            data = cfxr.decode_compress_to_multi_index(data, "location")
            # data = data.sel(location=~data.get_index("location").duplicated())
        print(f"{name}.zarr loaded.")
        return data

    def _load_eco_clusters(self):
        eco_clusters_path = self.config.saving_path / "eco_clusters_0.zarr"
        if not os.path.exists(eco_clusters_path):
            eco_clusters_path = self.config.saving_path / "bins_0.zarr"
            if not os.path.exists(eco_clusters_path):
                eco_clusters_path = self.config.saving_path / "bins.zarr"
                if not os.path.exists(eco_clusters_path):
                    printt(f"The file {eco_clusters_path} not found.")
                    return None

        data = xr.open_zarr(eco_clusters_path)
        # data = data.stack(location=["longitude", "latitude"])
        data = cfxr.decode_compress_to_multi_index(data, "location")

        # Determine the actual variable name ('eco_clusters' or 'bins')
        var_name = "eco_clusters" if "eco_clusters" in data.variables else "bins"

        # Access the variable dynamically
        data = getattr(data, var_name).transpose("location", "component", ...)
        # data = data.eco_clusters.transpose("location", "component", ...)
        condition = ~data.isnull().any(dim="component").compute()
        data = data.where(condition, drop=True)
        printt("eco_clusters loaded.")
        return data

    def _load_thresholds(self):
        """Saves the threshold quantiles to a file."""
        thresholds_path = self.config.saving_path / "thresholds_0.zarr"
        if not os.path.exists(thresholds_path):
            printt(f"The file {thresholds_path} not found.")
            return None
        thresholds = xr.open_zarr(thresholds_path)
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
