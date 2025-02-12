from .common_imports import *


class Saver:
    def __init__(
        self,
        config: InitializationConfig,
    ):
        self.config = config

    def update_saving_path(self, name):
        self.config.saving_path = self.config.saving_path / name
        # index_position = self.config.saving_path.parts.index(self.config.index)
        # self.config.saving_path = (
        #     self.config.saving_path.parents[
        #         len(self.config.saving_path) - index_position - 1
        #     ]
        #     / self.config.index
        #     / name
        # )

    def _generate_unique_save_path(self, base_name):
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
            name="pca_projection",
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

        # path = self._generate_unique_save_path("pca_projection")
        path = self.config.saving_path / "pca_projection.zarr"
        # Save the file with the unique path
        pca_projection.to_zarr(path, mode="w")
        printt(f"PCA Projection computed and saved to {path}.")

    def _save_limits_eco_clusters(self, limits_eco_clusters: list[np.ndarray]) -> None:
        """Saves the limits eco_clusters to a file."""
        limits_eco_clusters_path = self.config.saving_path / "limits_eco_clusters.npz"
        if os.path.exists(limits_eco_clusters_path):
            raise FileExistsError(
                f"The file {limits_eco_clusters_path} already exists. Rewriting is not allowed."
            )
        np.savez(limits_eco_clusters_path, *limits_eco_clusters)
        printt(f"Limits eco_clusters saved to {limits_eco_clusters_path}")

    def _save_data(self, data, name, basepath=None, location=True, eco_cluster=False):
        """Saves the data to a file."""
        if basepath is None:
            basepath = self.config.saving_path
        path = basepath / f"{name}.zarr"
        # Unstack location for longitude and latitude as dimensions
        if isinstance(data, xr.DataArray):
            data.name = name
            data = data.to_dataset()
        if location:
            data = cfxr.encode_multi_index_as_compress(data, "location")

        if eco_cluster:
            data = cfxr.encode_multi_index_as_compress(data, "eco_cluster")
        if "thresholds" in data.dims:
            # data = data.chunk({"location": 1000, "quantile": -1})
            chunk_size = 1000
            encoding = {
                "thresholds": {"chunks": (chunk_size, -1)},
                "component_1": {"chunks": (chunk_size,)},
                "component_2": {"chunks": (chunk_size,)},
                "component_3": {"chunks": (chunk_size,)},
            }
            if "location" in data.dims:
                data = data.chunk({"location": chunk_size, "quantile": -1})
            data.to_zarr(path, mode="w", encoding=encoding)

        else:
            data = data.chunk("auto")
            data.to_zarr(path, mode="w")
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
