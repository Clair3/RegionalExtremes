from .common_imports import *


class Saver:
    def __init__(
        self,
        config: InitializationConfig,
    ):
        self.config = config

    def update_saving_path(self, name):
        self.config.saving_path = self.config.saving_path / name

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

        path = self._generate_unique_save_path("pca_projection")
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
        if isinstance(data, xr.DataArray):
            data.name = name
            data = data.to_dataset()

        data = cfxr.encode_multi_index_as_compress(data, "location")
        data = data.chunk("auto")
        path = self._generate_unique_save_path(name)

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
