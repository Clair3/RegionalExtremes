from .common_imports import *
from .base import DatasetHandler
from .sentinel2 import Sentinel2DatasetHandler
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from pyproj import Transformer
import cf_xarray as cfxr


class ModisSamplingDatasetHandler(Sentinel2DatasetHandler):
    def _calculate_evi(self, ds):
        """Calculates the Enhanced Vegetation Index (EVI)."""
        return (ds["250m_16_days_EVI"] + 2000) / 12000

    def load_minicube(self, minicube_path, process_entire_minicube=False):
        filepath = Path(minicube_path)  # EARTHNET_FILEPATH + minicube_path
        with xr.open_zarr(filepath, chunks="auto") as ds:
            # Add landcover
            if "esa_worldcover_2021" not in ds.data_vars:
                ds = self.loader._load_and_add_landcover(filepath, ds)
            # Transform UTM to lat/lon
            ds = self._transform_utm_to_latlon(ds)

            if not process_entire_minicube:
                # Select a random vegetation location
                ds = self._get_random_vegetation_pixel_series(ds)
                if ds is None:
                    return None

            self.variable_name = "evi"  # ds.attrs["data_id"]
            # Filter based on vegetation occurrence

            # Calculate EVI and apply cloud/vegetation mask
            evi = self._calculate_evi(ds)
            evi = evi.rename({"start_range": "time"})
            data = xr.Dataset(
                data_vars={
                    f"{self.variable_name}": evi,  # Adding 'evi' as a variable
                    # "landcover": ds[
                    #    "esa_worldcover_2021"
                    # ],  # Adding 'landcover' as another variable
                },
                coords={
                    "source_path": minicube_path,  # Add the path as a coordinate
                },
            )
            if process_entire_minicube:
                self.saver.update_saving_path(filepath.stem)

            return data

    def compute_msc(self, data):
        return data.groupby("time.dayofyear").mean("time", skipna=True)

    def preprocess_data(
        self,
        scale=True,
        reduce_temporal_resolution=True,
        return_time_serie=False,
        remove_nan=True,
        minicube_path=None,
    ):
        """
        Preprocess data based on the index.
        """
        printt("start of the preprocess")

        if minicube_path:
            self._minicube_specific_loading(minicube_path=minicube_path)
        else:
            self._dataset_specific_loading()
        self.data = self.data[self.variable_name]
        # Randomly select n indices from the location dimension

        printt(
            f"Computation on the entire dataset. {self.data.sizes['location']} samples"
        )

        self.msc = self.compute_msc(self.data)
        self.msc = self.msc.transpose("location", "dayofyear", ...)
        self.saver._save_data(self.msc, "msc")

        self.msc = self.msc.persist()
        if return_time_serie:
            self.data = self.data.persist()
            self.data = self.data.transpose("location", "time", ...)
            return self.msc, self.data
        else:
            return self.msc
