from .common_imports import *
import cf_xarray as cfxr
from .data_processing.helpers import _ensure_time_chunks, circular_rolling_mean
from datetime import datetime, timedelta, date
import os

from .sentinel2 import Sentinel2Dataloader


class Sentinel2CoarseDataloader(Sentinel2Dataloader):
    def _select_random_pixel(self, ds, lon_idx, lat_idx):
        """
        Select a small 12x12 patch (MODIS resolution).
        """
        lon_size, lat_size = ds.sizes["longitude"], ds.sizes["latitude"]
        lon_start = min(lon_idx, lon_size - 12)
        lat_start = min(lat_idx, lat_size - 12)
        return ds.isel(
            longitude=slice(lon_start, lon_start + 12),
            latitude=slice(lat_start, lat_start + 12),
        )

    def generate_masked_vegetation_index(self, ds, filename=None):
        """
        Generate a coarse-resolution masked vegetation index dataset.

        If a coarse MODIS map exists for the given filename, align fine-resolution reflectance
        data to that map; otherwise, perform spatial coarsening directly. Applies mask
        fraction thresholds and returns the masked vegetation index dataset.

        Args:
            ds (xarray.Dataset): Input dataset with reflectance bands and coordinates.
            filename (str, optional): Used to locate an external coarse map.

        Returns:
            xarray.Dataset or None: Masked vegetation index dataset, or None if too many NaNs.
        """
        # --- Compute initial high-res mask ---
        mask = self._compute_masks(ds)
        ds = ds.unstack("location")  # Ensure unstacked for coarsening or grouping

        # --- Check for existing coarse map ---
        coarse_path = (
            f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/{filename}.zip"
        )
        if os.path.exists(coarse_path):
            try:
                ds_coarse, mask = self._align_with_coarse_map(ds, mask, coarse_path)
            except (OSError, KeyError, ValueError) as e:
                # Log the issue and gracefully fall back
                printt(f"Failed to load coarse map '{coarse_path}': {e}")
                ds_coarse, mask = self._coarsen_without_map(ds, mask)
        else:
            ds_coarse, mask = self._coarsen_without_map(ds, mask)

        # --- Compute coarse vegetation index ---
        evi = self._calculate_vegetation_index(ds_coarse).compute()
        mask = mask.chunk({"time": -1, "location": 3906}).compute()

        # --- Apply mask and validate ---
        masked_evi = evi * mask
        data = xr.Dataset({self.config.index.lower(): masked_evi})

        return None if self._has_excessive_nan(masked_evi) else data

    def _align_with_coarse_map(self, ds, mask, coarse_path):
        """Align fine-resolution dataset and mask to a precomputed coarse MODIS map."""
        coarse_map = xr.open_zarr(coarse_path, consolidated=True)
        coarse_map = (
            coarse_map["250m_16_days_EVI"]
            .isel(start_range=slice(-100, -1))
            .mean(dim="start_range")
        )
        coarse_map = coarse_map.rename({"x": "longitude", "y": "latitude"})
        coarse_map = coarse_map.assign_coords(
            latitude=ds.latitude, longitude=ds.longitude
        )

        # Group fine-resolution reflectance bands by coarse_map bins
        ds_coarse = (
            ds[["B8A", "B04", "B02"]]
            .groupby(coarse_map)
            .map(lambda x: x.mean(dim=("stacked_latitude_longitude"), skipna=True))
        )
        ds_coarse = ds_coarse.sel({coarse_map.name: coarse_map})
        ds_coarse = ds_coarse.stack(location=("latitude", "longitude"))
        ds_coarse = ds_coarse.chunk({"time": -1, "location": 3606})

        # Coarsen mask accordingly
        mask = mask.unstack("location")
        mask_frac = mask.groupby(coarse_map).map(
            lambda x: x.mean(dim=("stacked_latitude_longitude"), skipna=True)
        )
        mask_frac = mask_frac.sel({coarse_map.name: coarse_map})
        mask = xr.where(mask_frac > 0.5, 1, np.nan).stack(
            location=("latitude", "longitude")
        )
        return ds_coarse, mask

    def _coarsen_without_map(self, ds, mask):
        """Perform fixed-size spatial coarsening when no external coarse map is available."""
        ds_coarse = (
            ds[["B8A", "B04", "B02"]]
            .coarsen(latitude=12, longitude=12, boundary="trim")
            .mean(skipna=True)
        )

        mask = mask.unstack("location")
        mask_frac = mask.coarsen(latitude=12, longitude=12, boundary="trim").mean(
            skipna=True
        )
        mask = xr.where(mask_frac > 0.5, 1, np.nan).stack(
            location=("latitude", "longitude")
        )

        return ds_coarse, mask
