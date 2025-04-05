import xarray as xr
import numpy as np
import dask.array as da
from scipy.spatial.distance import cdist
from dask import delayed
import cf_xarray as cfxr
import os
import pandas as pd
from dask import delayed, compute


def compute_raoq(sample: str) -> xr.DataArray:
    # Paths
    path_pca = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/pca_projection.zarr"
    path_thresh = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_13:04:42_full_fluxnet_therightone_modis/EVI_MODIS/{sample}/thresholds.zarr"

    # Load PCA projection and decode multi-index
    pca_projection = xr.open_zarr(path_pca)
    pca_projection = cfxr.decode_compress_to_multi_index(
        pca_projection, "location"
    ).pca_projection
    pca_projection = pca_projection.transpose("location", "component", ...)

    # Load thresholds and decode multi-index
    thresholds_modis = xr.open_zarr(path_thresh)
    thresholds_modis = cfxr.decode_compress_to_multi_index(
        thresholds_modis, "location"
    ).thresholds
    ds_tr = thresholds_modis.sel(quantile=0.50, location=pca_projection.location.values)

    # Prepare output with original structure (but initially NaN)
    # raoq = xr.full_like(ds_tr, np.nan)

    raoq = xr.DataArray(
        np.full(
            (len(ds_tr.location),),
            np.nan,
        ),
        dims=["location"],
        coords={
            "location": ds_tr.location,
        },
    )

    unique_values = np.unique(ds_tr.values)

    def process_group(value):
        # Mask locations where the threshold == value
        mask = ds_tr == value
        masked = ds_tr.where(mask.compute(), drop=True)
        patch_locations = masked.location.values

        if len(patch_locations) < 2:
            return xr.full_like(masked, np.nan)

        patch_pca = pca_projection.sel(location=patch_locations)
        patch_pca_np = patch_pca.values
        dists = cdist(patch_pca_np, patch_pca_np, metric="euclidean").flatten()
        mean_dist = np.nanmean(dists)

        # Create a DataArray with the same coords and dims (matching the location order)
        filled = xr.full_like(masked, mean_dist)
        return filled

    # Parallel compute across unique threshold values
    results = [delayed(process_group)(val) for val in unique_values]

    # Combine results and ensure location is sorted
    combined_results = delayed(xr.concat)(results, dim="location")

    # Ensure location is sorted (since it can be unsorted across different results)
    combined_results_sorted = combined_results.sortby("location")

    # Compute the full result
    raoq_filled = combined_results_sorted.compute()

    # Combine filled and original structure (ensures all locations are present)
    raoq.loc[raoq_filled.location] = raoq_filled
    if isinstance(raoq, xr.DataArray):
        raoq_ds = raoq.to_dataset(name="raoq")
    raoq_ds = cfxr.encode_multi_index_as_compress(raoq_ds, "location")
    saving_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/raoq_lowres.zarr"
    raoq_ds.to_zarr(saving_path, mode="w")
    print("RoaQ computed for sample:", sample)
    return


def rmse(sample):
    path_thresh = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_13:04:42_full_fluxnet_therightone_modis/EVI_MODIS/{sample}/thresholds.zarr"
    # Load thresholds and decode multi-index
    thresholds_modis = xr.open_zarr(path_thresh)
    thresholds_modis = cfxr.decode_compress_to_multi_index(
        thresholds_modis, "location"
    ).thresholds
    unique_values = np.unique(thresholds_modis.values)

    path_thresh = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/thresholds.zarr"
    # Load thresholds and decode multi-index
    thresholds_s2 = xr.open_zarr(path_thresh)
    thresholds_s2 = cfxr.decode_compress_to_multi_index(
        thresholds_s2, "location"
    ).thresholds

    thresholds_modis, thresholds_s2 = xr.align(
        thresholds_modis, thresholds_s2, join="inner"
    )
    print(thresholds_s2)

    def process_group(value):
        # Mask locations where the threshold == value
        mask = thresholds_modis == value
        masked = thresholds_modis.where(mask.compute(), drop=True)
        patch_locations = masked.location.values
        try:
            if not thresholds_s2.indexes["location"].is_unique:
                raise ValueError("thresholds_s2 has duplicate locations")
        except:
            print(thresholds_s2)
        available_locs = np.intersect1d(patch_locations, thresholds_s2.location.values)

        rmse_array = xr.DataArray(
            np.full(
                (
                    len(thresholds_s2["quantile"]),
                    len([value]),
                ),
                np.nan,
            ),
            dims=["quantile", "location"],
            coords={
                "location": value,
                # pd.MultiIndex.from_product(
                #    [[masked.latitude.mean().item()], [masked.longitude.mean().item()]],
                #    names=["lat", "lon"],
                # ),
                "quantile": thresholds_s2["quantile"],
                # "dayofyear": data.time.dt.dayofyear,
            },
        )
        if len(available_locs) < 1:
            return rmse_array

        try:
            patch_s2 = thresholds_s2.sel(location=available_locs)
        except:
            print(
                (
                    thresholds_s2.location.values == thresholds_modis.location.values
                ).all()
            )
            print(available_locs.shape)
            print("Available locations:", available_locs)
            print(available_locs.shape)
            print("Available locations:", available_locs)
            if not thresholds_s2.indexes["location"].is_unique:
                thresholds_s2 = thresholds_s2.sel(
                    location=~thresholds_s2.get_index("location").duplicated()
                )

            patch_s2 = thresholds_s2.sel(location=available_locs)
        patch_modis = thresholds_modis.sel(location=available_locs)

        for quantile in thresholds_modis["quantile"]:
            diff = (
                patch_modis.sel(quantile=quantile).values
                - patch_s2.sel(quantile=quantile).values
            )
            rmse = np.nanmean(np.sqrt(diff**2))
            rmse_array.loc[dict(quantile=quantile)] = rmse
        return rmse_array

    results = [delayed(process_group)(val) for val in unique_values]
    # results = [process_group(val) for val in unique_values]
    # Combine results and ensure location is sorted
    combined_results = delayed(xr.concat)(results, dim="location")

    # Ensure location is sorted (since it can be unsorted across different results)
    # combined_results_sorted = combined_results.sortby("location")

    # Compute the full result
    rmse_array = combined_results.compute()
    if isinstance(rmse_array, xr.DataArray):
        rmse_array = rmse_array.to_dataset(name="rmse")
    rmse_array = cfxr.encode_multi_index_as_compress(rmse_array, "location")
    saving_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/rmse.zarr"
    rmse_array.to_zarr(saving_path, mode="w")
    print("RMSE computed for sample:", sample)


if __name__ == "__main__":
    # Example usage
    parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/"
    subfolders = [folder[:-4] for folder in os.listdir(parent_folder)]

    # sample = subfolders[0]
    # rmse(sample)
    @delayed
    def process_sample(sample):
        try:
            base_path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN"
            sample_path = os.path.join(base_path, sample)
            # if "rmse.zarr" not in os.listdir(sample_path):
            rmse(sample)
        except Exception as e:
            print(f"Error processing sample: {sample} – {e}")

    rmse(subfolders[0])
    # Create delayed tasks
    # tasks = [process_sample(sample) for sample in subfolders]

    # Trigger execution
    # compute(*tasks, scheduler="threads")  # or "processes" depending on workload

    # for sample in subfolders:
    #     try:
    #         if "rmse.zarr" not in os.listdir(
    #             f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/"
    #         ):
    #             rmse(sample)
    #     except:
    #         print(f"Error processing sample: {sample}")
    #         continue
#
