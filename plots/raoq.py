import xarray as xr
import numpy as np
import dask.array as da
from scipy.spatial.distance import cdist
from dask import delayed
import cf_xarray as cfxr
import os
import pandas as pd
import copy
from dask import delayed, compute


def compute_raoq(sample: str) -> xr.DataArray:
    # Paths
    path_pca = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/pca_projection.zarr"
    path_thresh = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_13:17:58_full_fluxnet_therightone_highveg_modis/EVI_MODIS/{sample}/thresholds.zarr"

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
    saving_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/raoq.zarr"
    raoq_ds.to_zarr(saving_path, mode="w")
    print("RoaQ computed for sample:", sample)
    return


def compute_rmse(sample: str) -> xr.DataArray:
    # Load datasets
    def load(path):
        ds = xr.open_zarr(path, chunks={})
        return cfxr.decode_compress_to_multi_index(ds, "location")

    path_modis = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_13:17:58_full_fluxnet_therightone_highveg_modis/EVI_MODIS/{sample}/thresholds.zarr"
    path_s2 = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/thresholds.zarr"

    path_raoq = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/raoq.zarr"

    thresholds_modis = load(path_modis).thresholds
    thresholds_s2 = load(path_s2).thresholds
    raoq = load(path_raoq).raoq

    thresholds_modis, thresholds_s2 = xr.align(
        thresholds_modis, thresholds_s2, join="inner"
    )
    thresholds_modis, raoq = xr.align(thresholds_modis, raoq, join="inner")

    quantiles = thresholds_modis["quantile"]
    locations = thresholds_modis["location"]

    # Create group variable: same shape as (location,)
    group_labels = raoq.values

    # Convert group labels to a DataArray for groupby
    group_da = xr.DataArray(
        group_labels, coords={"location": locations}, dims="location"
    )

    # Prepare output array
    rmse_array = xr.DataArray(
        np.full((len(quantiles), len(thresholds_modis.location)), np.nan),
        dims=["quantile", "location"],
        coords={"quantile": quantiles, "location": thresholds_modis.location},
    )
    # Loop over quantiles, apply groupby
    for q in quantiles.values:
        diff = thresholds_modis.sel(quantile=q) - thresholds_s2.sel(quantile=q)

        # Add group labels to diff
        diff.coords["raoq"] = group_da

        def rmse_func(x, axis=None):
            rmse = np.sqrt(np.nanmean(x**2, axis=axis))
            filled = xr.full_like(x, rmse)
            return filled

        grouped_rmse = diff.groupby("raoq").map(rmse_func)  # , dim="location")
        rmse_array.loc[dict(quantile=q, location=grouped_rmse.location)] = (
            grouped_rmse  # .reindex(
        )
        #    raoq=unique_values
        # )

    # Convert to dataset and save
    rmse_ds = rmse_array.to_dataset(name="rmse")
    rmse_ds = cfxr.encode_multi_index_as_compress(rmse_ds, "location")

    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/rmse_loc.zarr"
    rmse_ds.to_zarr(save_path, mode="w", consolidated=True)

    print("RMSE computed for:", sample)

    return


def rmse_raoq(sample):
    # Load datasets
    def load(path):
        ds = xr.open_zarr(path, chunks={})
        return cfxr.decode_compress_to_multi_index(ds, "location")

    path_modis = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_13:17:58_full_fluxnet_therightone_highveg_modis/EVI_MODIS/{sample}/thresholds.zarr"
    path_s2 = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/thresholds.zarr"

    path_raoq = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/raoq.zarr"

    thresholds_modis = load(path_modis).thresholds
    thresholds_s2 = load(path_s2).thresholds
    raoq = load(path_raoq).raoq

    thresholds_modis, thresholds_s2 = xr.align(
        thresholds_modis, thresholds_s2, join="inner"
    )
    thresholds_modis, raoq = xr.align(thresholds_modis, raoq, join="inner")

    quantiles = thresholds_modis["quantile"]
    locations = thresholds_modis["location"]

    # Create group variable: same shape as (location,)
    group_labels = raoq.values
    unique_values = np.unique(group_labels)

    # Convert group labels to a DataArray for groupby
    group_da = xr.DataArray(
        group_labels, coords={"location": locations}, dims="location"
    )

    # Prepare output array
    rmse_array = xr.DataArray(
        np.full((len(quantiles), len(unique_values)), np.nan),
        dims=["quantile", "raoq"],
        coords={"quantile": quantiles, "raoq": unique_values},
    )

    # Loop over quantiles, apply groupby
    for q in quantiles.values:
        diff = thresholds_modis.sel(quantile=q) - thresholds_s2.sel(quantile=q)

        # Add group labels to diff
        diff.coords["raoq"] = group_da

        def rmse_func(x, axis=None):
            return np.sqrt(np.nanmean(x**2, axis=axis))

        grouped_rmse = diff.groupby("raoq").reduce(rmse_func, dim="location")
        rmse_array.loc[dict(quantile=q)] = grouped_rmse.reindex(
            raoq=unique_values
        )  # .values

    # Convert to dataset and save
    rmse_ds = rmse_array.to_dataset(name="rmse")

    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/rmse.zarr"
    rmse_ds.to_zarr(save_path, mode="w", consolidated=True)

    print("RMSE computed for:", sample)


if __name__ == "__main__":
    # Example usage
    parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/"
    subfolders = [folder[:-4] for folder in os.listdir(parent_folder)]

    # parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_test/"
    subfolders = [
        "DE-Hai_51.08_10.45_v0.zarr",
        "DE-RuS_50.87_6.45_v0.zarr",
        "ES-Cnd_37.91_-3.23_v0.zarr",
        "FR-LGt_47.32_2.28_v0.zarr",
        "DE-Lnf_51.33_10.37_v0.zarr",
        "DE-Geb_51.10_10.91_v0.zarr",
        "DE-Wet_50.45_11.46_v0.zarr",
        "DE-Bay_50.14_11.87_v0.zarr",
        "DE-Meh_51.28_10.66_v0.zarr",
        "custom_cube_44.17_5.24.zarr",
        "custom_cube_44.24_5.14.zarr",
        "custom_cube_47.31_0.18.zarr",
        "custom_cube_50.90_11.56.zarr",
        # "UK-ESa_55.91_-2.86_v0.zarr",
        # "AT-Neu_47.12_11.32_v0.zarr",
    ]
    # "custom_cube_44.17_5.24.zarr",
    # "custom_cube_44.24_5.14.zarr",
    # "custom_cube_47.31_0.18.zarr",
    # "custom_cube_50.90_11.56.zarr",
    # "UK-ESa_55.91_-2.86_v0.zarr",
    # "AT-Neu_47.12_11.32_v0.zarr",

    # compute_rmse(subfolders[0])

    @delayed
    def process_sample(sample):
        try:
            base_path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN"
            sample_path = os.path.join(base_path, sample)
            # if "rmse.zarr" not in os.listdir(sample_path):
            compute_raoq(sample)
            compute_rmse(sample)
            rmse_raoq(sample)
        except Exception as e:
            print(f"Error processing sample: {sample} – {e}")

    # Create delayed tasks
    # rmse_raoq(subfolders[0])
    tasks = [process_sample(sample) for sample in subfolders]

    # Trigger execution
    compute(*tasks, scheduler="threads")  # or "processes" depending on workload
