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
from pyproj import Transformer
from collections import Counter
import pickle as pk
from pathlib import Path


def compute_extremes(
    sample: str,
    path_data1=str,
    path_data2=str,
    savename=str,
    type="common",
    dim="time",
    threshold=0.2,
) -> xr.DataArray:
    path = f"{path_data1}/{sample}/extremes.zarr"
    # path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-09-20_12:06:49_S2_low_res/EVI_EN/{sample}/extremes.zarr"
    ds = xr.open_zarr(path)
    s2 = cfxr.decode_compress_to_multi_index(ds, "location").extremes

    path = f"{path_data2}/{sample}/extremes.zarr"
    # path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_13:17:58_full_fluxnet_therightone_highveg_modis/EVI_MODIS/{sample}/extremes.zarr"

    ds = xr.open_zarr(path)
    modis = cfxr.decode_compress_to_multi_index(ds, "location").extremes

    path_thresh = os.path.abspath(
        f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_13:17:58_full_fluxnet_therightone_highveg_modis/EVI_MODIS/{sample}/thresholds.zarr"
    )

    thresholds_modis = xr.open_zarr(path_thresh)
    # Load thresholds and decode multi-index
    # thresholds_modis = xr.open_zarr(path_thresh)
    thresholds_modis = cfxr.decode_compress_to_multi_index(
        thresholds_modis, "location"
    ).thresholds

    modis = modis.drop_duplicates("location")  # .unstack("location")
    thresholds_modis = thresholds_modis.drop_duplicates(
        "location"
    )  # .unstack("location")
    s2 = s2.drop_duplicates("location")  # .unstack("location")
    # s2, modis = xr.align(s2, modis, join="inner")
    common_locations_modis = xr.align(
        thresholds_modis.location, modis.location, join="inner"
    )[0]
    common_locations = xr.align(s2.location, common_locations_modis, join="inner")[0]
    common_time = xr.align(s2.time, modis.time, join="inner")[0]
    ds_tr = thresholds_modis.sel(quantile=0.10, location=common_locations.location)
    unique_values = np.unique(ds_tr.values)

    s2 = s2.sel(location=common_locations, time=common_time)
    modis = modis.sel(location=common_locations, time=common_time)

    valid_mask = (~np.isnan(modis.values)) & (~np.isnan(s2.values))

    modis = modis.where(valid_mask)
    s2 = s2.where(valid_mask)
    if threshold < 0.501:
        s2_extreme = s2 <= threshold
        modis_extreme = modis <= threshold
    else:
        s2_extreme = s2 >= threshold
        modis_extreme = modis >= threshold

    # Per-day missed count
    def process_group(modis_pixel_indice):
        # Mask locations where the threshold == value
        mask = ds_tr == modis_pixel_indice
        masked = ds_tr.where(mask.compute(), drop=True)
        modis_pixel = masked.location  # .values
        if len(modis_pixel) < 100:
            mean_lon = modis_pixel.location.longitude.mean().item()
            mean_lat = modis_pixel.location.latitude.mean().item()

            # Create a DataArray filled with NaN
            sos_std = xr.full_like(s2.sel(location=modis_pixel).mean(), np.nan)

            # Expand to have lon/lat coordinates
            sos_std = sos_std.expand_dims(
                longitude=[mean_lon],
                latitude=[mean_lat],
            )

            # Stack into a multi-index for location
            sos_std = sos_std.stack(location=["longitude", "latitude"])
            return sos_std

        if type == "missed":
            missed_detection = s2_extreme.sel(location=modis_pixel) & (
                ~modis_extreme.sel(location=modis_pixel)
            )
            missed_detection = missed_detection.where(
                ~modis_extreme.sel(location=modis_pixel)
            )  # set to nan where modis is extreme
        elif type == "common":
            missed_detection = s2_extreme.sel(location=modis_pixel) & (
                modis_extreme.sel(location=modis_pixel)
            )
            missed_detection = missed_detection.where(
                modis_extreme.sel(location=modis_pixel)
            )

        n_missed = missed_detection.sum(dim="location")
        n_total = missed_detection.count(dim="location")
        n_total = n_total.where(n_total > 0)
        missed_fraction = n_missed / n_total

        no_extreme_days = (
            ~s2_extreme.sel(location=modis_pixel)
            & ~modis_extreme.sel(location=modis_pixel)
        ).all(dim="location")
        missed_fraction = missed_fraction.where(~no_extreme_days)
        if dim == "avg":
            missed_fraction = missed_fraction.mean(dim="time", skipna=True)
        mean_lon = modis_pixel.location.longitude.mean().item()
        mean_lat = modis_pixel.location.latitude.mean().item()
        # Expand scalar to have these coordinates
        missed_fraction = missed_fraction.expand_dims(
            longitude=[mean_lon],
            latitude=[mean_lat],
        )
        # Stack into a multi-index for location
        missed_fraction = missed_fraction.stack(location=["longitude", "latitude"])
        return missed_fraction

    # Parallel compute across unique threshold values
    results = [delayed(process_group)(val) for val in unique_values]
    # Combine results and ensure location is sorted
    combined_results = delayed(xr.concat)(results, dim="location", coords="minimal")
    # Ensure location is sorted (since it can be unsorted across different results)
    combined_results_sorted = combined_results.sortby("location")
    missed_fraction = combined_results_sorted.compute()
    print("missed", missed_fraction.mean().values)

    ds = missed_fraction.to_dataset(name=f"{type}_fraction_{dim}")
    ds = cfxr.encode_multi_index_as_compress(ds, "location")
    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/comparison/results2/{sample}/{savename}"
    ds = ds.chunk("auto")
    ds.to_zarr(save_path, mode="w", consolidated=True)
    print(f"agreement index computed for:", sample)


if __name__ == "__main__":
    # Example usage
    parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/"
    subfolders = [folder[:-4] for folder in os.listdir(parent_folder)]
    path_data1 = Path(
        "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/comparison/S2_regional_20_lowcloud/EVI_EN"
    )
    path_data2 = Path(
        "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/comparison/S2LR_regional_20_lowcloud/EVI"
    )
    # savename_dice = f"dice_{str(path_data1.parent.name)}_{str(path_data2.parent.name)}"
    savename = (
        f"common_fraction_{str(path_data1.parent.name)}_{str(path_data2.parent.name)}"
    )

    # existing = set(os.listdir(path_data2))
    #
    # subfolders = [
    #     folder
    #     for folder in subfolders
    #     if not os.path.isdir(os.path.join(path_data2, folder, f"{savename}.zarr"))
    # ]

    print(f"Processing {len(subfolders)} samples...")

    @delayed
    def process_sample(sample):
        try:
            compute_extremes(
                sample,
                path_data1=path_data1,
                path_data2=path_data2,
                savename=savename + "_0.01",
                type="common",
                dim="time",
                threshold=0.1,
            )
        except Exception as e:
            print(f"Error processing sample: {sample} – {e}")

    # Create delayed tasks
    # variance_raoq(subfolders[0])
    tasks = [process_sample(sample) for sample in subfolders]
    # Trigger execution
    for i in range(0, len(tasks), 5):
        if i < len(tasks) - 5:
            compute(
                *tasks[i : i + 5], scheduler="threads"
            )  # or "processes" depending on workload
        else:
            compute(
                *tasks[i:], scheduler="threads"
            )  # or "processes" depending on workload
