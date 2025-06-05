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


def kl_divergence_gaussians(mu_p, sigma_p, mu_q, sigma_q):
    """
    KL divergence between two univariate Gaussians: P || Q
    """
    return (
        np.log(sigma_q / sigma_p)
        + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2)
        - 0.5
    )


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

    path_lc = f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/{sample}.zip"
    ds = xr.open_zarr(path_lc)
    epsg = ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(ds.x.values, ds.y.values)
    ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)}).rename(
        {"x": "longitude", "y": "latitude"}
    )
    ds = ds.stack(location=["longitude", "latitude"])
    thresholds_modis = thresholds_modis.where(
        ~ds.esa_worldcover_2021.isin([50, 60, 70, 80]).compute(), drop=True
    )

    common_locations = xr.align(
        thresholds_modis.location, pca_projection.location, join="inner"
    )[0]

    ds_tr = thresholds_modis.sel(quantile=0.20, location=common_locations.location)

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


def compute_diversity(sample: str, metric="simpson") -> xr.DataArray:
    # Paths
    path_eco_cluster = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/eco_clusters.zarr"
    path_thresh = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_13:04:42_full_fluxnet_therightone_modis/EVI_MODIS/{sample}/thresholds.zarr"

    # Load PCA projection and decode multi-index
    eco_cluster = xr.open_zarr(path_eco_cluster)
    eco_cluster = cfxr.decode_compress_to_multi_index(
        eco_cluster, "location"
    ).eco_clusters

    # Load thresholds and decode multi-index
    thresholds_modis = xr.open_zarr(path_thresh)
    thresholds_modis = cfxr.decode_compress_to_multi_index(
        thresholds_modis, "location"
    ).thresholds

    # path_lc = f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/{sample}.zip"
    # ds = xr.open_zarr(path_lc)
    # epsg = ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
    # transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    # lon, lat = transformer.transform(ds.x.values, ds.y.values)
    # ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)}).rename(
    #     {"x": "longitude", "y": "latitude"}
    # )
    # ds = ds.stack(location=["longitude", "latitude"])
    # thresholds_modis = thresholds_modis.where(
    #     ~ds.esa_worldcover_2021.isin([50, 60, 70, 80]).compute(), drop=True
    # )

    common_locations = xr.align(
        thresholds_modis.location, eco_cluster.location, join="inner"
    )[0]

    ds_tr = thresholds_modis.sel(quantile=0.20, location=common_locations.location)

    # Prepare output with original structure (but initially NaN)
    simpson = xr.DataArray(
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

    def _create_cluster_labels(eco_clusters):
        """Create cluster labels for grouping."""
        unique_clusters, _ = np.unique(eco_clusters.values, axis=0, return_counts=True)
        labels = xr.DataArray(
            data=np.argmax(
                np.all(
                    eco_clusters.values[:, :, None] == unique_clusters.T[None, :, :],
                    axis=1,
                ),
                axis=1,
            ),
            dims=("location",),
            coords={"location": eco_clusters.location},
        )
        return unique_clusters, labels

    unique_clusters, eco_cluster_labels = _create_cluster_labels(eco_cluster)

    def process_group(value):
        # Mask locations where the threshold == value
        mask = ds_tr == value
        masked = ds_tr.where(mask.compute(), drop=True)
        patch_locations = masked.location.values

        if len(patch_locations) < 2:
            return xr.full_like(masked, np.nan)

        patch_eco_cluster = eco_cluster_labels.sel(location=patch_locations)

        # Step 2: count how many times each unique vector appears
        unique_vectors, inverse_indices, counts = np.unique(
            patch_eco_cluster, axis=0, return_counts=True, return_inverse=True
        )

        # Step 3: compute relative abundances
        total = counts.sum()  # Total number of pixels
        p = counts / total  # Relative abundances
        if metric == "relative_abundance":
            diversity = p[inverse_indices]
            filled = xr.DataArray(
                data=diversity,
                coords=masked.coords,
                dims=masked.dims,
                name=f"relative_abundance",
            )
        else:
            # Step 4: Simpson diversity
            if metric == "shannon":
                # Shannon diversity
                diversity = -np.sum(p * np.log(p))
            if metric == "simpson":
                # Simpson diversity
                diversity = 1 - np.sum(p**2)
            if metric == "berger":
                diversity = p.max()  # 1 / np.sum(p**2)
            if metric == "hill":
                # Hill diversity
                diversity = 1 / np.sum(p**2)
            # Create a DataArray with the same coords and dims (matching the location order)
            filled = xr.full_like(masked, diversity)
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
    simpson.loc[raoq_filled.location] = raoq_filled
    if isinstance(simpson, xr.DataArray):
        raoq_ds = simpson.to_dataset(name=metric)
    raoq_ds = cfxr.encode_multi_index_as_compress(raoq_ds, "location")
    saving_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/{metric}.zarr"
    raoq_ds.to_zarr(saving_path, mode="w")
    print(f"{metric} computed for sample:", sample)
    return


def compute_kl_div(sample: str, metric="raoq") -> xr.DataArray:
    # Load datasets
    def load(path, var_name):
        ds = xr.open_zarr(path, chunks={})
        return cfxr.decode_compress_to_multi_index(ds, "location")[var_name]

    path_data_modis = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_13:04:42_full_fluxnet_therightone_modis/EVI_MODIS/{sample}/deseasonalized.zarr"
    path_eco_clusters_sample = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/eco_clusters.zarr"
    path_eco_clusters_training = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/eco_clusters.zarr"
    path_train_data_s2 = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/deseasonalized.zarr"

    data_modis = load(path_data_modis, "deseasonalized")  # .deseasonalized
    eco_cluster_sample = load(path_eco_clusters_sample, "eco_clusters")  # .eco_clust
    eco_cluster_training_s2 = load(path_eco_clusters_training, "eco_clusters")
    data_training_s2 = load(path_train_data_s2, "deseasonalized")

    path_raoq = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/{metric}.zarr"
    # raoq = load(path_raoq, metric)

    # path_lc = f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/{sample}.zip"
    # ds = xr.open_zarr(path_lc)
    # epsg = ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
    # transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    # lon, lat = transformer.transform(ds.x.values, ds.y.values)
    # ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)}).rename(
    #     {"x": "longitude", "y": "latitude"}
    # )
    # ds = ds.stack(location=["longitude", "latitude"])
    # data_modis = data_modis.where(
    #     ~ds.esa_worldcover_2021.isin([50, 60, 70, 80]).compute(), drop=True
    # )

    # Calculate mean and variance for each modis pixel
    mean_modis = data_modis.mean(dim="time")
    std_modis = data_modis.std(dim="time")

    # Calculate mean and variance for each eco-cluster in the Sentinel-2 data
    def _create_cluster_labels(eco_clusters):
        """Create cluster labels for grouping."""
        unique_clusters, _ = np.unique(eco_clusters.values, axis=0, return_counts=True)
        labels = xr.DataArray(
            data=np.argmax(
                np.all(
                    eco_clusters.values[:, :, None] == unique_clusters.T[None, :, :],
                    axis=1,
                ),
                axis=1,
            ),
            dims=("location",),
            coords={"location": eco_clusters.location},
        )
        return unique_clusters, labels

    unique_clusters, eco_cluster_labels = _create_cluster_labels(
        eco_cluster_training_s2
    )
    mean_s2_training = data_training_s2.groupby(eco_cluster_labels).mean(
        dim=["location", "time"]
    )
    std_s2_training = data_training_s2.groupby(eco_cluster_labels).std(
        dim=["location", "time"]
    )

    # Get the mean and variance of each unique eco-cluster in eco_cluster from the training data
    def match_test_to_training_clusters(eco_cluster_test, unique_clusters):
        # Find which training cluster each test location matches
        matched_labels = np.argmax(
            np.all(
                eco_cluster_test.values[:, :, None] == unique_clusters.T[None, :, :],
                axis=1,
            ),
            axis=1,
        )
        return xr.DataArray(
            data=matched_labels,
            dims=("location",),
            coords={"location": eco_cluster_test.location},
        )

    eco_cluster_test_labels = match_test_to_training_clusters(
        eco_cluster_sample, unique_clusters
    )

    mean_s2_sample = mean_s2_training.sel(group=eco_cluster_test_labels)
    std_s2_sample = std_s2_training.sel(group=eco_cluster_test_labels)

    def wasserstein_distance(mu0, sigma0, mu1, sigma1):
        return np.sqrt((mu0 - mu1) ** 2 + (sigma0 - sigma1) ** 2)

    kl_div = wasserstein_distance(
        mu0=mean_modis, sigma0=std_modis, mu1=mean_s2_sample, sigma1=std_s2_sample
    )

    # kl_div = kl_divergence_gaussians(
    #     mu_p=mean_modis,  # modis,
    #     sigma_p=std_modis,
    #     mu_q=mean_s2_sample,
    #     sigma_q=std_s2_sample,
    # )

    # Convert to dataset and save
    kl_div_ds = kl_div.to_dataset(name="wasserstein")
    kl_div_ds = cfxr.encode_multi_index_as_compress(kl_div_ds, "location")
    kl_div_ds = kl_div_ds.chunk("auto")
    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/wasserstein.zarr"
    kl_div_ds.to_zarr(save_path, mode="w", consolidated=True)

    # # Create group variable: same shape as (location,)
    # group_labels = raoq.values
    #
    # # Convert group labels to a DataArray for groupby
    # locations = raoq["location"]
    # group_da = xr.DataArray(
    #     group_labels, coords={"location": locations}, dims="location"
    # )
    #
    # def mean_kl_func(x, axis=None):
    #     return np.mean(x, axis=axis)
    #
    # # Add group labels to diff
    # kl_div.coords[metric] = group_da
    # mean_kl = kl_div.groupby(metric).reduce(mean_kl_func, dim="location")
    #
    # # Convert to dataset and save
    # mean_kl_ds = mean_kl.to_dataset(name="kl_div")
    # save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/kl_div_{metric}_s2.zarr"
    # mean_kl_ds.to_zarr(save_path, mode="w", consolidated=True)
    # print("KL divergence computed for:", sample)

    return


def compute_kl_div_loc2(sample: str, metric="simpson") -> xr.DataArray:
    # Load datasets
    def load(path, var_name):
        ds = xr.open_zarr(path, chunks={})
        return cfxr.decode_compress_to_multi_index(ds, "location")[var_name]

    path_data_modis = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_13:17:58_full_fluxnet_therightone_highveg_modis/EVI_MODIS/{sample}/deseasonalized.zarr"
    path_eco_clusters_sample = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/eco_clusters.zarr"
    path_eco_clusters_training = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/eco_clusters.zarr"
    path_train_data_s2 = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/deseasonalized.zarr"

    data_modis = load(path_data_modis, "deseasonalized")  # .deseasonalized
    eco_cluster_sample = load(path_eco_clusters_sample, "eco_clusters")  # .eco_clust
    eco_cluster_training_s2 = load(path_eco_clusters_training, "eco_clusters")
    data_training_s2 = load(path_train_data_s2, "deseasonalized")

    path_raoq = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/{metric}.zarr"
    raoq = load(path_raoq, metric)

    path_lc = f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/{sample}.zip"
    ds = xr.open_zarr(path_lc)
    epsg = ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(ds.x.values, ds.y.values)
    ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)}).rename(
        {"x": "longitude", "y": "latitude"}
    )
    ds = ds.stack(location=["longitude", "latitude"])
    data_modis = data_modis.where(
        ~ds.esa_worldcover_2021.isin([50, 60, 70, 80]).compute(), drop=True
    )

    # Calculate mean and variance for each modis pixel
    mean_modis = data_modis.mean(dim="time")
    std_modis = data_modis.std(dim="time")

    # Calculate mean and variance for each eco-cluster in the Sentinel-2 data
    def _create_cluster_labels(eco_clusters):
        """Create cluster labels for grouping."""
        unique_clusters, _ = np.unique(eco_clusters.values, axis=0, return_counts=True)
        labels = xr.DataArray(
            data=np.argmax(
                np.all(
                    eco_clusters.values[:, :, None] == unique_clusters.T[None, :, :],
                    axis=1,
                ),
                axis=1,
            ),
            dims=("location",),
            coords={"location": eco_clusters.location},
        )
        return unique_clusters, labels

    unique_clusters, eco_cluster_labels = _create_cluster_labels(
        eco_cluster_training_s2
    )
    mean_s2_training = data_training_s2.groupby(eco_cluster_labels).mean(
        dim=["location", "time"]
    )
    std_s2_training = data_training_s2.groupby(eco_cluster_labels).std(
        dim=["location", "time"]
    )

    # Get the mean and variance of each unique eco-cluster in eco_cluster from the training data
    def match_test_to_training_clusters(eco_cluster_test, unique_clusters):
        # Find which training cluster each test location matches
        matched_labels = np.argmax(
            np.all(
                eco_cluster_test.values[:, :, None] == unique_clusters.T[None, :, :],
                axis=1,
            ),
            axis=1,
        )
        return xr.DataArray(
            data=matched_labels,
            dims=("location",),
            coords={"location": eco_cluster_test.location},
        )

    eco_cluster_test_labels = match_test_to_training_clusters(
        eco_cluster_sample, unique_clusters
    )

    mean_s2_sample = mean_s2_training.sel(group=eco_cluster_test_labels)
    std_s2_sample = std_s2_training.sel(group=eco_cluster_test_labels)

    def kl_divergence_gaussians(mu_p, sigma_p, mu_q, sigma_q):
        """
        KL divergence between two univariate Gaussians: P || Q
        """
        return (
            np.log(sigma_q / sigma_p)
            + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2)
            - 0.5
        )

    kl_div = kl_divergence_gaussians(
        mu_p=mean_modis,  # modis,
        sigma_p=std_modis,
        mu_q=mean_s2_sample,
        sigma_q=std_s2_sample,
    )

    # Convert to dataset and save
    kl_div_ds = kl_div.to_dataset(name="kl_div")
    kl_div_ds = cfxr.encode_multi_index_as_compress(kl_div_ds, "location")
    kl_div_ds = kl_div_ds.chunk("auto")

    # Create group variable: same shape as (location,)
    group_labels = raoq.values

    # Convert group labels to a DataArray for groupby
    locations = raoq["location"]
    group_da = xr.DataArray(
        group_labels, coords={"location": locations}, dims="location"
    )

    def mean_kl_func(x):
        mean_kl = x.mean(dim="location", skipna=True)
        filled = xr.full_like(x, mean_kl)
        return filled

    # Add group labels to diff
    kl_div.coords[metric] = group_da
    mean_kl = kl_div.groupby(metric).map(mean_kl_func)

    # Convert to dataset and save
    mean_kl_ds = mean_kl.to_dataset(name="kl_div")
    mean_kl_ds = cfxr.encode_multi_index_as_compress(mean_kl_ds, "location")

    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/kl_div_modisres_{metric}.zarr"
    mean_kl_ds = mean_kl_ds.chunk("auto")
    mean_kl_ds.to_zarr(save_path, mode="w", consolidated=True)

    print("KL divergence computed for:", sample)

    return


def compute_kl_div_loc(sample: str, metric="berger") -> xr.DataArray:
    # Load datasets
    def load(path, var_name):
        ds = xr.open_zarr(path, chunks={})
        return cfxr.decode_compress_to_multi_index(ds, "location")[var_name]

    path_data_modis = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_13:17:58_full_fluxnet_therightone_highveg_modis/EVI_MODIS/{sample}/deseasonalized.zarr"
    path_eco_clusters_sample = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/eco_clusters.zarr"
    path_eco_clusters_training = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/eco_clusters.zarr"
    path_train_data_s2 = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/deseasonalized.zarr"

    data_modis = load(path_data_modis, "deseasonalized")  # .deseasonalized
    eco_cluster_sample = load(path_eco_clusters_sample, "eco_clusters")  # .eco_clust
    eco_cluster_training_s2 = load(path_eco_clusters_training, "eco_clusters")
    data_training_s2 = load(path_train_data_s2, "deseasonalized")

    path_raoq = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/{metric}.zarr"
    raoq = load(path_raoq, metric)

    path_lc = f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/{sample}.zip"
    ds = xr.open_zarr(path_lc)
    epsg = ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(ds.x.values, ds.y.values)
    ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)}).rename(
        {"x": "longitude", "y": "latitude"}
    )
    ds = ds.stack(location=["longitude", "latitude"])
    data_modis = data_modis.where(
        ~ds.esa_worldcover_2021.isin([50, 60, 70, 80]).compute(), drop=True
    )

    # Calculate mean and variance for each modis pixel
    mean_modis = data_modis.mean(dim="time")
    std_modis = data_modis.std(dim="time")

    # Calculate mean and variance for each eco-cluster in the Sentinel-2 data
    def _create_cluster_labels(eco_clusters):
        """Create cluster labels for grouping."""
        unique_clusters, _ = np.unique(eco_clusters.values, axis=0, return_counts=True)
        labels = xr.DataArray(
            data=np.argmax(
                np.all(
                    eco_clusters.values[:, :, None] == unique_clusters.T[None, :, :],
                    axis=1,
                ),
                axis=1,
            ),
            dims=("location",),
            coords={"location": eco_clusters.location},
        )
        return unique_clusters, labels

    unique_clusters, eco_cluster_labels = _create_cluster_labels(
        eco_cluster_training_s2
    )
    mean_s2_training = data_training_s2.groupby(eco_cluster_labels).mean(
        dim=["location", "time"]
    )
    std_s2_training = data_training_s2.groupby(eco_cluster_labels).std(
        dim=["location", "time"]
    )

    # Get the mean and variance of each unique eco-cluster in eco_cluster from the training data
    def match_test_to_training_clusters(eco_cluster_test, unique_clusters):
        # Find which training cluster each test location matches
        matched_labels = np.argmax(
            np.all(
                eco_cluster_test.values[:, :, None] == unique_clusters.T[None, :, :],
                axis=1,
            ),
            axis=1,
        )
        return xr.DataArray(
            data=matched_labels,
            dims=("location",),
            coords={"location": eco_cluster_test.location},
        )

    eco_cluster_test_labels = match_test_to_training_clusters(
        eco_cluster_sample, unique_clusters
    )

    mean_s2_sample = mean_s2_training.sel(group=eco_cluster_test_labels)
    std_s2_sample = std_s2_training.sel(group=eco_cluster_test_labels)

    def kl_divergence_gaussians(mu_p, sigma_p, mu_q, sigma_q):
        """
        KL divergence between two univariate Gaussians: P || Q
        """
        try:
            return (
                np.log(sigma_q / sigma_p)
                + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2)
                - 0.5
            )
        except:
            print("Error in KL divergence calculation: ", mu_p, sigma_p, mu_q, sigma_q)
            return np.nan

    kl_div = kl_divergence_gaussians(
        mu_p=mean_modis,  # s2_sample,  # modis,
        sigma_p=std_modis,  # std_modis,
        mu_q=mean_s2_sample,
        sigma_q=std_s2_sample,
    )

    # Convert to dataset and save
    kl_div_ds = kl_div.to_dataset(name="kl_div")
    kl_div_ds = cfxr.encode_multi_index_as_compress(kl_div_ds, "location")

    # Create group variable: same shape as (location,)
    group_labels = raoq.values

    # Convert group labels to a DataArray for groupby
    locations = raoq["location"]
    group_da = xr.DataArray(
        group_labels, coords={"location": locations}, dims="location"
    )

    # Add group labels to diff
    kl_div.coords[metric] = group_da

    def mean_kl_func(x):
        mean_kl = x.mean(dim="location", skipna=True)
        filled = xr.full_like(x, mean_kl)
        return filled
        # return np.nanmean(x, axis=axis)

    mean_kl = kl_div.groupby(metric).map(mean_kl_func)
    # Convert to dataset and save
    mean_kl_ds = mean_kl.to_dataset(name="kl_div")
    mean_kl_ds = cfxr.encode_multi_index_as_compress(mean_kl_ds, "location")
    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/kl_div_mean_{metric}.zarr"
    mean_kl_ds = mean_kl_ds.chunk("auto")
    mean_kl_ds.to_zarr(save_path, mode="w", consolidated=True)

    print("KL divergence computed for:", sample)

    return


def compute_robin(sample: str) -> xr.DataArray:
    # Load datasets
    def load(path, var_name):
        ds = xr.open_zarr(path, chunks={})
        return cfxr.decode_compress_to_multi_index(ds, "location")[var_name]

    path_thresh = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_13:04:42_full_fluxnet_therightone_modis/EVI_MODIS/{sample}/thresholds.zarr"

    path_msc_sample = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/msc.zarr"
    path_pca_projection = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/pca_projection.zarr"
    path_pca = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/pca_matrix.pkl"
    with open(path_pca, "rb") as f:
        pca = pk.load(f)

    msc = load(path_msc_sample, "msc")
    pca_projection = load(path_pca_projection, "pca_projection")
    thresholds_modis = load(path_thresh, "thresholds")

    common_locations = xr.align(
        thresholds_modis.location, pca_projection.location, join="inner"
    )[0]

    ds_tr = thresholds_modis.sel(quantile=0.20, location=common_locations.location)

    # Prepare output with original structure (but initially NaN)

    unique_values = np.unique(ds_tr.values)

    def process_group(value):
        # Mask locations where the threshold == value
        mask = ds_tr == value
        masked = ds_tr.where(mask.compute(), drop=True)
        patch_locations = masked.location  # .values

        if len(patch_locations) < 2:
            return xr.full_like(masked, np.nan)

        patch_msc = msc.sel(location=patch_locations)
        mean_msc = patch_msc.mean(dim="location")
        mean_pca_projection = pca.transform(mean_msc.values.reshape(1, -1))

        patch_pca_projection = pca_projection.sel(location=patch_locations).values

        pca_variance = np.var(patch_pca_projection, axis=0)  # per PCA dimension
        total_variance = np.nanmean(pca_variance)
        # Create a DataArray with the same coords and dims (matching the location order)
        filled = xr.full_like(masked, total_variance)

        return filled  # robin

    # Parallel compute across unique threshold values
    results = [delayed(process_group)(val) for val in unique_values]

    # Combine results and ensure location is sorted
    combined_results = delayed(xr.concat)(results, dim="location")
    # Ensure location is sorted (since it can be unsorted across different results)
    combined_results = combined_results.sortby("location").compute()

    ds = combined_results.to_dataset(name="variance")
    ds = cfxr.encode_multi_index_as_compress(ds, "location")
    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/variance.zarr"
    ds = ds.chunk("auto")
    ds.to_zarr(save_path, mode="w", consolidated=True)
    print("Robin index computed for:", sample)


def compute_diff_sigma(sample: str, metric="raoq") -> xr.DataArray:
    # Load datasets
    def load(path, var_name):
        ds = xr.open_zarr(path, chunks={})
        return cfxr.decode_compress_to_multi_index(ds, "location")[var_name]

    path_data_modis = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_13:04:42_full_fluxnet_therightone_modis/EVI_MODIS/{sample}/deseasonalized.zarr"
    path_eco_clusters_sample = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/eco_clusters.zarr"
    path_eco_clusters_training = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/eco_clusters.zarr"
    path_train_data_s2 = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/deseasonalized.zarr"

    data_modis = load(path_data_modis, "deseasonalized")  # .deseasonalized
    eco_cluster_sample = load(path_eco_clusters_sample, "eco_clusters")  # .eco_clust
    eco_cluster_training_s2 = load(path_eco_clusters_training, "eco_clusters")
    data_training_s2 = load(path_train_data_s2, "deseasonalized")

    path_raoq = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/{metric}.zarr"
    raoq = load(path_raoq, metric)

    # path_lc = f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/{sample}.zip"
    # ds = xr.open_zarr(path_lc)
    # epsg = ds.attrs.get("spatial_ref") or ds.attrs.get("EPSG") or ds.attrs.get("CRS")
    # transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    # lon, lat = transformer.transform(ds.x.values, ds.y.values)
    # ds = ds.assign_coords({"x": ("x", lon), "y": ("y", lat)}).rename(
    #     {"x": "longitude", "y": "latitude"}
    # )
    # ds = ds.stack(location=["longitude", "latitude"])
    # data_modis = data_modis.where(
    #     ~ds.esa_worldcover_2021.isin([50, 60, 70, 80]).compute(), drop=True
    # )

    # Calculate mean and variance for each modis pixel
    mean_modis = data_modis.mean(dim="time")
    std_modis = data_modis.std(dim="time")

    # Calculate mean and variance for each eco-cluster in the Sentinel-2 data
    def _create_cluster_labels(eco_clusters):
        """Create cluster labels for grouping."""
        unique_clusters, _ = np.unique(eco_clusters.values, axis=0, return_counts=True)
        labels = xr.DataArray(
            data=np.argmax(
                np.all(
                    eco_clusters.values[:, :, None] == unique_clusters.T[None, :, :],
                    axis=1,
                ),
                axis=1,
            ),
            dims=("location",),
            coords={"location": eco_clusters.location},
        )
        return unique_clusters, labels

    unique_clusters, eco_cluster_labels = _create_cluster_labels(
        eco_cluster_training_s2
    )
    mean_s2_training = data_training_s2.groupby(eco_cluster_labels).mean(
        dim=["location", "time"]
    )
    std_s2_training = data_training_s2.groupby(eco_cluster_labels).std(
        dim=["location", "time"]
    )

    # Get the mean and variance of each unique eco-cluster in eco_cluster from the training data
    def match_test_to_training_clusters(eco_cluster_test, unique_clusters):
        # Find which training cluster each test location matches
        matched_labels = np.argmax(
            np.all(
                eco_cluster_test.values[:, :, None] == unique_clusters.T[None, :, :],
                axis=1,
            ),
            axis=1,
        )
        return xr.DataArray(
            data=matched_labels,
            dims=("location",),
            coords={"location": eco_cluster_test.location},
        )

    eco_cluster_test_labels = match_test_to_training_clusters(
        eco_cluster_sample, unique_clusters
    )

    mean_s2_sample = mean_s2_training.sel(group=eco_cluster_test_labels)
    std_s2_sample = std_s2_training.sel(group=eco_cluster_test_labels)

    def l1_norm_sigma(sigma_p, sigma_q):
        """
        KL divergence between two univariate Gaussians: P || Q
        """
        return np.abs(sigma_p - sigma_q)

    # kl_div = kl_divergence_gaussians(
    #     mu_p=mean_s2_sample,  # modis,
    #     sigma_p=std_s2_sample,
    #     mu_q=mean_modis,
    #     sigma_q=std_modis,
    # )

    kl_div = l1_norm_sigma(
        sigma_p=std_s2_sample,
        sigma_q=std_modis,
    )

    # Convert to dataset and save
    kl_div_ds = kl_div.to_dataset(name="kl_div")
    kl_div_ds = cfxr.encode_multi_index_as_compress(kl_div_ds, "location")
    kl_div_ds = kl_div_ds.chunk("auto")

    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/{sample}/diff_sigma.zarr"
    kl_div_ds.to_zarr(save_path, mode="w", consolidated=True)

    # Create group variable: same shape as (location,)
    group_labels = raoq.values

    # Convert group labels to a DataArray for groupby
    locations = raoq["location"]
    group_da = xr.DataArray(
        group_labels, coords={"location": locations}, dims="location"
    )

    def mean_kl_func(x, axis=None):
        return np.mean(x, axis=axis)

    # Add group labels to diff
    kl_div.coords[metric] = group_da
    mean_kl = kl_div.groupby(metric).reduce(mean_kl_func, dim="location")

    # Convert to dataset and save
    mean_kl_ds = mean_kl.to_dataset(name="kl_div")
    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/diff_sigma.zarr"
    mean_kl_ds.to_zarr(save_path, mode="w", consolidated=True)
    print("KL divergence computed for:", sample)

    return


def compute_extremes(
    sample: str, type="missed", dim="time", threshold=0.2
) -> xr.DataArray:
    path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/extremes.zarr"
    ds = xr.open_zarr(path)
    s2 = cfxr.decode_compress_to_multi_index(ds, "location").extremes

    path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-05-12_11:36:04_new/EVI_MODIS/{sample}/extremes.zarr"
    ds = xr.open_zarr(path)
    modis = cfxr.decode_compress_to_multi_index(ds, "location").extremes

    path_thresh = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-05-12_11:36:04_new/EVI_MODIS/{sample}/thresholds.zarr"
    # Load thresholds and decode multi-index
    thresholds_modis = xr.open_zarr(path_thresh)
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
        modis_pixel = masked.location.values
        if len(modis_pixel) < 100:
            return xr.full_like(
                s2_extreme.sel(location=modis_pixel).mean(dim="location"), np.nan
            )
        if type == "missed":
            missed_detection = s2_extreme.sel(location=modis_pixel) & (
                ~modis_extreme.sel(location=modis_pixel)
            )
        elif type == "common":
            missed_detection = s2_extreme.sel(location=modis_pixel) & (
                modis_extreme.sel(location=modis_pixel)
            )
        # Per-day missed count
        elif type == "all":
            missed_detection = s2_extreme.sel(location=modis_pixel)

        n_missed = missed_detection.sum(dim="location")
        n_total = (
            missed_detection.count(dim="location")
            # missed_detection.size
        )  # len(modis_pixel)  # (~s2_extreme.sel(location=modis_pixel)).sum()
        missed_fraction = n_missed / n_total
        if dim == "avg":
            # 1. Identify time steps (days) where any extreme is detected at any location
            extreme_days = missed_detection.any(dim="location")

            # 2. Mask the missed_detection to keep only the extreme days
            missed_fraction = missed_fraction.where(extreme_days)

            missed_fraction = missed_fraction.mean(
                dim="time", skipna=True
            )  # dim="time")
        return missed_fraction

    # Parallel compute across unique threshold values
    results = [delayed(process_group)(val) for val in unique_values]
    # Combine results and ensure location is sorted
    combined_results = delayed(xr.concat)(results, dim="location", coords="minimal")
    # Ensure location is sorted (since it can be unsorted across different results)
    combined_results_sorted = combined_results.sortby("location")
    missed_fraction = combined_results_sorted.compute()

    ds = missed_fraction.to_dataset(name=f"{type}_fraction_{dim}")
    # ds = cfxr.encode_multi_index_as_compress(ds, "location")
    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/{type}_fraction_{dim}_{threshold}_v6.zarr"
    ds = ds.chunk("auto")
    ds.to_zarr(save_path, mode="w", consolidated=True)
    print(f"{type}_fraction_{dim}_{threshold} index computed for:", sample)


def compute_extremes_s2(
    sample: str, type="missed", dim="time", threshold=0.2
) -> xr.DataArray:
    path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/extremes.zarr"
    ds = xr.open_zarr(path)
    s2 = cfxr.decode_compress_to_multi_index(ds, "location").extremes

    path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-05-12_11:36:04_new/EVI_MODIS/{sample}/extremes.zarr"
    ds = xr.open_zarr(path)
    modis = cfxr.decode_compress_to_multi_index(ds, "location").extremes

    path_thresh = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-05-12_11:36:04_new/EVI_MODIS/{sample}/thresholds.zarr"
    # Load thresholds and decode multi-index
    thresholds_modis = xr.open_zarr(path_thresh)
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

    if type == "missed":
        missed_detection = s2_extreme & ~modis_extreme
    elif type == "common":
        missed_detection = s2_extreme & modis_extreme
    # Per-day missed count
    elif type == "all":
        missed_detection = s2_extreme

    n_missed = missed_detection.sum(dim="location")
    n_total = (
        missed_detection.count(dim="location")
        # missed_detection.size
    )  # len(modis_pixel)  # (~s2_extreme.sel(location=modis_pixel)).sum()
    missed_fraction = n_missed / n_total
    if dim == "avg":
        # 1. Identify time steps (days) where any extreme is detected at any location
        extreme_days = missed_detection.any(dim="location")
        # 2. Mask the missed_detection to keep only the extreme days
        missed_fraction = missed_fraction.where(extreme_days)
        missed_fraction = missed_fraction.mean(dim="time", skipna=True)  # dim="time")

    ds = missed_fraction.to_dataset(name=f"{type}_fraction_{dim}")
    # ds = cfxr.encode_multi_index_as_compress(ds, "location")
    save_path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN/{sample}/s2_{type}_fraction_{dim}_{threshold}_v5.zarr"
    ds = ds.chunk("auto")
    ds.to_zarr(save_path, mode="w", consolidated=True)
    print(f"{type}_fraction_{dim}_{threshold} index computed for:", sample)


if __name__ == "__main__":
    # Example usage
    parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/"
    subfolders = [folder[:-4] for folder in os.listdir(parent_folder)]

    # # parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_test/"
    # subfolders = [
    #     # "custom_cube_50.90_11.56.zarr",
    #     "DE-Hai_51.08_10.45_v0.zarr",
    # ]
    #     #   "DE-RuS_50.87_6.45_v0.zarr",
    #     "ES-Cnd_37.91_-3.23_v0.zarr",
    #     "DE-Lnf_51.33_10.37_v0.zarr",
    #     "UK-ESa_55.91_-2.86_v0.zarr",
    #     "FR-LGt_47.32_2.28_v0.zarr",
    # ]
    # # #     # "DE-Geb_51.10_10.91_v0.zarr",
    #     # "DE-Wet_50.45_11.46_v0.zarr",
    #     # "DE-Bay_50.14_11.87_v0.zarr",
    #     # "DE-Meh_51.28_10.66_v0.zarr",
    #     # "custom_cube_44.17_5.24.zarr",
    #     # "custom_cube_44.24_5.14.zarr",
    #     # "custom_cube_47.31_0.18.zarr",

    #     # "UK-ESa_55.91_-2.86_v0.zarr",
    #     # "AT-Neu_47.12_11.32_v0.zarr",
    # ]
    # "custom_cube_44.17_5.24.zarr",
    # "custom_cube_44.24_5.14.zarr",
    # "custom_cube_47.31_0.18.zarr",
    # "custom_cube_50.90_11.56.zarr",
    # "UK-ESa_55.91_-2.86_v0.zarr",
    # "AT-Neu_47.12_11.32_v0.zarr",
    # compute_extremes(subfolders[0], type="all", dim="time", threshold=0.2)

    @delayed
    def process_sample(sample):
        try:
            base_path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg/EVI_EN"
            sample_path = os.path.join(base_path, sample)
            # if "common_fraction_avg_v3.zarr" not in os.listdir(sample_path):
            compute_extremes(sample, type="missed", dim="time", threshold=0.2)
            #    # remove the file
            #    shutil.rmtree(os.path.join(sample_path, "kl_div_raoq.zarr"))
            ## compute_variance(sample)
            # compute_raoq(sample)
            ## compute_diversity(sample, metric="raoq")
            # compute_diversity(sample, metric="relative_abundance")
            # compute_kl_div(sample)

            # compute_kl_div_loc2(sampbfolders[0]le, metric="raoq")

            # compute_diff_sigma(sample, metric="raoq")
            # compute_robin(sample)

        except Exception as e:
            print(f"Error processing sample: {sample} – {e}")

    # Create delayed tasks
    # variance_raoq(subfolders[0])
    tasks = [process_sample(sample) for sample in subfolders]
    # Trigger execution
    for i in range(0, len(tasks), 20):
        if i != 160:
            compute(
                *tasks[i : i + 20], scheduler="threads"
            )  # or "processes" depending on workload
        else:
            compute(
                *tasks[i:], scheduler="threads"
            )  # or "processes" depending on workload
