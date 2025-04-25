import os

from common_imports import *
from RegionalExtremesPackage.plots.base import Plots
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import rioxarray as rio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cf_xarray as cfxr


plt.rcParams.update({"font.size": 20})


class PlotsSentinel2(Plots):

    def normalize(self, data, rgb=[1, 0, 2]):
        # Normalize the explained variance
        def _normalization(index):
            band = data.isel(component=index).values
            band_min = np.quantile(band, 0.1)
            band_max = np.quantile(band, 0.90)
            # return np.clip((band - band_min) / (band_max - band_min), 0, 1)
            return (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band))

        normalized_red = _normalization(rgb[0])  # Red is the first component
        normalized_green = _normalization(rgb[1])  # Green is the second component
        normalized_blue = _normalization(rgb[2])  # blue is the third component

        # Stack the components into a 3D array
        return np.stack((normalized_red, normalized_green, normalized_blue), axis=-1)

    def map_component(self, colored_by_eco_cluster=True):
        if colored_by_eco_cluster:
            data = self.loader._load_data("eco_clusters")
            data = data.eco_clusters
        else:
            data, explained_variance = self.loader._load_pca_projection(
                explained_variance=True
            )

        longitudes, latitudes = zip(*data.location.values)
        longitudes = np.array(longitudes)
        latitudes = np.array(latitudes)

        # Normalize the explained variance
        rgb_colors = self.normalize(data)

        # Compute bounds with a small margin
        margin = 0.1  # Adjust the margin as needed
        min_lon, max_lon = longitudes.min() - margin, longitudes.max() + margin
        min_lat, max_lat = latitudes.min() - margin, latitudes.max() + margin

        # Create the plot
        fig, ax = plt.subplots(
            figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.set_facecolor("black")

        ax.scatter(
            longitudes,
            latitudes,
            color=rgb_colors,
            s=1,  # Size of the point
            transform=ccrs.PlateCarree(),
        )

        # Set the extent of the map
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        # Add geographical features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, edgecolor="white")
        ax.add_feature(
            cfeature.OCEAN,
        )

        # Adjust the layout
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        if colored_by_eco_cluster:
            # Add a title
            plt.title("Colored by eco-clusters", fontsize=16)
            map_saving_path = self.saving_path / "map_eco_clusters.png"
        else:
            plt.title("Colored by pca component", fontsize=16)
            map_saving_path = self.saving_path / "map_pca.png"
        plt.savefig(map_saving_path)
        # Show the plot
        plt.show()

    def plot_3D_pca_with_lat_lon_gradient(self):
        # Load PCA projection
        pca_projection, explained_variance = self.loader._load_pca_projection(
            explained_variance=True
        )
        pca_projection = pca_projection.set_index(
            location=["longitude", "latitude"]
        ).unstack("location")

        latitude, longitude = pca_projection.latitude, pca_projection.longitude

        # Normalize latitude and longitude to [0, 1] for gradient mapping
        norm_latitude = Normalize()(latitude)
        norm_longitude = Normalize()(longitude)

        # Map lat/lon to RGB using gradient
        colors = np.stack(
            (norm_latitude, norm_longitude, np.zeros_like(norm_latitude)), axis=1
        )
        # Example: latitude -> red, longitude -> green, fixed blue

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot
        sc = ax.scatter(
            pca_projection.isel(component=0).values.T,
            pca_projection.isel(component=1).values.T,
            pca_projection.isel(component=2).values.T,
            c=colors,
            s=10,
            edgecolor="k",
        )

        # Adding labels and title
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        ax.set_title("3D PCA Projection with Lat/Lon Gradient Colors")

        # Save and show plot
        saving_path = self.saving_path / "3D_pca_lat_lon_gradient.png"
        plt.savefig(saving_path)
        plt.show()
        return

    def plot_3D_pca_landcover(self):
        # Load PCA projection
        pca_projection = self.loader._load_pca_projection()
        pca_projection = pca_projection.set_index(
            location=["longitude", "latitude"]
        ).unstack("location")

        landcover = self.loader._load_data("landcover").landcover

        # Define landcover flag values and meanings
        flag_meanings = [
            "Tree cover",
            "Shrubland",
            "Grassland",
            "Cropland",
            "Built-up",
            "Bare / sparse vegetation",
            "Snow and ice",
            "Permanent water bodies",
            "Herbaceous wetland",
            "Mangroves",
            "Moss and lichen",
        ]
        flag_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

        # Create a colormap and a normalization for the land cover values
        cmap = plt.cm.get_cmap("tab20", len(flag_values) + 1)
        boundaries = flag_values + [110]
        norm = plt.matplotlib.colors.BoundaryNorm(boundaries=boundaries, ncolors=cmap.N)

        # Calculate the tick positions as the midpoint of each bin
        tick_positions = [
            (boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(flag_values))
        ]

        # Create the 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            pca_projection.isel(component=0).values.T,
            pca_projection.isel(component=1).values.T,
            pca_projection.isel(component=2).values.T,
            c=landcover.values,
            cmap=cmap,
            norm=norm,
            s=10,
            edgecolor="k",
        )

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.05, aspect=8)  # Adjust padding and aspect
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(flag_meanings)
        cbar.ax.tick_params(labelsize=10)  # Adjust tick label size
        cbar.set_label(
            "Land Cover", fontsize=12, rotation=270, labelpad=20
        )  # Adjust labelpad

        # Add labels and title
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        ax.set_title("3D PCA Projection with Landcover Colors")

        # Save and show the plot
        saving_path = self.saving_path / "3D_pca_landcover.png"
        plt.savefig(saving_path, bbox_inches="tight")  # Ensures no text is cut off
        plt.show()
        return

    def plot_msc(self, colored_by_eco_cluster=False):
        # Load and preprocess the time series dataset
        data = self.loader._load_data("msc")
        data = data.chunk({"location": 50, "dayofyear": -1})

        # Randomly select n indices from the location dimension
        # random_indices = np.random.choice(len(data.location), size=10000, replace=False)

        # Use isel to select the subset of data based on the random indices
        subset = data  # .isel(location=random_indices)
        if colored_by_eco_cluster:
            cluster = self.loader._load_data("eco_clusters")
            cluster = cluster.eco_clusters
        else:
            cluster = self.loader._load_pca_projection()

        # Normalize the explained variance
        rgb_colors = self.normalize(cluster)

        # Plot the time series with corresponding colors
        fig, ax = plt.subplots(figsize=(15, 10))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        for i, loc in enumerate(subset.location):
            ax.plot(
                subset.dayofyear,
                subset.msc.isel(location=i),
                color=tuple(rgb_colors[i]),
                # label=f"Loc: {loc}",
                alpha=0.8,
            )

        # Calculate and plot the mean MSC across all locations
        mean_msc = subset.msc.mean(dim="location")
        ax.plot(
            subset.dayofyear,
            mean_msc,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Mean MSC",
        )

        # Add labels and optional legend
        if colored_by_eco_cluster:
            plt.title("MSC Time Series Colored by eco-clusters")
            saving_path = self.saving_path / "msc_eco_cluster_quantile_norm.png"
        else:
            plt.title("MSC Time Series Colored by PCA Components")
            saving_path = self.saving_path / "msc_pca.png"
        # Optional: Set tick and label colors to white for visibility
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

        ax.set_xlabel("Day Of Year")
        ax.set_ylabel("EVI")
        plt.savefig(saving_path)
        plt.show()

    def plot_minicube_eco_clusters(self):
        data = self.loader._load_data("eco_clusters")
        data = data.eco_clusters.transpose("location", "component", ...)
        bins = data.unstack("location")

        # Normalize the explained variance
        rgb_normalized = self.normalize(bins)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        ax.pcolormesh(
            bins.longitude.values,
            bins.latitude.values,
            rgb_normalized.transpose(1, 0, 2),
            # transform=projection,
        )
        ax.axis("off")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        # Add a title
        plt.title("Eco-clusters")
        saving_path = self.saving_path / "eco_clusters_2.png"
        plt.savefig(saving_path)
        plt.show()

    def plot_minicube_pca_projection(self):
        data = self.loader._load_pca_projection(explained_variance=False)
        data = data.transpose("location", "component", ...)
        bins = data.unstack("location")

        # Normalize the explained variance
        rgb_normalized = self.normalize(bins)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        ax.pcolormesh(
            bins.longitude.values,
            bins.latitude.values,
            rgb_normalized.transpose(1, 0, 2),
            # transform=projection,
        )
        ax.axis("off")

        # Add a title
        plt.title("PCA Projection")
        saving_path = self.saving_path / "pca_projection.png"
        plt.savefig(saving_path)
        plt.show()

    def plot_rgb(self):
        paths = glob.glob(
            f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/{self.minicube_name}*"
        ) or glob.glob(
            f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_test/{self.minicube_name}*"
        )
        print(paths)
        path = paths[0]

        ds = xr.open_zarr(path)
        if "time" not in ds.dims:
            ds = ds.rename({"time_sentinel-2-l2a": "time"})

        # The rgb channel need to be normalise and enlightened.
        def normalize(band):
            band_min, band_max = (band.min(), band.max())
            # band_min = np.quantile(band, 0.02)
            # band_max = np.quantile(band, 0.98)
            return (band - band_min) / (band_max - band_min)

        def brighten(band):
            alpha = 1  # .2
            beta = 0
            return np.clip(
                alpha * band + beta, 0, 255
            )  # np.clip(alpha*band+beta, 0,255)

        # mask = ds.cloudmask_en.where(ds.cloudmask_en == 0, np.nan)
        mask = ds.SCL.where(ds.SCL.isin([4, 5]), np.nan)
        mask = mask.where(mask != 0, 1)

        fig, axes = plt.subplots(
            nrows=5, ncols=5, constrained_layout=True, figsize=(20, 20)
        )
        t = 270  # 259 + 1
        for i in range(5):
            for j in range(5):
                axes[i, j].get_xaxis().set_visible(False)
                axes[i, j].get_yaxis().set_visible(False)
                current_date = ds.isel(time=t).time.dt.date.values
                axes[i, j].set_title(str(current_date), fontsize=30)
                # Remove the cloud masking
                red = ds.isel(time=t).B04
                green = ds.isel(time=t).B03
                blue = ds.isel(time=t).B02

                # red = red * mask.isel(time=t)
                # green = green * mask.isel(time=t)
                # blue = blue * mask.isel(time=t)

                red = brighten(normalize(red))
                green = brighten(normalize(green))
                blue = brighten(normalize(blue))

                rgb_composite = np.dstack((red, green, blue))
                axes[i, j].imshow(rgb_composite)
                t += 1
        # Add a title
        # plt.title("RBG")
        saving_path = self.saving_path / "rgb.png"
        plt.savefig(saving_path)

    def plot_landcover(self):
        # minicube_name = os.path.basename(path)
        worldcover_name = (
            "/Net/Groups/BGI/work_4/scratch/mzehner/DE_cube_redo/Worldcover/extended_"
            + minicube_name.rsplit(".", 1)[0]
            + "_esa_wc_only.tif"
        )

        # Open the worldcover raster
        data_wc = rio.open_rasterio(worldcover_name).squeeze()

        # Define land cover values and their corresponding names
        flag_meanings = [
            "Tree cover",
            "Shrubland",
            "Grassland",
            "Cropland",
            "Built-up",
            "Bare / sparse vegetation",
            "Snow and ice",
            "Permanent water bodies",
            "Herbaceous wetland",
            "Mangroves",
            "Moss and lichen",
        ]
        flag_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

        # Create a colormap and a normalization for the land cover values
        cmap = plt.cm.get_cmap(
            "tab20", len(flag_values) + 1
        )  # Use 'tab20' to ensure sufficient colors
        boundaries = flag_values + [110]
        norm = plt.matplotlib.colors.BoundaryNorm(
            boundaries=flag_values + [110], ncolors=cmap.N
        )

        # Calculate the tick positions as the midpoint of each bin
        tick_positions = [
            (boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(flag_values))
        ]

        # Plot the raster with a colorbar
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data_wc, cmap=cmap, norm=norm)
        cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01)

        # Customize the colorbar with land cover names
        cb.set_ticks(tick_positions)
        cb.set_ticklabels(flag_meanings)
        cb.ax.tick_params(labelsize=10)  # Adjust label size if necessary
        cb.set_label("Land Cover", fontsize=12)

        # Add titles and labels to the plot
        ax.set_title("Land Cover", fontsize=14)
        ax.axis("off")  # Hide axes if the focus is on the raster

        # Show the plot
        plt.tight_layout()
        saving_path = self.saving_path / "landcover_wc2.png"
        plt.savefig(saving_path)

    def plot_location_in_europe(self):
        # Load data
        data = self.loader._load_data("msc")
        longitude = data.longitude.mean().values
        latitude = data.latitude.mean().values

        # Create the plot
        fig, ax = plt.subplots(
            figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        # Set the extent to Europe (longitude and latitude ranges)
        ax.set_extent([-25, 45, 35, 75], crs=ccrs.PlateCarree())

        # Scatter plot for the location
        ax.scatter(
            longitude,
            latitude,
            color="red",  # Point color for better visibility
            s=20,  # Adjusted size for better visibility
            transform=ccrs.PlateCarree(),
        )

        # Add geographical features
        ax.coastlines(resolution="10m")
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.LAKES, facecolor="lightblue")
        ax.add_feature(cfeature.RIVERS)

        # Add gridlines for better context
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Set a title
        ax.set_title("Point Location", fontsize=14)

        saving_path = self.saving_path / "location.png"
        plt.savefig(saving_path)
        plt.show()

    def plot_thresholds(self, quantile):
        data = self.loader._load_data("thresholds").thresholds
        data = data.unstack("location")

        fig, ax = plt.subplots(figsize=(12, 10))
        # Adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Plot the data with a colormap
        pcm = ax.pcolormesh(
            data.longitude.values.T,
            data.latitude.values.T,
            data.sel(quantile=quantile).values.T,
            cmap="inferno",  # viridis",  # Choose a colormap, e.g., 'viridis', 'plasma', 'coolwarm'
        )

        # Add a title
        plt.title(f"Quantile {quantile*100}%")
        ax.axis("off")  # Hide axes if the focus is on the raster

        # Add a colorbar
        pcm.set_clim()  # -0.15, -0.05)

        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("Value")  # Set the label for the colorbar
        saving_path = self.saving_path / f"quantile_{quantile}.png"
        plt.savefig(saving_path)
        plt.show()

    def plot_thresholds_rmse(self, quantile):
        data = self.loader._load_data("rmse_loc").rmse
        data = data.unstack("location")

        fig, ax = plt.subplots(figsize=(12, 10))
        # Adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Plot the data with a colormap
        pcm = ax.pcolormesh(
            data.longitude.values.T,
            data.latitude.values.T,
            data.sel(quantile=quantile).values.T,
            cmap="Reds",  # viridis",  # Choose a colormap, e.g., 'viridis', 'plasma', 'coolwarm'
        )

        # Add a title
        plt.title(f"RMSE between Modis and S2 - Quantile {quantile*100}%")
        ax.axis("off")  # Hide axes if the focus is on the raster

        # Add a colorbar
        pcm.set_clim()  # -0.15, -0.05)

        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("Value")  # Set the label for the colorbar
        saving_path = self.saving_path / f"rmse_modisres_{quantile}.png"
        plt.savefig(saving_path)
        plt.show()

    def plot_thresholds_error(self, sample, quantile):
        data = self.loader._load_data("thresholds").thresholds
        data = data.unstack("location")

        path = f"/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_13:04:42_full_fluxnet_therightone_modis/EVI_MODIS/{sample}/thresholds.zarr"
        data_modis = xr.open_zarr(path)
        data_modis = cfxr.decode_compress_to_multi_index(
            data_modis, "location"
        ).thresholds
        data_modis = data_modis.unstack("location")

        rmse = np.sqrt(
            (
                data.sel(quantile=quantile).values.T
                - data_modis.sel(quantile=quantile).values.T
            )
            ** 2
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        # Adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Plot the data with a colormap
        pcm = ax.pcolormesh(
            data.longitude.values.T,
            data.latitude.values.T,
            rmse,
            cmap="YlOrRd",  # viridis",  # Choose a colormap, e.g., 'viridis', 'plasma', 'coolwarm'
        )

        # Add a title
        plt.title(f"Rmse between Modis and Sentinel for the Quantile {quantile*100}%")
        ax.axis("off")  # Hide axes if the focus is on the raster

        # Add a colorbar
        pcm.set_clim(0, 0.1)

        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("Value")  # Set the label for the colorbar
        saving_path = self.saving_path / f"quantile_rmse{quantile}.png"
        plt.savefig(saving_path)
        plt.show()

    def plot_raoq(self):
        data = self.loader._load_data("raoq").raoq
        data = data.unstack("location")

        fig, ax = plt.subplots(figsize=(12, 10))
        # Adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Plot the data with a colormap
        pcm = ax.pcolormesh(
            data.longitude.values.T,
            data.latitude.values.T,
            data.values.T,
            cmap="Reds",  # viridis",  # Choose a colormap, e.g., 'viridis', 'plasma', 'coolwarm'
        )

        # Add a title
        plt.title(f"Raoq")

        # Add a colorbar
        pcm.set_clim(0, 0.8)
        ax.axis("off")  # Hide axes if the focus is on the raster

        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("RaoQ")  # Set the label for the colorbar
        saving_path = self.saving_path / "raoq.png"
        plt.savefig(saving_path)
        plt.show()

    # path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_12:13:03_full_fluxnet_therightone/EVI_EN/UK-ESa_55.91_-2.86_v0.zarr/raoq.zarr"
    # data = xr.open_zarr(path)
    # data = cfxr.decode_compress_to_multi_index(data, "location").raoq
    # data = data.unstack("location")

    def plot_extremes(self):
        extremes = self.loader._load_data("extremes").extremes
        # extremes = extremes.unstack("location")

        extremes = extremes.sel(time=slice("2017-01-01", None))

        # Get all unique quantile values present in extremes, ignoring NaN values
        unique_quantiles = np.sort(
            np.unique(extremes.values[~np.isnan(extremes.values)])
        )
        # Split quantiles into two groups
        low_quantiles = [q for q in unique_quantiles if q <= 0.5]
        high_quantiles = [q for q in unique_quantiles if q > 0.5]

        # Total locations
        # total_locations = extremes.notnull().sum(dim="location")
        # total_locations = len(extremes.location)
        total_locations = extremes.notnull().sum(dim="location")

        # Compute daily percentage for each quantile
        daily_percentages = {}
        for q in unique_quantiles:
            daily_percentages[q] = (
                (extremes == q).sum(dim="location").values / total_locations * 100
            )  # Convert to NumPy array

        # Define color maps with correct intensity
        reds = cm.get_cmap(
            "Reds", len(low_quantiles)
        )  # Normal Reds (more intense for larger values)
        blues = cm.get_cmap(
            "Blues", len(high_quantiles)
        )  # Normal Blues (more intense for larger values)

        # Convert time values to NumPy array
        time_values = extremes["time"].values

        # Create a single figure
        plt.figure(figsize=(25, 12))
        plt.rcParams.update({"font.size": 18})
        # Stack the lower quantiles in the negative direction
        bottom = np.zeros_like(time_values, dtype=float)
        low_handles, low_labels = [], []
        for i, q in enumerate(
            low_quantiles[::-1]
        ):  # Reverse order to have most intense at the bottom
            percentage_values = np.array(
                daily_percentages[q]
            )  # Ensure it's a NumPy array
            handle = plt.fill_between(
                time_values,
                -bottom,
                -(bottom + percentage_values),
                color=reds((i + 2) / (len(low_quantiles) + 2)),
            )
            low_handles.append(handle)
            low_labels.append(f"{q*100} %")
            bottom += percentage_values

        # Stack the higher quantiles in the positive direction
        bottom = np.zeros_like(time_values, dtype=float)
        high_handles, high_labels = [], []
        for i, q in enumerate(
            high_quantiles
        ):  # Keep normal order to have most intense at the top
            percentage_values = np.array(
                daily_percentages[q]
            )  # Ensure it's a NumPy array
            handle = plt.fill_between(
                time_values,
                bottom,
                bottom + percentage_values,
                color=blues((i + 2) / (len(high_quantiles) + 2)),
            )
            high_handles.append(handle)
            high_labels.append(f"{q*100} %")
            bottom += percentage_values

        plt.ylim(-100, 100)
        plt.xlabel("Time")
        plt.ylabel("Percentage (%)")
        plt.title("Stacked Daily Percentage of Quantiles")
        plt.axhline(0, color="black", linewidth=1)
        plt.grid()

        # Add separate legends on the right
        # Create legends with precise positioning
        high_legend = plt.legend(
            handles=high_handles,
            labels=high_labels,
            loc="upper right",
            bbox_to_anchor=(0.14, 1),
            title="Quantiles (Above)",
        )
        low_legend = plt.legend(
            handles=low_handles,
            labels=low_labels,
            loc="lower right",
            bbox_to_anchor=(1, 0),
            title="Quantiles (Bellow)",
        )

        # Add both legends to the plot
        plt.gca().add_artist(high_legend)
        plt.gca().add_artist(low_legend)

        plt.savefig(self.saving_path / "combined_quantiles_percentages.png")
        plt.show()


if __name__ == "__main__":
    args = parser_arguments().parse_args()

    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-14_12:48:54_full_fluxnet_therightone_highveg"

    subfolders = [
        # "30TVK_157southwest1260_combine.zarr",
        # "32SME_345southwest1260_combine.zarr",
        # "33SVB_095southwest1260_combine.zarr",
        # "37TDL_255southwest1260_combine.zarr",
        # "37UDB_063southwest1260_combine.zarr",
        "S2_38.3598_22.1618_34SEH_390.zarr",
        "S2_55.0510_-1.8846_30UWG_261.zarr",
    ]

    # subfolders = [
    #    # "DE-Tha_50.96_13.57_v0.zarr",
    #    # "DE-HoH_52.09_11.22_v0.zarr",
    #    # "DE-Obe_50.79_13.72_v0.zarr",
    #    # "DE-Hzd_50.96_13.49_v0.zarr",
    #    "custom_cube_50.90_11.56.zarr",
    #    "DE-Hai_51.08_10.45_v0.zarr",
    #    # "DE-RuS_50.87_6.45_v0.zarr",
    #    ##    "ES-LM1_39.94_-5.78_v0.zarr",
    #    ##    # "ES-LM2_39.93_-5.78_v0.zarr",
    #    ##    # # "ES-LMa_39.94_-5.77_v0.zarr",
    #    "ES-Cnd_37.91_-3.23_v0.zarr",
    #    "FR-LGt_47.32_2.28_v0.zarr",
    #    "DE-Lnf_51.33_10.37_v0.zarr",
    #    "DE-Geb_51.10_10.91_v0.zarr",
    #    "DE-Wet_50.45_11.46_v0.zarr",
    #    "DE-Bay_50.14_11.87_v0.zarr",
    #    "DE-Meh_51.28_10.66_v0.zarr",
    #    "custom_cube_44.17_5.24.zarr",
    #    "custom_cube_44.24_5.14.zarr",
    #    "custom_cube_47.31_0.18.zarr",
    #    # "UK-ESa_55.91_-2.86_v0.zarr",
    #    # "AT-Neu_47.12_11.32_v0.zarr",
    # ]  #
    ## subfolders = [
    ##    "ES-LM1_39.94_-5.78_v0.zarr",
    #    "ES-LM2_39.93_-5.78_v0.zarr",
    #    "ES-LMa_39.94_-5.77_v0.zarr",
    #    "DE-Hai_51.08_10.45_v0.zarr",
    #    "ES-Cnd_37.91_-3.23_v0.zarr",
    #    "FR-LGt_47.32_2.28_v0.zarr",
    #    # "DE-Lnf_51.33_10.37_v0.zarr",
    #    # "DE-Geb_51.10_10.91_v0.zarr",
    #    # "DE-Wet_50.45_11.46_v0.zarr",
    #    # "DE-Bay_50.14_11.87_v0.zarr",
    #    # "DE-Meh_51.28_10.66_v0.zarr",
    #    # "custom_cube_50.90_11.56.zarr",
    #    # "custom_cube_44.17_5.24.zarr",
    #    # "custom_cube_44.24_5.14.zarr",
    #    # "custom_cube_47.31_0.18.zarr",
    # ]
    # subfolders = [
    # #  subfolders = [
    #      ""
    #      # "IT-Tor_45.84_7.58_v0.zarr"
    #      # "customcube_CO-MEL_1.95_-72.60_S2_v0.zarr/customcube_CO-MEL_1.95_-72.60_S2_v0.zarr"
    #      "ES-Cnd_37.91_-3.23_v0.zarr",
    #      "DE-RuS_50.87_6.45_v0.zarr",
    #  ]
    quantiles = [
        #    0.025,
        #    0.05,
        0.10,
        #    0.2,
        #    0.3,
        #    0.4,
        #    0.50,
        #    0.501,
        #    0.6,
        0.7,
        #    0.8,
        #    0.9,
        #    0.95,
        #    0.975,
    ]
    for minicube_name in subfolders:
        config = InitializationConfig(args)
        plot = PlotsSentinel2(config=config, minicube_name=minicube_name)
        # try:
        #    plot.plot_raoq()
        # except:
        #    print(f"error with {minicube_name}")

        plot.plot_location_in_europe()
        # for quantile in quantiles:
        #     # plot.plot_thresholds_rmse(quantile)
        #     plot.plot_thresholds(quantile)
        plot.plot_minicube_eco_clusters()
        plot.plot_minicube_pca_projection()
        plot.plot_extremes()
        # plot.plot_rgb()
        plot.plot_msc(colored_by_eco_cluster=True)

# for minicube_name in subfolders:
#     config = InitializationConfig(args)
#     plot = PlotsSentinel2(config=config, minicube_name=minicube_name)
#     # plot.plot_msc(colored_by_eco_cluster=True)
#     plot.plot_rgb()
#     # plot.plot_3D_pca()
# plot.map_component()
#
# plot.map_component(colored_by_eco_cluster=False)
# plot.plot_3D_pca_landcover()
