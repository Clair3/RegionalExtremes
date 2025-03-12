import os

from common_imports import *
from RegionalExtremesPackage.plots.base import Plots
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import rioxarray as rio

plt.rcParams.update({"font.size": 20})


class PlotsSentinel2(Plots):

    def normalize(self, data):
        # Normalize the explained variance
        def _normalization(index):
            band = data.isel(component=index).values
            band_min = np.quantile(band, 0.1)
            band_max = np.quantile(band, 0.90)
            # return np.clip((band - band_min) / (band_max - band_min), 0, 1)
            return (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band))

        normalized_red = _normalization(1)  # Red is the first component
        normalized_green = _normalization(0)  # Green is the second component
        normalized_blue = _normalization(2)  # blue is the third component

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
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.OCEAN)

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

        plt.xlabel("Day of Year")
        plt.ylabel("MSC Value")
        plt.legend()
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
        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        ax.pcolormesh(
            bins.longitude.values,
            bins.latitude.values,
            rgb_normalized.transpose(1, 0, 2),
            # transform=projection,
        )

        # Add a title
        plt.title("Eco-clusters")
        saving_path = self.saving_path / "eco_clusters_2.png"
        plt.savefig(saving_path)
        plt.show()

    def plot_rgb(self):
        path = glob.glob(
            f"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_test/{self.minicube_name}*"
        )[0]
        ds = xr.open_zarr(path)
        if "time" not in ds.dims:
            ds = ds.rename({"time_sentinel-2-l2a": "time"})

        # The rgb channel need to be normalise and enlightened.
        def normalize(band):
            # band_min, band_max = (band.min(), band.max())
            band_min = np.quantile(band, 0.02)
            band_max = np.quantile(band, 0.98)
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
                green = ds.isel(time=t).B8A
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

    def plot_thresholds(self):
        data = self.loader._load_data("thresholds").thresholds
        data = data.unstack("location")

        fig, ax = plt.subplots(figsize=(12, 10))
        # Adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Plot the data with a colormap
        pcm = ax.pcolormesh(
            data.longitude.values.T,
            data.latitude.values.T,
            data.sel(quantile=0.05).values.T,
            cmap="inferno",  # viridis",  # Choose a colormap, e.g., 'viridis', 'plasma', 'coolwarm'
        )

        # Add a title
        plt.title("Quantile 5%")

        # Add a colorbar
        pcm.set_clim(-0.15, -0.05)

        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("Value")  # Set the label for the colorbar
        print(self.saving_path)
        saving_path = self.saving_path / "thresholds.png"
        plt.savefig(saving_path)
        plt.show()


if __name__ == "__main__":
    args = parser_arguments().parse_args()

    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-02-25_00:52:19_Final_20"

    subfolders = [
        "FR-LGt_47.32_2.28_v0.zarr"
        # "custom_cube_50.90_11.56.zarr",
        # "custom_cube_44.17_5.24.zarr",
        # "custom_cube_44.24_5.14.zarr",
        # "custom_cube_47.31_0.18.zarr",
        # "DE-Hai_51.08_10.45_v0.zarr",
        # "ES-Cnd_37.91_-3.23_v0.zarr",
        # "DE-Geb_51.10_10.91_v0.zarr",
        # "DE-Wet_50.45_11.46_v0.zarr",
        # "DE-Bay_50.14_11.87_v0.zarr",
        # "DE-Meh_51.28_10.66_v0.zarr",
        # "DE-Lnf_51.33_10.37_v0.zarr",
        # ]
        # subfolders = [
    ]
    # #  subfolders = [
    #      ""
    #      # "IT-Tor_45.84_7.58_v0.zarr"
    #      # "customcube_CO-MEL_1.95_-72.60_S2_v0.zarr/customcube_CO-MEL_1.95_-72.60_S2_v0.zarr"
    #      "ES-Cnd_37.91_-3.23_v0.zarr",
    #      "DE-RuS_50.87_6.45_v0.zarr",
    #  ]
    for minicube_name in subfolders:
        config = InitializationConfig(args)
        plot = PlotsSentinel2(config=config, minicube_name=minicube_name)
        plot.plot_location_in_europe()
        plot.plot_msc(colored_by_eco_cluster=True)
        plot.plot_thresholds()
        plot.plot_minicube_eco_clusters()
        plot.plot_rgb()

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
