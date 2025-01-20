import os

from common_imports import *
from RegionalExtremesPackage.plots.base import Plots
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class PlotsSentinel2(Plots):
    def normalize(self, data):
        # Normalize the explained variance
        def _normalization(index):
            band = data.isel(component=index).values
            return (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band))

        normalized_red = _normalization(0)  # Red is the first component
        normalized_green = _normalization(1)  # Green is the second component
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

        # Create the plot
        fig, ax = plt.subplots(
            figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        ax.scatter(
            longitudes,
            latitudes,
            color=rgb_colors,
            s=0.1,  # Size of the point
            transform=ccrs.PlateCarree(),
        )

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
        saving_path = self.saving_path / "landcover.png"
        plt.savefig(saving_path, bbox_inches="tight")  # Ensures no text is cut off
        plt.show()
        return

    def plot_msc(self):
        # Load and preprocess the time series dataset
        data = self.loader._load_data("msc")

        # Randomly select n indices from the location dimension
        print(len(data.location))
        random_indices = np.random.choice(len(data.location), size=1000, replace=False)

        # Use isel to select the subset of data based on the random indices
        subset = data.isel(location=random_indices)

        data = self.loader._load_pca_projection()
        data = data.isel(location=random_indices)

        # Normalize the explained variance
        rgb_colors = self.normalize(data)

        # Plot the time series with corresponding colors
        fig, ax = plt.subplots(figsize=(15, 10))
        # plt.figure(figsize=(15, 10))
        for i, loc in enumerate(subset.location):
            ax.plot(
                subset.dayofyear,
                subset.msc.isel(location=i),
                color=tuple(rgb_colors[i]),
                label=f"Loc: {loc}",
                alpha=0.8,
            )

        # Add labels and optional legend
        plt.title("MSC Time Series Colored by PCA Components")
        plt.xlabel("Day of Year")
        plt.ylabel("MSC Value")
        saving_path = self.saving_path / "msc4.png"
        plt.savefig(saving_path)
        plt.show()

    def plot_minicube_eco_clusters(self, minicube_name):
        data = self.loader._load_data("eco_clusters")
        data = data.eco_clusters
        data = data.unstack("location")

        rgb_colors = self.normalize(data)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        ax.pcolormesh(
            data.longitude.values,
            data.latitude.values,
            rgb_colors.T,
        )
        # Add a title
        plt.title("Eco-clusters")
        saving_path = self.saving_path / "eco_clusters.png"
        plt.savefig(saving_path)
        plt.show()


if __name__ == "__main__":
    args = parser_arguments().parse_args()

    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-01-16_12:19:12_deep_extreme_global"  # "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/RegionalExtremesPackage/experiments/2024-12-19_13:52:48_deep_extreme_HR"
    config = InitializationConfig(args)
    plot = PlotsSentinel2(config=config, minicube_name="mc_25.61_44.32_1.3_20231018_0")
    # plot.plot_minicube_eco_clusters("mc_25.61_44.32_1.3_20231018_0")
    # plot.map_component()
    # plot.map_component(colored_by_eco_cluster=False)
    plot.plot_msc()
    # plot.plot_3D_pca_landcover()
