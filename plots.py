###
# plot file to plot data, experiments etc.
# Draft, the code is not made to be shared and reused!
###
import os


import xarray as xr
import numpy as np
import matplotlib
import datetime

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider
import cartopy
import random

from config import InitializationConfig
from RegionalExtremesPackage.utils.loader import Loader, Saver
from main import parser_arguments
from RegionalExtremesPackage.utils.logger import printt
from RegionalExtremesPackage.Datahandlers.base import create_handler


class PlotExtremes(InitializationConfig):
    def __init__(
        self,
        config: InitializationConfig,
    ):
        """
        Initialize PlotExtremes.

        """
        self.config = config
        self.saving_path = self.config.saving_path / "plots"
        self.saving_path.mkdir(parents=True, exist_ok=True)
        self.loader = Loader(config)

    def _load_pca_projection(self):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        projection_path = self.config.saving_path / "pca_projection.zarr"
        data = xr.open_zarr(projection_path)
        pca_projection = data.pca
        explained_variance = data.explained_variance
        printt("Projection loaded from {}".format(projection_path))
        return pca_projection, explained_variance

    def map_component(self, normalization=False):
        """Map of the PCA component in RBG."""
        pca_projection, explained_variance = self.loader._load_pca_projection(
            explained_variance=True
        )
        pca_projection = pca_projection.unstack("location")
        # Normalize the explained variance
        normalized_variance = (
            explained_variance.explained_variance
            / explained_variance.explained_variance.sum()
        )

        # Normalize the data to the range [0, 1]
        def _normalization(index, normalization):
            band = pca_projection.isel(component=index).values

            normalized_band = (band - np.nanmin(band)) / (
                np.nanmax(band) - np.nanmin(band)
            )
            ## We normalize with the 5% and 95% quantiles due to outliers.
            # normalized_band = (band - np.nanquantile(band, q=0.05)) / (
            #    np.nanquantile(band, q=0.95) - np.nanquantile(band, q=0.05)
            # )
            # print(
            #     np.nanmin(band),
            #     np.nanmax(band),
            #     np.nanquantile(band, q=0.05),
            #     np.nanquantile(band, q=0.95),
            # )
            # # Normalization of the color by feature importance
            if normalization:
                normalized_band = (
                    normalized_band * normalized_variance.sel(component=index).values
                )

            return normalized_band

        normalized_red = _normalization(
            0, normalization=normalization
        )  # Red is the first component
        normalized_green = _normalization(
            1, normalization=normalization
        )  # Green is the second component
        normalized_blue = _normalization(
            2, normalization=normalization
        )  # blue is the third component

        # Stack the components into a 3D array
        rgb_normalized = np.dstack((normalized_red, normalized_green, normalized_blue))
        # Transpose the array
        rgb_normalized = np.transpose(rgb_normalized, (1, 0, 2))

        # Set up the map projection
        projection = cartopy.crs.PlateCarree()

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={"projection": projection})

        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Add coastlines and set global extent
        ax.coastlines()
        ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")

        # Plot the RGB data
        img_extent = (
            pca_projection.longitude.min(),
            pca_projection.longitude.max(),
            pca_projection.latitude.min(),
            pca_projection.latitude.max(),
        )

        ax.set_extent(img_extent, crs=projection)

        ax.pcolormesh(
            pca_projection.longitude.values,
            pca_projection.latitude.values,
            rgb_normalized,
            transform=projection,
        )

        # Add a title
        plt.title("RGB Components on Earth Map")

        map_saving_path = self.saving_path / "map_components_rgb_quantile_2.png"
        plt.savefig(map_saving_path)
        printt("Plot saved")

        # Show the plot
        # plt.show()

    def map_bins(self, normalization=False):
        """Map of the bins in RBG."""
        bins = self.loader._load_bins().T
        bins = bins.unstack("location")

        # Normalize the explained variance
        # Normalize the data to the range [0, 1]
        def _normalization(index):
            band = bins.isel(component=index).values
            return (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band))

        normalized_red = _normalization(0)  # Red is the first component
        normalized_green = _normalization(1)  # Green is the second component
        normalized_blue = _normalization(2)  # blue is the third component
        # For the blue component, we'll use a constant value or a combination of red and green
        # Option 1: Constant blue (e.g., 0)
        # normalized_blue = np.zeros_like(normalized_red)
        # Stack the components into a 3D array
        rgb_normalized = np.dstack((normalized_red, normalized_green, normalized_blue))
        # Transpose the array
        rgb_normalized = np.transpose(rgb_normalized, (1, 0, 2))

        # Set up the map projection
        projection = cartopy.crs.PlateCarree()

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={"projection": projection})

        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Add coastlines and set global extent
        # ax.coastlines()
        # ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")

        # Plot the RGB data
        # img_extent = (
        #     bins.longitude.min(),
        #     bins.longitude.max(),
        #     bins.latitude.min(),
        #     bins.latitude.max(),
        # )

        ax.pcolormesh(
            bins.longitude.values,
            bins.latitude.values,
            rgb_normalized,
            # transform=projection,
        )

        # Add a title
        plt.title("RGB Components on Earth Map")

        map_saving_path = self.saving_path / "map_bins.png"
        plt.savefig(map_saving_path)
        printt("Plot saved")

        # Show the plot
        # plt.show()

    def map_modis(self):
        "Map vegetation index per month of modis."

        dataset_processor = create_handler(config=self.config, n_samples=None)
        data = dataset_processor.preprocess_data(scale=False, remove_nan=False)

        data = data.sel(
            time=slice(datetime.date(2018, 1, 1), datetime.date(2018, 12, 31))
        )
        data["dayofyear"] = data.time.dt.dayofyear

        msc = dataset_processor.preprocess_data().msc

        msc = msc.sel(dayofyear=data.dayofyear)
        data = data - msc
        # Set up the map projection
        projection = cartopy.crs.PlateCarree()

        for month in range(5, 9):
            if month < 10:
                data_day = data.sel(time=f"2018-0{month}-15T12:00:00.000000000")
            else:
                data_day = data.sel(time=f"2018-{month}-15T12:00:00.000000000")

            # Create the figure and axes
            fig, ax = plt.subplots(
                figsize=(12, 5),
                subplot_kw={"projection": projection},
            )

            # adjust the plot
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

            # Add coastlines and set global extent
            ax.coastlines()
            # ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")

            # Plot the RGB data
            img_extent = (
                data_day.longitude.min(),
                data_day.longitude.max(),
                data_day.latitude.min(),
                data_day.latitude.max(),
            )
            ax.set_extent(img_extent, crs=projection)

            print(
                data_day.longitude.values.shape,
                data_day.latitude.values.shape,
                data_day.values.shape,
            )

            im = ax.pcolormesh(
                data_day.longitude.values,
                data_day.latitude.values,
                data_day.values,
                transform=projection,
                cmap="viridis",
                vmin=-0.15,
                vmax=0.15,
            )

            # Add a colorbar
            cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.08)
            cbar.set_label(f"{self.config.index} Anomaly (EVI - MSC)")

            # Add a title
            title = ax.set_title(f"Modis data - Month {month}")

            map_saving_path = (
                self.saving_path / f"modis_2018/map_anomaly_medsc_{month}.png"
            )
            plt.savefig(map_saving_path)
            printt("Plot saved")

            # Show the plot
            # plt.show()

    def map_modis_slidder(self):
        "Map modis with a slidder. In progress do not work."
        initial_day = 0
        dataset_processor = create_handler(config=self.config, n_samples=10)
        data = dataset_processor.preprocess_data(scale=False).msc
        data = data.set_index(location=["longitude", "latitude"]).unstack("location")
        # Set up the map projection
        projection = cartopy.crs.PlateCarree()

        # Create the figure and axes
        fig, (ax, slider_ax) = plt.subplots(
            nrows=2,
            figsize=(12, 6),
            gridspec_kw={"height_ratios": [20, 1]},
            subplot_kw={"projection": projection},
        )

        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Add coastlines and set global extent
        ax.coastlines()
        # ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")

        # Plot the RGB data
        img_extent = (
            data.longitude.min(),
            data.longitude.max(),
            data.latitude.min(),
            data.latitude.max(),
        )
        ax.set_extent(img_extent, crs=projection)

        im = ax.imshow(
            data.isel(dayofyear=initial_day).values,
            origin="lower",
            extent=img_extent,
            transform=projection,
            cmap="viridis",
        )

        ax.imshow(
            data.isel(dayofyear=initial_day).values,
            origin="lower",
            extent=img_extent,
            transform=projection,
            cmap="viridis",
        )
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.08)
        cbar.set_label(f"{self.config.index}")

        # Add a title
        title = ax.set_title(f"Modis data - Day of Year: {initial_day}")

        # Create the slider
        slider = Slider(slider_ax, "Day of Year", 0, 73, valinit=initial_day, valstep=1)

        # Update function for slider
        def update(val):
            day = int(slider.val)
            im.set_array(data.isel(dayofyear=day).values)
            title.set_text(f"Modis data - Day of Year: {day}")
            fig.canvas.draw_idle()

        slider.on_changed(update)

        map_saving_path = self.saving_path / "map_modis.png"
        plt.savefig(map_saving_path)
        printt("Plot saved")

        # Show the plot
        # plt.show()

    def region(self, indices=None):
        """plot the samples of a single region"""
        boxes = self.loader._load_bins().T
        # Select randomly a first location
        if indices is None:
            lon = random.choice(boxes.longitude.values).item()
            lat = random.choice(boxes.latitude.values).item()
            # Get the boxe indices of the location
            indices = boxes.sel(longitude=lon, latitude=lat).values
        # Create a boolean mask for the subset
        mask = np.all(boxes.values == indices[:, np.newaxis], axis=0)

        # Get the lat and lon values where the mask is true
        masked_lons = boxes.longitude.values[mask]
        masked_lats = boxes.latitude.values[mask]
        masked_lons_lats = list(zip(masked_lons, masked_lats))
        # locations = self.find_bins_origin()
        # masked_lons_lats = list(
        #    zip(locations.longitude.values, locations.latitude.values)
        # )
        # mask = locations

        # indices = "center"

        printt(f"Number of samples in the region {indices}: {len(masked_lons_lats)}.")
        if len(masked_lons_lats) == 0:
            printt(f"No samples in the region {indices}.")
            return

        def map_single_region(mask):
            # Get the subset of data using the mask
            subset = boxes.isel(location=mask)
            # Plot the subset
            # Set up the map projection
            projection = cartopy.crs.PlateCarree()

            # Create the figure and axis
            fig, ax = plt.subplots(
                figsize=(12, 5), subplot_kw={"projection": projection}
            )

            # adjust the plot
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

            # Add coastlines and set global extent
            ax.coastlines()
            ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")

            # Plot the RGB data
            img_extent = (
                boxes.longitude.min(),
                boxes.longitude.max(),
                boxes.latitude.min(),
                boxes.latitude.max(),
            )

            ax.set_extent(img_extent, crs=projection)

            # Plot the points
            # Plot each point individually with a unique color
            for i, (lon, lat) in enumerate(zip(subset.longitude, subset.latitude)):
                ax.scatter(
                    lon,
                    lat,
                    transform=projection,
                    s=10,
                    label=f"({lon:.2f}, {lat:.2f})",
                )

            # Add a title
            plt.title(f"Location of samples of the region {indices}.")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Adjust layout to prevent cutting off the legend
            plt.tight_layout()

            # Save the figure
            map_saving_path = (
                self.saving_path / f"map_location_single_region_{indices}.png"
            )
            plt.savefig(map_saving_path)
            plt.close()
            printt("Plot saved")

            # plt.show()

        def time_series_single_region(masked_lons_lats):
            if len(masked_lons_lats) > 10:
                masked_lons_lats = random.choices(masked_lons_lats, k=36)
                printt(f"Selected locations: {masked_lons_lats}")

            dataloader = create_handler(
                config=config, n_samples=None  # args.n_samples,  # all the dataset
            )
            data = (
                dataloader._dataset_specific_loading()
                .stack(location=("longitude", "latitude"))
                .to_array()
                .squeeze(axis=0)
            )
            data = data.sel(location=masked_lons_lats)
            data = data.chunk({"time": len(data.time), "location": 1})
            # Remove NaNs

            def time_series(data):
                # Create a figure and axis
                fig, ax = plt.subplots(figsize=(12, 8))
                # Plot each location
                for loc in range(data.location.size):
                    lon, lat = masked_lons_lats[loc]
                    ax.plot(
                        data.time,
                        data.isel(location=loc),
                        label=f"({lon:.3f}, {lat:.3f})",
                    )

                # Customize the plot
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{self.config.index}")
                ax.set_ylim(0, 1)
                ax.set_title(f"Time series of locations in the region {indices}")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                # Adjust layout to prevent cutting off the legend
                plt.tight_layout()

                # Show the plot
                saving_path = self.saving_path / f"ts_single_region_{indices}.png"
                plt.savefig(saving_path)

                # plt.show()

            def msc_vsc(data):
                msc = (
                    data.groupby("time.dayofyear")
                    .mean("time", skipna=True)
                    .isel(dayofyear=slice(1, 365))
                )
                msc = msc.chunk({"dayofyear": len(msc.dayofyear), "location": 1})
                vsc = data.groupby("time.dayofyear").var("time", skipna=True)
                vsc = vsc.chunk({"dayofyear": len(vsc.dayofyear), "location": 1}).isel(
                    dayofyear=slice(1, 365)
                )
                # Create a figure and axis
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

                # Plot each location for the first variable
                for loc in range(data.location.size):
                    lon, lat = masked_lons_lats[loc]
                    ax1.plot(
                        msc.dayofyear,
                        msc.isel(location=loc),  # .transpose(),
                        label=f"Location {round(lon, 3), round(lat, 3)}",
                    )
                    ax2.plot(
                        vsc.dayofyear,
                        vsc.isel(location=loc),  # .transpose(),
                        label=f"({lon:.3f}, {lat:.3f})",
                    )

                # Customize the plot
                ax1.set_xlabel("Day of Year")
                ax1.set_ylabel("Mean Seasonal Cycle")
                ax1.set_ylim(0, 1)
                ax1.set_title(f"MSC or Each Location of the region {indices}")
                ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                ax2.set_xlabel("Day of Year")
                ax2.set_ylabel("Variance Seasonal Cycle")
                ax2.set_title(f"VSC for Each Location of the region {indices}")
                ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                plt.tight_layout()
                plt.subplots_adjust(hspace=0.3)

                saving_path = self.saving_path / f"msc_vsc_single_region_{indices}.png"
                plt.savefig(saving_path)

                # plt.show()

            time_series(data)
            msc_vsc(data)

        def plot_3D_pca(masked_lons_lats):
            pca_projection, explained_variance = self.loader._load_pca_projection(
                explained_variance=True
            )
            # pca_subset = pca_projection.sel(location=masked_lons_lats)
            # n_bins = self.config.n_bins
            # box_indices = self.loader._load_bins()

            # Convert box indices to RGB colors
            # Normalize indices to the range [0, 1] for RGB
            # colors = box_indices / (n_bins + 1)
            # Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Scatter plot

            for loc in masked_lons_lats:
                lon, lat = loc
                sc = ax.scatter(
                    pca_projection.sel(component=0, location=loc).values,
                    pca_projection.sel(component=1, location=loc).values,
                    pca_projection.sel(component=2, location=loc).values,
                    s=50,
                    label=f"({lon:.3f}, {lat:.3f})",
                )

            # Adding labels and title
            ax.set_xlabel("PCA Component 1")
            ax.set_xlim(
                pca_projection.sel(component=0).min(),
                pca_projection.sel(component=0).max(),
            )
            ax.set_ylabel("PCA Component 2")
            ax.set_ylim(
                pca_projection.sel(component=1).min(),
                pca_projection.sel(component=1).max(),
            )
            ax.set_zlabel("PCA Component 3")
            ax.set_zlim(
                pca_projection.sel(component=2).min(),
                pca_projection.sel(component=2).max(),
            )
            ax.set_title("3D PCA Projection")
            # Position the legend to the right of the plot
            ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

            saving_path = self.saving_path / f"3D_pca_{indices}.png"
            plt.savefig(saving_path)

            # plt.show()

        # plot_3D_pca(masked_lons_lats)
        map_single_region(mask)
        time_series_single_region(masked_lons_lats)

    def find_bins_origin(self):
        pca_projection, explained_variance = self.loader._load_pca_projection(
            explained_variance=True
        )
        lower_bound = -0.2
        upper_bound = 0.2
        condition = (
            (pca_projection >= lower_bound) & (pca_projection <= upper_bound)
        ).compute()

        pca_projection = pca_projection.where(condition, drop=True)
        condition = ~pca_projection.isnull().any(dim="component").compute()
        pca_projection = pca_projection.where(condition, drop=True)
        return pca_projection.location

    def plot_3D_pca(self):
        pca_projection, explained_variance = self.loader._load_pca_projection(
            explained_variance=True
        )

        n_bins = self.config.n_bins
        box_indices = self.loader._load_bins()
        # Convert box indices to RGB colors
        # Normalize indices to the range [0, 1] for RGB
        colors = box_indices / (n_bins + 1)
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot
        sc = ax.scatter(
            pca_projection.isel(component=0).values.T,
            pca_projection.isel(component=1).values.T,
            pca_projection.isel(component=2).values.T,
            c=colors.values,
            s=50,
            edgecolor="k",
        )

        # Adding labels and title
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        ax.set_title("3D PCA Projection with RGB Colors")
        ax.legend()

        saving_path = self.saving_path / "3D_pca.png"
        plt.savefig(saving_path)

        plt.show()
        return

    def plot_3D_pca_other(self):
        data = xr.open_zarr(
            "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-18_10:05:59_eco_regional_2000_hr_50bins/EVI/pca_projection.zarr"
        )
        pca_projection = data.pca.stack(location=("longitude", "latitude")).transpose(
            "location", "component", ...
        )

        mask = xr.open_zarr(
            "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-17_11:14:50_eco_pca_2016_500bins_regional/EVI/mask.zarr"
        ).EVIgapfilled_QCdyn
        mask = mask.stack(location=("longitude", "latitude"))
        mask_broadcasted = mask.broadcast_like(pca_projection)
        pca_projection = pca_projection.where(mask_broadcasted.compute(), drop=True)
        # pca_projection = pca_projection.sel(location=common_locations)

        # Error computation
        target = xr.open_zarr(
            "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-17_14:40:51_eco_local_2000_hr/EVI/thresholds.zarr"
        ).threshold.sel(quantile=0.025)
        target = target.stack(location=("longitude", "latitude"))
        target = target.sel(location=pca_projection.location)
        prediction = xr.open_zarr(
            "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-18_10:05:59_eco_regional_2000_hr_50bins/EVI/thresholds.zarr"
        ).threshold.sel(quantile=0.025)
        prediction = prediction.stack(location=("longitude", "latitude"))
        prediction = prediction.sel(location=pca_projection.location)
        error = target - prediction
        # Convert box indices to RGB colors
        # Normalize indices to the range [0, 1] for RGB

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Scatter plot
        sc = ax.scatter(
            pca_projection.isel(component=0).values.T,
            pca_projection.isel(component=1).values.T,
            pca_projection.isel(component=2).values.T,
            c=error.values.T,
            s=5,
            cmap="seismic",
            vmin=-0.2,
            vmax=0.2,
        )
        # Adding labels and title
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        ax.set_title("3D PCA Projection with RGB Colors")
        ax.legend()
        plt.show()
        return

    def plot_2D_component(self):
        pca_projection, explained_variance = self.loader._load_pca_projection(
            explained_variance=True
        )

        n_bins = self.config.n_bins
        box_indices = self.loader._load_bins()
        # Convert box indices to RGB colors
        # Normalize indices to the range [0, 1] for RGB
        colors = box_indices / (n_bins + 1)
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plot
        sc = ax.scatter(
            pca_projection.isel(component=0).values.T,
            pca_projection.isel(component=1).values.T,
            c=colors.values,
            s=75,
            edgecolor="k",
        )

        # Adding labels and title
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        # ax.set_zlabel("PCA Component 3")
        ax.set_title("2D PCA Projection with RGB Colors")
        ax.legend()

        saving_path = self.saving_path / "2D_pca.png"
        plt.savefig(saving_path)

    def distribution_per_region(self):
        boxes = self.loader._load_bins()
        boxes = boxes

        # Count occurrences
        unique, counts = np.unique(boxes.values, axis=0, return_counts=True)

        # Sort the unique values and counts in descending order of counts
        sorted_indices = np.argsort(counts)[::-1]
        unique_sorted = unique[sorted_indices][:50]
        counts_sorted = counts[sorted_indices][:50]
        # Convert unique values to strings for labels
        labels = [f"({int(u[0])}, {int(u[1])})" for u in unique_sorted]

        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(counts_sorted)), counts_sorted)
        plt.xlabel("Regions")
        plt.ylabel("Number of samples")
        plt.title("Number of Samples per Region (first 50 regions).")

        # Set x-axis ticks and labels
        plt.xticks(range(len(labels)), labels, rotation=90)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        saving_path = self.saving_path / "nb_samples_per_region.png"
        plt.savefig(saving_path)

        plt.show()

    def region_distribution(self):
        boxes = self.loader._load_bins()
        unique, counts = np.unique(boxes.values, axis=0, return_counts=True)

        # Get unique counts and their frequencies
        unique_counts, count_frequencies = np.unique(counts, return_counts=True)

        # Sort the unique counts and their frequencies
        sort_indices = np.argsort(unique_counts)
        unique_counts_sorted = unique_counts[sort_indices][:50]
        count_frequencies_sorted = count_frequencies[sort_indices][:50]

        # Create the bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(
            unique_counts_sorted, count_frequencies_sorted, edgecolor="black", width=0.8
        )

        plt.xlabel("Region Size (Number of Samples)")
        plt.ylabel("Number of Regions")
        plt.title("Distribution of Region Sizes")

        # Set x-ticks to be exactly at the unique count values
        plt.xticks(unique_counts_sorted)

        # Add grid for better readability
        plt.grid(axis="y", alpha=0.75)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the figure
        saving_path = self.saving_path / "region_size_distribution_discrete_first50.png"
        plt.savefig(saving_path)

        # Show the plot
        plt.show()

    def distribution_per_component(self):
        boxes = self._load_boxes()
        boxes = boxes.T

        # Count occurrences
        pca_projection = self.loader._load_pca_projection()
        n_bins = self.config.n_bins
        box_indices = self.loader._load_bins()

        # Convert box indices to RGB colors
        # Normalize indices to the range [0, 1] for RGB
        colors = box_indices / (n_bins + 1)
        unique, counts = np.unique(boxes.values, axis=0, return_counts=True)

        pass

    def threshold_time_serie(self):

        # quantile_array = quantile_array.where(mask, subset_quantiles_array)
        # quantile_array.isel(location=mask) = subset_quantiles_array
        # Define the colormap for different quantile levels
        colors = {
            0.025: "blue",  # Color for below the 0.025 quantile
            0.975: "red",  # Color for above the 0.975 quantile
            # Add more color mappings as needed for other quantile levels
        }

        # Create a base figure
        plt.figure(figsize=(10, 6))
        subset_data = subset_data[0, :]

        plt.plot(
            subset_data["time"],
            subset_data,
            label="Original Data",
            color="black",
        )

        # Plot the deseasonalized data
        plt.plot(
            subset_data["time"],
            deseasonalized[0, :],
            label="Deseasonalized Data",
            color="black",
            zorder=1,
        )
        plt.scatter(
            subset_data["time"],
            subset_quantiles_array[0, :],
            label="Quantiles",
            color="red",
            s=20,
        )

        # Loop over time intervals and apply color spans for extreme quantiles
        # for i in range(len(subset_data["time"]) - 1):
        #    # Check the quantile value at each time step
        #    quantile_level = quantile_values[i]
        #
        #    # If the quantile level corresponds to one of the defined extremes (e.g., 0.025 or 0.975)
        #    if quantile_level in colors:
        #        # Highlight the time span between two time points
        #        plt.axvspan(
        #            subset_data["time"].values[i],  # Start of the time interval
        #            subset_data["time"].values[
        #                i + 1
        #            ],  # End of the time interval
        #            color=colors[
        #                quantile_level
        #            ],  # Color based on the quantile level
        #            alpha=0.3,  # Transparency of the shaded region
        #        )
        #
        ## Add title and labels
        plt.title("Deseasonalized Data with Extreme Quantile Color Coding")
        plt.xlabel("Time")
        plt.ylabel("Deseasonalized Values")

        # Add legend manually for quantile colors
        # handles = [
        #     plt.Line2D([0], [0], color=color, lw=4) for color in colors.values()
        # ]
        # labels = [f"Quantile {level}" for level in colors.keys()]
        plt.legend()

        # Show the plot
        plt.savefig(
            "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-09-04_15:18:47_eco_50bins/EVI/plots/extremes.png"
        )

    def extremes_plots(self, date):
        extremes = self.loader._load_extremes().EVIgapfilled_QCdyn
        print(extremes)
        # mask = self.loader._load_spatial_masking().EVIgapfilled_QCdyn
        # print(extremes)
        # print(mask)
        # mask_broadcasted = mask.broadcast_like(extremes)
        # extremes = extremes.where(mask_broadcasted, np.nan)
        # Set up the map projection
        projection = cartopy.crs.PlateCarree()
        data_day = extremes.sel(time=f"{date}T12:00:00.000000000", method="nearest")

        fig, ax = plt.subplots(
            figsize=(12, 5),
            subplot_kw={"projection": projection},
        )
        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # Add coastlines and set global extent
        ax.coastlines()
        ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")

        # Plot the RGB data
        img_extent = (
            data_day.longitude.min(),
            data_day.longitude.max(),
            data_day.latitude.min(),
            data_day.latitude.max(),
        )

        # Define a colormap with distinct colors for each data value
        colors = [
            "#d73027",
            "#fc8d59",
            "#fee08b",
            "#d9ef8b",
            "#91cf60",
            "#1a9850",
        ]
        cmap = ListedColormap(colors)

        # Define boundaries based on your specific data values
        boundaries = [0.01, 0.025, 0.05, 0.95, 0.975, 0.99, 1]
        norm = BoundaryNorm(boundaries, cmap.N)

        ax.set_extent(img_extent, crs=projection)
        im = ax.pcolormesh(
            data_day.longitude.values,  # Test with a smaller subset
            data_day.latitude.values,
            data_day.values.T,  # Ensure C is 1 smaller
            transform=projection,
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        # Add a colorbar
        cbar = plt.colorbar(
            im, ax=ax, orientation="vertical", pad=0.08, ticks=boundaries
        )
        cbar.set_label(f"{self.config.index} Extremes")
        cbar.set_ticks(boundaries[:-1])  # Use the first 6 boundaries (exclude 1.0)
        cbar.set_ticklabels([f"{b:.3f}" for b in boundaries[:-1]])

        # Add a title
        title = ax.set_title(f"Regional extremes {date}")
        map_saving_path = self.saving_path / f"extremes_{date}.png"
        plt.savefig(map_saving_path)
        printt("Plot saved")
        # Show the plot
        plt.show()

    def check_number(self, config, date):
        extremes = self.loader._load_extremes().EVIgapfilled_QCdyn
        dataset_processor = create_handler(
            config=config, n_samples=None
        )  # all the dataset
        condition = ~extremes.isnull().all(dim="time").compute()
        extremes = extremes.where(condition, drop=True)
        extremes = dataset_processor._spatial_filtering(extremes)

        extremes = extremes.sel(time=f"{date}T12:00:00.000000000", method="nearest")
        print(extremes.values.shape)
        unique, counts = np.unique(extremes.values, return_counts=True)
        print(unique)
        print((counts / sum(counts)) * 100)

    def threshold(self):
        args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-02_15:04:53_eco_threshold_uniform_2000"
        config = InitializationConfig(args)
        print("first", config.path_load_experiment)
        threshold_local2000 = self.loader._load_thresholds().threshold.sel(
            quantile=0.05
        )
        # mask = self.loader._load_spatial_masking().EVIgapfilled_QCdyn
        # mask_broadcasted = mask.broadcast_like(threshold_local2000)
        # threshold_local2000 = threshold_local2000.where(mask_broadcasted, np.nan)
        threshold_local2000 = threshold_local2000.stack(
            location=("longitude", "latitude")
        )
        print(threshold_local2000.values[5300:5500])
        args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-01_14:53:47_eco_threshold_2016"
        config = InitializationConfig(args)
        print("2", config.path_load_experiment)
        threshold_regional2000 = self.loader._load_thresholds().threshold.sel(
            quantile=0.05
        )
        threshold_regional2000 = threshold_regional2000.stack(
            location=("longitude", "latitude")
        )
        print(threshold_regional2000.values[5300:5500])
        common_locations = np.intersect1d(
            threshold_local2000.location, threshold_regional2000.location
        )
        threshold = threshold_local2000 - threshold_regional2000
        threshold = threshold.set_index(location=["longitude", "latitude"]).unstack(
            "location"
        )

        projection = cartopy.crs.PlateCarree()

        # Create the figure and axes
        fig, ax = plt.subplots(
            figsize=(12, 5),
            subplot_kw={"projection": projection},
        )
        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # Add coastlines and set global extent
        ax.coastlines()
        # ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")
        # Plot the RGB data
        img_extent = (
            threshold.longitude.min(),
            threshold.longitude.max(),
            threshold.latitude.min(),
            threshold.latitude.max(),
        )
        ax.set_extent(img_extent, crs=projection)

        im = ax.pcolormesh(
            threshold.longitude.values,
            threshold.latitude.values,
            threshold.values.T,
            transform=projection,
            cmap="viridis",
            # vmin=-0.15,
            # vmax=0.15,
        )
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.08)
        cbar.set_label(f"Threshold")
        # Add a title
        title = ax.set_title(f"Regional thresholds")
        map_saving_path = self.saving_path / f"threshold_diff.png"
        print(map_saving_path)
        plt.savefig(map_saving_path)
        print("Plot saved")
        # Show the plot
        plt.show()

    def plot_mask(self):
        mask = self.loader._load_spatial_masking().EVIgapfilled_QCdyn
        projection = cartopy.crs.PlateCarree()

        # Create the figure and axes
        fig, ax = plt.subplots(
            figsize=(12, 5),
            subplot_kw={"projection": projection},
        )
        # adjust the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # Add coastlines and set global extent
        ax.coastlines()
        # ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")
        # Plot the RGB data
        img_extent = (
            mask.longitude.min(),
            mask.longitude.max(),
            mask.latitude.min(),
            mask.latitude.max(),
        )
        ax.set_extent(img_extent, crs=projection)
        print(
            mask.values.shape, mask.longitude.values.shape, mask.latitude.values.shape
        )
        im = ax.pcolormesh(
            mask.longitude.values,
            mask.latitude.values,
            mask.values.T,
            transform=projection,
            cmap="viridis",
            # vmin=-0.15,
            # vmax=0.15,
        )
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.08)
        cbar.set_label(f"Threshold")
        # Add a title
        title = ax.set_title(f"Regional thresholds")
        map_saving_path = self.saving_path / f"mask.png"
        plt.savefig(map_saving_path)
        print("Plot saved")
        # Show the plot
        plt.show()

    def barplot_error_thresholds(self):
        args = parser_arguments().parse_args()

        args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-01_14:53:47_eco_threshold_2016"
        config = InitializationConfig(args)
        print(args.path_load_experiment)
        threshold_regional2000 = self.loader._load_thresholds().threshold
        threshold_regional2000 = threshold_regional2000.stack(
            location=("longitude", "latitude")
        )
        threshold_regional2000 = threshold_regional2000.dropna(
            dim="location", how="all"
        )
        print(threshold_regional2000.values.shape)
        # args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-01_14:53:47_eco_threshold_2016"
        # config = InitializationConfig(args)
        # threshold_regional2016 = self.loader._load_thresholds().threshold
        # threshold_regional2016 = threshold_regional2016.stack(
        #     location=("longitude", "latitude")
        # )
        # args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-02_14:53:56_eco_threshold_uniform_2016"
        # config = InitializationConfig(args)
        # threshold_local2016 = self.loader._load_thresholds().threshold
        # threshold_local2016 = threshold_local2016.stack(
        #     location=("longitude", "latitude")
        # )

        # Ground Truth
        args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-10-02_15:04:53_eco_threshold_uniform_2000"
        config = InitializationConfig(args)
        print(args.path_load_experiment)
        threshold_local2000 = self.loader._load_thresholds().threshold
        # mask = self.loader._load_spatial_masking().EVIgapfilled_QCdyn
        # mask_broadcasted = mask.broadcast_like(threshold_local2000)
        # threshold_local2000 = threshold_local2000.where(mask_broadcasted, np.nan)
        threshold_local2000 = threshold_local2000.stack(
            location=("longitude", "latitude")
        )
        threshold_local2000 = threshold_local2000.dropna(dim="location", how="all")

        rmse_data1 = calculate_rmse(
            threshold_local2000,
            threshold_regional2000,
        )
        print(rmse_data1)

        # Get quantile values
        quantiles = threshold_local2000.coords["quantile"].values
        print(quantiles)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(quantiles, rmse_data1, label="Regional 2016")
        plt.plot(quantiles, rmse_data2, label="Local 2016")
        plt.xlabel("Quantile")
        plt.ylabel("RMSE")
        plt.title("RMSE vs Quantile for Different Datasets")
        plt.legend()
        plt.grid(True)

        plt.show()


def calculate_rmse(truth, data):
    # Calculate the squared difference
    common_quantiles = np.intersect1d(truth.location, data.location)
    return np.mean(
        (
            truth.sel(location=common_quantiles, quantile=0.05).values
            - data.sel(location=common_quantiles, quantile=0.05).values
        )
    )


if __name__ == "__main__":
    args = parser_arguments().parse_args()

    args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-11-25_18:41:43_deep_extreme_HR"
    config = InitializationConfig(args)
    # loader = Loader(config)
    # print(loader._load_pca_matrix().explained_variance_ratio_)
    # limits_bins = loader._load_limits_bins()
    # print(limits_bins)
    plot = PlotExtremes(config=config)
    plot.map_bins()
    # plot.plot_3D_pca()
    # plot.plot_2D_component()
    # plot.map_component()
    # plot.map_bins()

    # plot.plot_2D_component()

    # plot.find_bins_origin()

    # indices = np.array([20, 1, 1])
    # plot.region(indices=indices)
    #
    # indices = np.array([21, 1, 1])
    # plot.region(indices=indices)
    #
    # indices = np.array([3, 2, 1])
    # plot.region(indices=indices)
    #
    # indices = np.array([6, 2, 1])
    # plot.region(indices=indices)
    #
    # indices = np.array([6, 2, 2])
    # plot.region(indices=indices)
    #
    # indices = np.array([9, 1, 2])
    # plot.region(indices=indices)
    # indices = np.array([12, 1, 1])
    # plot.region(indices=indices)load_thresholds().__xarray_dataarray_variable__

    plot.region_distribution()
    plot.distribution_per_region()
    # plot.map_modis()
    # plot.map_bins()
    # plot.threshold_time_serie()
    # plot.check_number(config, "2018-08-15")
    # plot.barplot_error_thresholds()
    # plot.threshold()
    # # plot.plot_mask()
    # plot.extremes_plots("2022-08-15")
    # plot.extremes_plots("2021-08-15")
    # plot.extremes_plots("2020-08-15")
    # plot.extremes_plots("2019-08-15")
    # plot.extremes_plots("2018-08-15")
    # plot.extremes_plots("2017-08-15")
    # plot.extremes_plots("2016-08-15")
    # plot.extremes_plots("2003-08-15")
    # plot.extremes_plots("2004-08-15")
