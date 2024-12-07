from .common_imports import *
from .base import DatasetHandler


class EcologicalDatasetHandler(DatasetHandler):

    def _dataset_specific_loading(self):
        """
        Preprocess data based on the index.
        """
        if self.config.index in ECOLOGICAL_INDICES:
            filepath = ECOLOGICAL_FILEPATH(self.config.index)
            self.load_data(filepath)
            # self.reduce_resolution()
        else:
            raise NotImplementedError(
                f"Index {self.config.index} unavailable. Ecological Index available: {ECOLOGICAL_INDICES}."
            )
        return self.data

    def load_data(self, filepath=None):
        """
        Load data from the specified filepath.

        Parameters:
        filepath (str): Path to the data file.
        """
        if not filepath:
            filepath = ECOLOGICAL_FILEPATH(self.config.index)

        self.variable_name = VARIABLE_NAME(self.config.index)
        self.data = xr.open_zarr(filepath, consolidated=False)[[self.variable_name]]
        printt("Data loaded from {}".format(filepath))
        self._stackdims()

    def _stackdims(self):
        self.data = self.data.stack(
            {
                "latitude": ["latchunk", "latstep_modis"],
                "longitude": ["lonchunk", "lonstep_modis"],
            }
        )
        self.data = self.data.reset_index(["latitude", "longitude"])
        self.data["latitude"] = self.data.latchunk + self.data.latstep_modis
        self.data["longitude"] = self.data.lonchunk + self.data.lonstep_modis

        self.data = self.data.set_index(latitude="latitude", longitude="longitude")
        self.data = self.data.drop(
            ["latchunk", "latstep_modis", "lonchunk", "lonstep_modis"]
        )

    def reduce_resolution(self):
        res_lat, res_lon = len(self.data.latitude), len(self.data.longitude)
        self.data = self.data.coarsen(latitude=5, longitude=5, boundary="trim").mean()
        printt(
            f"Reduce the resolution from ({res_lat}, {res_lon}) to ({len(self.data.latitude)}, {len(self.data.longitude)})."
        )

    def filter_dataset_specific(self):
        """
        Standarize the ecological xarray. Remove irrelevant area, and reshape for the PCA.
        """
        assert (
            self.config.index in ECOLOGICAL_INDICES
        ), f"Index {self.config.index} unavailable. Index available: {ECOLOGICAL_INDICES}."

        assert self.data is not None, "Data not loaded."

        # Assert dimensions are as expected after loading and transformation
        assert all(
            dim in self.data.sizes for dim in ("time", "latitude", "longitude")
        ), "Dimension missing"

        # Temporal filtering.
        self.data = self.data.sel(
            time=slice(
                datetime.date(self.start_year, 1, 1), datetime.date(2022, 12, 31)
            )
        )
        # Then remove specific years using boolean indexing
        # years_to_exclude = [2003, 2018, 2019, 2020, 2022]
        # self.data = self.data.sel(time=~self.data.time.dt.year.isin(years_to_exclude))

        self.data = self._spatial_filtering(self.data)

        printt(f"Ecological data loaded with dimensions: {self.data.sizes}")

    def _remove_low_vegetation_location(self, threshold, msc):
        # Calculate mean data across the dayofyear dimension
        mean_msc = msc.mean("dayofyear", skipna=True)

        # Create a boolean mask for locations where mean is greater than or equal to the threshold
        mask = mean_msc >= threshold

        return mask
