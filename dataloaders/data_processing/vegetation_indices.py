from ..sentinel2 import Sentinel2Dataloader
from ..modis import ModisDataloader

# TODO


class VegetationIndices:
    def __init__(self, dataloader):
        """
        Initialize the class with a specific dataloader (MODIS or Sentinel2).

        Args:
            dataloader: The data loader object (MODIS or Sentinel2).
        """
        self.dataloader = dataloader

        # Checking the type of dataloader to adjust index computation accordingly
        if isinstance(dataloader, ModisDataloader):
            self.band_mapping = {
                "NDVI": ("sur_refl_b02", "sur_refl_b01"),  # MODIS specific bands
                "EVI": ("sur_refl_b02", "sur_refl_b01", "sur_refl_b07"),
            }
            self.normalization = {
                "NDVI": (0, 1),
                "EVI": (0, 1),
            }  # Define normalization for MODIS
        elif isinstance(dataloader, Sentinel2Dataloader):
            self.band_mapping = {
                "NDVI": ("B8", "B4"),  # Sentinel 2 specific bands
                "EVI": ("B8", "B4", "B2"),
            }
            self.normalization = {
                "NDVI": (0, 1),
                "EVI": (0, 1),
            }  # Define normalization for Sentinel2
        else:
            raise ValueError("Unsupported dataloader type")
