from .common_imports import *
from .generic import GenericDataloader
from .modis_gapfilled import ModisGapfilledDataloader
from .era5 import Era5Dataloader
from .sentinel2 import Sentinel2Dataloader
from .modis import ModisDataloader
from .sentinel2_coarse import Sentinel2CoarseDataloader


@staticmethod
def dataloader(config: InitializationConfig, n_samples: int):
    # if config.index in ECOLOGICAL_INDICES:
    #     return ModisGapfilledDataloader(config, n_samples=n_samples)
    # elif config.index in CLIMATIC_INDICES:
    #     return Era5Dataloader(config, n_samples=n_samples)
    if config.data_source == "S2":  # config.index in EARTHNET_INDICES
        return Sentinel2Dataloader(config, n_samples=n_samples)
    elif config.data_source == "S2" and (config.modis_resolution is True):
        return Sentinel2CoarseDataloader(config, n_samples=n_samples)
    elif config.data_source == "MODIS":
        return ModisDataloader(config, n_samples=n_samples)
    else:
        return GenericDataloader(config, n_samples=n_samples)
        # raise ValueError("Invalid index")
