from .common_imports import *
from .generic import GenericDataloader
from .modis_gapfilled import ModisGapfilledDataloader
from .era5 import Era5Dataloader
from .sentinel2 import Sentinel2Dataloader
from .modis import ModisDataloader


@staticmethod
def dataloader(config: InitializationConfig, n_samples: int):
    if config.index in ECOLOGICAL_INDICES:
        return ModisGapfilledDataloader(config, n_samples=n_samples)
    elif config.index in CLIMATIC_INDICES:
        return Era5Dataloader(config, n_samples=n_samples)
    elif config.index in EARTHNET_INDICES:
        return Sentinel2Dataloader(config, n_samples=n_samples)
    elif config.index in MODIS_INDICES:
        return ModisDataloader(config, n_samples=n_samples)
    else:
        return GenericDataloader(config, n_samples=n_samples)
        # raise ValueError("Invalid index")
