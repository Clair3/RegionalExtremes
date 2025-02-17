from .common_imports import *
from .generic import GenericDatasetHandler
from .modis import EcologicalDatasetHandler
from .climatic import ClimaticDatasetHandler
from .sentinel2 import Sentinel2DatasetHandler
from .modis_sampling import ModisSamplingDatasetHandler


@staticmethod
def create_handler(config: InitializationConfig, n_samples: int):
    if config.is_generic_xarray_dataset:
        return GenericDatasetHandler(config, n_samples=n_samples)
    elif config.index in ECOLOGICAL_INDICES:
        return EcologicalDatasetHandler(config, n_samples=n_samples)
    elif config.index in CLIMATIC_INDICES:
        return ClimaticDatasetHandler(config, n_samples=n_samples)
    elif config.index in EARTHNET_INDICES:
        return Sentinel2DatasetHandler(config, n_samples=n_samples)
    elif config.index in MODIS_INDICES:
        return ModisSamplingDatasetHandler(config, n_samples=n_samples)
    else:
        raise ValueError("Invalid index")
