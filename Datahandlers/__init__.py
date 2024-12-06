from .common_imports import *
from .generic import GenericDatasetHandler
from .modis import EcologicalDatasetHandler
from .climatic import ClimaticDatasetHandler
from .sentinel2 import EarthnetDatasetHandler


@staticmethod
def create_handler(config, n_samples):
    if config.is_generic_xarray_dataset:
        return GenericDatasetHandler(config=config, n_samples=n_samples)
    elif config.index in ECOLOGICAL_INDICES:
        return EcologicalDatasetHandler(config=config, n_samples=n_samples)
    elif config.index in CLIMATIC_INDICES:
        return ClimaticDatasetHandler(config=config, n_samples=n_samples)
    elif config.index in EARTHNET_INDICES:
        return EarthnetDatasetHandler(config=config, n_samples=n_samples)
    else:
        raise ValueError("Invalid index")
