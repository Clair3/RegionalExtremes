import os
import numpy as np
import xarray as xr
import pickle as pk
from pathlib import Path
import cf_xarray as cfxr
from RegionalExtremesPackage.utils.logging_config import printt
from RegionalExtremesPackage.utils.config import InitializationConfig


CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
ECOLOGICAL_FILEPATH = (
    lambda index: f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{index}/Groups_{index}gapfilled_QCdyn.zarr"
)
VARIABLE_NAME = lambda index: f"{index}gapfilled_QCdyn"
