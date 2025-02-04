from typing import Union, Optional
import xarray as xr

import dask.array as da
import numpy as np
import datetime
import regionmask
from scipy.signal import savgol_filter
from typing import Union, Optional
import warnings
import sys
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from RegionalExtremesPackage.utils.config import (
    InitializationConfig,
    CLIMATIC_INDICES,
    ECOLOGICAL_INDICES,
    EARTHNET_INDICES,
)
from RegionalExtremesPackage.utils import Loader, Saver
from RegionalExtremesPackage.utils.logger import printt


np.random.seed(2024)


NORTH_POLE_THRESHOLD = 66.5
SOUTH_POLE_THRESHOLD = -66.5
MAX_NAN_PERCENTAGE = 0.7
CLIMATIC_FILEPATH = "/Net/Groups/BGI/scratch/mweynants/DeepExtremes/v3/PEICube.zarr"
ECOLOGICAL_FILEPATH = (
    lambda index: f"/Net/Groups/BGI/work_1/scratch/fluxcom/upscaling_inputs/MODIS_VI_perRegion061/{index}/Groups_{index}gapfilled_QCdyn.zarr"
)
VARIABLE_NAME = lambda index: f"{index}gapfilled_QCdyn"
EARTHNET_FILEPATH = "/Net/Groups/BGI/work_2/scratch/DeepExtremes/dx-minicubes"
