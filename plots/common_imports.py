###
# plot file to plot data, experiments etc.
# Draft, the code is not made to be shared and reused!
###
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import xarray as xr
import numpy as np
import matplotlib
import datetime

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider
import cartopy
import random

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

from RegionalExtremesPackage.utils.config import InitializationConfig
from RegionalExtremesPackage.utils import Loader, Saver
from main import parser_arguments
from RegionalExtremesPackage.utils.logger import printt
from RegionalExtremesPackage.dataloaders import dataloader
from abc import ABC, abstractmethod
