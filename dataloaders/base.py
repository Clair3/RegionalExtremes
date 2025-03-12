from abc import ABC, abstractmethod
from RegionalExtremesPackage.utils.config import InitializationConfig
from typing import Union, Optional
from .common_imports import *


class Dataloader(ABC):
    def __init__(
        self,
        config: InitializationConfig,
        n_samples: Union[int, None],
    ):
        """
        Initialize DatasetHandler.

        Parameters:
        n_samples (Union[int, None]): Number of samples to select.
        time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload. Defaults to 5.
        """
        # Config class to deal with loading and saving the model.
        self.config = config
        # Number of samples to load. If None, the full dataset is loaded.
        self.n_samples = n_samples
        self.loader = Loader(config)
        self.saver = Saver(config)
        self.noise_removal = NoiseRemovalBase()

        self.start_year = self.config.start_year

        # data loaded from the dataset
        self.data = None
        # Mean seasonal cycle
        self.msc = None

        # minimum and maximum of the data for normalization
        self.max_data = None
        self.min_data = None
