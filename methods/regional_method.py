import xarray as xr

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
import sys
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import dask.array as da


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from RegionalExtremesPackage.methods.eco_cluster import EcoCluster
from RegionalExtremesPackage.methods.quantiles import Quantiles
from RegionalExtremesPackage.utils.logger import printt
from RegionalExtremesPackage.utils import Loader, Saver
from RegionalExtremesPackage.dataloaders import dataloader
from RegionalExtremesPackage.utils.config import (
    InitializationConfig,
    CLIMATIC_INDICES,
    ECOLOGICAL_INDICES,
    EARTHNET_INDICES,
)

np.set_printoptions(threshold=sys.maxsize)


class RegionalExtremes:
    def __init__(
        self,
        config: InitializationConfig,
    ):
        """
        Compute the regional extremes by defining boxes of similar region using a PCA computed on the mean seasonal cycle of the samples.
        Each values of the msc is considered as an independent component.

        Args:
            config (InitializationConfig): Shared attributes across the classes.
            n_components (int): number of components of the PCA
            n_eco_clusters (int): Number of eco_clusters per component to define the boxes. Number of boxes = n_eco_clusters**n_components
        """
        self.config = config
        self.n_components = self.config.n_components
        self.n_eco_clusters = self.config.n_eco_clusters
        self.lower_quantiles = self.config.lower_quantiles
        self.upper_quantiles = self.config.upper_quantiles
        # Loader class to load intermediate steps.
        self.loader = Loader(config)
        # Saver class to save intermediate steps.
        self.saver = Saver(config)

        if self.config.path_load_experiment:
            # Load every variable if already available, otherwise return None.
            self.pca = self.loader._load_pca_matrix()
            self.projected_data = self.loader._load_pca_projection()
            self.limits_eco_clusters = self.loader._load_limits_eco_clusters()
            self.eco_clusters = self.loader._load_data("eco_clusters")
            self.thresholds = self.loader._load_data(
                "thresholds", location=False, cluster=True
            )

        else:
            # Initialize a new PCA.
            if self.config.k_pca:
                self.pca = KernelPCA(n_components=self.n_components, kernel="rbf")
            else:
                self.pca = PCA(n_components=self.n_components)

            self.projected_data = None
            self.limits_eco_clusters = None
            self.eco_clusters = None
            self.thresholds = None


def regional_extremes_method(args):
    """Fit the PCA with a subset of the data, then project the full dataset,
    then define the eco_clusters on the full dataset projected."""
    # Initialization of the configs, load and save paths, log.txt.
    # minicube_path = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_test/customcube_CO-MEL_1.95_-72.60_S2_v0.zarr.zip"
    config = InitializationConfig(args)

    assert config.method == "regional"
    # Initialization of EcoCluster, load data if already computed.
    eco_cluster = EcoCluster(
        config=config,
    )
    quantiles_processor = Quantiles(config=config)

    # Load a subset of the dataset and fit the PCA
    if not hasattr(eco_cluster.pca, "explained_variance_"):
        # Initialization of the climatic or ecological DatasetHandler
        dataset_processor = dataloader(
            config=config,
            n_samples=None,  # 10000,  # config.n_samples,  # all the dataset
        )
        # Load and preprocess the dataset
        data = dataset_processor.preprocess_data()  # minicube_path=minicube_path)
        # Fit the PCA on the data
        eco_cluster.compute_pca_and_transform(scaled_data=data)

    # Apply the PCA to the entire dataset
    # if eco_cluster.projected_data is None:

    # Define the boundaries of the eco_clusters
    if eco_cluster.limits_eco_clusters is None:
        dataset_processor = dataloader(config=config, n_samples=None)  # all the dataset
        data = dataset_processor.preprocess_data()  # minicube_path=minicube_path)
        eco_cluster.apply_pca(scaled_data=data)
        eco_cluster.define_limits_eco_clusters()

    # Apply the regional threshold and compute the extremes
    # Load the data
    dataset_processor = dataloader(
        config=config,
        n_samples=None,
    )
    msc, data = dataset_processor.preprocess_data(
        return_time_series=True,
    )
    eco_cluster.apply_pca(scaled_data=msc)
    eco_cluster.find_eco_clusters()
    # Compute the quantiles per eco_clusters
    quantiles_processor.compute_regional_threshold(data)

    return eco_cluster


def regional_extremes_minicube(args, minicube_path):
    """Fit the PCA with a subset of the data, then project the full dataset,
    then define the eco_clusters on the full dataset projected."""
    # Initialization of the configs, load and save paths, log.txt.
    config = InitializationConfig(args)
    assert config.method == "regional"

    # Initialization of EcoCluster, load data if already computed.
    eco_cluster = EcoCluster(
        config=config,
    )
    quantiles_processor = Quantiles(config=config)
    if eco_cluster.limits_eco_clusters is None:
        raise FileNotFoundError("limits_eco_clusters file unavailable.")
    if eco_cluster.thresholds is None:
        raise FileNotFoundError("thresholds file unavailable.")

    # Apply the regional threshold and compute the extremes
    # Load the data
    dataset_processor = dataloader(config=config, n_samples=None)  # config.n_samples
    msc, data = dataset_processor.preprocess_data(
        return_time_series=True,
        minicube_path=minicube_path,
    )
    if data is None:
        return

    eco_cluster.apply_pca(scaled_data=msc)
    eco_cluster.find_eco_clusters()
    # Compute the quantiles per regions/biome (=eco_clusters)
    quantiles_processor.apply_regional_threshold(data)
