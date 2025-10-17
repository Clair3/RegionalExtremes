import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import sys
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from RegionalExtremesPackage.methods.eco_cluster import EcoCluster
from RegionalExtremesPackage.methods.quantiles import quantiles
from RegionalExtremesPackage.utils.logging_config import printt
from RegionalExtremesPackage.utils import Loader, Saver
from RegionalExtremesPackage.dataloaders import dataloader
from RegionalExtremesPackage.utils.config import InitializationConfig

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
    config = InitializationConfig(args)
    assert config.method == "regional"
    # Initialization of EcoCluster, load data if already computed.
    eco_cluster_processor = EcoCluster(
        config=config,
    )
    quantiles_processor = quantiles(config=config)

    # Load a subset of the dataset and fit the PCA
    if not hasattr(eco_cluster_processor.pca, "explained_variance_"):
        # Fit the PCA on the data
        # Initialization of the climatic or ecological DatasetHandler
        dataset_processor = dataloader(
            config=config,
            n_samples=config.n_samples_clustering,  # all the dataset
        )
        # Load and preprocess the dataset
        msc = dataset_processor.preprocess_data()  # minicube_path=minicube_path)
        eco_cluster_processor.compute_pca_and_transform(scaled_data=msc)

    # Apply the PCA to the entire dataset
    # if eco_cluster.projected_data is None:

    # Define the boundaries of the eco_clusters
    if eco_cluster_processor.limits_eco_clusters is None:
        dataset_processor = dataloader(
            config=config, n_samples=config.n_samples_clustering
        )  # all the dataset
        msc, data = dataset_processor.preprocess_data(
            return_time_series=True
        )  # minicube_path=minicube_path)
        eco_cluster_processor.apply_pca(scaled_data=msc)
        eco_cluster_processor.define_limits_eco_clusters()
        eco_clusters = eco_cluster_processor.find_eco_clusters()
    else:
        # Apply the regional threshold and compute the extremes
        # Load the data
        dataset_processor = dataloader(
            config=config,
            n_samples=config.n_samples_clustering,
        )
        # msc, data = dataset_processor.preprocess_data(
        #     return_time_series=True,
        # )
        msc, data = dataset_processor.preprocess_data(
            return_time_series=True
        )  # minicube_path=minicube_path)
        # eco_cluster_processor.apply_pca(scaled_data=msc)
        eco_cluster_processor.define_limits_eco_clusters()
        eco_clusters = eco_cluster_processor.find_eco_clusters()

    quantiles_processor.generate_regional_threshold(
        data, eco_clusters_load=eco_cluster_processor.eco_clusters
    )


def regional_extremes_minicube(args, minicube_path):
    """Fit the PCA with a subset of the data, then project the full dataset,
    then define the eco_clusters on the full dataset projected."""
    # Initialization of the configs, load and save paths, log.txt.
    config = InitializationConfig(args)
    assert config.method == "regional"

    # Initialization of EcoCluster, load data if already computed.
    eco_cluster_processor = EcoCluster(
        config=config,
    )
    quantiles_processor = quantiles(config=config)
    if eco_cluster_processor.limits_eco_clusters is None:
        raise FileNotFoundError("limits_eco_clusters file unavailable.")
    # if eco_cluster_processor.thresholds is None:
    #     raise FileNotFoundError("thresholds file unavailable.")

    # Apply the regional threshold and compute the extremes
    # Load the data
    dataset_processor = dataloader(config=config, n_samples=None)  # config.n_samples
    msc, data = dataset_processor.preprocess_data(
        return_time_series=True,
        minicube_path=minicube_path,
    )
    if data is None:
        return

    eco_cluster_processor.apply_pca(scaled_data=msc)
    eco_clusters = eco_cluster_processor.find_eco_clusters()
    # Compute the quantiles per regions/biome (=eco_clusters)
    quantiles_processor.apply_regional_threshold(data, eco_clusters_load=eco_clusters)
