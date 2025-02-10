import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from RegionalExtremesPackage.utils.logger import int_or_none
from RegionalExtremesPackage.datahandlers import create_handler


import argparse
from RegionalExtremesPackage.methods import (
    regional_extremes_method,
    local_extremes_method,
    regional_extremes_minicube,
)


# Argparser for all configuration needs
def parser_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="id of the experiment is time of the job launch and job_id",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="name_of_the_experiment",
    )

    parser.add_argument(
        "--index",
        type=str,
        default="pei_180",
        help=" The climatic or ecological index to be processed (default: pei_180). "
        "Index available:\n -Climatic: 'pei_30', 'pei_90', 'pei_180'. \n Ecological: 'None.",
    )

    parser.add_argument(
        "--compute_variance",
        type=bool,
        default=False,
        help="Compute variance of the seasonal cycle in addition of the mean seasonal cycle (default: False).",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="globe",
        help="Region of the globe to apply the regional extremes."
        "Region available: 'globe', 'europe'.",
    )
    parser.add_argument(
        "--time_resolution",
        type=int,
        default=5,
        help="time_resolution (int, optional): temporal resolution of the msc, to reduce computationnal workload (default: 5). ",
    )

    parser.add_argument(
        "--n_components", type=int, default=3, help="Number of component of the PCA."
    )

    parser.add_argument(
        "--n_samples",
        type=int_or_none,
        default=100,
        help="Select randomly n_samples. Use 'None' for no limit.",
    )

    parser.add_argument(
        "--n_eco_clusters",
        type=int,
        default=25,
        help="number of eco_clusters to define the regions of similar seasonal cycle. n_eco_clusters is proportional. ",
    )

    parser.add_argument(
        "--kernel_pca",
        type=str,
        default=False,
        help="Using a Kernel PCA instead of a PCA.",
    )

    parser.add_argument(
        "--saving_path",
        type=str,
        default=None,
        help="Absolute path to save the experiments 'path/to/experiment'. "
        "If None, the experiment will be save in a folder /experiment in the parent folder.",
    )

    parser.add_argument(
        "--path_load_experiment",
        type=str,
        default=None,
        help="Path of the trained model folder.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="regional",
        help="Type of method to compute extremes. Either 'regional' or 'uniform'.",
    )
    return parser


if __name__ == "__main__":
    args = parser_arguments().parse_args()
    args.name = "deep_extreme_HR"
    args.index = "EVI_EN"
    args.k_pca = False
    args.n_samples = 10
    args.n_components = 3
    args.n_eco_clusters = 50
    args.compute_variance = False
    args.method = "regional"
    args.start_year = 2000
    args.is_generic_xarray_dataset = False
    args.lower_quantiles = [0.01, 0.025, 0.05]
    args.upper_quantiles = [0.95, 0.975, 0.99]

    # args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-01-23_10:01:46_deep_extreme_global"  # "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-01-23_10:01:46_deep_extreme_global"

    if args.method == "regional":
        # Train the regional extreme method on a subset of locations
        regional_extremes_method(args)
        # Apply the regional extremes method on a single minicube
        # parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final"
        # subfolders = [
        #     folder
        #     for folder in os.listdir(parent_folder)
        #     if os.path.isdir(os.path.join(parent_folder, folder))
        #     and folder.startswith("mc_")
        # ]
        # regional_extremes_minicube(
        #     args,
        #     minicube_path="/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_test/customcube_CO-MEL_1.95_-72.60_S2_v0.zarr.zip",  # "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/CH-Oe1_47.29_7.73_v0.zarr.zip",
        # )
    elif args.method == "local":
        # Apply the uniform threshold method
        local_extremes_method(args)
    elif args.method == "global":
        raise NotImplementedError("the global method is not yet implemented.")
