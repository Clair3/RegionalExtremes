import numpy as np
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from RegionalExtremesPackage.utils.logging_config import int_or_none
from dask import delayed, compute
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
        help="id_of_the_experiment. If no id, the id is set to the current date and time.",
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
        "--data_source",
        type=str,
        default="S2",
        help=" The climatic or ecological index to be processed (default: pei_180). "
        "Data Source available:\n : 'S2', 'MODIS'",
    )

    parser.add_argument(
        "--data_source_path",
        type=str,
        default="path/to/data",
        help=" Absolute path to the clustering dataset (default: path/to/data).",
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
        "--n_samples_pca",
        type=int_or_none,
        default=15000,
        help="Select randomly n_samples to train the PCA (ideally around 10 or 20000, PCA has O(n²) complexity, no need for a large dataset). Use 'None' for no limit.",
    )

    parser.add_argument(
        "--n_samples_clustering",
        type=int_or_none,
        default=50000,
        help="Select randomly n_samples to compute eco-cluster and percentiles (ideally as large as possible). Use 'None' for no limit.",
    )

    parser.add_argument(
        "--n_eco_clusters",
        type=int,
        default=25,
        help="number of eco_clusters to define the regions of similar seasonal cycle. n_eco_clusters is proportional. ",
    )

    parser.add_argument(
        "--saving_path",
        type=str,
        default=None,
        help="Absolute path to save the experiments 'path/to/experiment'. "
        "If None, the experiment will be save in a folder /experiment in the parent folder.",
    )

    parser.add_argument(
        "--dayofyear_extreme",
        type=bool,
        default=False,
        help="If True, compute the extremes per Day Of Year.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="regional",
        help="Type of method to compute extremes. Either 'regional' or 'uniform'.",
    )
    return parser

def run_extremes_pipeline(args, parent_folder, subfolders, batch_size=20):

    print(f"Processing {len(subfolders)} minicubes...")

    if args.method == "regional":

        # Train model if needed
        if args.saving_path is None:
            regional_extremes_method(args)

        @delayed
        def process_sample(folder):
            try:
                regional_extremes_minicube(
                    args,
                    minicube_path=os.path.join(parent_folder, folder),
                )
            except Exception as e:
                print(f"Error with {folder}: {e}")

        tasks = [process_sample(folder) for folder in subfolders]

        for i in range(0, len(tasks), batch_size):
            compute(*tasks[i:i + batch_size], scheduler="threads")

    elif args.method == "local":

        for folder in subfolders:
            try:
                local_extremes_method(
                    args,
                    minicube_path=os.path.join(parent_folder, folder),
                )
            except Exception as e:
                print(f"error with {folder} - {e}")
    else:
        raise NotImplementedError(f"{args.method} method not implemented")
        
if __name__ == "__main__":

    args = parser_arguments().parse_args()
    args.name = "S2LR_20_highmask_local"  # "large_training_set"
    args.modis_resolution = False  # True  # True
    args.index = "EVI"
    args.dayofyear_extreme = False  # True
    args.data_source = "S2" #"MODIS"
    args.vci = False
    args.time_resolution = 16
    args.data_source_path = [
        # "/Net/Groups/BGI/work_5/scratch/EU_Minicubes/final_modis/"
        # "/Net/Groups/BGI/work_5/scratch/Somalia_VCI_test/S2_samples/",
        # "/Net/Groups/BGI/work_2/scratch/DeepExtremes/dx-minicubes/full/",
        "/Net/Groups/BGI/work_5/scratch/EU_Minicubes/_final/",
    ]
    args.n_samples_pca = 15000
    args.n_samples_clustering = 50000
    args.n_components = 3
    args.n_eco_clusters = 20
    args.method = "regional" # "local"  # "local"  # "regional"
    args.lower_quantiles = [0, 0.025, 0.05, 0.10, 0.2, 0.3, 0.4, 0.50]
    args.upper_quantiles = [0.501, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1]

    args.saving_path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2026-04-14_15:54:09_S2LR_20_highmask_local/EVI/"  # "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-12-10_17:45:30_modis_2017_32y/EVI/"  # "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-12-09_02:53:15_modis_regional/EVI/"  # "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/PassageProject/2025-10-21_10:05:17_somalia_5d/NDVI"

    parent_folder ="/Net/Groups/BGI/tscratch/crobin/ContrastiveEarthnetProject/datasets/" # ß"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/"  #
    subfolders = [
        folder for folder in os.listdir(parent_folder) if folder[-4:] == ".zip"
    ]
    subfolders = ["ES-LM2_39.93_-5.78_v0/modis-13Q1-061.zarr.zip"]
    subfolders = ["aligned.zarr"]
    
    run_extremes_pipeline(
        args=args,
        parent_folder=parent_folder,
        subfolders=subfolders,
    )
