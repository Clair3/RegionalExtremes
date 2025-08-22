import numpy as np
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from RegionalExtremesPackage.utils.logging_config import int_or_none, printt

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
        "--compute_only_thresholds",
        type=bool,
        default=False,
        help="If True, only compute the thresholds and not the extremes.",
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
    args.name = "big_dataset_20"
    # args.modis_resolution = True
    args.index = "EVI_EN"
    args.k_pca = False
    args.n_samples = 50000  # 40000
    args.n_components = 3
    args.n_eco_clusters = 20
    args.compute_variance = False
    args.method = "local"
    args.start_year = 2000
    args.lower_quantiles = [0.025, 0.05, 0.10, 0.2, 0.3, 0.4, 0.50]
    args.upper_quantiles = [0.501, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]

    # args.saving_path = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2025-04-04_13:04:42_full_fluxnet_therightone_modis/EVI_MODIS"  #

    if args.method == "regional":
        # Train the regional extreme method on a subset of locations
        regional_extremes_method(args)
        # Apply the regional extremes method on a single minicube
        parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/"

        subfolders = [
            folder for folder in os.listdir(parent_folder) if folder[-4:] == ".zip"
        ]
        # parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_test/"
        subfolders = [
            "DE-Hai_51.08_10.45_v0.zarr.zip",
            "FR-LGt_47.32_2.28_v0.zarr.zip",
            "ES-Cnd_37.91_-3.23_v0.zarr.zip",
            "ES-LM1_39.94_-5.78_v0.zarr.zip",
            "ES-LM2_39.93_-5.78_v0.zarr.zip",
            "ES-LMa_39.94_-5.77_v0.zarr.zip",
            "DE-Geb_51.10_10.91_v0.zarr.zip",
            "DE-Wet_50.45_11.46_v0.zarr.zip",
            "DE-Bay_50.14_11.87_v0.zarr.zip",
            "DE-Meh_51.28_10.66_v0.zarr.zip",
            "DE-Lnf_51.33_10.37_v0.zarr.zip",
        ]
        # subfolders = [
        #    "DE-Hai_51.08_10.45_v0.zarr.zip",
        #    # "ES-LM1_39.94_-5.78_v0.zarr.zip",
        #    # "ES-LM2_39.93_-5.78_v0.zarr.zip",
        #    # # "ES-LMa_39.94_-5.77_v0.zarr.zip",
        #    # "ES-Cnd_37.91_-3.23_v0.zarr.zip",
        #    # "FR-LGt_47.32_2.28_v0.zarr.zip",
        #    # "DE-Lnf_51.33_10.37_v0.zarr.zip",
        #    # "DE-Geb_51.10_10.91_v0.zarr.zip",
        #    # "DE-Wet_50.45_11.46_v0.zarr.zip",
        #    # "DE-Bay_50.14_11.87_v0.zarr.zip",
        #    # "DE-Meh_51.28_10.66_v0.zarr.zip",
        # ]

        # subfolders = [
        #     "custom_cube_50.90_11.56.zarr.zip",
        #     "custom_cube_44.17_5.24.zarr.zip",
        #     "custom_cube_44.24_5.14.zarr.zip",
        #     "custom_cube_47.31_0.18.zarr.zip",
        # ]
        # regional_extremes_minicube(
        #     args,
        #     minicube_path="/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/32UNB_Hainich.zarr",
        # )
        # cube = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/32UNB_Hainich.zarr"

        # regional_extremes_minicube(
        #    args,
        #    minicube_path=cube,
        # )
        for folder in subfolders:
            if folder[:-4] not in os.listdir(
                f"{args.path_load_experiment}/{args.index}/"
            ):
                try:
                    regional_extremes_minicube(
                        args,
                        minicube_path=parent_folder + folder,
                    )
                except:
                    print(f"error with {folder}")
    #
    # regional_extremes_minicube(
    #     args,
    #     minicube_path="/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/DE-RuS_50.87_6.45_v0.zarr.zip",  # ES-Cnd_37.91_-3.23_v0.zarr.zip",
    # )
    elif args.method == "local":
        # Apply the uniform threshold method
        parent_folder = "/Net/Groups/BGI/work_5/scratch/EU_Minicubes/_final/" #"/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/final/"

        # subfolders = [
        #     "DE-Hai_51.08_10.45_v0.zarr.zip",
        #     "FR-LGt_47.32_2.28_v0.zarr.zip",
        #     "ES-Cnd_37.91_-3.23_v0.zarr.zip",
        #     # "ES-LM1_39.94_-5.78_v0.zarr.zip",
        #     # "ES-LM2_39.93_-5.78_v0.zarr.zip",
        #     # "ES-LMa_39.94_-5.77_v0.zarr.zip",
        #     "DE-Geb_51.10_10.91_v0.zarr.zip",
        #     "DE-Wet_50.45_11.46_v0.zarr.zip",
        #     "DE-Bay_50.14_11.87_v0.zarr.zip",
        #     "DE-Meh_51.28_10.66_v0.zarr.zip",
        #     "DE-Lnf_51.33_10.37_v0.zarr.zip",
        # ]


        subfolders = [folder for folder in os.listdir(parent_folder)]

        # parent_folder = "/Net/Groups/BGI/work_5/scratch/FluxSitesMiniCubes/_test/"
        # subfolders = [
        #     "custom_cube_44.17_5.24.zarr.zip",
        #     "custom_cube_44.24_5.14.zarr.zip",
        #     "custom_cube_47.31_0.18.zarr.zip",
        #     "custom_cube_50.90_11.56.zarr.zip",
        # ]

        for folder in subfolders:
            #   if folder[:-4] not in os.listdir(
            #       f"{args.path_load_experiment}/{args.index}/"
            #   ):
            try:
                local_extremes_method(
                    args,
                    minicube_path=parent_folder + folder)
            except Exception as e:
                printt(f"Error with {folder}: {e}")

    elif args.method == "global":
        raise NotImplementedError("the global method is not yet implemented.")
