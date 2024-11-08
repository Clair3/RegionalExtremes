import xarray as xr
import dask.array as da
from argparse import Namespace
import numpy as np
import json
import random
import datetime
from sklearn.decomposition import PCA
import pandas as pd
import pickle as pk
from pathlib import Path
from typing import Union
import time
import sys
import os
from utils import initialize_logger, printt, int_or_none

CURRENT_DIRECTORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIRECTORY_PATH = os.path.abspath(os.path.join(CURRENT_DIRECTORY_PATH, os.pardir))
CLIMATIC_INDICES = ["pei_30", "pei_90", "pei_180"]
ECOLOGICAL_INDICES = ["EVI", "NDVI", "kNDVI"]
EARTHNET_INDICES = ["EVI_EN"]


class InitializationConfig:
    def __init__(self, args: Namespace):
        """
        Initialize InitializationConfig with the provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        if args.path_load_experiment is None:
            self.path_load_experiment = None
            self._initialize_new_experiment(args)
        else:
            self.path_load_experiment = Path(args.path_load_experiment)
            self._load_existing_experiment()

    def _initialize_new_experiment(self, args: Namespace):
        """
        Initialize settings for a new model when no model is loaded.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        print(args)
        for key, value in vars(args).items():
            setattr(self, key, value)
        # self.time_resolution = args.time_resolution
        # self.index = args.index
        # self.compute_variance = args.compute_variance
        # self.k_pca = args.k_pca
        # self.start_year = args.start_year
        # self.method = args.method
        self.is_generic_xarray_dataset = args.is_generic_xarray_dataset is True

        if self.is_generic_xarray_dataset:
            return

        self._set_saving_path(args)
        initialize_logger(self.saving_path)
        printt("Initialisation of a new model, no path provided for an existing model.")
        printt(f"The saving path is: {self.saving_path}")
        self._save_args(args)

    def _set_saving_path(self, args: Namespace):
        """
        Set the saving path for the new model.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        # Model launch with the command line. If model launch with sbatch, the id can be define using the id job + date
        if not args.id:
            args.id = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")

        if args.saving_path:
            self.saving_path = Path(args.saving_path) / {args.id} / self.index
        else:
            if args.name:
                self.saving_path = (
                    Path(PARENT_DIRECTORY_PATH)
                    / "experiments/"
                    / f"{args.id}_{args.name}"
                    / self.index
                )
            else:
                self.saving_path = (
                    Path(PARENT_DIRECTORY_PATH) / "experiments/" / args.id / self.index
                )
        self.saving_path.mkdir(parents=True, exist_ok=True)
        args.saving_path = str(self.saving_path)

    def _save_args(self, args: Namespace):
        """
        Save the arguments to a JSON file for future reference.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        assert self.path_load_experiment is None

        # Saving path
        args_path = self.saving_path / "args.json"

        # Convert to a dictionnary
        args_dict = vars(args)
        del args_dict["path_load_experiment"]

        if not args_path.exists():
            with open(args_path, "w") as f:
                json.dump(args_dict, f, indent=4)
        else:
            raise f"{args_path} already exist."
        printt(f"args saved, path: {args_path}")

    def _load_existing_experiment(self):
        """
        Load an existing model's PCA matrix and min-max data from files.
        """
        # Filter out 'slurm_files' in the path to load experiment to find the index used.
        self.index = [
            folder
            for folder in os.listdir(self.path_load_experiment)
            if folder != "slurm_files"
        ][0]
        self.saving_path = self.path_load_experiment / self.index

        # Initialise the logger
        initialize_logger(self.saving_path)
        printt(f"Loading of the model path: {self.path_load_experiment}")
        self._load_args()

    def _load_args(self):
        """
        Load args data from the file.
        """
        args_path = self.saving_path / "args.json"
        if args_path.exists():
            with open(args_path, "r") as f:
                args = json.load(f)
                for key, value in args.items():
                    setattr(self, key, value)
            self.saving_path = Path(self.saving_path)
        else:
            raise FileNotFoundError(f"{args_path} does not exist.")
