import xarray as xr
import dask.array as da
from argparse import Namespace
import numpy as np
import json
import datetime
import sys
import os
from pathlib import Path


from RegionalExtremesPackage.utils.logging_config import initialize_logger, printt

sys.path.append(str(Path(__file__).resolve().parent.parent))

GRANDPARENT_DIRECTORY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
CLIMATIC_INDICES = ["pei_30", "pei_90", "pei_180"]
ECOLOGICAL_INDICES = ["EVI", "NDVI", "kNDVI"]
EARTHNET_INDICES = ["EVI_EN"]
MODIS_INDICES = ["EVI_MODIS"]


class InitializationConfig:
    def __init__(self, args: Namespace):
        """
        Initialize InitializationConfig with the provided arguments.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        if args.saving_path is None:
            # self.saving_path = None
            self._initialize_new_experiment(args)
        else:

            self._load_existing_experiment(args)

    def _initialize_new_experiment(self, args: Namespace):
        """
        Initialize settings for a new model when no model is loaded.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        for key, value in vars(args).items():
            setattr(self, key, value)

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
                    Path(GRANDPARENT_DIRECTORY_PATH)
                    / "experiments/"
                    / f"{args.id}_{args.name}"
                    / self.index
                )
            else:
                self.saving_path = (
                    Path(GRANDPARENT_DIRECTORY_PATH)
                    / "experiments/"
                    / args.id
                    / self.index
                )
        self.saving_path.mkdir(parents=True, exist_ok=True)
        args.saving_path = str(self.saving_path)

    def _save_args(self, args: Namespace):
        """
        Save the arguments to a JSON file for future reference.

        Args:
            args (argparse.Namespace): Parsed arguments from argparse.ArgumentParser().parse_args()
        """
        # assert self.saving_path is None

        # Saving path
        args_path = self.saving_path / "args.json"

        # Convert to a dictionnary
        args_dict = vars(args)
        # del args_dict["saving_path"]

        if not args_path.exists():
            with open(args_path, "w") as f:
                json.dump(args_dict, f, indent=4)
        else:
            raise f"{args_path} already exist."

    def _load_existing_experiment(self, args):
        """
        Load an existing model's PCA matrix and min-max data from files.
        """
        self.saving_path = Path(args.saving_path)
        print(f"loading existing experiment from {self.saving_path}")

        # Initialise the logger
        initialize_logger(self.saving_path)
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
