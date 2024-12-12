import sys

sys.path.append(
    "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/RegionalExtremesPackage"
)

from main import (
    parser_arguments,
    regional_extremes_method,
    local_extremes_method,
    regional_extremes_minicube,
)

import numpy as np
from dask_jobqueue import SLURMCluster
from dask.distributed import Client as daskClient
import argparse

cluster = SLURMCluster(
    queue="work",  # Specify the SLURM queue
    processes=1,  # Number of processes per job
    cores=1,  # Number of cores per job
    memory="50GB",  # Memory per job
    walltime="15:00:00",  # Job duration (hh:mm:ss)
    job_script_prologue=[
        "module load BGC-easybuilded",
        "module load  GCC",
    ],
)

# Scale up the number of workers
# cluster.scale(jobs=20)  # Adjust the number of jobs/workers
cluster.adapt(minimum=0, maximum=100)

# Create a Dask client that connects to the cluster
client = daskClient(cluster)

# Check cluster status
print(cluster)

# Remove Jupyter's arguments before parsing your own
sys.argv = sys.argv[:1]

args = parser_arguments().parse_args()
args.name = "deep_extreme_HR"
args.index = "EVI_EN"
args.k_pca = False
args.n_samples = 1000
args.n_components = 3
args.n_bins = 50
args.compute_variance = False
args.method = "regional"
args.start_year = 2000
args.is_generic_xarray_dataset = False
# args.path_load_experiment = "/Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/experiments/2024-11-25_18:41:43_deep_extreme_HR"

LOWER_QUANTILES_LEVEL = np.array([0.01, 0.025, 0.05])
UPPER_QUANTILES_LEVEL = np.array([0.95, 0.975, 0.99])
if args.method == "regional":
    # Apply the regional extremes method
    regional_extremes_method(args, (LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL))
    regional_extremes_minicube(args, (LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL))
elif args.method == "local":
    # Apply the uniform threshold method
    local_extremes_method(args, (LOWER_QUANTILES_LEVEL, UPPER_QUANTILES_LEVEL))
elif args.method == "global":
    raise NotImplementedError("the global method is not yet implemented.")

client.close()
cluster.close()
