
# RegionalExtremes
---
## Installation

1. **Set up the Conda environment**:

   To install the required environment, run the following command:

   ```bash
   conda env create python=3.12 -n ExtremesEnv --file environment.yaml
   ```

2. **Activate the environment**:

   After the environment is created, ensure that it's active:

   ```bash
   conda activate ExtremesEnv
   ```

## Running the Script

To execute the main script, use the following command:

```bash
/Net/Groups/BGI/scratch/crobin/miniconda3/envs/ExtremesEnv2/bin/python /Net/Groups/BGI/scratch/crobin/PythonProjects/ExtremesProject/RegionalExtremesPackage/regional_extremes.py
```

### Important Notes

1. **Argument Configuration**:
   - You can pass the script arguments in two ways:
     - Directly within the `__main__` function of the script.
     - Via the command line when executing the script.
   - I often do it within the `__main__` function. If using command-line arguments, ensure no legacy code is overwriting them.

2. **Environment Variables**:
   - If you encounter issues with libraries not being found, you might need to set the `LD_LIBRARY_PATH`. You can do this by running:
   
     ```bash
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/Environment/lib
     ```

   - For example:

     ```bash
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Net/Groups/BGI/scratch/crobin/miniconda3/envs/ExtremesEnv2/lib
     ```

3. **Resuming from Saved Experiments**:
   - If the argument `path_load_experiment` is provided, the script will load saved intermediate steps and only compute the remaining steps.
  
- **Execution Time**:
   - Processing MODIS data for Europe typically takes around 20 minutes.

---


