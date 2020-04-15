#!/bin/bash

# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017 

#SBATCH --exclusive
#SBATCH --time 48:00:00
#SBATCH --mem 100000 
#SBATCH --mail-type ALL
#SBATCH --mail-user mail@mail.com
#SBATCH --partition gpu_p100
#SBATCH --gres gpu:2
#SBATCH --nodes 4
#SBATCH --output /path/to/logs/%J.out
#SBATCH --error /pth/to/logs/%J.err
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=14

# Loading modules if your environement define some
module load cuda/9.0
module load python3-DL/3.6.1

# Custom python path
export PYTHONUSERBASE=/path/to/python/env

# required training variables
export EXPERIMENTS_OUTPUT_DIRECTORY=/path/to/output/dir
export DATASET_PATH_TRAIN=/path/to/train/dir
export DATASET_PATH_VAL=/path/to/val/dir

cd /path/to/sl_script

# Start the calculation
srun python3 /path/to/training.py -c /path/to/config/vggD_dct/ --horovod 
