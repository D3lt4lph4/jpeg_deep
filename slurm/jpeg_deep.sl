#!/bin/bash

# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

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
export PYTHONUSERBASE=/home/2017018/bdegue01/.virtualenvs/vgg_jpeg

# required training variables
export EXPERIMENTS_OUTPUT_DIRECTORY=$LOCAL_WORK_DIR/experiment
export DATASET_PATH_TRAIN=/save/2017018/bdegue01/datasets
export DATASET_PATH_VAL=/dlocal/home/2017018/bdegue01

cd /path/to/sl_script

# Start the calculation
srun python3 /path/to/training.py -c /path/to/config/vggD_dct/ --horovod 

# Optionaly remove the verbose to run faster
# srun python3 /path/to/training.py -c /path/to/config/vggD_dct/ --horovod --no-verbose