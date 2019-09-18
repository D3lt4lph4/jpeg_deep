#!/bin/bash

# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

#SBATCH --exclusive
#SBATCH --time 48:00:00
#SBATCH --mem 100000 
#SBATCH --mail-type ALL
#SBATCH --mail-user benjamin.deguerre@insa-rouen.fr
#SBATCH --partition gpu_p100
#SBATCH --gres gpu:2
#SBATCH --nodes 4
#SBATCH --output /home/2017018/bdegue01/logs/%J.out
#SBATCH --error /home/2017018/bdegue01/logs/%J.err
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=14

module load cuda/9.0
module load python3-DL/3.6.1

export PYTHONUSERBASE=/home/2017018/bdegue01/.virtualenvs/vgg_jpeg_test
export EXPERIMENTS_OUTPUT_DIRECTORY=$LOCAL_WORK_DIR/experiment
export LOG_DIRECTORY=$LOCAL_WORK_DIR/logs
export DATASET_PATH_TRAIN=/save/2017018/bdegue01/datasets
export DATASET_PATH_VAL=/dlocal/home/2017018/bdegue01

cd /home/2017018/bdegue01/git/vgg_jpeg/keras

# We re install the package
srun python3 /home/2017018/bdegue01/git/vgg_jpeg/keras/training.py -c /home/2017018/bdegue01/git/vgg_jpeg/keras/config/vggD_dct/ --horovod 
