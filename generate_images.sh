#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --array=0-12
#SBATCH --time=4-0:0:0       # Time: D-H:M:S
#SBATCH --account=rrg-keli      # Account: def-keli/rrg-keli
#SBATCH --mem=32G               # Memory in total
#SBATCH --nodes=1               # Number of nodes requested.
#SBATCH --cpus-per-task=8       # Number of cores per task.
#SBATCH --gres=gpu:a100:1       # 40G A100

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.

#SBATCH --output=job_results/coco_generations_%j.txt

# Below sets the email notification, swap to your email to receive notifications
#SBATCH --mail-user=tristanengst@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Print some info for context.
pwd
hostname
date

echo "Starting job number $SLURM_ARRAY_TASK_ID"

source ~/.bashrc
conda activate py39SPS

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

# Do all the research.
python GenerateAugmentations.py --data_dir ~/scratch/SPS/data --data coco_captions_images --split train --gpu 0 --start_stop_idxs 0 10000 --start_epoch $SLURM_ARRAY_TASK_ID --epochs 1

date
