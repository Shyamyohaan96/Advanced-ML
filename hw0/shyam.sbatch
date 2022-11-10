#!/bin/bash

#
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=xor_%04a_stdout.txt
#SBATCH --error=xor_%04a_stderr.txt
#SBATCH --time=00:02:00
#SBATCH --job-name=xor_test
#SBATCH --mail-user=shyamkrishnan@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504311/hw0/
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# Change this line to start an instance of your experiment
python shyam_hw0.py


