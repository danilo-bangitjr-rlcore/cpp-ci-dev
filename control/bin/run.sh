#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --array=1-162%20
#SBATCH --exclusive=user
#SBATCH --output=out/adam_rmsprop-%j.out

eval "$(conda shell.bash hook)"
conda activate rlcore

# Fetch command on line $SLURM_ARRAY_TASK_ID from test file
EXE=`cat ../out/scripts/tasks_0.sh | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`
`$EXE`