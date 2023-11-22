#!/bin/bash

start="${1:-0}"
end="${2:-0}"
paral="${3:-5}"
ompt="${4:-4}"

echo "Double check inputs"
echo "First job ID:" $start
echo "Last job ID:" $end
echo "Number of parallel jobs:" $paral
echo "OMP thread:" $ompt

chmod +x ../out/scripts/tasks_*.sh
export OMP_NUM_THREADS=$ompt
parallel -j $paral ../out/scripts/tasks_{}.sh ::: $(seq $start $end) &
