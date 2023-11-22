#!/bin/bash

start="${1:-0}"
end="${2:-0}"
paral="${3:-5}"
ompt="${4:-4}"

echo $start
echo $end
echo $paral
echo $ompt

chmod +x ../out/scripts/tasks_*.sh
export OMP_NUM_THREADS=$ompt
parallel -j $paral ../out/scripts/tasks_{}.sh ::: $(seq $start $end)
