#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$HERE:$HERE/../../src/sr_benchmark" HERE
for t in stefan_boltzmann monin_obukhov_length log_wind_profile bulk_sensible_heat; do
 for n in 0.0 0.05; do for s in 0 1; do for m in units nounits; do echo "$t $n $s $m"; done; done; done; done |
xargs -P 4 -L 1 bash -c 'd=runs/$0_$1_$2_$3; mkdir -p $d; cd $d && python $HERE/physo_units.py $0 $1 $2 $3 $HERE/results.csv > log.txt 2>&1; tail -1 log.txt'
