#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$HERE:$HERE/../../src/sr_benchmark" HERE
for t in temperature_tendency relative_humidity net_radiation; do for s in 0 1; do for m in units nounits; do echo "$t $s $m"; done; done; done |
xargs -P 4 -L 1 bash -c 'd=jruns/$0_$1_$2; mkdir -p $d; cd $d && python $HERE/physo_jaru.py $0 $1 $2 $HERE/jaru_results.csv > log.txt 2>&1; tail -1 log.txt'
