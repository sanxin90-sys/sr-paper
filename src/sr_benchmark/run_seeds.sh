#!/bin/bash
# Seeds 0-2 for the engines that had only seed 0 in the original benchmark.
cd "$(dirname "$0")"
export SR_THREADS=2
for s in 0 1 2; do
  python run_benchmark.py --suite static  --methods sindy_stlsq sindy_poly2 --noise 0 0.05 --seed $s --out results_seeds/static_sindy_s$s.csv
  python run_benchmark.py --suite dynamic --methods sindy_stlsq sindy_poly2 --noise 0 0.01 0.05 --seed $s --out results_seeds/dynamic_sindy_s$s.csv
done > results_seeds/sindy.log 2>&1
op() { python run_benchmark.py --suite static --methods $1 --noise 0 0.05 --budget 45 --seed $2 --out results_seeds/static_$1_s$2.csv
       python run_benchmark.py --suite dynamic --methods $1 --noise 0 0.01 0.05 --budget 30 --seed $2 --out results_seeds/dynamic_$1_s$2.csv; }
(op operon 1 > results_seeds/operon_s1.log 2>&1) &
(op operon 2 > results_seeds/operon_s2.log 2>&1) &
wait
(op gplearn 1 > results_seeds/gplearn_s1.log 2>&1) &
(op gplearn 2 > results_seeds/gplearn_s2.log 2>&1) &
wait
echo ALLDONE
