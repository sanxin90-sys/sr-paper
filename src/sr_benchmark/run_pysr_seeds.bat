@echo off
rem Multi-seed PySR benchmark. Run from src\sr_benchmark with the jaru env active:
rem   run_pysr_seeds.bat
rem Writes results\static_pysr_s<seed>.csv and results\dynamic_pysr_s<seed>.csv.
rem Expected duration: about 4 hours in total.

for %%s in (0 1 2) do (
    python run_benchmark.py --suite static --methods pysr --noise 0 0.05 --budget 120 --seed %%s --out results\static_pysr_s%%s.csv
    python run_benchmark.py --suite dynamic --methods pysr --noise 0 0.01 0.05 --budget 60 --seed %%s --out results\dynamic_pysr_s%%s.csv
)

echo Done.
