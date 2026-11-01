"""
PySR step of the AI-Lorenz experiments (E1-E3), for a machine with Julia.

Reads every pysr_inputs/*.npz written by e1/e2/e3 (features X, target y = the
learned unknown term f, variable names), fits PySR with the benchmark's
settings (methods.fit_pysr), and writes pysr_outputs/ailorenz_pysr.csv with
one row per (dataset, seed). Scoring is done by score_pysr.py.

    python run_pysr_device.py                 # 60 s per fit, seed 0 (+ seeds 1, 2 for E2)
    python run_pysr_device.py --budget 120    # longer fits
    python run_pysr_device.py --quick         # first dataset only, to test the setup
    python run_pysr_device.py --inputs pysr_inputs_seeds --out ailorenz_pysr_seeds.csv

Rows are capped at 3000 per dataset (evenly spaced) to bound the run time.
Expected duration with the defaults: about 50 minutes.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAX_ROWS = 3000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--inputs", default="pysr_inputs")
    ap.add_argument("--out", default="ailorenz_pysr.csv")
    args = ap.parse_args()
    from pysr import PySRRegressor  # first import installs Julia

    files = sorted((HERE / args.inputs).glob("*.npz"))
    if args.quick:
        files = files[:1]
    out_dir = HERE / "pysr_outputs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / args.out
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            done = {(r["dataset"], r["seed"]) for r in csv.DictReader(fh)}
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["dataset", "seed", "n_rows", "seconds", "expression"])
        for f in files:
            d = np.load(f)
            X, y, names = d["X"], d["y"], [str(n) for n in d["names"]]
            if len(y) > MAX_ROWS:
                idx = np.linspace(0, len(y) - 1, MAX_ROWS).astype(int)
                X, y = X[idx], y[idx]
            if "_k" in f.stem and f.stem.rsplit("_k", 1)[1].isdigit():
                seeds = [int(f.stem.rsplit("_k", 1)[1])]  # joint X-TFC/PySR seed
            else:
                seeds = [0, 1, 2] if f.stem.startswith("e2_") else [0]
            for seed in seeds:
                if (f.stem, str(seed)) in done:
                    continue
                t0 = time.perf_counter()
                reg = PySRRegressor(
                    niterations=1000, populations=15,
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["exp", "log", "sqrt", "square"],
                    maxsize=30, timeout_in_seconds=args.budget, model_selection="best",
                    random_state=seed, deterministic=True, parallelism="serial",
                    progress=False, verbosity=0, temp_equation_file=True,
                )
                reg.fit(X, y, variable_names=names)
                expr = str(reg.sympy())
                w.writerow([f.stem, seed, len(y), round(time.perf_counter() - t0, 1), expr])
                fh.flush()
                print(f"{f.stem} seed {seed}: {expr}", flush=True)
    print(f"written {out}")


if __name__ == "__main__":
    main()
