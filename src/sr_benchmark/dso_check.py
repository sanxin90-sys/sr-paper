"""
Run Deep Symbolic Regression (DSO, Petersen et al. 2019) on one static task.

DSO's reference implementation needs TensorFlow 1.14, numpy<=1.19 and numba 0.53,
so it runs in its own Python 3.7 environment, separate from the jaru env.
See the setup section in LEIA-ME.md (section "Testar o DSO").

Examples (run from src/sr_benchmark, with the dso env active):

    python dso_check.py --smoke
    python dso_check.py --task stefan_boltzmann --noise 0 --n-samples 100000
    python dso_check.py --task tetens_es --noise 0.05 --out results/static_dso.csv

--dry-run builds the benchmark data and prints its shape without importing DSO,
so the data side can be checked in the jaru env.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import sympy as sp

from tasks import STATIC_TASKS, add_noise, sample_static
from methods import complexity

FIELDS = ["suite", "task", "target", "noise", "estimator", "method", "seed", "status",
          "r2_extrap", "nrmse_extrap", "complexity", "seconds", "deriv_rel_err", "expression"]

# DSO token names (see dso/functions.py). Protected operators are on so that
# log and sqrt never return NaN during the search.
FUNCTION_SET = ["add", "sub", "mul", "div", "sqrt", "log", "exp", "neg", "inv"]


def r2(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nrmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.std(y_true) + 1e-12))


def make_data(task_name: str, noise: float, n_train: int, seed: int):
    """Same sampling as run_benchmark.static_suite, so DSO sees the same data as the other methods."""
    task = next(t for t in STATIC_TASKS if t.name == task_name)
    rng = np.random.default_rng(seed)
    X, y_clean = sample_static(task, n_train, rng)
    Xt, yt = sample_static(task, 2000, rng, test=True)
    y = add_noise(y_clean, noise, np.random.default_rng(seed + 1))
    return task, X, y, Xt, yt


def make_config(n_samples: int, seed: int, const: bool = False) -> dict:
    # "const" adds a free-constant token fitted by BFGS inside DSO; without it,
    # expressions can only use the inputs and the operators above.
    return {
        "task": {
            "task_type": "regression",
            "function_set": FUNCTION_SET + (["const"] if const else []),
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "protected": True,
            "threshold": 1e-12,
        },
        "training": {"n_samples": n_samples, "epsilon": 0.05},
        "experiment": {"seed": seed, "logdir": None, "verbose": False},
    }


def smoke_data(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 1))
    y = X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0]  # Nguyen-1 style target
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stefan_boltzmann")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--n-samples", type=int, default=100_000,
                    help="expression budget; DSO stops after this many sampled expressions")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--const", action="store_true", help="add DSO's fitted-constant token")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run to check the install")
    ap.add_argument("--dry-run", action="store_true", help="build data only; do not import DSO")
    ap.add_argument("--out", type=Path, default=None, help="append one row in run_benchmark format")
    args = ap.parse_args()

    if args.smoke:
        X, y = smoke_data(seed=args.seed)
        names, Xt, yt, label = ["x1"], X, y, "smoke"
        noise = 0.0
    else:
        task, X, y, Xt, yt = make_data(args.task, args.noise, args.n_train, args.seed)
        names, label, noise = task.variables, task.name, args.noise

    if args.dry_run:
        print(f"{label}: train X {X.shape}, y {y.shape}; test X {Xt.shape}; variables {names}")
        return

    # Import here so that --dry-run works without TensorFlow.
    from dso.task.regression.sklearn import DeepSymbolicRegressor

    n_samples = 2000 if args.smoke else args.n_samples
    model = DeepSymbolicRegressor(make_config(n_samples, args.seed, args.const))
    t0 = time.perf_counter()
    model.fit(X, y)
    seconds = time.perf_counter() - t0

    # Convert DSO's x1..xn symbols to the benchmark's variable names.
    sym = model.program_.sympy_expr
    subs = {sp.Symbol(f"x{i + 1}"): sp.Symbol(v) for i, v in enumerate(names)}
    expr = str(sp.simplify(sym.subs(subs)))
    y_hat = np.asarray(model.predict(Xt), dtype=float)
    y_hat = np.broadcast_to(y_hat, yt.shape)
    row = dict(suite="smoke" if args.smoke else "static", task=label, target="y", noise=noise,
               estimator="-", method="dso_const" if args.const else "dso", seed=args.seed, status="ok",
               r2_extrap=round(r2(yt, y_hat), 8), nrmse_extrap=round(nrmse(yt, y_hat), 8),
               complexity=complexity(expr), seconds=round(seconds, 1), deriv_rel_err="",
               expression=expr)

    print(f"{label} noise={noise} seed={args.seed}  R2_extrap={row['r2_extrap']}  "
          f"complexity={row['complexity']}  time={row['seconds']} s")
    print(f"expression: {expr}")

    if args.out and not args.smoke:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        new = not args.out.exists()
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
