"""
Benchmark runner: symbolic-regression engines on environmental equations.

Static suite : recover a closed-form diagnostic relation y = f(x) from noisy
               samples; scored on a wider (extrapolation) box, noise-free.
Dynamic suite: recover ODE right-hand sides from a noisy trajectory, after a
               derivative-estimation step (finite differences, Savitzky-Golay
               or the X-TFC stage of AI-Lorenz); scored against the exact RHS
               on an independent trajectory.

Results are appended row by row to a CSV so a long run can be interrupted.

Usage
-----
    python run_benchmark.py --suite static  --noise 0 0.05 --budget 45
    python run_benchmark.py --suite dynamic --noise 0 0.01 0.05 --budget 45
    python run_benchmark.py --suite static --methods pysr   # on a machine with Julia
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

import numpy as np

from derivatives import ESTIMATORS
from methods import METHODS, Unavailable, complexity
from tasks import JARU_T, LORENZ, STATIC_TASKS, add_noise, sample_static, simulate_jaru, simulate_lorenz

FIELDS = [
    "suite", "task", "target", "noise", "estimator", "method", "seed", "status",
    "r2_extrap", "nrmse_extrap", "complexity", "seconds", "deriv_rel_err", "expression",
]


def r2(y_true, y_pred) -> float:
    y_pred = np.asarray(y_pred, float)
    if not np.all(np.isfinite(y_pred)):
        return float("nan")
    return float(1.0 - np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def nrmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - np.asarray(y_pred, float)) ** 2)) / np.std(y_true))


def _run_one(method, X, y, Xt, yt, names, budget, seed):
    try:
        fit = METHODS[method](X, y, names, budget, seed)
        pred = fit.predict(Xt)
        return dict(status="ok", r2_extrap=r2(yt, pred), nrmse_extrap=nrmse(yt, pred),
                    complexity=complexity(fit.expression), seconds=round(fit.seconds, 1),
                    expression=fit.expression)
    except Unavailable as exc:
        return dict(status=f"unavailable: {exc}")
    except Exception as exc:
        return dict(status=f"error: {type(exc).__name__}: {str(exc)[:120]}")


def static_suite(methods, noises, budget, seed, n_train, writer, only=None):
    for task in STATIC_TASKS:
        if only and task.name not in only:
            continue
        rng = np.random.default_rng(seed)
        X, y_clean = sample_static(task, n_train, rng)
        Xt, yt = sample_static(task, 2000, rng, test=True)
        for noise in noises:
            y = add_noise(y_clean, noise, np.random.default_rng(seed + 1))
            for m in methods:
                row = dict(suite="static", task=task.name, target="y", noise=noise, estimator="-",
                           method=m, seed=seed)
                row.update(_run_one(m, X, y, Xt, yt, task.variables, budget, seed))
                writer(row)


def _dynamic_data(task, seed):
    if task is LORENZ:
        train = simulate_lorenz(dt=0.01, t_final=20.0, x0=(-8.0, 7.0, 27.0))
        test = simulate_lorenz(dt=0.01, t_final=10.0, x0=(5.0, -3.0, 20.0))
    else:
        train = simulate_jaru(n_days=60, seed=seed)
        test = simulate_jaru(n_days=30, seed=seed + 100)
    return train, test


# Per task: estimator -> (kwargs for noise == 0, kwargs for noise > 0); windows in samples.
# Lorenz values follow the AI-Lorenz paper; Jaru values were chosen
# by a small grid search per estimator on the training trajectory (5 % noise).
ESTIMATOR_SETTINGS = {
    "lorenz63": {
        "finite_difference": ({}, {}),
        "savitzky_golay": ({"window": 5, "polyorder": 3}, {"window": 21, "polyorder": 3}),
        "xtfc_ai_lorenz": ({"samples_per_subdomain": 10, "n_neurons": 10},
                           {"samples_per_subdomain": 100, "n_neurons": 30, "smooth_window": 31}),
    },
    "jaru_temperature": {
        "finite_difference": ({}, {}),
        "savitzky_golay": ({"window": 5, "polyorder": 3}, {"window": 7, "polyorder": 2}),
        "xtfc_ai_lorenz": ({"samples_per_subdomain": 10, "n_neurons": 10},
                           {"samples_per_subdomain": 48, "n_neurons": 12}),
    },
}


def dynamic_suite(methods, noises, budget, seed, writer):
    for task in (JARU_T, LORENZ):
        train, test = _dynamic_data(task, seed)
        names = task.state_names + task.forcing_names
        for noise in noises:
            rng = np.random.default_rng(seed + 2)
            s = train["states"]
            s_noisy = s + rng.normal(0.0, noise * s.std(0), s.shape)
            f = train["forcing"]
            f_noisy = f + rng.normal(0.0, noise * f.std(0), f.shape) if f.size else f
            for est, fn in ESTIMATORS.items():
                kw = ESTIMATOR_SETTINGS[task.name][est][int(noise > 0)]
                s_hat, ds_hat = fn(train["t"], s_noisy, **kw)
                deriv_err = np.mean(np.abs(ds_hat - train["derivs"]), 0) / np.mean(np.abs(train["derivs"]), 0)
                X = np.column_stack([s_hat, f_noisy]) if f.size else s_hat
                Xt = np.column_stack([test["states"], test["forcing"]]) if f.size else test["states"]
                # trim the ends, where every estimator is least reliable
                X, ds_hat = X[10:-10], ds_hat[10:-10]
                for k, target in enumerate(task.state_names):
                    for m in methods:
                        row = dict(suite="dynamic", task=task.name, target=f"d{target}/dt", noise=noise,
                                   estimator=est, method=m, seed=seed, deriv_rel_err=round(float(deriv_err[k]), 4))
                        row.update(_run_one(m, X, ds_hat[:, k], Xt, test["derivs"][:, k], names, budget, seed))
                        writer(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["static", "dynamic"], required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.05])
    ap.add_argument("--budget", type=int, default=45, help="seconds per (task, method)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", help="static task names to run (default: all)")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/benchmark.csv"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    fh = args.out.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    def writer(row):
        w.writerow({k: row.get(k, "") for k in FIELDS})
        fh.flush()
        print(f"{time.strftime('%H:%M:%S')} {row['task']:<24} {row.get('target',''):<6} "
              f"noise={row['noise']:<5} {row.get('estimator','-'):<18} {row['method']:<12} "
              f"{row['status'][:30]:<30} R2={row.get('r2_extrap', float('nan')):.4f}", flush=True)

    try:
        if args.suite == "static":
            static_suite(args.methods, args.noise, args.budget, args.seed, args.n_train, writer, args.tasks)
        else:
            dynamic_suite(args.methods, args.noise, args.budget, args.seed, writer)
    except Exception:
        traceback.print_exc()
    finally:
        fh.close()


if __name__ == "__main__":
    main()
