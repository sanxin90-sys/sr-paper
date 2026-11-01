"""
kan_eval.py
===========

Focused evaluation of KAN (pykan) as a symbolic-regression engine.

Two configurations are compared:

* ``v1``  - the configuration used in the benchmark (sr_benchmark.methods.fit_kan):
            raw inputs, width [d, 3, 1], L-BFGS, auto_symbolic, no refit.
* ``v2``  - the workflow recommended by the pykan authors: standardised inputs
            and target, held-out validation split, sparsity penalty, pruning,
            symbolic snapping with a restricted library, and an affine refit of
            the snapped network before the formula is read out.

For every fit we also keep the *spline* (pre-snapping) prediction, so the loss
caused by snapping is measured separately from the loss of the network itself.

Usage
-----
    python kan_eval.py static  --seeds 0 1 2 --out results/kan_static.csv
    python kan_eval.py real    --xlsx "Dados Rebio Jaru 2006 a 2008.xlsx" --seeds 0 1 2 --out results/kan_real.csv
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "sr_benchmark"))
from methods import _sympy_predict, complexity  # noqa: E402

CKPT = tempfile.mkdtemp(prefix="kan_ckpt_")
LIB = ["x", "x^2", "x^3", "exp", "log", "sqrt", "1/x", "tanh", "sin"]


def _r2(y, p):
    p = np.asarray(p, float)
    if not np.all(np.isfinite(p)):
        return float("nan")
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def _tensor(a):
    return torch.tensor(np.asarray(a, float), dtype=torch.float32)


def fit_v1(X, y, names, seed=0, steps=50):
    from kan import KAN
    torch.manual_seed(seed)
    data = {"train_input": _tensor(X), "train_label": _tensor(y[:, None]),
            "test_input": _tensor(X), "test_label": _tensor(y[:, None])}
    model = KAN(width=[X.shape[1], 3, 1], grid=5, k=3, seed=seed, device="cpu", auto_save=False, ckpt_path=CKPT)
    model.fit(data, opt="LBFGS", steps=steps, lamb=0.001, log=10**6)
    net = model.copy()  # keep the spline network before snapping
    spline_pred = lambda Z: net(_tensor(Z)).detach().numpy().ravel()
    model.auto_symbolic(verbose=0)
    expr = str(model.symbolic_formula(var=list(sp.symbols(names)))[0][0])
    return expr, spline_pred


def fit_v2(X, y, names, seed=0, hidden=3, steps=100, lamb=0.003, lib=LIB):
    """Standardise -> train (val split) -> prune -> snap -> affine refit -> de-standardise."""
    from kan import KAN
    rng = np.random.default_rng(seed)
    mx, sx = X.mean(0), X.std(0) + 1e-12
    my, sy = y.mean(), y.std() + 1e-12
    Z, u = (X - mx) / sx, (y - my) / sy
    idx = rng.permutation(len(Z))
    n_val = max(len(Z) // 5, 1)
    va, tr = idx[:n_val], idx[n_val:]
    data = {"train_input": _tensor(Z[tr]), "train_label": _tensor(u[tr, None]),
            "test_input": _tensor(Z[va]), "test_label": _tensor(u[va, None])}
    torch.manual_seed(seed)
    model = KAN(width=[X.shape[1], hidden, 1], grid=5, k=3, seed=seed, device="cpu", auto_save=False, ckpt_path=CKPT)
    model.fit(data, opt="LBFGS", steps=steps, lamb=lamb, lamb_entropy=2.0, log=10**6)
    try:
        model = model.prune(node_th=1e-2, edge_th=3e-2)
        model.fit(data, opt="LBFGS", steps=steps // 2, lamb=0.0, log=10**6)
    except Exception:
        pass
    net = model.copy()

    def spline_pred(Xn):
        return net(_tensor((Xn - mx) / sx)).detach().numpy().ravel() * sy + my

    model.auto_symbolic(lib=lib, verbose=0, weight_simple=0.8)
    model.fit(data, opt="LBFGS", steps=steps // 2, lamb=0.0, update_grid=False, log=10**6)  # refit affine params
    zs = sp.symbols([f"z{i}" for i in range(X.shape[1])])
    f = model.symbolic_formula(var=list(zs))[0][0]
    xs = sp.symbols(names)
    f = f.subs({z: (x - float(m)) / float(s) for z, x, m, s in zip(zs, xs, mx, sx)})
    expr = sp.nsimplify(sp.expand(float(sy) * f + float(my)), rational=False)
    expr = sp.simplify(expr) if complexity(str(expr)) < 200 else expr
    return str(sp.N(expr, 6)), spline_pred


CONFIGS = {"v1": fit_v1, "v2": fit_v2}


def evaluate(X, y, Xt, yt, names, config, seed):
    t0 = time.perf_counter()
    try:
        expr, spline = CONFIGS[config](X, y, names, seed)
        sec = time.perf_counter() - t0
        pred = _sympy_predict(expr, names)(Xt)
        return dict(status="ok", r2_spline=_r2(yt, spline(Xt)), r2_symbolic=_r2(yt, pred),
                    rmse_symbolic=float(np.sqrt(np.mean((yt - pred) ** 2))),
                    complexity=complexity(expr), seconds=round(sec, 1), expression=expr)
    except Exception as exc:
        return dict(status=f"error: {type(exc).__name__}: {str(exc)[:120]}", seconds=round(time.perf_counter() - t0, 1))


# ---------------------------------------------------------------------------
def run_static(seeds, configs, noises, out):
    from tasks import STATIC_TASKS, add_noise, sample_static
    rows = []
    for task in STATIC_TASKS:
        for noise in noises:
            for seed in seeds:
                X, y, Xt, yt = _static_data(task, noise, seed, sample_static, add_noise)
                for c in configs:
                    r = evaluate(X, y, Xt, yt, task.variables, c, seed)
                    rows.append(dict(suite="static", task=task.name, noise=noise, config=c, seed=seed, **r))
                    _log(rows[-1]); pd.DataFrame(rows).to_csv(out, index=False)


def _static_data(task, noise, seed, sample_static, add_noise, n_train=1000):
    rng = np.random.default_rng(seed)  # same draw order as run_benchmark.run_static
    X, y_clean = sample_static(task, n_train, rng)
    Xt, yt = sample_static(task, 2000, rng, test=True)
    return X, add_noise(y_clean, noise, np.random.default_rng(seed + 1)), Xt, yt


def run_real(xlsx, seeds, configs, out, seasons=(None,)):
    from real_discovery import TARGETS, load_jaru, split
    df = load_jaru(xlsx)
    rows = []
    for name, t in TARGETS.items():
        for season in seasons:
            for seed in seeds:
                X, y, Xt, yt = split(df, t, season, seed=seed)
                for c in configs:
                    r = evaluate(X, y, Xt, yt, t.features, c, seed)
                    rows.append(dict(suite="real", task=name, season=season or "all", config=c, seed=seed,
                                     n_train=len(y), n_test=len(yt), **r))
                    _log(rows[-1]); pd.DataFrame(rows).to_csv(out, index=False)


def _log(r):
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "expression"},
          "|", str(r.get("expression", ""))[:140], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["static", "real"])
    ap.add_argument("--xlsx", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--configs", nargs="+", default=["v1", "v2"])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--seasons", nargs="+", default=["all"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.suite == "static":
        run_static(a.seeds, a.configs, a.noise, a.out)
    else:
        run_real(a.xlsx, a.seeds, a.configs, a.out, [None if s == "all" else s for s in a.seasons])
