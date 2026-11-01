"""
pysr_pareto_seeds.py
====================

PySR on the real Rebio Jaru targets with the three fixes proposed in
methods/pysr.md (run locally: PySR needs Julia):

* several seeds and training subsamples instead of a single run;
* the whole Pareto front is saved, never only the `model_selection="best"`
  equation (that rule picked complexity-3 forms in the first local run);
* the equation is chosen on a validation year (2007) and only then scored on
  the test year (2008): train 2006 -> select 2007 -> test 2008.

Output: one CSV row per (target, seed, Pareto candidate) with train, validation
and test RMSE, plus a flag for the candidate chosen on validation.

Usage
-----
    python src/pysr_pareto_seeds.py "Dados Rebio Jaru 2006 a 2008.xlsx"
    python src/pysr_pareto_seeds.py <xlsx> --targets net_radiation --seeds 0 1 2 3 4 --budget 600
    python src/pysr_pareto_seeds.py <xlsx> --dry-run     # data split only, no Julia needed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from real_discovery import TARGETS, load_jaru


def split3(df: pd.DataFrame, target, season: str | None, n_max: int, seed: int):
    d = df.dropna(subset=[target.y, *target.features])
    if season:
        d = d[d.season == season]
    train, val, test = d[d.year == 2006], d[d.year == 2007], d[d.year == 2008]
    if len(train) > n_max:  # a different subsample per seed: seed and data variability together
        train = train.sample(n_max, random_state=seed)
    arr = lambda a: np.array(a.to_numpy(dtype=float), dtype=float, order="C", copy=True)
    return [(arr(p[target.features]), arr(p[target.y])) for p in (train, val, test)]


def run(df, name, seed, budget, season, n_max, dry_run):
    t = TARGETS[name]
    (X, y), (Xv, yv), (Xt, yt) = split3(df, t, season, n_max, seed)
    print(f"{name} seed={seed}: train {len(y)}, val {len(yv)}, test {len(yt)}")
    if dry_run:
        return pd.DataFrame()

    from pysr import PySRRegressor

    reg = PySRRegressor(
        niterations=100000, populations=30, maxsize=25,
        binary_operators=["+", "-", "*", "/"], unary_operators=["exp", "log", "sqrt", "square"],
        timeout_in_seconds=budget, model_selection="accuracy", random_state=seed,
        deterministic=True, parallelism="serial", progress=False, verbosity=0,
    )
    reg.fit(X, y, variable_names=t.features)
    eqs = reg.equations_[["complexity", "loss", "equation"]].copy()
    rmse = lambda Xs, ys, i: float(np.sqrt(np.mean((ys - reg.predict(Xs, index=i)) ** 2)))
    eqs["rmse_train"] = [rmse(X, y, i) for i in range(len(eqs))]
    eqs["rmse_val"] = [rmse(Xv, yv, i) for i in range(len(eqs))]
    eqs["rmse_test"] = [rmse(Xt, yt, i) for i in range(len(eqs))]
    eqs["chosen_on_val"] = eqs.rmse_val == eqs.rmse_val.min()
    eqs.insert(0, "seed", seed)
    eqs.insert(0, "season", season or "all")
    eqs.insert(0, "target", name)
    return eqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--targets", nargs="+", default=["net_radiation", "relative_humidity"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--season", choices=["wet", "dry"], default=None)
    ap.add_argument("--n-max", type=int, default=3000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/pysr_pareto_seeds.csv"))
    args = ap.parse_args()

    df = load_jaru(args.xlsx)
    fronts = [run(df, n, s, args.budget, args.season, args.n_max, args.dry_run)
              for n in args.targets for s in args.seeds]
    if args.dry_run:
        return
    out = pd.concat(fronts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out[out.chosen_on_val].to_string(index=False, max_colwidth=80))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
