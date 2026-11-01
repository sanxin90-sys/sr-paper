"""
pysr_clean_rn_rh.py
===================

PySR on the clean Rn ~ PAR and RH ~ T, Tmin_day comparisons of EXP-017/017b
(EXPERIMENT_LOG), so that PySR sits in the same table as Operon and SINDy.

Data and splits are those of src/clean_rerun_rn_rh.py, unchanged:
* Rn ~ PAR (spreadsheet, Oct/2006 removed), train and test on the same side
  of the 2007 radiometer step: A 2006 Jan-Jun -> Jul-Dec, B 2008 Jan-Jun ->
  Jul-Dec; C 2006 -> 2008 across the step (reference only).
* RH ~ T, Tmin_day: FLUXNET RH and TA at 62 m (QC = 0), Tmin_day from full
  days only; train 2006, test 2005, 2008, 2009, 2010.
* Classical laws re-fitted on the same 3000-row subsample: Rn = a*PAR + b;
  Tetens T0 (Td = a*Tmin + b) and T1 (Td = a*Tmin + c*T + b, EXP-017b).

PySR (fixed before running, 2026-10-29): 5 seeds (the seed also changes the
3000-row subsample, as for Operon), 120 s each, operators + - * / exp log
sqrt square, maxsize 25. The equation is chosen on a validation block = the
last 20 % of the training days (time order), never on the test set; the
whole Pareto front is saved with its test RMSE.

Rule (same as EXP-017): PySR beats the classical law in a split only if the
95 % CI of RMSE(PySR) - RMSE(classical) by daily-block bootstrap (500) lies
entirely below zero. For RH the comparison is made against T0 and T1.

Usage
-----
    python pysr_clean_rn_rh.py "<xlsx>" <AMF_BR-Ji3_FLUXNET_FLUXMET_HH_*.csv> [--no-pysr]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real_discovery import load_jaru, _es  # noqa: E402

SEEDS = range(5)
N_TRAIN = 3000
OUT = HERE / "sr_benchmark" / "results"


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def boot_diff(y, p_sr, p_cl, days, n=500, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    idx = {d: np.where(days == d)[0] for d in uniq}
    out = []
    for _ in range(n):
        ii = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq))])
        out.append(rmse(y[ii], p_sr[ii]) - rmse(y[ii], p_cl[ii]))
    return np.percentile(out, [2.5, 97.5])


def arr(a):
    return np.array(a, dtype=float, order="C", copy=True)


def split_val(tr: pd.DataFrame):
    """Last 20 % of the training days (time order) as validation block."""
    days = np.sort(tr.date.unique())
    cut = days[int(0.8 * len(days))]
    return tr[tr.date < cut], tr[tr.date >= cut]


def run(task, tests, tr, feats, ycol, laws, use_pysr, budget, rows, fronts):
    """tests: {split name: test frame}. laws: {name: (f, p0)}; the first law is the reference."""
    for seed in SEEDS:
        sub = tr.sample(min(N_TRAIN, len(tr)), random_state=seed)
        Xtr, ytr = arr(sub[feats]), arr(sub[ycol])
        fitted = {n: curve_fit(f, Xtr, ytr, p0=p0, maxfev=20000)[0] for n, (f, p0) in laws.items()}
        if seed == 0:
            for split, te in tests.items():
                X, y = arr(te[feats]), arr(te[ycol])
                for n, (f, _) in laws.items():
                    p = f(X, *fitted[n])
                    rows.append(dict(task=task, split=split, method=n, seed=0, rmse=rmse(y, p), r2=r2(y, p),
                                     expression=" ".join(f"{v:.4g}" for v in fitted[n])))
                    print(task, split, n, f"RMSE {rows[-1]['rmse']:.3f} R2 {rows[-1]['r2']:.3f}", flush=True)
        if not use_pysr:
            continue
        from pysr import PySRRegressor

        fit_part, val_part = split_val(sub)
        reg = PySRRegressor(
            niterations=100000, populations=30, maxsize=25,
            binary_operators=["+", "-", "*", "/"], unary_operators=["exp", "log", "sqrt", "square"],
            timeout_in_seconds=budget, model_selection="accuracy", random_state=seed,
            deterministic=True, parallelism="serial", progress=False, verbosity=0,
        )
        reg.fit(arr(fit_part[feats]), arr(fit_part[ycol]), variable_names=feats)
        eqs = reg.equations_[["complexity", "loss", "equation"]].copy()
        Xv, yv = arr(val_part[feats]), arr(val_part[ycol])
        with np.errstate(all="ignore"):
            eqs["rmse_val"] = [rmse(yv, reg.predict(Xv, index=i)) for i in range(len(eqs))]
        eqs["rmse_val"] = eqs.rmse_val.fillna(np.inf)
        k = int(eqs.rmse_val.idxmin())
        for split, te in tests.items():
            X, y, days = arr(te[feats]), arr(te[ycol]), te.date.to_numpy()
            with np.errstate(all="ignore"):
                eqs[f"rmse_test_{split}"] = [rmse(y, reg.predict(X, index=i)) for i in range(len(eqs))]
                p = reg.predict(X, index=k)
            ok = np.isfinite(p).all()
            row = dict(task=task, split=split, method="pysr", seed=seed, rmse=rmse(y, p) if ok else np.nan,
                       r2=r2(y, p) if ok else np.nan, complexity=int(eqs.complexity[k]), expression=eqs.equation[k])
            for n, (f, _) in laws.items():
                p_cl = f(X, *fitted[n])
                lo, hi = boot_diff(y, p, p_cl, days) if ok else (np.nan, np.nan)
                row[f"rmse_{n}_same_seed"] = rmse(y, p_cl)
                row[f"d_rmse_vs_{n}_lo"], row[f"d_rmse_vs_{n}_hi"] = lo, hi
            rows.append(row)
            print(task, split, "seed", seed, f"pysr RMSE {row['rmse']:.3f} (complexity {row['complexity']})",
                  {n: (round(row[f'd_rmse_vs_{n}_lo'], 3), round(row[f'd_rmse_vs_{n}_hi'], 3)) for n in laws},
                  eqs.equation[k], flush=True)
        eqs.insert(0, "seed", seed)
        eqs.insert(0, "task", task)
        fronts.append(eqs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("flx")
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--no-pysr", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rows, fronts = [], []

    # Rn ~ PAR (spreadsheet), same splits as clean_rerun_rn_rh.py
    j = load_jaru(args.xlsx).dropna(subset=["Rn", "PAR"])
    j = j[~((j.year == 2006) & (j.month == 10))].copy()
    j["date"] = j.timestamp.dt.date
    lin = {"classical": (lambda X, a, b: a * X[:, 0] + b, [2.0, -20.0])}
    run("Rn~PAR", {"A_2006_H1toH2": j[(j.year == 2006) & (j.month >= 7)]},
        j[(j.year == 2006) & (j.month <= 6)], ["PAR"], "Rn", lin, not args.no_pysr, args.budget, rows, fronts)
    run("Rn~PAR", {"B_2008_H1toH2": j[(j.year == 2008) & (j.month >= 7)]},
        j[(j.year == 2008) & (j.month <= 6)], ["PAR"], "Rn", lin, not args.no_pysr, args.budget, rows, fronts)
    run("Rn~PAR", {"C_2006to2008_acrossstep": j[j.year == 2008]},
        j[j.year == 2006], ["PAR"], "Rn", lin, not args.no_pysr, args.budget, rows, fronts)

    # RH ~ T, Tmin_day (FLUXNET, 62 m)
    f = pd.read_csv(args.flx, na_values=[-9999], usecols=["TIMESTAMP_END", "TA_F_MDS", "TA_F_MDS_QC", "RH"])
    f["timestamp"] = pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")
    f["T"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0)
    f["date"] = (f.timestamp - pd.Timedelta("30min")).dt.date
    full = f.groupby("date")["T"].transform("count") == 48
    f["Tmin_day"] = f.groupby("date")["T"].transform("min").where(full)
    f["year"] = f.timestamp.dt.year
    f = f.dropna(subset=["RH", "T", "Tmin_day"])
    tetens = {
        "T0": (lambda X, a, b: 100 * _es(a * X[:, 1] + b) / _es(X[:, 0]), [1.0, -1.0]),
        "T1": (lambda X, a, c, b: 100 * _es(a * X[:, 1] + c * X[:, 0] + b) / _es(X[:, 0]), [1.0, 0.0, -1.0]),
    }
    run("RH~T,Tmin", {f"train2006_test{y}": f[f.year == y] for y in (2005, 2008, 2009, 2010)},
        f[f.year == 2006], ["T", "Tmin_day"], "RH", tetens, not args.no_pysr, args.budget, rows, fronts)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "baselines" if args.no_pysr else "pysr"
    pd.DataFrame(rows).to_csv(args.out / f"pysr_clean_rn_rh_{tag}.csv", index=False)
    if fronts:
        pd.concat(fronts, ignore_index=True).to_csv(args.out / "pysr_clean_rn_rh_fronts.csv", index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
