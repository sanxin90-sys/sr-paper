"""
clean_rerun_extra.py
====================

Extends clean_rerun_rn_rh.py (same data decisions, splits, training subsample, baselines,
bootstrap and decision rule) to the engines that were not in that run:

* gplearn, 5 seeds, 45 s per fit;
* SINDy with the extended library (cubic, log, sqrt, 1/x, ratio terms), deterministic;
* KAN v2 (kan_eval.fit_v2), 3 seeds, when torch and pykan import;
* PhySO without units, 2 seeds, 5 epochs, when torch and physo import.

Rule (unchanged): an engine "beats" the re-fitted classical law in a split only if the 95 % CI of
RMSE(engine) - RMSE(classical) from a daily block bootstrap (500) lies entirely below zero.

Usage: python src/clean_rerun_extra.py "Dados Rebio Jaru 2006 a 2008.xlsx" \
       data/fluxnet_BR-Ji3/AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv [engines...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sr_benchmark"))
from clean_rerun_rn_rh import boot_diff, rmse, r2, N_TRAIN  # noqa: E402
from real_discovery import load_jaru, _es  # noqa: E402
from methods import fit_gplearn, fit_sindy, fit_physo, complexity, _sympy_predict  # noqa: E402

OUT = None  # set in main: one file per engine set, e.g. clean_rerun_extra_gplearn.csv
BUDGET = int(os.environ.get("BUDGET", 45))


def _kan(X, y, names, seed):
    from kan_eval import fit_v2
    expr, _ = fit_v2(X, y, names, seed=seed)
    return expr, _sympy_predict(expr, names)


def _physo(X, y, names, seed):
    fit = fit_physo(X, y, names, time_budget_s=250, seed=seed)
    return fit.expression, fit.predict


def _gplearn(X, y, names, seed):
    fit = fit_gplearn(X, y, names, time_budget_s=BUDGET, seed=seed)
    return fit.expression, fit.predict


def _sindy_ext(X, y, names, seed):
    fit = fit_sindy(X, y, names)
    return fit.expression, fit.predict


ENGINES = {"gplearn": (_gplearn, range(5)), "sindy_ext": (_sindy_ext, [0]),
           "kan_v2": (_kan, range(3)), "physo": (_physo, range(2))}


def run(task, split, tr, te, feats, ycol, law, p0, engines, rows):
    days = te.timestamp.dt.date.to_numpy()
    Xte = te[feats].to_numpy(dtype=float)
    yte = te[ycol].to_numpy(dtype=float)
    for name in engines:
        fn, seeds = ENGINES[name]
        for seed in seeds:
            sub = tr.sample(min(N_TRAIN, len(tr)), random_state=seed)  # same subsample as the Operon run
            Xtr, ytr = sub[feats].to_numpy(dtype=float), sub[ycol].to_numpy(dtype=float)
            popt, _ = curve_fit(law, Xtr, ytr, p0=p0, maxfev=20000)
            p_cl = law(Xte, *popt)
            try:
                expr, pred = fn(Xtr, ytr, feats, seed)
                with np.errstate(all="ignore"):
                    p = np.asarray(pred(Xte), float) * np.ones(len(yte))
                ok = bool(np.isfinite(p).all())
            except Exception as exc:  # record failures, do not drop them
                expr, ok, p = f"error: {type(exc).__name__}: {str(exc)[:100]}", False, None
            lo, hi = boot_diff(yte, p, p_cl, days) if ok else (np.nan, np.nan)
            rows.append(dict(task=task, split=split, method=name, seed=seed, n_train=len(sub), n_test=len(te),
                             rmse=rmse(yte, p) if ok else np.nan, r2=r2(yte, p) if ok else np.nan,
                             rmse_classical_same_seed=rmse(yte, p_cl),
                             complexity=complexity(expr) if ok else np.nan,
                             d_rmse_lo=lo, d_rmse_hi=hi, beats=bool(ok and hi < 0), expression=expr))
            print(task, split, name, seed, f"RMSE {rows[-1]['rmse']:.3f} vs classical {rmse(yte, p_cl):.3f}",
                  f"CI [{lo:.3f}, {hi:.3f}]", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)


def main(xlsx, flx, engines):
    global OUT
    OUT = HERE / "sr_benchmark" / "results" / f"clean_rerun_extra_{'_'.join(engines)}.csv"
    rows = []
    j = load_jaru(xlsx).dropna(subset=["Rn", "PAR"])
    j = j[~((j.year == 2006) & (j.month == 10))]
    lin = lambda X, a, b: a * X[:, 0] + b  # noqa: E731
    splits = {
        "A_2006_H1toH2": (j[(j.year == 2006) & (j.month <= 6)], j[(j.year == 2006) & (j.month >= 7)]),
        "B_2008_H1toH2": (j[(j.year == 2008) & (j.month <= 6)], j[(j.year == 2008) & (j.month >= 7)]),
        "C_2006to2008_acrossstep": (j[j.year == 2006], j[j.year == 2008]),
    }
    for name, (tr, te) in splits.items():
        run("Rn~PAR", name, tr, te, ["PAR"], "Rn", lin, [2.0, -20.0], engines, rows)

    f = pd.read_csv(flx, na_values=[-9999], usecols=["TIMESTAMP_END", "TA_F_MDS", "TA_F_MDS_QC", "RH"])
    f["timestamp"] = pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")
    f["T"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0)
    f["date"] = (f.timestamp - pd.Timedelta("30min")).dt.date
    full = f.groupby("date")["T"].transform("count") == 48
    f["Tmin_day"] = f.groupby("date")["T"].transform("min").where(full)
    f["year"] = f.timestamp.dt.year
    f = f.dropna(subset=["RH", "T", "Tmin_day"])
    tetens = lambda X, a, b: 100 * _es(a * X[:, 1] + b) / _es(X[:, 0])  # noqa: E731
    tr = f[f.year == 2006]
    for y in (2005, 2008, 2009, 2010):
        run("RH~T,Tmin", f"train2006_test{y}", tr, f[f.year == y], ["T", "Tmin_day"], "RH", tetens, [1.0, -1.0],
            engines, rows)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or list(ENGINES))
