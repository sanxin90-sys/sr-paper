"""
clean_rerun_rn_rh.py
====================

Redo of the paper/03 comparison (symbolic regression vs re-fitted classical law) for
net radiation and relative humidity, with the data problems found on 2026-10-28 removed.

Data decisions (fixed before running; see EXPERIMENT_LOG DQ-001..003):
* Rn ~ PAR (spreadsheet; PAR in W m-2). The net radiometer has a step change in 2007 and
  Oct/2006 and most of 2007 are suspect, so train and test always lie on the same side of the step:
    A. 2006, Jan-Jun -> Jul-Dec (Oct excluded)          [pre-step]
    B. 2008, Jan-Jun -> Jul-Dec                          [post-step]
    C. 2006 (Oct excluded) -> 2008                       [across the step; reference only]
* RH ~ T, Tmin_day: FLUXNET RH and TA (62 m, QC = 0), consistent across years and checked against
  the spreadsheet RH (r = 0.97 in 2006 and 2008). Dew-point proxy = daily minimum of TA.
  Train 2006 (3000 random half-hours), test 2005, 2008, 2009, 2010 (2007 excluded: spreadsheet RH r = 0.73
  with FLUXNET, T22 defective).
Methods: classical law re-fitted on the training set; SINDy quadratic (deterministic); Operon, 5 seeds
(seed also changes the 3000-row training subsample). PySR cannot run in the cloud (Julia blocked).
Scores: test RMSE and R2; 95 % CI of RMSE(SR) - RMSE(classical) by daily block bootstrap (500).
Decision rule: SR "beats" the classical law only if that CI lies entirely below zero.

Usage: python src/clean_rerun_rn_rh.py "Dados Rebio Jaru 2006 a 2008.xlsx" data/fluxnet_BR-Ji3/AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv
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
from real_discovery import load_jaru, _es  # noqa: E402
from methods import fit_operon, fit_sindy_poly, complexity  # noqa: E402

OUT = HERE / "sr_benchmark" / "results" / "clean_rerun_rn_rh.csv"
BUDGET = int(os.environ.get("BUDGET", 45))
SEEDS = range(5)
N_TRAIN = 3000


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


def run(task, split, tr, te, feats, ycol, law, p0, rows):
    days = te.timestamp.dt.date.to_numpy()
    Xte = np.array(te[feats].to_numpy(), dtype=float, order="C", copy=True)  # Operon rejects non-contiguous arrays
    yte = te[ycol].to_numpy(dtype=float)
    for seed in SEEDS:
        sub = tr.sample(min(N_TRAIN, len(tr)), random_state=seed)
        Xtr, ytr = sub[feats].to_numpy(), sub[ycol].to_numpy()
        popt, _ = curve_fit(law, Xtr, ytr, p0=p0, maxfev=20000)
        p_cl = law(Xte, *popt)
        if seed == 0:
            rows.append(dict(task=task, split=split, method="classical", seed=0, n_train=len(sub), n_test=len(te),
                             rmse=rmse(yte, p_cl), r2=r2(yte, p_cl), n_const=len(popt),
                             expression=" ".join(f"{v:.4g}" for v in popt)))
            fit = fit_sindy_poly(Xtr, ytr, feats)
            p = fit.predict(Xte)
            lo, hi = boot_diff(yte, p, p_cl, days)
            rows.append(dict(task=task, split=split, method="sindy_poly2", seed=0, n_train=len(sub), n_test=len(te),
                             rmse=rmse(yte, p), r2=r2(yte, p), complexity=complexity(fit.expression),
                             d_rmse_lo=lo, d_rmse_hi=hi, expression=fit.expression))
        fit = fit_operon(Xtr, ytr, feats, time_budget_s=BUDGET, seed=seed)
        p = fit.predict(Xte)
        ok = np.isfinite(p).all()
        lo, hi = boot_diff(yte, p, p_cl, days) if ok else (np.nan, np.nan)
        rows.append(dict(task=task, split=split, method="operon", seed=seed, n_train=len(sub), n_test=len(te),
                         rmse=rmse(yte, p) if ok else np.nan, r2=r2(yte, p) if ok else np.nan,
                         rmse_classical_same_seed=rmse(yte, p_cl), complexity=complexity(fit.expression),
                         d_rmse_lo=lo, d_rmse_hi=hi, expression=fit.expression))
        print(task, split, "seed", seed, f"operon RMSE {rows[-1]['rmse']:.3f} vs classical {rmse(yte, p_cl):.3f}",
              f"CI [{lo:.3f}, {hi:.3f}]", flush=True)


def main(xlsx, flx):
    rows = []
    # ---------------- Rn ~ PAR (spreadsheet) ----------------
    j = load_jaru(xlsx).dropna(subset=["Rn", "PAR"])
    j = j[~((j.year == 2006) & (j.month == 10))]
    lin = lambda X, a, b: a * X[:, 0] + b  # noqa: E731
    splits = {
        "A_2006_H1toH2": (j[(j.year == 2006) & (j.month <= 6)], j[(j.year == 2006) & (j.month >= 7)]),
        "B_2008_H1toH2": (j[(j.year == 2008) & (j.month <= 6)], j[(j.year == 2008) & (j.month >= 7)]),
        "C_2006to2008_acrossstep": (j[j.year == 2006], j[j.year == 2008]),
    }
    for name, (tr, te) in splits.items():
        run("Rn~PAR", name, tr, te, ["PAR"], "Rn", lin, [2.0, -20.0], rows)

    # ---------------- RH ~ TA, Tmin_day (FLUXNET, 62 m) ----------------
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
        run("RH~T,Tmin", f"train2006_test{y}", tr, f[f.year == y], ["T", "Tmin_day"], "RH", tetens, [1.0, -1.0], rows)

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
