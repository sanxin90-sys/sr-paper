"""
xtfc_jaru_sensitivity.py
========================

Does the AI-Lorenz derivative stage (black-box X-TFC) change the conclusions
about the real Jaru temperature tendency dT/dt compared with the
Savitzky-Golay estimate used in real_discovery.py / dtdt_robustness.py?

No symbolic regression here: the same parametric candidates as
dtdt_robustness.py (relaxation, saturating, each with/without 3 diurnal
harmonics, and harmonics alone) are fitted on 2006-2007 and scored on 2008,
once per derivative estimator. All estimators are evaluated on the same rows
(gap-free runs of >= MIN_RUN half-hours), so differences come only from the
estimator.

Usage:  python src/ailorenz/xtfc_jaru_sensitivity.py "Dados Rebio Jaru 2006 a 2008.xlsx"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "sr_benchmark"))
from real_discovery import load_jaru  # noqa: E402
from derivatives import xtfc  # noqa: E402
from dtdt_robustness import fit_parametric, fit_harmonic_only, r2  # noqa: E402

DT_H = 0.5
MIN_RUN = 96  # two days: X-TFC needs at least one full sub-domain
N_BOOT = 500

ESTIMATORS = {
    "savgol_w7p2": lambda t, y: savgol_filter(y, 7, 2, deriv=1, delta=DT_H),
    "finite_diff": lambda t, y: np.gradient(y, t),
    # settings tuned on the synthetic Jaru task at 5 % noise (run_benchmark.py)
    "xtfc_48s12n": lambda t, y: xtfc(t, y, samples_per_subdomain=48, n_neurons=12)[1][:, 0],
    # finer setting, closer to the clean-data configuration of the paper
    "xtfc_10s10n": lambda t, y: xtfc(t, y, samples_per_subdomain=10, n_neurons=10)[1][:, 0],
    # paper's noisy-data recipe: long sub-domain, many neurons, SG post-filter (order 5, frame 31)
    "xtfc_96s30n_sg": lambda t, y: xtfc(t, y, samples_per_subdomain=96, n_neurons=30, smooth_window=31)[1][:, 0],
}


def derivatives(s: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=s.index, columns=list(ESTIMATORS))
    run_id = s.isna().cumsum()
    for _, seg in s.dropna().groupby(run_id[s.notna()]):
        if len(seg) < MIN_RUN:
            continue
        t = np.arange(len(seg)) * DT_H
        y = seg.to_numpy(float)
        for name, f in ESTIMATORS.items():
            d = np.asarray(f(t, y)).ravel()
            d[:3] = d[-3:] = np.nan  # edges are unreliable for every estimator
            out.loc[seg.index, name] = d
    return out


def main(xlsx):
    df = load_jaru(xlsx)
    D = derivatives(df["T22"])
    df = pd.concat([df, D], axis=1).dropna(subset=list(ESTIMATORS) + ["Rn", "T22", "WS"])
    print(f"rows used: {len(df)}  (train {int((df.year < 2008).sum())}, test {int((df.year == 2008).sum())})")
    print("\nCorrelation between estimators:")
    print(df[list(ESTIMATORS)].corr().round(3).to_string())
    print("\nStd of each estimate (K/h):", df[list(ESTIMATORS)].std().round(3).to_dict())

    rng = np.random.default_rng(0)
    rows = []
    for season in ["all", "wet", "dry"]:
        ds = df if season == "all" else df[df.season == season]
        tr, te = ds[ds.year < 2008], ds[ds.year == 2008]
        X = tr[["Rn", "T22", "WS", "hour"]].to_numpy(float)
        Xt = te[["Rn", "T22", "WS", "hour"]].to_numpy(float)
        days = te.timestamp.dt.date.to_numpy()
        for est in ESTIMATORS:
            y, yt = tr[est].to_numpy(float), te[est].to_numpy(float)
            preds = {
                "harmonic_only": fit_harmonic_only(X, y, Xt)[0],
                "relaxation": fit_parametric("relaxation", X, y, Xt, False)[0],
                "saturating": fit_parametric("saturating", X, y, Xt, False)[0],
                "relax+harm": fit_parametric("relaxation", X, y, Xt, True)[0],
                "sat+harm": fit_parametric("saturating", X, y, Xt, True)[0],
            }
            ud = np.unique(days)
            by_day = {d: np.flatnonzero(days == d) for d in ud}
            boot = []
            for _ in range(N_BOOT):
                idx = np.concatenate([by_day[d] for d in rng.choice(ud, len(ud))])
                boot.append(r2(yt[idx], preds["sat+harm"][idx]) - r2(yt[idx], preds["relax+harm"][idx]))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            row = {"season": season, "estimator": est, "n_test": len(yt)}
            row.update({k: round(r2(yt, p), 4) for k, p in preds.items()})
            row.update({"dR2_sat_vs_relax_harm": round(r2(yt, preds["sat+harm"]) - r2(yt, preds["relax+harm"]), 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
            rows.append(row)
            print(row, flush=True)
    out = pd.DataFrame(rows)
    dest = HERE.parent / "sr_benchmark" / "results" / "real_xtfc_sensitivity.csv"
    out.to_csv(dest, index=False)
    print(f"\nwritten {dest}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
