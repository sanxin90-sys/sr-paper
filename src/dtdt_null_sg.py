"""
dtdt_null_sg.py
===============

Null check for EXP-001/002 (EXP-003): can the Savitzky-Golay (window 7, 30-min)
derivative alone create the apparent saturating Rn term?

A synthetic temperature is generated from the *linear* relaxation law
dT/dt = a*Rn - lam*T + c (no saturation, no hour term) driven by the real
2006-2008 Rn, integrated at 1-min steps on gap-free Rn runs (initial value =
observed T22), sampled every 30 min, optionally with white sensor noise,
differentiated with the same SG filter, then analysed as in EXP-002 (exclude
variant). If saturating+harm beats relaxation+harm by a dR2 comparable to the
real-data +0.01..0.04, the real finding is attributable to the derivative.

Usage:  python src/dtdt_null_sg.py "Dados Rebio Jaru 2006 a 2008.xlsx"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_discovery import load_jaru, _sg_derivative  # noqa: E402
from dtdt_robustness import SUSPECT_RN, fit_parametric, r2, relax  # noqa: E402

SUB = 30  # integration sub-steps per half hour (1 min)


def simulate(df, a, lam, c, noise_K, rng):
    T = pd.Series(np.nan, index=df.index)
    ok = df.Rn.notna() & df.T22.notna()
    run = (~ok).cumsum()
    for _, seg in df[ok].groupby(run[ok]):
        if len(seg) < 8:
            continue
        rn = seg.Rn.to_numpy()
        t, out = seg.T22.iloc[0], [seg.T22.iloc[0]]
        h = 0.5 / SUB
        for i in range(len(rn) - 1):
            for s in range(SUB):
                r = rn[i] + (rn[i + 1] - rn[i]) * s / SUB
                t += h * (a * r - lam * t + c)
            out.append(t)
        T.loc[seg.index] = np.array(out) + rng.normal(0, noise_K, len(out))
    return T


def main(xlsx):
    df = load_jaru(xlsx)
    keep = ~np.array([(y, m) in SUSPECT_RN for y, m in zip(df.year, df.month)])
    rng = np.random.default_rng(0)
    # relaxation constants fitted on the real exclude-variant training data
    d = df[keep].dropna(subset=["dTdt", "Rn", "T22"])
    tr = d[d.year <= 2007]
    (a, lam, c), _ = curve_fit(lambda Xf, *p: relax(Xf.reshape(-1, 2), *p),
                               tr[["Rn", "T22"]].to_numpy(float).ravel(), tr.dTdt.to_numpy(float), p0=[0.002, 0.05, 1.2])
    print(f"relaxation constants: a={a:.5f} lam={lam:.4f} c={c:.3f}")
    rows = []
    for noise in [0.0, 0.05]:
        sim = df.copy()
        sim["T22"] = simulate(df, a, lam, c, noise, rng)
        sim["dTdt"] = _sg_derivative(sim["T22"], dt_h=0.5, window=7, polyorder=2)
        sim["dTdt_true"] = a * sim.Rn - lam * sim.T22 + c
        s = sim[keep].dropna(subset=["dTdt", "Rn", "T22", "WS"])
        for season in ["all", "wet", "dry"]:
            ss = s if season == "all" else s[s.season == season]
            trs, te = ss[ss.year <= 2007], ss[ss.year == 2008]
            Xt = te[["Rn", "T22", "WS", "hour"]].to_numpy(float)
            for target in ["dTdt", "dTdt_true"]:
                yt = te[target].to_numpy(float)
                for seed in range(5):
                    t = trs.sample(min(3000, len(trs)), random_state=seed)
                    X, y = t[["Rn", "T22", "WS", "hour"]].to_numpy(float), t[target].to_numpy(float)
                    pr, _ = fit_parametric("relaxation", X, y, Xt, True)
                    try:
                        ps, _ = fit_parametric("saturating", X, y, Xt, True)
                        conv = True
                    except RuntimeError:  # b -> 0: saturating collapses onto the linear law
                        ps, conv = pr, False
                    rows.append({"sat_converged": conv, "noise_K": noise, "season": season, "target": target, "seed": seed,
                                 "r2_relax_harm": r2(yt, pr), "r2_sat_harm": r2(yt, ps), "dr2": r2(yt, ps) - r2(yt, pr)})
        print(f"noise {noise} done", flush=True)
    out = pd.DataFrame(rows)
    path = Path(__file__).resolve().parent / "sr_benchmark" / "results" / "dtdt_null_sg.csv"
    out.to_csv(path, index=False)
    print(out.groupby(["noise_K", "target", "season"])[["r2_relax_harm", "r2_sat_harm", "dr2"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
