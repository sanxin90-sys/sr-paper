"""
E2 - Gray-box X-TFC on the real Jaru 22 m temperature record.

Known part: relaxation dT/dt = a*Rn - lam*T22 + c, fitted by OLS on the
Savitzky-Golay tendency of 2006, then frozen.
Unknown term f(t): learned on 2006 as well, by three routes, per gap-free run
of >= 96 half-hours with T22 and Rn present. 2007 is not used: its T22 record
is fragmented (896 runs, median length 8 half-hours; only 788 rows lie in runs
>= 2 days), too short for X-TFC sub-domains. Learning f on the year that fixed
the known part is deliberate (October 2006 is dropped: corrupted Rn): f is then the part of the tendency that the
frozen linear relaxation cannot express; 2008 stays fully out of sample.
  sg        - SG(7,2) tendency minus the known part (the current pipeline);
  blackbox  - X-TFC (48 samples, 12 neurons) derivative minus the known part;
  graybox   - gray-box X-TFC, w_phys in {1, 0.1, 0.03} and the truth-free choice.
SR of f on {Rn, T, WS, RH, VPD, hs, hc} (hs, hc = sin, cos of 2*pi*hour/24):
SINDy poly2 on every route; Operon (60 s, seeds 0-2) on sg, blackbox, graybox_auto.
Reference residual models fitted to the same f: none (relaxation only),
3 diurnal harmonics, saturating A*(1-exp(-b*Rn)) + d.

Test: 2008, target = SG tendency (as in dtdt_robustness.py) and, as a
sensitivity check, the black-box X-TFC tendency. Score R2 of known part +
residual model, all/wet/dry seasons; dR2 against "relaxation + harmonics" with
a daily block bootstrap (500). Decision rule (fixed before running): an SR
residual is supported only if its dR2 CI excludes zero (> 0) in all three
seasons for both targets.

Usage:  python src/ailorenz/e2_jaru_graybox_real.py "Dados Rebio Jaru 2006 a 2008.xlsx"
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from common import RESULTS, discrepancy_choice, export_for_pysr, r2, run_sr, complexity
from gbxtfc import scalar_relaxation
from real_discovery import load_jaru

FEATS = ["Rn", "T", "WS", "RH", "VPD", "hs", "hc"]
W_GRID = [1.0, 0.1, 0.03]
MIN_RUN = 96
N_BOOT = 500
XTFC = dict(s=48, m=12)


def harm(h, K=3):
    w = 2 * np.pi * h / 24
    return np.column_stack([f(k * w) for k in range(1, K + 1) for f in (np.cos, np.sin)])


def add_features(df):
    es = 0.6108 * np.exp(17.27 * df.T22 / (df.T22 + 237.3))
    df = df.assign(T=df.T22, VPD=es * (1 - df.RH / 100),
                   hs=np.sin(2 * np.pi * df.hour / 24), hc=np.cos(2 * np.pi * df.hour / 24))
    return df


def gapfree_runs(mask):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    return [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(r) >= MIN_RUN]


def residuals(df, known, route, seed=0):
    """f over df (NaN where unusable) and the data misfit of the X-TFC fit."""
    a, lam, c = known
    f = np.full(len(df), np.nan)
    if route == "sg":
        return (df.dTdt - (a * df.Rn - lam * df.T22 + c)).to_numpy(), 0.0
    w = 0.0 if route == "blackbox" else float(route.split("_w")[1])
    mis = []
    T, R = df.T22.to_numpy(float), df.Rn.to_numpy(float)
    for r in gapfree_runs(np.isfinite(T) & np.isfinite(R)):
        tt = np.arange(len(r)) * 0.5
        Th, dT, fr = scalar_relaxation(tt, T[r], R[r], a, lam, c, w_phys=w, seed=seed, **XTFC)
        fr = fr.copy(); fr[:3] = fr[-3:] = np.nan
        f[r] = fr
        mis.append((Th - T[r])[3:-3])
    return f, float(np.sqrt(np.mean(np.concatenate(mis) ** 2)))


def xtfc_tendency(df):
    out = np.full(len(df), np.nan)
    T = df.T22.to_numpy(float)
    for r in gapfree_runs(np.isfinite(T)):
        _, dT, _ = scalar_relaxation(np.arange(len(r)) * 0.5, T[r], np.zeros(len(r)), 0, 0, 0, w_phys=0.0, **XTFC)
        dT = dT.copy(); dT[:3] = dT[-3:] = np.nan
        out[r] = dT
    return out


def fit_reference(name, D, y):
    if name == "none":
        return lambda E: np.zeros(len(E)), "0"
    if name == "harmonics":
        A = np.column_stack([np.ones(len(D)), harm(D.hour.to_numpy())])
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        return lambda E: np.column_stack([np.ones(len(E)), harm(E.hour.to_numpy())]) @ b, "harmonics K=3"
    g = lambda rn, A_, b_, d_: A_ * (1 - np.exp(-b_ * rn)) + d_
    p, _ = curve_fit(g, D.Rn.to_numpy(), y, p0=[2.0, 0.005, -1.0], maxfev=40000)
    return lambda E: g(E.Rn.to_numpy(), *p), f"{p[0]:.4g}*(1-exp(-{p[1]:.4g}*Rn)) + {p[2]:.4g}"


def main(xlsx):
    df = add_features(load_jaru(xlsx))
    # Oct 2006: Rn/PAR median -1.10 (sign/scale fault; methods/pysr.md 4.1) -> Rn removed
    df.loc[(df.year == 2006) & (df.month == 10), "Rn"] = np.nan
    d06 = df[(df.year == 2006)].dropna(subset=["dTdt", "Rn", "T22"])
    known = tuple(np.linalg.lstsq(np.column_stack([d06.Rn, -d06.T22, np.ones(len(d06))]), d06.dTdt, rcond=None)[0])
    print(f"known relaxation (2006): a={known[0]:.5g}  lam={known[1]:.5g}  c={known[2]:.5g}", flush=True)
    a, lam, c = known

    d07 = df[df.year == 2006].reset_index(drop=True)  # training year for f (see docstring)
    te = df[df.year == 2008].reset_index(drop=True)
    te = te.assign(dTdt_xtfc=xtfc_tendency(te))
    te = te.dropna(subset=FEATS + ["dTdt", "dTdt_xtfc"]).reset_index(drop=True)
    days = te.timestamp.dt.date.to_numpy()
    by_day = {d: np.flatnonzero(days == d) for d in np.unique(days)}
    known_te = a * te.Rn - lam * te.T22 + c

    fs = {r: residuals(d07, known, r) for r in ["sg", "blackbox"] + [f"graybox_w{w}" for w in W_GRID]}
    w_star = discrepancy_choice({w: fs[f"graybox_w{w}"][1] for w in W_GRID}, fs["blackbox"][1])
    fs["graybox_auto"] = fs[f"graybox_w{w_star}"]
    print(f"gray-box truth-free weight: {w_star}; misfits", {k: round(v[1], 4) for k, v in fs.items()}, flush=True)
    fr = pd.DataFrame({k: v[0] for k, v in fs.items()})
    print("correlation of f between routes:\n", fr.corr().round(3).to_string(), flush=True)

    rows = []
    rng = np.random.default_rng(0)
    for route, (f, _) in fs.items():
        ok = np.isfinite(f) & d07[FEATS].notna().all(axis=1).to_numpy()
        D, y = d07[ok], f[ok]
        models = {}
        for ref in ["none", "harmonics", "saturating"]:
            models[ref] = fit_reference(ref, D, y)
        main_route = route in ("sg", "blackbox", "graybox_auto")
        if main_route:
            export_for_pysr(f"e2_{route}", D[FEATS].to_numpy(float), y, FEATS, a=a, lam=lam, c=c)
        X = D[FEATS].to_numpy(float)
        for seed in (range(3) if main_route else [0]):
            engines = ("sindy_poly2", "operon") if main_route else ("sindy_poly2",)
            if seed > 0:
                engines = ("operon",)
            for eng, fit in run_sr(X, y, FEATS, engines, budget=60, seed=seed).items():
                models[f"{eng}_s{seed}"] = (lambda E, fit=fit: fit.predict(E[FEATS].to_numpy(float)), fit.expression)
        Xte = te
        for target in ["dTdt", "dTdt_xtfc"]:
            yt = te[target].to_numpy()
            preds = {k: known_te.to_numpy() + m[0](Xte) for k, m in models.items()}
            for season in ["all", "wet", "dry"]:
                sel = np.ones(len(te), bool) if season == "all" else (te.season == season).to_numpy()
                sdays = np.unique(days[sel])
                boots = {k: [] for k in preds}
                for _ in range(N_BOOT):
                    idx = np.concatenate([by_day[d][sel[by_day[d]]] for d in rng.choice(sdays, len(sdays))])
                    ref = r2(yt[idx], preds["harmonics"][idx])
                    for k, p in preds.items():
                        boots[k].append(r2(yt[idx], p[idx]) - ref)
                ref_full = r2(yt[sel], preds["harmonics"][sel])
                for k, p in preds.items():
                    lo, hi = np.percentile(boots[k], [2.5, 97.5])
                    rows.append(dict(route=route, model=k, target=target, season=season, n_train=int(ok.sum()),
                                     r2_test=round(r2(yt[sel], p[sel]), 4),
                                     dR2_vs_relax_harm=round(r2(yt[sel], p[sel]) - ref_full, 4),
                                     ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                     complexity=complexity(models[k][1]) if k not in ("none", "harmonics", "saturating") else "",
                                     expression=models[k][1]))
        print(route, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ailorenz_e2_jaru_real.csv", index=False)
    piv = out.pivot_table(index=["route", "model"], columns=["target", "season"], values="dR2_vs_relax_harm")
    print(piv.round(4).to_string())
    sup = out.groupby(["route", "model"]).apply(lambda g: bool((g.ci_lo > 0).all()), include_groups=False)
    print("\nsupported by decision rule:\n", sup[sup].to_string() if sup.any() else "none")
    print(out[(out.target == "dTdt") & (out.season == "all")][["route", "model", "r2_test", "complexity", "expression"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1])
