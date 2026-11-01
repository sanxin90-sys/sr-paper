"""
sindy_jaru_check.py
===================

Focused audit of quadratic SINDy (STLSQ) on the real Rebio Jaru targets of
real_discovery.py. It asks whether the earlier SINDy result for dT/dt
(R2 = 0.374, 10 terms, the best test R2 of all engines) is a sparse,
stable, physically interpretable model or a dense polynomial that mainly
absorbs the diurnal cycle.

Design (fixed before running): train 2006-2007, test 2008, as in
real_discovery.py; n_max = 3000 random training rows.
  1. reproduce the original sindy_poly2 fit (seed 0, threshold 0.1);
  2. threshold sweep -> Pareto front (number of terms vs test R2);
  3. term-inclusion frequency over 20 training-subsample seeds;
  4. baselines: linear relaxation, relaxation + 3 diurnal harmonics,
     harmonics alone; and SINDy + harmonics;
  5. seasons all / wet / dry;
  6. 95% CI of dR2 (SINDy - relaxation+harmonics) by daily block bootstrap;
  7. sensitivity to the Savitzky-Golay window of the dT/dt estimate.
Decision rule: SINDy is credited with structure beyond the diurnal cycle only
if SINDy (+harmonics) beats relaxation+harmonics with a CI excluding zero in
all three season settings.

Usage:  python src/sindy_jaru_check.py "Dados Rebio Jaru 2006 a 2008.xlsx" [--cache df.pkl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_discovery import TARGETS, _sg_derivative, load_jaru, split  # noqa: E402
from sr_benchmark.methods import fit_sindy_poly  # noqa: E402

N_BOOT = 1000
FEATS = ["Rn", "T22", "WS"]


def harm(h, K=3):
    w = 2 * np.pi * h / 24
    return np.column_stack([f(k * w) for k in range(1, K + 1) for f in (np.cos, np.sin)])


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def ols(A, y, At):
    A1, At1 = np.column_stack([np.ones(len(A)), A]), np.column_stack([np.ones(len(At)), At])
    beta, *_ = np.linalg.lstsq(A1, y, rcond=None)
    return At1 @ beta


def stlsq_poly(X, y, Xt, threshold, extra=None, extra_t=None, names=None):
    """Quadratic SINDy as in methods.fit_sindy_poly, optionally with unpenalised-free extra columns."""
    import pysindy as ps
    from sklearn.preprocessing import PolynomialFeatures

    pf = PolynomialFeatures(2, include_bias=True)
    th, tht = pf.fit_transform(X), pf.transform(Xt)
    nm = [n.replace(" ", "*").replace("^", "**") for n in pf.get_feature_names_out(names or FEATS)]
    if extra is not None:
        th, tht = np.column_stack([th, extra]), np.column_stack([tht, extra_t])
        nm += [f"h{i}" for i in range(extra.shape[1])]
    sc = np.linalg.norm(th, axis=0); sc[sc == 0] = 1
    opt = ps.STLSQ(threshold=threshold * np.linalg.norm(y), alpha=1e-8, max_iter=50)
    opt.fit(th / sc, y.reshape(-1, 1))
    coef = np.ravel(opt.coef_) / sc
    return tht @ coef, {n: c for n, c in zip(nm, coef) if c != 0}


def data(df, season=None, seed=0, n_max=3000, ycol="dTdt"):
    d = df.dropna(subset=[ycol, *FEATS])
    if season:
        d = d[d.season == season]
    tr, te = d[d.year <= 2007], d[d.year == 2008]
    if len(tr) > n_max:
        tr = tr.sample(n_max, random_state=seed)
    f = lambda a: a.to_numpy(dtype=float).copy()
    return (f(tr[FEATS]), f(tr[ycol]), f(tr["hour"]), f(te[FEATS]), f(te[ycol]), f(te["hour"]),
            te.timestamp.dt.date.to_numpy())


def models(df, season, seed, thr=0.1, ycol="dTdt"):
    X, y, h, Xt, yt, ht, days = data(df, season, seed, ycol=ycol)
    P = {}
    P["relaxation"] = ols(X[:, :2], y, Xt[:, :2])
    P["harmonics"] = ols(harm(h), y, harm(ht))
    P["relaxation+harm"] = ols(np.column_stack([X[:, :2], harm(h)]), y, np.column_stack([Xt[:, :2], harm(ht)]))
    P["sindy"], tsindy = stlsq_poly(X, y, Xt, thr)
    P["sindy+harm"], tsh = stlsq_poly(X, y, Xt, thr, harm(h), harm(ht))
    return P, yt, days, tsindy, tsh


def block_boot(yt, preds, days, ref, rng):
    ud = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in ud}
    out = {k: [] for k in preds}
    for _ in range(N_BOOT):
        s = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
        r_ref = r2(yt[s], preds[ref][s])
        for k, p in preds.items():
            out[k].append(r2(yt[s], p[s]) - r_ref)
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "sr_benchmark" / "results")
    a = ap.parse_args()
    df = pd.read_pickle(a.cache) if a.cache and a.cache.exists() else load_jaru(a.xlsx)
    rng = np.random.default_rng(0)

    # 1. reproduce original
    t = TARGETS["temperature_tendency"]
    X, y, Xt, yt = split(df, t, None, seed=0)
    fit = fit_sindy_poly(X, y, t.features)
    print("1. reproduction sindy_poly2 seed0 thr0.1: R2_test =", round(r2(yt, fit.predict(Xt)), 4),
          "| terms =", fit.expression.count("+") + 1)
    print("   ", fit.expression)

    # 2. threshold sweep
    rows = []
    for thr in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        p, terms = stlsq_poly(X, y, Xt, thr)
        rows.append({"threshold": thr, "n_terms": len(terms), "r2_test": r2(yt, p), "terms": " + ".join(f"{c:.3g}*{n}" for n, c in terms.items())})
    sweep = pd.DataFrame(rows)
    print("\n2. threshold sweep (dT/dt, all seasons)\n", sweep[["threshold", "n_terms", "r2_test"]].to_string(index=False))

    # 3. term-inclusion frequency across subsample seeds
    inc, signs, r2s = {}, {}, []
    for s in range(20):
        X_, y_, _, Xt_, yt_, _, _ = data(df, None, s)
        p, terms = stlsq_poly(X_, y_, Xt_, 0.1)
        r2s.append(r2(yt_, p))
        for n, c in terms.items():
            inc[n] = inc.get(n, 0) + 1
            signs.setdefault(n, []).append(np.sign(c))
    freq = pd.DataFrame({"term": list(inc), "freq": [inc[k] / 20 for k in inc],
                         "sign_agreement": [abs(np.mean(signs[k])) for k in inc]}).sort_values("freq", ascending=False)
    print(f"\n3. 20 seeds: R2_test {np.mean(r2s):.3f} +- {np.std(r2s):.3f}\n", freq.to_string(index=False))

    # 4-6. baselines, seasons, bootstrap
    res = []
    for season in [None, "wet", "dry"]:
        P, yt_, days, ts, tsh = models(df, season, 0)
        ci = block_boot(yt_, P, days, "relaxation+harm", rng)
        for k, p in P.items():
            res.append({"season": season or "all", "model": k, "r2_test": r2(yt_, p),
                        "dR2_vs_relax+harm_lo": ci[k][0], "dR2_vs_relax+harm_hi": ci[k][1],
                        "n_terms": {"relaxation": 3, "harmonics": 7, "relaxation+harm": 9,
                                    "sindy": len(ts), "sindy+harm": len(tsh)}[k]})
        print(f"\n   {season or 'all'} sindy terms: {list(ts)}\n   {season or 'all'} sindy+harm terms: {list(tsh)}")
    res = pd.DataFrame(res)
    print("\n4-6. baselines/seasons/bootstrap\n", res.round(4).to_string(index=False))

    # 7. derivative window sensitivity
    win = []
    for w in [5, 7, 11, 15]:
        d2 = df.copy()
        d2["dTdt_w"] = _sg_derivative(d2["T22"], dt_h=0.5, window=w, polyorder=2)
        P, yt_, *_ = models(d2, None, 0, ycol="dTdt_w")
        win.append({"sg_window": w, **{k: r2(yt_, p) for k, p in P.items()}})
    win = pd.DataFrame(win)
    print("\n7. SG window sensitivity (all seasons, R2_test)\n", win.round(4).to_string(index=False))

    # Rn: why no intercept
    Xr = df.dropna(subset=["Rn", "PAR"])
    tr, te = Xr[Xr.year <= 2007].sample(3000, random_state=0), Xr[Xr.year == 2008]
    rn = []
    for thr in [0.0, 0.01, 0.05, 0.1]:
        p, terms = stlsq_poly(tr[["PAR"]].to_numpy(float), tr.Rn.to_numpy(float), te[["PAR"]].to_numpy(float), thr, names=["PAR"])
        night = te.PAR.to_numpy() < 1
        rn.append({"threshold": thr, "terms": terms, "r2_test": r2(te.Rn.to_numpy(float), p),
                   "frac_night_Rn<0": float(np.mean(p[night] < 0))})
    print("\nRn~PAR sweep\n", pd.DataFrame(rn).to_string(index=False))

    a.outdir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(a.outdir / "sindy_jaru_sweep.csv", index=False)
    freq.to_csv(a.outdir / "sindy_jaru_term_freq.csv", index=False)
    res.to_csv(a.outdir / "sindy_jaru_baselines.csv", index=False)
    win.to_csv(a.outdir / "sindy_jaru_sgwindow.csv", index=False)
    pd.DataFrame(rn).to_csv(a.outdir / "sindy_jaru_rn.csv", index=False)


if __name__ == "__main__":
    main()
