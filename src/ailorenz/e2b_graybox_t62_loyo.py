"""
E2b - Gray-box X-TFC on real data, redone after DQ-001..003 (ailorenz.md 5.3 is Blocked).

State: T62 = FLUXNET BR-Ji3 TA_F_MDS with QC = 0 (consistent height in all
years; DQ-002). Forcing: SW_IN_F_MDS with QC = 0 (FLUXNET), because the
spreadsheet Rn has a calibration step in 2007 (DQ-003); same choice as the
SINDy thread (sindy.md 8). WS from the spreadsheet. Excluded months: the
DQ-001 list plus Feb-Mar 2007 (sindy.md 8.1), for every variable.

Leave-one-year-out over 2006, 2007, 2008. For each held-out year:
  known part  dT/dt = a*SW - lam*T + c, OLS on the SG(7,2) tendency of the two
              training years, then frozen;
  unknown f   learned on the training years by three routes, per gap-free run
              (>= 96 half-hours, within one year) of T62 and SW:
                sg        - SG tendency minus the known part;
                blackbox  - X-TFC (48 samples, 12 neurons) derivative minus the known part;
                graybox   - gray-box X-TFC, w_phys chosen by the truth-free rule
                            (common.discrepancy_choice over {1, 0.1, 0.03});
  residual    models fitted to f on {SW, T, WS, hs, hc}: harmonics K=3 (reference),
              sat+harm  A(1-exp(-b*SW)) + harmonics (b on a grid),
              sat+wind+harm  sat+harm + k*SW*WS (sindy.md 8 D),
              SINDy poly2, Operon (60 s, seeds 0-2).
Held-out scoring: R2 of known + residual against the SG tendency (primary) and
the X-TFC tendency of T62; pooled over the three held-out years and by year;
dR2 vs "known + harmonics" with a daily block bootstrap (1000) of the pooled
predictions.

Decision rules (fixed before running):
  R1 (any residual structure): a residual model beats known + harmonics with
     pooled CI > 0 in all, wet and dry, dR2 > 0 in each held-out year, for both
     targets (for Operon: in >= 2 of 3 seeds).
  R2 (the AI-Lorenz question): the gray-box route is better than the
     black-box route for the same residual model: pooled dR2(graybox -
     blackbox) > 0 with CI > 0 (season all, primary target) for sat+wind+harm
     and for >= 2/3 Operon seeds.

Usage:  python src/ailorenz/e2b_graybox_t62_loyo.py "Dados Rebio Jaru 2006 a 2008.xlsx"
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from common import HERE, RESULTS, discrepancy_choice, r2, run_sr, complexity
from gbxtfc import scalar_relaxation
from real_discovery import load_jaru
from dtdt_robustness import SUSPECT_RN

FLX = HERE.parents[1] / "data/fluxnet_BR-Ji3/AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv"
YEARS = [2006, 2007, 2008]
EXCLUDE = set(SUSPECT_RN) | {(2007, 2), (2007, 3)}
FEATS = ["SW", "T", "WS", "hs", "hc"]
W_GRID = [1.0, 0.1, 0.03]
XTFC = dict(s=48, m=12)
MIN_RUN = 96
N_BOOT = 1000
B_GRID = np.array([0.0005, 0.001, 0.002, 0.003, 0.005, 0.01])


def harm(h, K=3):
    w = 2 * np.pi * h / 24
    return np.column_stack([f(k * w) for k in range(1, K + 1) for f in (np.cos, np.sin)])


def load(xlsx):
    df = load_jaru(xlsx)
    f = pd.read_csv(FLX, na_values=[-9999], usecols=["TIMESTAMP_END", "TA_F_MDS", "TA_F_MDS_QC",
                                                       "SW_IN_F_MDS", "SW_IN_F_MDS_QC"])
    f["timestamp"] = pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")
    df = df.merge(f.drop(columns="TIMESTAMP_END"), on="timestamp", how="left")
    df["T"] = df.TA_F_MDS.where(df.TA_F_MDS_QC == 0)
    df["SW"] = df.SW_IN_F_MDS.where(df.SW_IN_F_MDS_QC == 0)
    bad = np.array([(y, m) in EXCLUDE for y, m in zip(df.year, df.month)])
    df.loc[bad, ["T", "SW", "WS"]] = np.nan
    df["hs"] = np.sin(2 * np.pi * df.hour / 24)
    df["hc"] = np.cos(2 * np.pi * df.hour / 24)
    df = df[df.year.isin(YEARS)].reset_index(drop=True)
    df["day"] = df.timestamp.dt.date
    # tendencies per gap-free run inside a year
    df["d_sg"], df["d_xtfc"] = np.nan, np.nan
    for _, r in runs(df, ["T"], min_len=7):
        T = df["T"].to_numpy(float)[r]
        df.loc[r, "d_sg"] = savgol_filter(T, 7, 2, deriv=1, delta=0.5)
        if len(r) >= MIN_RUN:
            _, dT, _ = scalar_relaxation(np.arange(len(r)) * 0.5, T, np.zeros(len(r)), 0, 0, 0, w_phys=0.0, **XTFC)
            dT = dT.copy(); dT[:3] = dT[-3:] = np.nan
            df.loc[r, "d_xtfc"] = dT
    return df


def runs(df, cols, min_len):
    ok = df[cols].notna().all(axis=1).to_numpy()
    yr = df.year.to_numpy()
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    cut = np.flatnonzero((np.diff(idx) > 1) | (np.diff(yr[idx]) != 0)) + 1
    return [(yr[r[0]], r) for r in np.split(idx, cut) if len(r) >= min_len]


def residual_f(df, rows_mask, known, route, w=None):
    a, lam, c = known
    f = np.full(len(df), np.nan)
    if route == "sg":
        f[:] = df.d_sg - (a * df.SW - lam * df["T"] + c)
        f[~rows_mask] = np.nan
        return f, 0.0
    ww = 0.0 if route == "blackbox" else w
    mis = []
    T, SW = df["T"].to_numpy(float), df.SW.to_numpy(float)
    for _, r in runs(df, ["T", "SW"], MIN_RUN):
        if not rows_mask[r].all():
            continue
        Th, _, fr = scalar_relaxation(np.arange(len(r)) * 0.5, T[r], SW[r], a, lam, c, w_phys=ww, **XTFC)
        fr = fr.copy(); fr[:3] = fr[-3:] = np.nan
        f[r] = fr
        mis.append((Th - T[r])[3:-3])
    return f, float(np.sqrt(np.mean(np.concatenate(mis) ** 2)))


def fit_residual(name, D, y):
    H = harm(D.hour.to_numpy())
    one = np.ones(len(D))
    if name == "harmonics":
        X = lambda E: np.column_stack([np.ones(len(E)), harm(E.hour.to_numpy())])
    elif name in ("sat+harm", "sat+wind+harm"):
        best = None
        for b in B_GRID:
            def X(E, b=b):
                cols = [np.ones(len(E)), 1 - np.exp(-b * E.SW.to_numpy()), harm(E.hour.to_numpy())]
                if name == "sat+wind+harm":
                    cols.append(E.SW.to_numpy() * E.WS.to_numpy())
                return np.column_stack(cols)
            beta = np.linalg.lstsq(X(D), y, rcond=None)[0]
            sse = np.sum((y - X(D) @ beta) ** 2)
            if best is None or sse < best[0]:
                best = (sse, X, beta, b)
        _, X, beta, b = best
        return (lambda E: X(E) @ beta), f"{name} b={b}"
    beta = np.linalg.lstsq(X(D), y, rcond=None)[0]
    return (lambda E: X(E) @ beta), name


def main(xlsx):
    df = load(xlsx)
    print("rows with T62 & SW:", int(df[["T", "SW"]].notna().all(axis=1).sum()),
          {y: int(df[(df.year == y)][["T", "SW", "WS"]].notna().all(axis=1).sum()) for y in YEARS}, flush=True)
    preds, info = {}, []
    test_rows = []
    for ho in YEARS:
        trm = (df.year != ho).to_numpy()
        d = df[trm].dropna(subset=["d_sg", "SW", "T"])
        known = tuple(np.linalg.lstsq(np.column_stack([d.SW, -d["T"], np.ones(len(d))]), d.d_sg, rcond=None)[0])
        fs = {"sg": residual_f(df, trm, known, "sg"), "blackbox": residual_f(df, trm, known, "blackbox")}
        gb = {w: residual_f(df, trm, known, "graybox", w) for w in W_GRID}
        w_star = discrepancy_choice({w: v[1] for w, v in gb.items()}, fs["blackbox"][1])
        fs["graybox"] = gb[w_star]
        print(f"held-out {ho}: known a={known[0]:.5g} lam={known[1]:.4g} c={known[2]:.4g}; w*={w_star}; "
              f"misfit bb={fs['blackbox'][1]:.4f} gb={fs['graybox'][1]:.4f}; "
              f"corr(f_bb,f_gb)={pd.Series(fs['blackbox'][0]).corr(pd.Series(fs['graybox'][0])):.3f}", flush=True)
        te = df[(df.year == ho)].dropna(subset=FEATS + ["d_sg", "d_xtfc"])
        known_te = (known[0] * te.SW - known[1] * te["T"] + known[2]).to_numpy()
        for route, (f, _) in fs.items():
            ok = np.isfinite(f) & df[FEATS].notna().all(axis=1).to_numpy() & trm
            D, y = df[ok], f[ok]
            models = {k: fit_residual(k, D, y) for k in ["harmonics", "sat+harm", "sat+wind+harm"]}
            X = D[FEATS].to_numpy(float)
            for seed in range(3):
                engines = ("sindy_poly2", "operon") if seed == 0 else ("operon",)
                for eng, fit in run_sr(X, y, FEATS, engines, budget=60, seed=seed).items():
                    models[f"{eng}_s{seed}"] = ((lambda E, fit=fit: fit.predict(E[FEATS].to_numpy(float))), fit.expression)
            for k, (fn, expr) in models.items():
                preds[(route, k, ho)] = known_te + fn(te)
                info.append(dict(heldout=ho, route=route, model=k, n_train=int(ok.sum()), w_star=w_star,
                                 complexity=complexity(expr) if ("operon" in k or "sindy" in k) else "", expression=expr))
            print(f"  {route} done ({int(ok.sum())} rows)", flush=True)
        test_rows.append(te.assign(heldout=ho))
    TE = pd.concat(test_rows, ignore_index=True)
    days = (TE.heldout.astype(str) + "_" + TE.day.astype(str)).to_numpy()
    ud = np.unique(days)
    by_day = {u: np.flatnonzero(days == u) for u in ud}
    P = {key[:2]: None for key in preds}
    for rk in P:
        P[rk] = np.concatenate([preds[(rk[0], rk[1], ho)] for ho in YEARS])
    rng = np.random.default_rng(0)
    rows = []
    for target in ["d_sg", "d_xtfc"]:
        yt = TE[target].to_numpy()
        for season in ["all", "wet", "dry"]:
            sel = np.ones(len(TE), bool) if season == "all" else (TE.season == season).to_numpy()
            sdays = np.unique(days[sel])
            boot_idx = [np.concatenate([by_day[u][sel[by_day[u]]] for u in rng.choice(sdays, len(sdays))]) for _ in range(N_BOOT)]
            for (route, k), p in P.items():
                ref = P[(route, "harmonics")]
                bb = P.get(("blackbox", k))
                dr = [r2(yt[i], p[i]) - r2(yt[i], ref[i]) for i in boot_idx]
                lo, hi = np.percentile(dr, [2.5, 97.5])
                row = dict(target=target, season=season, route=route, model=k, r2=round(r2(yt[sel], p[sel]), 4),
                           dR2_vs_harm=round(r2(yt[sel], p[sel]) - r2(yt[sel], ref[sel]), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4))
                if route == "graybox" and bb is not None:
                    dg = [r2(yt[i], p[i]) - r2(yt[i], bb[i]) for i in boot_idx]
                    glo, ghi = np.percentile(dg, [2.5, 97.5])
                    row.update(dR2_gb_vs_bb=round(r2(yt[sel], p[sel]) - r2(yt[sel], bb[sel]), 4), gb_ci_lo=round(glo, 4), gb_ci_hi=round(ghi, 4))
                if season == "all":
                    for ho in YEARS:
                        m = (TE.heldout == ho).to_numpy()
                        row[f"dR2_{ho}"] = round(r2(yt[m], p[m]) - r2(yt[m], ref[m]), 4)
                rows.append(row)
        print(f"scored {target}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ailorenz_e2b_t62_loyo.csv", index=False)
    pd.DataFrame(info).to_csv(RESULTS / "ailorenz_e2b_t62_loyo_models.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out.target == "d_sg"].to_string(index=False))
    # decision rules
    r1 = []
    for (route, k), g in out.groupby(["route", "model"]):
        if k == "harmonics":
            continue
        ok = all((g[(g.target == t) & (g.season == s)].ci_lo > 0).all() for t in ["d_sg", "d_xtfc"] for s in ["all", "wet", "dry"])
        ok &= all((g[(g.season == "all")][f"dR2_{ho}"] > 0).all() for ho in YEARS)
        r1.append(dict(route=route, model=k, R1=ok))
    r1 = pd.DataFrame(r1)
    print("\nR1 per (route, model):\n", r1.to_string(index=False))
    g = out[(out.target == "d_sg") & (out.season == "all") & (out.route == "graybox")]
    print("\nR2 graybox vs blackbox (d_sg, all):\n", g[["model", "dR2_gb_vs_bb", "gb_ci_lo", "gb_ci_hi"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
