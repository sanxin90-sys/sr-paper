"""
asymmetry_test.py
=================

Dedicated test of the morning/afternoon asymmetry suggested by PySR in E2:
the residual of the frozen relaxation was ~ c * Rn * sin(2*pi*h/24), i.e. the
temperature tendency at 22 m responds more strongly to net radiation in the
morning than in the afternoon.

Data: Rebio Jaru, train 2006 (October dropped: corrupted Rn), test 2008;
2007 dropped (fragmented T22, suspicious Rn). Targets: dT/dt from
Savitzky-Golay (7, 2; the project's default), centred finite differences, and
black-box X-TFC (48 samples, 12 neurons). All targets on the same rows.

Linear models (OLS), all with 3 diurnal harmonics ("H") unless stated:
  M0  a*Rn - lam*T + c                              (relaxation, no H)
  M1  M0 + H                                        (baseline)
  M2  M1 + b_s*Rn*sin(w h) + b_c*Rn*cos(w h)        (asymmetry, the PySR form)
  M3  a*Rn(t - tau) - lam*T + c + H                 (pure lag; tau on 2006 grid -1.5..3 h)
  M4  M1 + g*dRn/dt                                 (rate / storage term)
  M5  M4 + Rn*sin + Rn*cos                          (asymmetry beyond the rate term)
  M6  M3(tau) + Rn(t - tau)*sin + Rn(t - tau)*cos   (asymmetry beyond the lag)
Direct check: daytime (Rn > 20 W m-2) slopes of dT/dt on Rn, separately
for morning (06:00-11:30) and afternoon (12:30-18:00), controlling for T:
ratio r = slope_morning / slope_afternoon.
Timing check: hour of the Rn and T22 diurnal maxima vs local solar noon
(10.08 S, 61.93 W; UTC-4 -> about 12:08 local mean time).

Uncertainty: daily block bootstrap (500), on the training year for
coefficients and on 2008 for out-of-sample dR2.

Decision rules (fixed before running):
  A (asymmetry exists): b_s > 0 with 95 % CI excluding 0 in 2006 and in 2008
    (each year fitted on its own), in wet and dry seasons, for all three
    targets; and M2 beats M1 on 2008 (dR2 CI > 0) for all three targets.
  B (explained by lag/rate): A holds but M5 vs M4 or M6 vs M3 has a dR2 CI
    that includes 0 (or is negative) for at least two targets.
  C (timing artifact suspected): the fitted tau is within 0.25 h of the
    Rn-maximum offset from solar noon, or the Rn peak is > 30 min from
    solar noon.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from common import RESULTS, r2
import e2_jaru_graybox_real as e2

N_BOOT = 500
TAUS = np.arange(-3, 7) * 0.5  # hours; Rn lagged by tau
W = 2 * np.pi / 24


def harm(h, K=3):
    return np.column_stack([f(k * W * h) for k in range(1, K + 1) for f in (np.cos, np.sin)])


def per_run(s: pd.Series, fn, min_len):
    out = pd.Series(np.nan, index=s.index)
    run = s.isna().cumsum()
    for _, seg in s.dropna().groupby(run[s.notna()]):
        if len(seg) >= min_len:
            out.loc[seg.index] = fn(seg.to_numpy(float))
    return out


def prepare(xlsx):
    df = e2.add_features(e2.load_jaru(xlsx))
    df.loc[(df.year == 2006) & (df.month == 10), "Rn"] = np.nan
    df = df[df.year.isin([2006, 2008])].copy()
    parts = []
    for y, d in df.groupby("year"):
        d = d.reset_index(drop=True)
        d["t_sg"] = d.dTdt
        d["t_fd"] = per_run(d.T22, lambda v: np.gradient(v, 0.5), 3)
        d["t_xtfc"] = e2.xtfc_tendency(d)
        d["dRn"] = per_run(d.Rn, lambda v: savgol_filter(v, 7, 2, deriv=1, delta=0.5), 7)
        for tau in TAUS:
            d[f"Rn_lag{tau:g}"] = d.Rn.shift(int(round(tau / 0.5)))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


TARGETS = ["t_sg", "t_fd", "t_xtfc"]


def design(d, model, tau=0.0):
    H = harm(d.hour.to_numpy())
    rn = d.Rn.to_numpy()
    s, c = np.sin(W * d.hour.to_numpy()), np.cos(W * d.hour.to_numpy())
    base = [rn, -d.T22.to_numpy(), np.ones(len(d))]
    if model == "M0":
        cols = base
    elif model == "M1":
        cols = base + [H]
    elif model == "M2":
        cols = base + [H, rn * s, rn * c]
    elif model == "M4":
        cols = base + [H, d.dRn.to_numpy()]
    elif model == "M5":
        cols = base + [H, d.dRn.to_numpy(), rn * s, rn * c]
    elif model in ("M3", "M6"):
        rl = d[f"Rn_lag{tau:g}"].to_numpy()
        cols = [rl, -d.T22.to_numpy(), np.ones(len(d)), H] + ([rl * s, rl * c] if model == "M6" else [])
    return np.column_stack(cols)


def fit(d, y, model, tau=0.0):
    X = design(d, model, tau)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def day_blocks(d):
    days = d.timestamp.dt.date.to_numpy()
    return [np.flatnonzero(days == u) for u in np.unique(days)]


def main(xlsx):
    df = prepare(xlsx)
    need = ["Rn", "T22", "dRn"] + TARGETS + [f"Rn_lag{t:g}" for t in TAUS]
    df = df.dropna(subset=need).reset_index(drop=True)
    tr, te = df[df.year == 2006].reset_index(drop=True), df[df.year == 2008].reset_index(drop=True)
    print(f"rows: 2006 {len(tr)}, 2008 {len(te)}")
    rng = np.random.default_rng(0)
    lines = []

    # --- timing check -------------------------------------------------------
    print("\nTiming (mean diurnal cycle, local clock of the file):")
    for y, d in df.groupby("year"):
        prof = d.groupby("hour")[["Rn", "T22"]].mean()
        hh = np.linspace(0, 24, 2401)[:-1]
        cen = {}
        for v in ["Rn"]:
            p = prof[v].clip(lower=0)
            cen[v] = float((p.index * p).sum() / p.sum())
        print(f"  {y}: Rn max at {prof.Rn.idxmax():.2f} h, daytime Rn centroid {cen['Rn']:.2f} h; T22 max at {prof.T22.idxmax():.2f} h")
        lines.append(dict(test="timing", year=y, rn_peak=prof.Rn.idxmax(), rn_centroid=cen["Rn"], t_peak=prof.T22.idxmax()))

    # --- coefficient of Rn*sin, each year on its own, by season -------------
    print("\nM2 coefficient b_s (Rn*sin) and b_c (Rn*cos), 95 % daily-block CI:")
    for y, d0 in df.groupby("year"):
        for season in ["all", "wet", "dry"]:
            d = d0 if season == "all" else d0[d0.season == season]
            d = d.reset_index(drop=True)
            blocks = day_blocks(d)
            for tg in TARGETS:
                y_ = d[tg].to_numpy()
                X = design(d, "M2")
                b = np.linalg.lstsq(X, y_, rcond=None)[0][-2:]
                bs = []
                for _ in range(N_BOOT):
                    idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
                    bs.append(np.linalg.lstsq(X[idx], y_[idx], rcond=None)[0][-2:])
                lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
                lines.append(dict(test="coef_M2", year=y, season=season, target=tg, b_s=b[0], b_s_lo=lo[0], b_s_hi=hi[0],
                                  b_c=b[1], b_c_lo=lo[1], b_c_hi=hi[1]))
                print(f"  {y} {season:4s} {tg:7s} b_s={b[0]:.2e} [{lo[0]:.2e}, {hi[0]:.2e}]  b_c={b[1]:.2e} [{lo[1]:.2e}, {hi[1]:.2e}]")

    # --- direct morning/afternoon slopes ------------------------------------
    print("\nDaytime slopes of dT/dt on Rn (controlling for T), morning vs afternoon:")
    for y, d0 in df.groupby("year"):
        for season in ["all", "wet", "dry"]:
            d = d0 if season == "all" else d0[d0.season == season]
            d = d[(d.Rn > 20)]
            m = ((d.hour >= 6) & (d.hour <= 11.5)).to_numpy()
            a = ((d.hour >= 12.5) & (d.hour <= 18)).to_numpy()
            d = d[m | a].reset_index(drop=True)
            m = ((d.hour >= 6) & (d.hour <= 11.5)).to_numpy().astype(float)
            a = 1 - m
            blocks = day_blocks(d)
            for tg in TARGETS:
                X = np.column_stack([d.Rn * m, d.Rn * a, -d.T22, m, a])
                y_ = d[tg].to_numpy()
                b = np.linalg.lstsq(X, y_, rcond=None)[0]
                rs = []
                for _ in range(N_BOOT):
                    idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
                    bb = np.linalg.lstsq(X[idx], y_[idx], rcond=None)[0]
                    rs.append(bb[0] / bb[1])
                lo, hi = np.percentile(rs, [2.5, 97.5])
                lines.append(dict(test="slopes", year=y, season=season, target=tg, slope_morning=b[0], slope_afternoon=b[1],
                                  ratio=b[0] / b[1], ratio_lo=lo, ratio_hi=hi))
                print(f"  {y} {season:4s} {tg:7s} morning {b[0]:.4f}  afternoon {b[1]:.4f}  ratio {b[0]/b[1]:.2f} [{lo:.2f}, {hi:.2f}]")

    # --- out-of-sample comparison (train 2006, test 2008) -------------------
    print("\nOut of sample (train 2006, test 2008):")
    blocks_te = day_blocks(te)
    for tg in TARGETS:
        ytr, yte = tr[tg].to_numpy(), te[tg].to_numpy()
        # lag chosen on 2006 (in-sample R2 of M3)
        r2_tau = {tau: r2(ytr, design(tr, "M3", tau) @ fit(tr, ytr, "M3", tau)) for tau in TAUS}
        tau = max(r2_tau, key=r2_tau.get)
        preds = {}
        for mdl in ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]:
            preds[mdl] = design(te, mdl, tau) @ fit(tr, ytr, mdl, tau)
        pairs = [("M2", "M1"), ("M3", "M1"), ("M4", "M1"), ("M5", "M4"), ("M6", "M3"), ("M2", "M4"), ("M2", "M3")]
        boots = {p: [] for p in pairs}
        for _ in range(N_BOOT):
            idx = np.concatenate([blocks_te[i] for i in rng.integers(0, len(blocks_te), len(blocks_te))])
            for p in pairs:
                boots[p].append(r2(yte[idx], preds[p[0]][idx]) - r2(yte[idx], preds[p[1]][idx]))
        print(f"  {tg}: tau chosen on 2006 = {tau:g} h (R2 by tau: " + ", ".join(f"{k:g}:{v:.3f}" for k, v in r2_tau.items()) + ")")
        print("    R2 2008: " + ", ".join(f"{k} {r2(yte, p):.4f}" for k, p in preds.items()))
        for p in pairs:
            lo, hi = np.percentile(boots[p], [2.5, 97.5])
            d = r2(yte, preds[p[0]]) - r2(yte, preds[p[1]])
            print(f"    dR2 {p[0]} - {p[1]} = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            lines.append(dict(test="oos", target=tg, tau=tau, model=p[0], ref=p[1], dR2=d, ci_lo=lo, ci_hi=hi,
                              r2_model=r2(yte, preds[p[0]]), r2_ref=r2(yte, preds[p[1]])))
        b = fit(tr, ytr, "M4", tau)
        lines.append(dict(test="coef_M4", target=tg, g_dRn=b[-1], a_Rn=b[0]))
        print(f"    M4 (2006): a={b[0]:.4g} per W m-2, g(dRn/dt)={b[-1]:.4g}, implied lag g/a = {b[-1]/b[0]:.2f} h")

    out = pd.DataFrame(lines)
    out.to_csv(RESULTS / "ailorenz_asymmetry_test.csv", index=False)
    print(f"\nwritten {RESULTS / 'ailorenz_asymmetry_test.csv'}")


if __name__ == "__main__":
    main(sys.argv[1])
