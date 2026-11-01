"""
aif_separability.py
===================

EXP-011: AI Feynman separability tests as a structural diagnostic of dT/dt
(methods/aifeynman.md, idea 7).

Question. Is the tower temperature tendency additively separable in the
forcing and state, dT/dt = g1(SW) + g2(T) + g3(WS), or do pairs interact?
EXP-008 (SINDy) found a stable SW*WS term; this test asks the same question
without choosing a library.

Data window (same as EXP-007/008 so results are comparable).
T = FLUXNET BR-Ji3 TA_F_MDS with QC = 0 (DQ-002); forcing SW = SW_IN_F_MDS with
QC = 0 (DQ-003: Rn changes scale in 2007, so Rn is not used); WS from the
spreadsheet; target = Savitzky-Golay window-7 derivative of T, never across a
year boundary. Strict window (main): DQ-001 months dropped, and in 2007 only
January and August are kept (DQ-003 update: Feb-Mar 2007 partly corrupted,
step Apr-Jul). Loose window (sensitivity): only DQ-001 months dropped.

Test (AI Feynman, S_separability.check_separability_plus). A network with the
AI Feynman architecture (SimpleNet 128-128-64-64, tanh) is trained on the two
training years; for a partition A|B of the inputs the separable reconstruction
is f(x_A, med_B) + f(med_A, x_B) - f(med_A, med_B), medians from training data.
On the held-out year:
  delta  = MSE(y, f_sep) - MSE(y, f)          (primary; data-referenced)
  I_nn   = mean((f - f_sep)^2) / var(f)       (interaction share inside the net)
  aif    = median 2|y - f_sep| / median 2|y - f|  (AI Feynman's own statistic)
Partitions: {SW}|{T,WS}, {T}|{SW,WS}, {WS}|{SW,T}, plus full additivity.
The multiplicative test is not used: dT/dt changes sign every day.
95% CI of delta by a daily block bootstrap of the pooled held-out predictions.

Calibration (same inputs, same noise). G0 = additive generator
c - lam*T + A(1 - exp(-b*SW)) + harmonics(hour); G1 = G0 + kappa*SW*WS. Both are
fitted by OLS on the real target; their residuals are added back by
reassigning whole days at random, so noise level, autocorrelation and the
correlation between inputs are the real ones.

Decision rule (fixed before running).
  flag(partition, seed) = pooled delta > 0 with CI excluding zero.
  Calibration passes if G0 flags each partition in <= 1 of 5 seeds AND G1 flags
  {SW}|{T,WS} and {WS}|{SW,T} in >= 4 of 5 seeds.
  On real data a partition is called non-separable if flagged in >= 4 of 5
  seeds AND the seed-median delta > 0 in >= 2 of 3 held-out years.
  If calibration fails, the real-data result is reported as inconclusive.

Usage:  python src/aif_separability.py "Dados Rebio Jaru 2006 a 2008.xlsx" [--cache df.pkl] [--seeds 5]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sindy_exp008 as E  # noqa: E402

OUT = Path(__file__).resolve().parent / "sr_benchmark" / "results"
YEARS = [2006, 2007, 2008]
FEATS = ["SW", "T", "WS"]
PARTS = {"SW|T,WS": [0], "T|SW,WS": [1], "WS|SW,T": [2], "additive": None}
# exploratory variant (--hour), added after the main run failed calibration:
# hour of day enters the net as one group H = (cos, sin), so the diurnal cycle is
# no longer a hidden variable shared by SW and T
PARTS_H = {"SW|T,WS,H": [0], "T|SW,WS,H": [1], "WS|SW,T,H": [2], "H|SW,T,WS": [3, 4]}
GROUPS_H = [[0], [1], [2], [3, 4]]
STRICT_2007 = {1, 8}
B_GRID = E.B_GRID
N_BOOT = 1000


def load(xlsx, cache, window):
    df, _, _ = E.load(xlsx, cache)
    if window == "strict":
        df.loc[(df.year == 2007) & ~df.month.isin(STRICT_2007), ["T", "SW"]] = np.nan
    dq1 = np.array([(y, m) in E.W.SUSPECT_RN for y, m in zip(df.year, df.month)])
    df.loc[dq1, ["T", "SW"]] = np.nan
    d = df[df.year.isin(YEARS)].dropna(subset=FEATS + ["dTdt", "hour"]).copy()
    return d.reset_index(drop=True)


def harm(h, K=3):
    w = 2 * np.pi * np.asarray(h, float) / 24
    return np.column_stack([f(k * w) for k in range(1, K + 1) for f in (np.cos, np.sin)])


def generator(d, interaction):
    """OLS fit of G0/G1 on the real target; returns the deterministic part and residuals."""
    best = None
    for b in B_GRID:
        cols = [np.ones(len(d)), d["T"], 1 - np.exp(-b * d["SW"]), *harm(d.hour).T]
        if interaction:
            cols.append(d["SW"] * d["WS"])
        A = np.column_stack(cols)
        xi = np.linalg.lstsq(A, d.dTdt.to_numpy(), rcond=None)[0]
        res = d.dTdt.to_numpy() - A @ xi
        if best is None or res @ res < best[0]:
            best = (res @ res, A @ xi, res, xi, b)
    return best[1], best[2], best[3], best[4]


def synthetic_target(d, interaction, seed):
    mean, res, xi, b = generator(d, interaction)
    rng = np.random.default_rng(1000 + seed)
    out = np.empty(len(d))
    # reassign whole days: residual of a random day of the same year at the same hour
    uniq = {yr: d.loc[d.year == yr, "day"].unique() for yr in YEARS}
    by_day = d.groupby("day").indices
    perm = {}
    for yr in YEARS:
        u = uniq[yr]
        perm.update(dict(zip(u, rng.permutation(u))))
    hour_of = d.hour.to_numpy()
    for day, idx in by_day.items():
        src = by_day[perm[day]]
        src_res = dict(zip(hour_of[src], res[src]))
        for i in idx:
            r = src_res.get(hour_of[i])
            out[i] = res[i] if r is None else r
    return mean + out, xi, b


def train_net(X, y, seed, val_frac=0.15, max_epochs=300, patience=20):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    class SimpleNet(nn.Module):  # AI Feynman S_NN_train.SimpleNet
        def __init__(self, ni):
            super().__init__()
            self.l1, self.l2 = nn.Linear(ni, 128), nn.Linear(128, 128)
            self.l3, self.l4, self.l5 = nn.Linear(128, 64), nn.Linear(64, 64), nn.Linear(64, 1)

        def forward(self, x):
            for l in (self.l1, self.l2, self.l3, self.l4):
                x = F.tanh(l(x))
            return self.l5(x)

    mu, sd = X.mean(0), X.std(0)
    ym, ys = y.mean(), y.std()
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(((y - ym) / ys)[:, None], dtype=torch.float32)
    n = len(X)
    idx = rng.permutation(n)
    nv = int(val_frac * n)
    va, tr = idx[:nv], idx[nv:]
    net = SimpleNet(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 512)):
            opt.zero_grad()
            loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = ((net(Xt[va]) - yt[va]) ** 2).mean().item()
        if v < best - 1e-5:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def f(Xn):
        with torch.no_grad():
            return net(torch.tensor((Xn - mu) / sd, dtype=torch.float32)).numpy().ravel() * ys + ym
    return f


def separable(f, X, med, part, groups=None):
    if part is None:  # full additivity: sum of 1-D slices minus (k-1) f(med)
        groups = groups or [[j] for j in range(X.shape[1])]
        s = -(len(groups) - 1) * f(np.tile(med, (len(X), 1)))
        for g in groups:
            Z = np.tile(med, (len(X), 1)); Z[:, g] = X[:, g]
            s = s + f(Z)
        return s
    A = np.zeros(X.shape[1], bool); A[part] = True
    ZA = X.copy(); ZA[:, ~A] = med[~A]
    ZB = X.copy(); ZB[:, A] = med[A]
    return f(ZA) + f(ZB) - f(np.tile(med, (len(X), 1)))


def one_fold(args):
    """Train on two years, score every partition on the held-out one."""
    X, y, years, hold, seed, parts = args
    tr, te = years != hold, years == hold
    f = train_net(X[tr], y[tr], seed)
    med = np.median(X[tr], axis=0)
    full = f(X[te])
    out = {"full": full}
    for name, part in parts.items():
        out[name] = separable(f, X[te], med, part, GROUPS_H if X.shape[1] == 5 else None)
    return hold, seed, out


def boot_ci(e_full, e_sep, days, rng):
    """95% CI of mean(e_sep - e_full) by daily block bootstrap."""
    diff = e_sep - e_full
    u, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, diff); c = np.bincount(inv)
    stats = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(u), len(u))
        stats.append(s[k].sum() / c[k].sum())
    return np.percentile(stats, [2.5, 97.5])


def evaluate(d, y, label, seeds, pool, seed0=0, hour=False):
    X = d[FEATS].to_numpy(float)
    parts = PARTS
    if hour:
        X = np.column_stack([X, harm(d.hour, K=1)])
        parts = {**PARTS_H, "additive": None}
    years = d.year.to_numpy()
    jobs = [(X, y, years, h, seed0 + s, parts) for s in range(seeds) for h in YEARS]
    res = pool.map(one_fold, jobs)
    rows = []
    rng = np.random.default_rng(7)
    for s in range(seeds):
        preds = {h: o for h, ss, o in res if ss == seed0 + s}
        for name in parts:
            e_full_all, e_sep_all, days_all = [], [], []
            for h in YEARS:
                te = years == h
                yt = y[te]
                ef = (yt - preds[h]["full"]) ** 2
                es = (yt - preds[h][name]) ** 2
                inn = np.mean((preds[h]["full"] - preds[h][name]) ** 2) / np.var(preds[h]["full"])
                aif = np.median(2 * np.abs(yt - preds[h][name])) / np.median(2 * np.abs(yt - preds[h]["full"]))
                r2 = 1 - ef.sum() / ((yt - yt.mean()) ** 2).sum()
                rows.append(dict(data=label, seed=seed0 + s, partition=name, held_out=h, n=int(te.sum()),
                                 r2_full=r2, delta=es.mean() - ef.mean(), I_nn=inn, aif_ratio=aif))
                e_full_all.append(ef); e_sep_all.append(es); days_all.append(d.day.to_numpy()[te])
            ef, es, dd = (np.concatenate(v) for v in (e_full_all, e_sep_all, days_all))
            lo, hi = boot_ci(ef, es, dd, rng)
            rows.append(dict(data=label, seed=seed0 + s, partition=name, held_out="pooled", n=len(ef),
                             r2_full=np.nan, delta=es.mean() - ef.mean(), I_nn=np.nan, aif_ratio=np.nan,
                             ci_lo=lo, ci_hi=hi, flag=bool(lo > 0)))
    return pd.DataFrame(rows)


def verdict(t, seeds):
    p = t[t.held_out == "pooled"].groupby(["data", "partition"]).agg(
        flags=("flag", "sum"), delta_med=("delta", "median"), ci_lo_med=("ci_lo", "median"), ci_hi_med=("ci_hi", "median"))
    y = t[t.held_out != "pooled"].groupby(["data", "partition", "held_out"]).delta.median().unstack()
    p["years_pos"] = (y > 0).sum(axis=1)
    p = p.join(y.add_prefix("delta_"))
    q = t[t.held_out != "pooled"].groupby(["data", "partition"])[["I_nn", "aif_ratio", "r2_full"]].median()
    return p.join(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--cache")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--window", default="strict", choices=["strict", "loose"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hour", action="store_true", help="exploratory: hour of day as an input group")
    a = ap.parse_args()
    d = load(a.xlsx, a.cache, a.window)
    print("rows per year:", d.groupby("year").size().to_dict(), "months 2007:", sorted(d[d.year == 2007].month.unique()))
    y0, xi0, b0 = synthetic_target(d, False, 0)
    _, xi1, b1 = synthetic_target(d, True, 0)
    print(f"G0 b={b0} xi={np.round(xi0, 5)}\nG1 b={b1} kappa={xi1[-1]:.3g}")
    with get_context("spawn").Pool(a.workers) as pool:
        tabs = [evaluate(d, d.dTdt.to_numpy(), "real", a.seeds, pool, hour=a.hour)]
        for lab, inter in (("G0_additive", False), ("G1_SWxWS", True)):
            ys = [synthetic_target(d, inter, s)[0] for s in range(a.seeds)]
            # each seed gets its own noise realisation and its own nets
            parts = []
            for s in range(a.seeds):
                parts.append(evaluate(d, ys[s], lab, 1, pool, seed0=s, hour=a.hour))
            tabs.append(pd.concat(parts))
    t = pd.concat(tabs, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.window + ("_hour" if a.hour else "")
    t.to_csv(OUT / f"aif_separability_{tag}.csv", index=False)
    v = verdict(t, a.seeds)
    v.to_csv(OUT / f"aif_separability_{tag}_summary.csv")
    P = {**PARTS_H, "additive": None} if a.hour else PARTS
    sw_ws = ("SW|T,WS,H", "WS|SW,T,H") if a.hour else ("SW|T,WS", "WS|SW,T")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(v.round(4))
    calib = (all(v.loc[("G0_additive", p), "flags"] <= 1 for p in P)
             and all(v.loc[("G1_SWxWS", p), "flags"] >= a.seeds - 1 for p in sw_ws))
    for p in P:  # per-partition calibration, reported alongside the global rule
        print(f"calib {p:10s}: G0 flags {int(v.loc[('G0_additive', p), 'flags'])}/{a.seeds}, "
              f"G1 flags {int(v.loc[('G1_SWxWS', p), 'flags'])}/{a.seeds}")
    print("calibration passes:", calib)
    for p in P:
        r = v.loc[("real", p)]
        ns = r["flags"] >= a.seeds - 1 and r["years_pos"] >= 2
        print(f"real {p:10s}: {'NON-SEPARABLE' if ns else 'separable / not detected'}"
              f"{'' if calib else ' (inconclusive: calibration failed)'}")


if __name__ == "__main__":
    main()
