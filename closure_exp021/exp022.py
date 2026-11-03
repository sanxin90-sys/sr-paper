"""EXP-022: Rn ~ SW_IN and RH ~ (T, Tmin_day) on several FLUXNET towers, SR vs re-fitted classical laws,
with a fixed audit, fixed physical-validity tests and leave-one-site-out transfer.
Rules registered in EXPERIMENT_LOG.md (EXP-022) before running.
Usage: python exp022.py RAW_DIR REPO_SRC OUT_DIR [--quick]"""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit
RAW, REPO, OUT = sys.argv[1:4]; QUICK = "--quick" in sys.argv
sys.path[:0] = [REPO, REPO + "/sr_benchmark"]
from methods import fit_operon, fit_sindy_poly, fit_sindy, fit_gplearn, complexity  # noqa: E402

SITES = {"Ji3": dict(train=[2006, 2008], test=[2005, 2009, 2010]),
         "Ma2": dict(train=[2002, 2003, 2004, 2005], test=[2006, 2007, 2008, 2009]),
         "Sa1": dict(train=[2002, 2003, 2004], test=[2009, 2010, 2011]),
         "Ma3": dict(train=[2009, 2010, 2011], test=[2013, 2018]),
         "Sa3": dict(train=[2001, 2002], test=[2000, 2003])}
N_TRAIN, NB, BUDGET = 3000, (200 if QUICK else 500), (15 if QUICK else 45)
SEEDS = [0, 1] if QUICK else list(range(5))
es = lambda t: 6.112 * np.exp(17.67 * t / (t + 243.5))
LAWS = {"Rn": (lambda X, a, b: a * X[:, 0] + b, [0.7, -20.0], ["SW"]),
        "RH": (lambda X, a, b: 100 * es(a * X[:, 1] + b) / es(X[:, 0]), [1.0, -1.0], ["T", "Tmin"])}

def load(site):
    f = pd.read_pickle(f"{RAW}/{site}.pkl")
    d = pd.DataFrame({"ts": pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")})
    d["T"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0).to_numpy()
    d["SW"] = f.SW_IN_F_MDS.where(f.SW_IN_F_MDS_QC == 0).to_numpy()
    d["Rn"] = f["NETRAD"].to_numpy(float) if "NETRAD" in f else np.nan
    rh = f.RH.to_numpy(float); d["RH"] = np.where((rh >= 1) & (rh <= 100.5), rh, np.nan)
    d["date"] = (d.ts - pd.Timedelta("30min")).dt.date
    full = d.groupby("date")["T"].transform("count") == 48
    d["Tmin"] = d.groupby("date")["T"].transform("min").where(full)
    d["year"] = d.ts.dt.year; d["month"] = d.ts.dt.to_period("M").astype(str); d["hour"] = d.ts.dt.hour
    return d[d.year.isin(SITES[site]["train"] + SITES[site]["test"])].reset_index(drop=True)

def audit(d, target):
    if target == "Rn":
        m = d[(d.SW > 100) & d.Rn.notna()].assign(r=lambda x: x.Rn / x.SW).groupby("month").r.median()
        bad = m[(m - m.median()).abs() > 0.15 * m.median()].index.tolist()
    else:
        m = d[d.hour.between(0, 5)].groupby("month").RH.median()
        bad = m[(m < 70) | (m > 100.5)].index.tolist()
    return bad

def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def boot(y, pa, pb, days, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(days); idx = {k: np.flatnonzero(days == k) for k in u}; o = []
    for _ in range(NB):
        ii = np.concatenate([idx[k] for k in rng.choice(u, len(u))]); o.append(rmse(y[ii], pa[ii]) - rmse(y[ii], pb[ii]))
    return np.percentile(o, [2.5, 97.5])

def valid(target, f, dom):
    with np.errstate(all="ignore"):
        if target == "Rn":
            g = np.linspace(0, dom["SWmax"], 300)[:, None]; p = f(g)
            return bool(np.isfinite(p).all() and -150 <= p[0] <= 50 and np.all(np.diff(p) >= -1e-6))
        T = np.linspace(dom["Tlo"], dom["Thi"], 40); Tm = np.linspace(dom["Tmlo"], dom["Tmhi"], 40)
        TT, MM = np.meshgrid(T, Tm); ok = MM <= TT; X = np.column_stack([TT[ok], MM[ok]]); p = f(X)
        if not (np.isfinite(p).all() and p.min() >= 0 and p.max() <= 105): return False
        P = np.full(TT.shape, np.nan); P[ok] = p
        dT = np.diff(P, axis=1); return bool(np.nanmax(dT) <= 1e-6) if np.isfinite(dT).any() else True

ENG = {"sindy_poly2": (lambda X, y, n, s: fit_sindy_poly(X, y, n), [0]),
       "sindy_ext": (lambda X, y, n, s: fit_sindy(X, y, n), [0]),
       "operon": (lambda X, y, n, s: fit_operon(X, y, n, time_budget_s=BUDGET, seed=s), SEEDS),
       "gplearn": (lambda X, y, n, s: fit_gplearn(X, y, n, time_budget_s=BUDGET, seed=s), SEEDS)}

rows, audits, data = [], [], {}
for target in ("Rn", "RH"):
    law, p0, feats = LAWS[target]
    sites = [s for s in SITES if not (target == "Rn" and s == "Ji3")]
    for s in sites:
        d = load(s); bad = audit(d, target); audits.append({"target": target, "site": s, "excluded_months": " ".join(bad)})
        d = d[~d.month.isin(bad)].dropna(subset=feats + [target])
        tr, te = d[d.year.isin(SITES[s]["train"])], d[d.year.isin(SITES[s]["test"])]
        data[(target, s)] = (tr, te)
        print(target, s, "train", len(tr), "test", len(te), "excluded", bad, flush=True)
    for s in sites:
        tr, te = data[(target, s)]; Xte = np.ascontiguousarray(te[feats].to_numpy(float)); yte = te[target].to_numpy(float)
        days = te.date.to_numpy()
        dom = dict(SWmax=float(pd.concat([tr, te]).SW.max())) if target == "Rn" else dict(
            Tlo=float(tr["T"].quantile(.01)), Thi=float(tr["T"].quantile(.99)), Tmlo=float(tr.Tmin.quantile(.01)), Tmhi=float(tr.Tmin.quantile(.99)))
        # local and LOSO training sets per seed
        for mode in ("local", "loso"):
            for eng, (fit, seeds) in ENG.items():
                for seed in seeds:
                    if mode == "local":
                        sub = tr.sample(min(N_TRAIN, len(tr)), random_state=seed)
                    else:
                        sub = pd.concat([data[(target, k)][0].sample(min(N_TRAIN, len(data[(target, k)][0])), random_state=seed)
                                         for k in sites if k != s])
                    Xtr = np.ascontiguousarray(sub[feats].to_numpy(float)); ytr = sub[target].to_numpy(float)
                    loc = tr.sample(min(N_TRAIN, len(tr)), random_state=seed)
                    Xl, yl = np.ascontiguousarray(loc[feats].to_numpy(float)), loc[target].to_numpy(float)
                    pc, _ = curve_fit(law, Xl, yl, p0=p0, maxfev=20000); p_cl = law(Xte, *pc)
                    try:
                        F = fit(Xtr, ytr, feats, seed)
                    except Exception as e:
                        rows.append(dict(target=target, site=s, mode=mode, engine=eng, seed=seed, status=f"fail {e}"[:120])); continue
                    pr = (lambda X, F=F: F.predict(np.array(X, dtype=float, order="C", copy=True)))
                    variants = {"": pr}
                    if mode == "loso":  # affine re-fit a + c f(x) on the target's training sample
                        with np.errstate(all="ignore"): fl = pr(Xl)
                        if np.isfinite(fl).all():
                            A = np.column_stack([np.ones_like(fl), fl]); ab = np.linalg.lstsq(A, yl, rcond=None)[0]
                            variants = {" frozen": pr, " refit": (lambda X, pr=pr, ab=ab: ab[0] + ab[1] * pr(X))}
                        else:
                            variants = {" frozen": pr}
                    for tag, f in variants.items():
                        with np.errstate(all="ignore"): p = f(Xte)
                        ok = np.isfinite(p).all(); lo, hi = boot(yte, p, p_cl, days, seed) if ok else (np.nan, np.nan)
                        rows.append(dict(target=target, site=s, mode=mode + tag, engine=eng, seed=seed, status="ok",
                                         rmse=rmse(yte, p) if ok else np.nan, rmse_classical=rmse(yte, p_cl), d_lo=lo, d_hi=hi,
                                         win=bool(ok and hi < 0), valid=valid(target, f, dom), complexity=complexity(F.expression),
                                         classical=" ".join(f"{v:.4g}" for v in pc), expression=F.expression))
                print(target, s, mode, eng, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/exp022_results.csv", index=False); pd.DataFrame(audits).to_csv(f"{OUT}/exp022_audit.csv", index=False)
R["valid_win"] = R.win & R.valid
print(pd.DataFrame(audits).to_string(index=False))
pd.set_option("display.width", 250)
print(R.groupby(["target", "mode", "engine", "site"]).agg(n=("seed", "size"), wins=("win", "sum"), valid_wins=("valid_win", "sum"),
      d_rmse=("rmse", lambda x: np.nan)).reset_index().pivot_table(index=["target", "mode", "engine"], columns="site", values="valid_wins").to_string())
R["d"] = R.rmse - R.rmse_classical
print(R.pivot_table(index=["target", "mode", "engine"], columns="site", values="d", aggfunc="median").round(3).to_string())
