"""EXP-031: score the laptop PySR equations with the same data, audit, classical law, bootstrap and validity tests as exp031_rn_rh.py.
Usage: python score.py RAW REPO_SRC OUT"""
import sys, os, numpy as np, pandas as pd, sympy as sp
here = os.path.dirname(os.path.abspath(__file__))
src = open("/mnt/project-files/code/article1_closure/exp031_rn_rh.py").read()
pre = src.split("ENG = {")[0].replace("from methods import", "#from methods import")
exec(pre)
from scipy.optimize import curve_fit
EQ = {}
for line in open("/tmp/claude-0/-home-claude-sr-paper/1bba3474-96b0-5868-a589-6868a680e616/scratchpad/e32/dso_chosen.txt"):
    k, e = [x.strip() for x in line.split("|", 1)]; t, s, m, sd = k.split("__"); EQ[(t, s, m, int(sd))] = e
def mk(expr, feats):
    sy = sp.symbols(" ".join(feats)); sy = sy if isinstance(sy, tuple) else (sy,)
    f = sp.lambdify(sy, sp.sympify(expr), "numpy")
    def g(X):
        with np.errstate(all="ignore"):
            o = f(*[X[:, i] for i in range(len(feats))])
        return np.broadcast_to(np.asarray(o, float), (len(X),)).copy()
    return g
rows, data = [], {}
for target in ("Rn", "RH"):
    law, p0, feats = LAWS[target]; sites = [s for s in SITES if not (target == "Rn" and s == "Ji3")]
    for s in sites:
        d = load(s); bad = audit(d, target); d = d[~d.month.isin(bad)].dropna(subset=feats + [target])
        data[(target, s)] = (d[d.year.isin(SITES[s]["train"])], d[d.year.isin(SITES[s]["test"])])
    for s in sites:
        tr, te = data[(target, s)]; Xte = te[feats].to_numpy(float); yte = te[target].to_numpy(float); days = te.date.to_numpy()
        dom = dict(SWmax=float(pd.concat([tr, te]).SW.max())) if target == "Rn" else dict(
            Tlo=float(tr["T"].quantile(.01)), Thi=float(tr["T"].quantile(.99)), Tmlo=float(tr.Tmin.quantile(.01)), Tmhi=float(tr.Tmin.quantile(.99)))
        for mode in ("local", "loso"):
            for seed in range(5):
                loc = tr.sample(min(N_TRAIN, len(tr)), random_state=seed); Xl, yl = loc[feats].to_numpy(float), loc[target].to_numpy(float)
                pc, _ = curve_fit(law, Xl, yl, p0=p0, maxfev=20000); p_cl = law(Xte, *pc)
                e = EQ[(target, s, mode, seed)]; pr = mk(e, feats); variants = {"": pr}
                if mode == "loso":
                    fl = pr(Xl); variants = {" frozen": pr}
                    if np.isfinite(fl).all():
                        ab = np.linalg.lstsq(np.column_stack([np.ones_like(fl), fl]), yl, rcond=None)[0]
                        variants[" refit"] = (lambda X, pr=pr, ab=ab: ab[0] + ab[1] * pr(X))
                for tag, f in variants.items():
                    p = f(Xte); ok = bool(np.isfinite(p).all()); lo, hi = boot(yte, p, p_cl, days, seed) if ok else (np.nan, np.nan)
                    rows.append(dict(target=target, site=s, mode=mode + tag, engine="dso", seed=seed, status="ok" if ok else "nonfinite",
                                     rmse=rmse(yte, p) if ok else np.nan, rmse_classical=rmse(yte, p_cl), d_lo=lo, d_hi=hi,
                                     win=bool(ok and hi < 0), valid=valid(target, f, dom), expression=e))
        print(target, s, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/exp031_dso.csv", index=False)
