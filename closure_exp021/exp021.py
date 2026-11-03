"""EXP-021: symbolic regression as the closure of a physical air-temperature model,
with a priori, 24-h simulation and leave-one-site-out (LOSO) evaluation.

Base physical model B (EXP-020, sat+rdot):
    dT/dt = c0 + c1*T + c2*(1 - exp(-b*SW)) + c3*dSW/dt
Closure s(x), x = (T, SW, WS, VPD), no hour of day. Engines: weak-form SINDy (quadratic
library, STLSQ), Operon on the pointwise residual, HistGradientBoosting as a black-box ceiling.
Decision rules are registered in EXPERIMENT_LOG.md (EXP-021) before this script was run.

Usage: python exp021.py RAW_DIR OUT_DIR [--quick]
RAW_DIR holds <site>.pkl from reduce.py.
"""
import sys, time, numpy as np, pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import HistGradientBoostingRegressor
from pyoperon.sklearn import SymbolicRegressor

RAW, OUT = sys.argv[1], sys.argv[2]
QUICK = "--quick" in sys.argv
DT, M, P = 0.5, 9, 4
B_GRID = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02]
THR_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SITES = {
    "Ji3": dict(train=[2006, 2008], test=[2005, 2009, 2010]),
    "Ma2": dict(train=[2002, 2003, 2004, 2005], test=[2006, 2007, 2008, 2009]),
    "Sa1": dict(train=[2002, 2003, 2004], test=[2009, 2010, 2011]),
    "Ma3": dict(train=[2009, 2010, 2011], test=[2013, 2018]),
    "Sa3": dict(train=[2001, 2002], test=[2000, 2003]),
}
XV = ["T", "SW", "WS", "VPD"]
SEEDS = [0] if QUICK else [0, 1, 2]
NB = 200 if QUICK else 1000

# ---------------------------------------------------------------- data
def load(site):
    f = pd.read_pickle(f"{RAW}/{site}.pkl")
    d = pd.DataFrame({"ts": pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")})
    d["T"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0).to_numpy()
    d["SW"] = f.SW_IN_F_MDS.where(f.SW_IN_F_MDS_QC == 0).to_numpy()
    d["WS"] = f.WS.to_numpy(float)
    d["VPD"] = f.VPD_F_MDS.to_numpy(float)
    d["year"] = d.ts.dt.year; d["hour"] = d.ts.dt.hour + d.ts.dt.minute / 60; d["day"] = d.ts.dt.date
    yrs = SITES[site]["train"] + SITES[site]["test"]
    return d[d.year.isin(yrs)].reset_index(drop=True)

def windows(d):
    ok = d[XV].notna().all(axis=1).to_numpy(); yr = d.year.to_numpy(); out = []; st = None
    for i in range(len(d) + 1):
        good = i < len(d) and ok[i] and (st is None or yr[i] == yr[st])
        if good and st is None: st = i
        if not good and st is not None:
            for a in range(st, i - M + 1, max(M // 2, 1)): out.append(np.arange(a, a + M))
            st = i if (i < len(d) and ok[i]) else None
    return np.array(out)

def sg_deriv(d):
    """Pointwise dT/dt (K/h) by Savitzky-Golay (7 points, order 2) on contiguous valid runs."""
    T = d["T"].to_numpy(float); ok = d[XV].notna().all(axis=1).to_numpy(); out = np.full(len(d), np.nan)
    i = 0
    while i < len(d):
        if not ok[i]: i += 1; continue
        j = i
        while j < len(d) and ok[j]: j += 1
        if j - i >= 7: out[i:j] = savgol_filter(T[i:j], 7, 2, deriv=1, delta=DT)
        i = j
    return out

s_ = np.linspace(0, 1, M); L_ = (M - 1) * DT
v = (s_ * (1 - s_)) ** P; dv = P * (s_ * (1 - s_)) ** (P - 1) * (1 - 2 * s_) / L_
w = np.full(M, DT); w[0] = w[-1] = DT / 2; v, dv = v * w, dv * w
weak = lambda f, R: v @ f[R].T          # ∫φ f
def weak_rdot(SW, R): return -(dv @ SW[R].T)  # ∫φ dSW/dt by parts

# ---------------------------------------------------------------- base model and closures
def base_cols(d, R, b):
    T, SW = d["T"].to_numpy(float), d["SW"].to_numpy(float)
    return np.column_stack([weak(np.ones_like(T), R), weak(T, R), weak(1 - np.exp(-b * SW), R), weak_rdot(SW, R)])

def harm_cols(d, R):
    wv = 2 * np.pi * d["hour"].to_numpy(float) / 24
    return np.column_stack([weak(f(k * wv), R) for k in range(1, 4) for f in (np.cos, np.sin)])

def lib_point(X, b, mu, sd):
    """Quadratic library on standardised (T, SWsat, WS, VPD); linear T and SWsat are already in B."""
    Z = np.column_stack([X[:, 0], 1 - np.exp(-b * X[:, 1]), X[:, 2], X[:, 3]])
    Z = (Z - mu) / sd
    cols, names = [Z[:, 2], Z[:, 3]], ["WS", "VPD"]
    nm = ["T", "SWs", "WS", "VPD"]
    for a in range(4):
        for c in range(a, 4):
            cols.append(Z[:, a] * Z[:, c]); names.append(f"{nm[a]}*{nm[c]}")
    return np.column_stack(cols), names

def zstats(d, b):
    Z = np.column_stack([d["T"], 1 - np.exp(-b * d["SW"]), d["WS"], d["VPD"]]).astype(float)
    Z = Z[np.isfinite(Z).all(1)]
    return Z.mean(0), Z.std(0)

def lstsq(G, y): return np.linalg.lstsq(G, y, rcond=None)[0]

def stlsq(GB, GL, y, thr, n_iter=10):
    """STLSQ on the library block only; base columns are never thresholded. Library columns are
    scaled to unit std so the threshold is on standardised coefficients."""
    sc = GL.std(0); sc[sc == 0] = 1; GLs = GL / sc
    keep = np.ones(GL.shape[1], bool)
    for _ in range(n_iter):
        xi = lstsq(np.column_stack([GB, GLs[:, keep]]), y)
        cl = np.zeros(GL.shape[1]); cl[keep] = xi[GB.shape[1]:]
        new = np.abs(cl) >= thr
        if (new == keep).all(): break
        keep = new
    xi = lstsq(np.column_stack([GB, GLs[:, keep]]), y)
    cl = np.zeros(GL.shape[1]); cl[keep] = xi[GB.shape[1]:]
    return xi[:GB.shape[1]], cl / sc, keep

def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

def ci_r2(y, pa, pb, days, rng):
    ud = np.unique(days); idx = {k: np.flatnonzero(days == k) for k in ud}; out = []
    for _ in range(NB):
        ss = np.concatenate([idx[k] for k in rng.choice(ud, len(ud))])
        out.append(r2(y[ss], pa[ss]) - r2(y[ss], pb[ss]))
    return np.percentile(out, [2.5, 97.5])

def operon(seed):
    return SymbolicRegressor(allowed_symbols="add,sub,mul,div,exp,constant,variable", max_length=25,
                             population_size=500, generations=200, optimizer_iterations=10,
                             max_time=20 if QUICK else 60, n_threads=4, random_state=seed)

# ---------------------------------------------------------------- per-site preparation
S = {}
for site in SITES:
    d = load(site); R = windows(d)
    mid = R[:, M // 2]; ry = d.year.to_numpy()[mid]; rd = d.day.to_numpy()[mid]
    tr = np.isin(ry, SITES[site]["train"]); te = np.isin(ry, SITES[site]["test"])
    y = -(dv @ d["T"].to_numpy(float)[R].T)
    best = None
    for b in B_GRID:
        GB = base_cols(d, R, b); c = lstsq(GB[tr], y[tr]); sse = np.sum((y[tr] - GB[tr] @ c) ** 2)
        if best is None or sse < best[0]: best = (sse, b, c)
    b, cB = best[1], best[2]
    GB = base_cols(d, R, b); GH = harm_cols(d, R)
    X = d[XV].to_numpy(float); dTdt = sg_deriv(d)
    T, SW = d["T"].to_numpy(float), d["SW"].to_numpy(float)
    # pointwise base prediction with the weak-form coefficients (rdot by central difference)
    rdot = np.gradient(SW, DT)
    Bpt = cB[0] + cB[1] * T + cB[2] * (1 - np.exp(-b * SW)) + cB[3] * rdot
    S[site] = dict(d=d, R=R, tr=tr, te=te, y=y, rd=rd, ry=ry, b=b, cB=cB, GB=GB, GH=GH, X=X,
                   res=dTdt - Bpt, ptyear=d.year.to_numpy())
    print(site, "windows", len(R), "train", tr.sum(), "test", te.sum(), "b", b, flush=True)

def pt_train_mask(site):
    s = S[site]; return np.isin(s["ptyear"], SITES[site]["train"]) & np.isfinite(s["res"]) & np.isfinite(s["X"]).all(1)

# ---------------------------------------------------------------- local fits
rows, expr = [], []
pred = {}  # (site, model) -> weak-form prediction on all windows
models_sim = {}  # (site, model) -> (cB, b, closure callable, closure coef)

for site, s in S.items():
    d, R, tr, te, y, b = s["d"], s["R"], s["tr"], s["te"], s["y"], s["b"]
    GB, GH = s["GB"], s["GH"]
    pred[(site, "B")] = GB @ lstsq(GB[tr], y[tr])
    GBH = np.column_stack([GB, GH]); pred[(site, "B+harm")] = GBH @ lstsq(GBH[tr], y[tr])
    models_sim[(site, "B")] = (lstsq(GB[tr], y[tr]), b, None, 0.0)

    # SINDy weak form: threshold by leave-one-train-year-out CV
    mu, sd = zstats(d[d.year.isin(SITES[site]["train"])], b)
    Lp, names = lib_point(s["X"], b, mu, sd)
    GL = np.column_stack([weak(Lp[:, k], R) for k in range(Lp.shape[1])])
    cv = []
    for thr in THR_GRID:
        e = 0.0
        for yo in SITES[site]["train"]:
            a = tr & (s["ry"] != yo); o = tr & (s["ry"] == yo)
            cb, cl, _ = stlsq(GB[a], GL[a], y[a], thr); e += np.sum((y[o] - GB[o] @ cb - GL[o] @ cl) ** 2)
        cv.append(e)
    thr = THR_GRID[int(np.argmin(cv))]
    cb, cl, keep = stlsq(GB[tr], GL[tr], y[tr], thr)
    pred[(site, "SINDy")] = GB @ cb + GL @ cl
    expr.append({"site": site, "model": "SINDy", "thr": thr, "n_terms": int(keep.sum()),
                 "expression": " + ".join(f"{c:.4g}*{n}" for c, n in zip(cl, names) if c != 0)})
    models_sim[(site, "SINDy")] = (cb, b, ("sindy", mu, sd, cl), 1.0)

    # Operon and GBM on the pointwise residual
    m = pt_train_mask(site); Xtr, rtr = s["X"][m], s["res"][m]
    rng = np.random.default_rng(0)
    if len(rtr) > 20000:
        k = rng.choice(len(rtr), 20000, replace=False); Xtr, rtr = Xtr[k], rtr[k]
    fin = np.isfinite(s["X"]).all(1)
    for seed in SEEDS:
        t0 = time.time(); op = operon(seed); op.fit(Xtr, rtr)
        sp = np.zeros(len(d)); sp[fin] = op.predict(s["X"][fin])
        Gs = np.column_stack([GB, weak(sp, R)]); c = lstsq(Gs[tr], y[tr])
        pred[(site, f"Operon_s{seed}")] = Gs @ c
        expr.append({"site": site, "model": f"Operon_s{seed}", "length": op.model_.Length if hasattr(op.model_, "Length") else np.nan,
                     "scale": c[-1], "seconds": round(time.time() - t0, 1),
                     "expression": op.get_model_string(op.model_, 4, ["T", "SW", "WS", "VPD"])})
        models_sim[(site, f"Operon_s{seed}")] = (c[:4], b, ("op", op), c[-1])
    gb = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, rtr)
    sp = np.zeros(len(d)); sp[fin] = gb.predict(s["X"][fin])
    Gs = np.column_stack([GB, weak(sp, R)]); c = lstsq(Gs[tr], y[tr])
    pred[(site, "GBM")] = Gs @ c
    models_sim[(site, "GBM")] = (c[:4], b, ("op", gb), c[-1])
    print(site, "local fits done", flush=True)

# ---------------------------------------------------------------- LOSO transfer (a priori)
def partial_out(GB, A):
    """Residualise the columns of A on the base block (Frisch-Waugh), within one site."""
    return A - GB @ np.linalg.lstsq(GB, A, rcond=None)[0]

for target in S:
    src = [k for k in S if k != target]
    t = S[target]
    # SINDy: pooled standardisation and pooled STLSQ on base-partialled columns
    dpool = pd.concat([S[k]["d"][S[k]["d"].year.isin(SITES[k]["train"])] for k in src])
    bpool = float(np.median([S[k]["b"] for k in src]))
    mu, sd = zstats(dpool, bpool)
    GLs, ys = [], []
    for k in src:
        sk = S[k]; Lp, names = lib_point(sk["X"], bpool, mu, sd)
        GL = np.column_stack([weak(Lp[:, j], sk["R"]) for j in range(Lp.shape[1])])[sk["tr"]]
        GBk = sk["GB"][sk["tr"]]; GLs.append(partial_out(GBk, GL)); ys.append(partial_out(GBk, sk["y"][sk["tr"]][:, None])[:, 0])
    GLp, yp = np.vstack(GLs), np.concatenate(ys)
    thr = float(pd.DataFrame(expr).query("model=='SINDy' and site in @src").thr.median())
    _, cl, keep = stlsq(np.zeros((len(yp), 0)), GLp, yp, thr)
    Lp, _ = lib_point(t["X"], bpool, mu, sd)
    GLt = np.column_stack([weak(Lp[:, j], t["R"]) for j in range(Lp.shape[1])])
    tr, y, GB = t["tr"], t["y"], t["GB"]
    cb = lstsq(GB[tr], y[tr] - GLt[tr] @ cl)                     # frozen closure, base refitted
    pred[(target, "SINDy LOSO frozen")] = GB @ cb + GLt @ cl
    Gr = np.column_stack([GB, GLt[:, keep]]); c = lstsq(Gr[tr], y[tr])   # same terms, refitted
    pred[(target, "SINDy LOSO refit")] = Gr @ c
    expr.append({"site": target, "model": "SINDy LOSO", "n_terms": int(keep.sum()),
                 "expression": " + ".join(f"{c_:.4g}*{n}" for c_, n in zip(cl, names) if c_ != 0)})
    # Operon / GBM: pooled pointwise residuals (each source site with its own base)
    Xp = np.vstack([S[k]["X"][pt_train_mask(k)] for k in src]); rp = np.concatenate([S[k]["res"][pt_train_mask(k)] for k in src])
    rng = np.random.default_rng(0); kk = rng.choice(len(rp), min(40000, len(rp)), replace=False)
    fin = np.isfinite(t["X"]).all(1)
    learners = [(f"Operon_s{sd_} LOSO", operon(sd_)) for sd_ in SEEDS] + [("GBM LOSO", HistGradientBoostingRegressor(max_iter=300, random_state=0))]
    for name, est in learners:
        est.fit(Xp[kk], rp[kk]); sp = np.zeros(len(t["d"])); sp[fin] = est.predict(t["X"][fin]); ws = weak(sp, t["R"])
        cb = lstsq(GB[tr], y[tr] - ws[tr]); pred[(target, name.replace("LOSO", "LOSO frozen"))] = GB @ cb + ws
        Gs = np.column_stack([GB, ws]); c = lstsq(Gs[tr], y[tr]); pred[(target, name.replace("LOSO", "LOSO refit"))] = Gs @ c
        if name.startswith("Operon"):
            expr.append({"site": target, "model": name, "expression": est.get_model_string(est.model_, 4, ["T", "SW", "WS", "VPD"])})
    print(target, "LOSO done", flush=True)

# ---------------------------------------------------------------- a priori scores
for site, s in S.items():
    te, y, rd = s["te"], s["y"], s["rd"]; rng = np.random.default_rng(0)
    for (st, m), p in pred.items():
        if st != site: continue
        row = {"site": site, "model": m, "r2_test": r2(y[te], p[te])}
        for ref in ("B", "B+harm"):
            lo, hi = ci_r2(y[te], p[te], pred[(site, ref)][te], rd[te], rng)
            row[f"dR2_vs_{ref}"] = row["r2_test"] - r2(y[te], pred[(site, ref)][te]); row[f"lo_{ref}"] = lo; row[f"hi_{ref}"] = hi
        rows.append(row)
A = pd.DataFrame(rows); A.to_csv(f"{OUT}/exp021_apriori.csv", index=False)
pd.DataFrame(expr).to_csv(f"{OUT}/exp021_expressions.csv", index=False)
pd.set_option("display.width", 250)
print(A.round(4).to_string(index=False), flush=True)

# ---------------------------------------------------------------- 24-h simulation (local models)
def closure_value(spec, T, sw, ws, vpd, b):
    if spec is None: return 0.0
    if spec[0] == "sindy":
        _, mu, sd, cl = spec; Lp, _ = lib_point(np.column_stack([T, sw, ws, vpd]), b, mu, sd); return Lp @ cl
    X = np.column_stack([T, sw, ws, vpd]); out = spec[1].predict(X); return out

def simulate(site, key):
    s = S[site]; cb, b, spec, scale = models_sim[(site, key)]; out = []
    for ty in SITES[site]["test"]:
        dy = s["d"][s["d"].year == ty].reset_index(drop=True)
        need = dy[XV].notna().all(axis=1).to_numpy()
        starts = [a for a in np.flatnonzero(np.isclose(dy.hour.to_numpy(), 0.0)) if a + 48 < len(dy) and need[a:a + 49].all()]
        if not starts: continue
        idx = np.array(starts)[:, None] + np.arange(49)[None, :]
        A_ = {c: dy[c].to_numpy(float)[idx] for c in XV}
        def rhs(T, j, fr):
            jj = min(j + 1, 48); it = lambda c: A_[c][:, j] + fr * (A_[c][:, jj] - A_[c][:, j])
            sw = it("SW"); rd = (A_["SW"][:, jj] - A_["SW"][:, j]) / DT
            base = cb[0] + cb[1] * T + cb[2] * (1 - np.exp(-b * sw)) + cb[3] * rd
            return base + scale * closure_value(spec, T, sw, it("WS"), it("VPD"), b)
        T = A_["T"][:, 0].copy(); sim = []; h = 0.05
        with np.errstate(all="ignore"):
            for j in range(48):
                for k in range(10):
                    fr = k / 10
                    k1 = rhs(T, j, fr); k2 = rhs(T + h / 2 * k1, j, fr + .05); k3 = rhs(T + h / 2 * k2, j, fr + .05); k4 = rhs(T + h * k3, j, fr + .1)
                    T = T + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                sim.append(T.copy())
        sim = np.array(sim).T; rm = np.sqrt(np.mean((sim - A_["T"][:, 1:]) ** 2, axis=1))
        rm[~np.all((sim > 0) & (sim < 60), axis=1)] = np.nan
        out += [{"site": site, "model": key, "year": ty, "day": dd, "rmse": r} for dd, r in zip(dy.day.to_numpy()[idx[:, 0]], rm)]
    return out

sims = []
for (site, key) in models_sim:
    sims += simulate(site, key); print("sim", site, key, flush=True)
Sm = pd.DataFrame(sims); Sm.to_csv(f"{OUT}/exp021_sim_days.csv", index=False)
out = []
for site in S:
    piv = Sm[Sm.site == site].pivot_table(index=["year", "day"], columns="model", values="rmse")
    rng = np.random.default_rng(1)
    for m in piv.columns:
        ok = piv[m].notna() & piv["B"].notna(); diff = (piv[m] - piv["B"])[ok].to_numpy()
        bs = [np.mean(rng.choice(diff, len(diff))) for _ in range(NB)]
        out.append({"site": site, "model": m, "days": int(ok.sum()), "blowups": int(piv[m].isna().sum()), "rmse": piv[m][ok].mean(),
                    "d_vs_B": diff.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
Sim = pd.DataFrame(out); Sim.to_csv(f"{OUT}/exp021_sim.csv", index=False)
print(Sim.round(3).to_string(index=False))
