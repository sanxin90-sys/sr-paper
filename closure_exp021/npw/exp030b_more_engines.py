"""EXP-021b (cloud part): gplearn and extended weak-form SINDy as closures of B, scored exactly like
exp021.py (a priori vs B and B+harm, LOSO frozen/refit, 24-h simulation for local fits).
Usage: python exp021b.py RAW_DIR OUT_DIR [--quick]"""
import sys, os, numpy as np, pandas as pd
from gplearn.genetic import SymbolicRegressor as GPL
here = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(here, "exp030_closure.py")).read()
exec(src.split("# ---------------------------------------------------------------- local fits")[0])
sim_src = src.split("# ---------------------------------------------------------------- 24-h simulation (local models)")[1].split("sims = []")[0]

def ext_point(X, b, mu, sd):
    """Quadratic library of exp021 plus cubes, log/sqrt of positive inputs, reciprocal and ratio."""
    Lq, nq = lib_point(X, b, mu, sd)
    T, SW, WS, VPD = (X[:, k] for k in range(4))
    Z = (np.column_stack([T, 1 - np.exp(-b * SW), WS, VPD]) - mu) / sd
    with np.errstate(all="ignore"):
        extra = [Z[:, k] ** 3 for k in range(4)] + [np.log(WS + 0.1), np.log(VPD + 0.01), np.sqrt(np.clip(WS, 0, None)),
                 np.sqrt(np.clip(VPD, 0, None)), 1 / (WS + 0.5), VPD / (WS + 0.5)]
    ne = ["T^3", "SWs^3", "WS^3", "VPD^3", "log(WS+0.1)", "log(VPD+0.01)", "sqrt(WS)", "sqrt(VPD)", "1/(WS+0.5)", "VPD/(WS+0.5)"]
    return np.column_stack([Lq] + extra), nq + ne

def gpl(seed):
    return GPL(population_size=1000, generations=20 if QUICK else 40, function_set=("add", "sub", "mul", "div", "log", "sqrt"),
               parsimony_coefficient=1e-5, max_samples=0.9, random_state=seed, n_jobs=4, init_depth=(2, 5))

pred, models_sim, expr = {}, {}, []
for site, s in S.items():
    d, R, tr, y, b, GB, GH = s["d"], s["R"], s["tr"], s["y"], s["b"], s["GB"], s["GH"]
    pred[(site, "B")] = GB @ lstsq(GB[tr], y[tr]); GBH = np.column_stack([GB, GH]); pred[(site, "B+harm")] = GBH @ lstsq(GBH[tr], y[tr])
    models_sim[(site, "B")] = (lstsq(GB[tr], y[tr]), b, None, 0.0)
    mu, sd = zstats(d[d.year.isin(SITES[site]["train"])], b)
    Lp, names = ext_point(s["X"], b, mu, sd); Lp[~np.isfinite(Lp)] = 0
    GL = np.column_stack([weak(Lp[:, k], R) for k in range(Lp.shape[1])])
    cv = []
    for thr in THR_GRID:
        e = 0.0
        for yo in SITES[site]["train"]:
            a = tr & (s["ry"] != yo); o = tr & (s["ry"] == yo)
            cb, cl, _ = stlsq(GB[a], GL[a], y[a], thr); e += np.sum((y[o] - GB[o] @ cb - GL[o] @ cl) ** 2)
        cv.append(e)
    thr = THR_GRID[int(np.argmin(cv))]; cb, cl, keep = stlsq(GB[tr], GL[tr], y[tr], thr)
    pred[(site, "SINDy-ext")] = GB @ cb + GL @ cl
    expr.append({"site": site, "model": "SINDy-ext", "thr": thr, "n_terms": int(keep.sum()),
                 "expression": " + ".join(f"{c:.4g}*{n}" for c, n in zip(cl, names) if c != 0)})
    class _Ext:
        def __init__(s_, mu, sd, cl): s_.mu, s_.sd, s_.cl = mu, sd, cl
        def predict(s_, X):
            L, _ = ext_point(X, b, s_.mu, s_.sd); L[~np.isfinite(L)] = 0; return L @ s_.cl
    models_sim[(site, "SINDy-ext")] = (cb, b, ("op", _Ext(mu, sd, cl)), 1.0)
    m = pt_train_mask(site); Xtr, rtr = s["X"][m], s["res"][m]
    k = np.random.default_rng(0).choice(len(rtr), min(20000, len(rtr)), replace=False); Xtr, rtr = Xtr[k], rtr[k]
    fin = np.isfinite(s["X"]).all(1)
    for seed in SEEDS:
        g = gpl(seed).fit(Xtr, rtr); spv = np.zeros(len(d)); spv[fin] = g.predict(s["X"][fin]); spv[~np.isfinite(spv)] = 0
        Gs = np.column_stack([GB, weak(spv, R)]); c = lstsq(Gs[tr], y[tr]); pred[(site, f"gplearn_s{seed}")] = Gs @ c
        expr.append({"site": site, "model": f"gplearn_s{seed}", "length": g._program.length_, "expression": str(g._program)})
        models_sim[(site, f"gplearn_s{seed}")] = (c[:4], b, ("op", g), c[-1])
    print(site, "local done", flush=True)

def partial_out(GB, A): return A - GB @ np.linalg.lstsq(GB, A, rcond=None)[0]
E0 = pd.DataFrame(expr)
for target in S:
    srcs = [k for k in S if k != target]; t = S[target]; tr, y, GB = t["tr"], t["y"], t["GB"]
    dpool = pd.concat([S[k]["d"][S[k]["d"].year.isin(SITES[k]["train"])] for k in srcs]); bpool = float(np.median([S[k]["b"] for k in srcs]))
    mu, sd = zstats(dpool, bpool); GLs, ys = [], []
    for k in srcs:
        sk = S[k]; Lp, names = ext_point(sk["X"], bpool, mu, sd); Lp[~np.isfinite(Lp)] = 0
        GL = np.column_stack([weak(Lp[:, j], sk["R"]) for j in range(Lp.shape[1])])[sk["tr"]]
        GBk = sk["GB"][sk["tr"]]; GLs.append(partial_out(GBk, GL)); ys.append(partial_out(GBk, sk["y"][sk["tr"]][:, None])[:, 0])
    thr = float(E0.query("model=='SINDy-ext' and site in @srcs").thr.median())
    _, cl, keep = stlsq(np.zeros((sum(len(a) for a in ys), 0)), np.vstack(GLs), np.concatenate(ys), thr)
    Lp, _ = ext_point(t["X"], bpool, mu, sd); Lp[~np.isfinite(Lp)] = 0
    GLt = np.column_stack([weak(Lp[:, j], t["R"]) for j in range(Lp.shape[1])])
    cb = lstsq(GB[tr], y[tr] - GLt[tr] @ cl); pred[(target, "SINDy-ext LOSO frozen")] = GB @ cb + GLt @ cl
    Gr = np.column_stack([GB, GLt[:, keep]]); c = lstsq(Gr[tr], y[tr]); pred[(target, "SINDy-ext LOSO refit")] = Gr @ c
    expr.append({"site": target, "model": "SINDy-ext LOSO", "n_terms": int(keep.sum()), "thr": thr})
    Xp = np.vstack([S[k]["X"][pt_train_mask(k)] for k in srcs]); rp = np.concatenate([S[k]["res"][pt_train_mask(k)] for k in srcs])
    kk = np.random.default_rng(0).choice(len(rp), min(40000, len(rp)), replace=False); fin = np.isfinite(t["X"]).all(1)
    for seed in SEEDS:
        g = gpl(seed).fit(Xp[kk], rp[kk]); spv = np.zeros(len(t["d"])); spv[fin] = g.predict(t["X"][fin]); spv[~np.isfinite(spv)] = 0
        ws = weak(spv, t["R"]); cb = lstsq(GB[tr], y[tr] - ws[tr]); pred[(target, f"gplearn_s{seed} LOSO frozen")] = GB @ cb + ws
        Gs = np.column_stack([GB, ws]); c = lstsq(Gs[tr], y[tr]); pred[(target, f"gplearn_s{seed} LOSO refit")] = Gs @ c
        expr.append({"site": target, "model": f"gplearn_s{seed} LOSO", "length": g._program.length_, "expression": str(g._program)})
    print(target, "LOSO done", flush=True)

rows = []
for site, s in S.items():
    te, y, rd = s["te"], s["y"], s["rd"]; rng = np.random.default_rng(0)
    for (st, m), p in pred.items():
        if st != site or m in ("B", "B+harm"): continue
        row = {"site": site, "model": m, "r2_test": r2(y[te], p[te])}
        for ref in ("B", "B+harm"):
            lo, hi = ci_r2(y[te], p[te], pred[(site, ref)][te], rd[te], rng)
            row[f"dR2_vs_{ref}"] = row["r2_test"] - r2(y[te], pred[(site, ref)][te]); row[f"lo_{ref}"] = lo; row[f"hi_{ref}"] = hi
        rows.append(row)
A = pd.DataFrame(rows); A.to_csv(f"{OUT}/exp021b_apriori.csv", index=False); pd.DataFrame(expr).to_csv(f"{OUT}/exp021b_expressions.csv", index=False)
pd.set_option("display.width", 250); print(A.round(4).to_string(index=False), flush=True)
exec(sim_src)
sims = []
for (site, key) in models_sim: sims += simulate(site, key)
Sm = pd.DataFrame(sims); out = []
for site in S:
    piv = Sm[Sm.site == site].pivot_table(index=["year", "day"], columns="model", values="rmse"); rng = np.random.default_rng(1)
    for m in piv.columns:
        if m == "B": continue
        ok = piv[m].notna() & piv["B"].notna(); diff = (piv[m] - piv["B"])[ok].to_numpy()
        bs = [np.mean(rng.choice(diff, len(diff))) for _ in range(NB)]
        out.append({"site": site, "model": m, "days": int(ok.sum()), "blowups": int(piv[m].isna().sum()), "rmse": piv[m][ok].mean(),
                    "d_vs_B": diff.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
Sim = pd.DataFrame(out); Sim.to_csv(f"{OUT}/exp021b_sim.csv", index=False); print(Sim.round(3).to_string(index=False))
