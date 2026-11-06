"""EXP-033(b): pooled wind closure s = -k*WS added to B, k from the other forests (LOSO-pooled) or all 5
(all-pooled), applied frozen with B refit at the target. Same weak-form a priori and 24-h simulation as EXP-030.
Usage: VPD_SIM=recalc python exp033_pooled_wind.py RAW OUT"""
import sys, os, numpy as np, pandas as pd
RAW, OUT = sys.argv[1:3]
A1 = "/mnt/project-files/code/article1_closure"
src = open(os.path.join(A1, "exp030_closure.py")).read()
sys.argv = ["exp030_closure.py", RAW, OUT]
exec(src.split("# ---------------------------------------------------------------- local fits")[0])
sim_src = src.split("# ---------------------------------------------------------------- 24-h simulation (local models)")[1].split("sims = []")[0]

class WindOnly:
    def predict(self, X): return -X[:, 2]
for site, s in S.items():
    ws = np.nan_to_num(s["X"][:, 2]); s["wWS"] = weak(ws, s["R"])

def pooled_k(sites):
    blocks, cols, ys = [], [], []
    n = len(sites)
    for i, st in enumerate(sites):
        s = S[st]; tr = s["tr"]; G = s["GB"][tr]
        Z = np.zeros((tr.sum(), 4 * n)); Z[:, 4 * i:4 * i + 4] = G
        blocks.append(np.column_stack([Z, -s["wWS"][tr]])); ys.append(s["y"][tr])
    c = lstsq(np.vstack(blocks), np.concatenate(ys)); return c[-1]

def pooled_k_pt(sites):  # exploratory: pointwise residual res = a - k*WS (the PySR target)
    xs, ys = [], []
    for st in sites:
        m = pt_train_mask(st); xs.append(S[st]["X"][m, 2]); ys.append(S[st]["res"][m])
    x, yy = np.concatenate(xs), np.concatenate(ys); c = lstsq(np.column_stack([np.ones_like(x), -x]), yy); return c[1]
K = {"all5": pooled_k(list(S)), "pt_all5": pooled_k_pt(list(S))}
for site in S: K[f"pt_loso_{site}"] = pooled_k_pt([x for x in S if x != site])
for site in S: K[f"loso_{site}"] = pooled_k([x for x in S if x != site])
for site in S:  # local slope (descriptive)
    s = S[site]; tr = s["tr"]; c = lstsq(np.column_stack([s["GB"], -s["wWS"]])[tr], s["y"][tr]); K[f"local_{site}"] = c[-1]
pd.Series(K).to_csv(f"{OUT}/exp033_k.csv"); print(pd.Series(K).round(4), flush=True)

pred, models_sim, rows = {}, {}, []
for site, s in S.items():
    tr, y, GB = s["tr"], s["y"], s["GB"]
    cB = lstsq(GB[tr], y[tr]); pred[(site, "B")] = GB @ cB; models_sim[(site, "B")] = (cB, s["b"], None, 0.0)
    for name, k in [("wind LOSO-pooled", K[f"loso_{site}"]), ("wind all-pooled", K["all5"]), ("wind local", K[f"local_{site}"]), ("wind LOSO-pooled pointwise", K[f"pt_loso_{site}"])]:
        cb = lstsq(GB[tr], y[tr] + k * s["wWS"][tr]); pred[(site, name)] = GB @ cb - k * s["wWS"]
        models_sim[(site, name)] = (cb, s["b"], ("op", WindOnly()), k)
for site, s in S.items():
    te, y, rd = s["te"], s["y"], s["rd"]; rng = np.random.default_rng(0)
    for (st, m), p in pred.items():
        if st != site or m == "B": continue
        lo, hi = ci_r2(y[te], p[te], pred[(site, "B")][te], rd[te], rng)
        rows.append({"site": site, "model": m, "r2_test": r2(y[te], p[te]), "dR2_vs_B": r2(y[te], p[te]) - r2(y[te], pred[(site, "B")][te]), "lo": lo, "hi": hi})
Ap = pd.DataFrame(rows); Ap.to_csv(f"{OUT}/exp033_apriori.csv", index=False); print(Ap.round(4).to_string(index=False), flush=True)

exec(sim_src)
sims = []
for key in models_sim: sims += simulate(*key)
Sm = pd.DataFrame(sims); out = []
for site in S:
    piv = Sm[Sm.site == site].pivot_table(index=["year", "day"], columns="model", values="rmse"); rng = np.random.default_rng(1)
    for m in piv.columns:
        if m == "B": continue
        ok = piv[m].notna() & piv["B"].notna(); diff = (piv[m] - piv["B"])[ok].to_numpy()
        bs = [np.mean(rng.choice(diff, len(diff))) for _ in range(NB)]
        out.append({"site": site, "model": m, "days": int(ok.sum()), "blowups": int(piv[m].isna().sum()), "rmse": piv[m][ok].mean(),
                    "d_vs_B": diff.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
Sim = pd.DataFrame(out); Sim.to_csv(f"{OUT}/exp033_sim.csv", index=False); print(Sim.round(3).to_string(index=False))
