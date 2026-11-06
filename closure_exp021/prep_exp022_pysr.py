"""Build the PySR inputs for EXP-022 (Rn ~ SW_IN, RH ~ T, Tmin) with exactly the training samples of
exp022.py: local = 3000 rows of the site's training years (random_state = seed); LOSO = 3000 rows of each
other site, pooled. 'val' marks the last 20 % of training days of each site (time order), used only to
pick the equation on the Pareto front (as in pysr_clean_rn_rh.py). Usage: python prep_exp022_pysr.py RAW OUT.npz"""
import sys, os, numpy as np, pandas as pd
RAW, OUTF = sys.argv[1:3]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp022.py")).read().split("ENG = {")[0]
src = src.replace("from methods import fit_operon, fit_sindy_poly, fit_sindy, fit_gplearn, complexity  # noqa: E402", "")
sys.argv = ["exp022.py", RAW, ".", "."]
exec(src)
out = {}
def mark_val(sub):
    v = np.zeros(len(sub), bool)
    for _, g in sub.groupby("site"):
        days = np.sort(g.date.unique()); cut = days[int(0.8 * len(days))]
        v[np.flatnonzero((sub.site == g.site.iat[0]).to_numpy() & (sub.date >= cut).to_numpy())] = True
    return v
for target in ("Rn", "RH"):
    feats = LAWS[target][2]; sites = [s for s in SITES if not (target == "Rn" and s == "Ji3")]; tr = {}
    for s in sites:
        d = load(s); bad = audit(d, target); d = d[~d.month.isin(bad)].dropna(subset=feats + [target])
        tr[s] = d[d.year.isin(SITES[s]["train"])].assign(site=s)
    for s in sites:
        for seed in range(5):
            for mode in ("local", "loso"):
                if mode == "local": sub = tr[s].sample(min(N_TRAIN, len(tr[s])), random_state=seed)
                else: sub = pd.concat([tr[k].sample(min(N_TRAIN, len(tr[k])), random_state=seed) for k in sites if k != s])
                key = f"{target}__{s}__{mode}__{seed}"
                out[key + "__X"] = sub[feats].to_numpy(float); out[key + "__y"] = sub[target].to_numpy(float); out[key + "__val"] = mark_val(sub)
np.savez_compressed(OUTF, **out); print(len(out) // 3, "tasks; checksum", round(float(sum(v.sum() for k, v in out.items() if k.endswith("__y"))), 3))
