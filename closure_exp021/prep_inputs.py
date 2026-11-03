"""Build the inputs for the local PySR closure runs (EXP-021): pointwise residual of the base
model B for each site, train years only, subsampled. Local tasks (one per site) and LOSO tasks
(pooled from the other four sites). Usage: python prep_inputs.py RAW_DIR OUT.npz"""
import sys, numpy as np
RAW, OUTF = sys.argv[1], sys.argv[2]
src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "exp021.py")).read().split("# ---------------------------------------------------------------- local fits")[0]
sys.argv = ["exp021.py", RAW, "."]
exec(src)
rng = np.random.default_rng(0); out = {}
for site in S:
    m = pt_train_mask(site); X, r = S[site]["X"][m], S[site]["res"][m]
    k = rng.choice(len(r), min(5000, len(r)), replace=False)
    out[f"local_{site}_X"], out[f"local_{site}_y"] = X[k], r[k]
for t in S:
    Xs, rs = [], []
    for s_ in S:
        if s_ == t: continue
        m = pt_train_mask(s_); X, r = S[s_]["X"][m], S[s_]["res"][m]
        k = rng.choice(len(r), min(2500, len(r)), replace=False); Xs.append(X[k]); rs.append(r[k])
    out[f"loso_{t}_X"], out[f"loso_{t}_y"] = np.vstack(Xs), np.concatenate(rs)
np.savez_compressed(OUTF, **out)
print({k: v.shape for k, v in out.items() if k.endswith("_y")})
