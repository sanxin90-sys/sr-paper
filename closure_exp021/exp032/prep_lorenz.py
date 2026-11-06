"""EXP-032, AI-Lorenz closure: X-TFC (Article 1 config: 48 samples, 12 neurons, runs >= 96 half-hours) estimates dT/dt on
the training years of each tower; target f = dT/dt_XTFC - B. Black box: data only, then B subtracted. Gray box: B imposed
(a=1, forcing = c2*sat(SW) + c3*dSW/dt, lam = -c1, c = c0), physics weight by common.discrepancy_choice over {1, 0.1, 0.03}.
Same tasks and sampling sizes as prep_inputs.py. Usage: python prep_lorenz.py RAW_DIR OUT.npz"""
import sys, os, numpy as np
RAW, OUTF = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "src", "ailorenz"))
from common import discrepancy_choice
from gbxtfc import scalar_relaxation
src = open(os.path.join(ROOT, "exp021.py")).read().split("# ---------------------------------------------------------------- local fits")[0]
sys.argv = ["exp021.py", RAW, "."]
exec(src)
XTFC, MIN_RUN, W_GRID = dict(s=48, m=12), 96, [1.0, 0.1, 0.03]

def runs(d):
    ok = d[XV].notna().all(axis=1).to_numpy(); yr = d.year.to_numpy(); idx = np.flatnonzero(ok)
    cut = np.flatnonzero((np.diff(idx) > 1) | (np.diff(yr[idx]) != 0)) + 1
    return [r for r in np.split(idx, cut) if len(r) >= MIN_RUN]

def route(site):
    s = S[site]; d, b, c = s["d"], s["b"], s["cB"]; T, SW = d["T"].to_numpy(float), d["SW"].to_numpy(float)
    rr = [r for r in runs(d) if d.year.iat[r[0]] in SITES[site]["train"]]
    bb, gb = np.full(len(d), np.nan), {w: np.full(len(d), np.nan) for w in W_GRID}; mb, mg = [], {w: [] for w in W_GRID}
    for r in rr:
        t = np.arange(len(r)) * DT; rdot = np.gradient(SW[r], DT); Fo = c[2] * (1 - np.exp(-b * SW[r])) + c[3] * rdot
        Th, dT, _ = scalar_relaxation(t, T[r], np.zeros(len(r)), 0, 0, 0, w_phys=0.0, **XTFC)
        f = dT - (c[0] + c[1] * T[r] + Fo); f[:3] = f[-3:] = np.nan; bb[r] = f; mb.append((Th - T[r])[3:-3])
        for w in W_GRID:
            Th, _, f = scalar_relaxation(t, T[r], Fo, 1.0, -c[1], c[0], w_phys=w, **XTFC)
            f = f.copy(); f[:3] = f[-3:] = np.nan; gb[w][r] = f; mg[w].append((Th - T[r])[3:-3])
    rms = lambda a: float(np.sqrt(np.mean(np.concatenate(a) ** 2)))
    ws = discrepancy_choice({w: rms(mg[w]) for w in W_GRID}, rms(mb))
    print(site, "runs", len(rr), "misfit bb %.4f" % rms(mb), {w: round(rms(mg[w]), 4) for w in W_GRID}, "w*", ws, flush=True)
    return bb, gb[ws], ws

F = {site: route(site) for site in S}
out = {f"w_{site}": np.array(F[site][2]) for site in S}
for name, k_ in (("blackbox", 0), ("graybox", 1)):
    rng = np.random.default_rng(0); get = lambda st: (lambda m: (S[st]["X"][m], F[st][k_][m]))(np.isfinite(F[st][k_]) & np.isfinite(S[st]["X"]).all(1))
    for site in S:
        X, r = get(site); k = rng.choice(len(r), min(5000, len(r)), replace=False)
        out[f"{name}__local_{site}__X"], out[f"{name}__local_{site}__y"] = X[k], r[k]
    for t in S:
        Xs, rs = [], []
        for st in S:
            if st == t: continue
            X, r = get(st); k = rng.choice(len(r), min(2500, len(r)), replace=False); Xs.append(X[k]); rs.append(r[k])
        out[f"{name}__loso_{t}__X"], out[f"{name}__loso_{t}__y"] = np.vstack(Xs), np.concatenate(rs)
np.savez_compressed(OUTF, **out)
print({k: v.shape for k, v in out.items() if k.endswith("__y")})
