"""EXP-021b, KAN part: fit the closure s(T, SW, WS, VPD) with the corrected KAN protocol of
Article 1 (v2: standardise, 80/20 split, prune, fixed library, affine refit after snapping) on the
same closure_inputs.npz as the PySR run. 10 tasks x 3 seeds. Writes kan_closure.csv (one symbolic
expression per fit). Restartable. Usage: python run_kan_closure.py [--quick]"""
import sys, os, time, tempfile, numpy as np, pandas as pd, sympy as sp, torch
from kan import KAN

QUICK = "--quick" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "closure_inputs.npz"))
OUT = os.path.join(HERE, "kan_closure_quick.csv" if QUICK else "kan_closure.csv")
CKPT = tempfile.mkdtemp(prefix="kan_ckpt_")
LIB = ["x", "x^2", "x^3", "exp", "log", "sqrt", "1/x", "tanh", "sin"]
NAMES = ["T", "SW", "WS", "VPD"]
T_ = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float32)

def fit_v2(X, y, seed, hidden=3, steps=100, lamb=0.003):
    rng = np.random.default_rng(seed)
    mx, sx = X.mean(0), X.std(0) + 1e-12; my, sy = y.mean(), y.std() + 1e-12
    Z, u = (X - mx) / sx, (y - my) / sy
    idx = rng.permutation(len(Z)); nv = max(len(Z) // 5, 1); va, tr = idx[:nv], idx[nv:]
    data = {"train_input": T_(Z[tr]), "train_label": T_(u[tr, None]), "test_input": T_(Z[va]), "test_label": T_(u[va, None])}
    torch.manual_seed(seed)
    m = KAN(width=[X.shape[1], hidden, 1], grid=5, k=3, seed=seed, device="cpu", auto_save=False, ckpt_path=CKPT)
    m.fit(data, opt="LBFGS", steps=steps, lamb=lamb, lamb_entropy=2.0, log=10**6)
    try:
        m = m.prune(node_th=1e-2, edge_th=3e-2); m.fit(data, opt="LBFGS", steps=steps // 2, lamb=0.0, log=10**6)
    except Exception:
        pass
    m.auto_symbolic(lib=LIB, verbose=0, weight_simple=0.8)
    m.fit(data, opt="LBFGS", steps=steps // 2, lamb=0.0, update_grid=False, log=10**6)
    zs = sp.symbols([f"z{i}" for i in range(X.shape[1])])
    f = m.symbolic_formula(var=list(zs))[0][0]
    xs = sp.symbols(NAMES)
    f = f.subs({z: (x - float(a)) / float(b) for z, x, a, b in zip(zs, xs, mx, sx)})
    return str(sp.N(sp.expand(float(sy) * f + float(my)), 6))

tasks = sorted({k.rsplit("_", 1)[0] for k in D.files}); seeds = [0, 1, 2]
if QUICK: tasks, seeds = tasks[:1], [0]
done = set()
if os.path.exists(OUT):
    o = pd.read_csv(OUT); done = set(zip(o.task, o.seed))
for task in tasks:
    X, y = D[task + "_X"], D[task + "_y"]
    for seed in seeds:
        if (task, seed) in done: continue
        t0 = time.time()
        try:
            eq, st = fit_v2(X, y, seed), "ok"
        except Exception as e:
            eq, st = "", f"fail: {type(e).__name__}: {e}"[:200]
        pd.DataFrame([{"task": task, "seed": seed, "equation": eq, "status": st, "seconds": round(time.time() - t0, 1)}]).to_csv(
            OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(task, seed, st, eq[:300], flush=True)
print("done ->", OUT)
