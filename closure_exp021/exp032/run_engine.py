"""EXP-032: DSO, PhySO and AI Feynman on the tower tasks, through the unchanged wrappers of
src/sr_benchmark/methods.py. Closure: closure_inputs.npz, 10 tasks x seeds 0-2. Rn/RH: exp022_pysr_inputs.npz,
fitted on the non-'val' rows as in run_pysr_exp022.py, with the task's own seed (AI Feynman: subsample seed 0, one seed).
One CSV per engine and shard, one row per fit, restartable (finished fits are skipped).
Usage (from an empty working folder): python run_engine.py ENGINE OUT_DIR SHARD NSHARDS [--quick]"""
import sys, os, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "src", "sr_benchmark"))
ENGINE, OUT, SHARD, NSHARDS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
QUICK = "--quick" in sys.argv
if os.name == "nt":  # POSIX calls used on timeout by fit_aifeynman; same effect on Windows
    import signal, subprocess
    if not hasattr(signal, "SIGKILL"): signal.SIGKILL = 9
    if not hasattr(os, "killpg"): os.killpg = lambda pid, sig: subprocess.call(["taskkill", "/T", "/F", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
from methods import fit_physo, fit_dso, fit_aifeynman
FIT, BUDGET = {"physo": (fit_physo, 250), "dso": (fit_dso, 120), "ai_feynman": (fit_aifeynman, 1800)}[ENGINE]
one_seed = ENGINE == "ai_feynman"
C = np.load(os.path.join(ROOT, "closure_inputs.npz")); E = np.load(os.path.join(ROOT, "exp022_pysr_inputs.npz"))
fits = []
for t in sorted({k.rsplit("_", 1)[0] for k in C.files}):
    for s in ([0] if one_seed else [0, 1, 2]): fits.append((t, s, C[t + "_X"], C[t + "_y"], ["T", "SW", "WS", "VPD"]))
for t in sorted({k.rsplit("__", 1)[0] for k in E.files}):
    target, site, mode, s = t.split("__"); s = int(s)
    if one_seed and s != 0: continue
    v = E[t + "__val"]
    fits.append((t, s, E[t + "__X"][~v], E[t + "__y"][~v], ["SW"] if target == "Rn" else ["T", "Tmin"]))
if QUICK: fits, BUDGET = fits[:1], 100 if ENGINE == "physo" else 20 if ENGINE == "dso" else 300
print(len(fits), "fits in all", flush=True)
path = os.path.join(OUT, f"{ENGINE}{'_quick' if QUICK else ''}_shard{SHARD}.csv")
done = set()
if os.path.exists(path):
    o = pd.read_csv(path); done = set(zip(o.task, o.seed))
for i, (t, s, X, y, names) in enumerate(fits):
    if i % NSHARDS != SHARD or (t, s) in done: continue
    t0 = time.time()
    try:
        eq, st = FIT(np.asarray(X, float), np.asarray(y, float), names, time_budget_s=BUDGET, seed=s).expression, "ok"
    except Exception as e:
        eq, st = "", f"fail: {type(e).__name__}: {e}"[:300]
    pd.DataFrame([{"task": t, "seed": s, "equation": eq, "is_best": st == "ok", "status": st, "seconds": round(time.time() - t0, 1)}]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(t, s, st, eq[:200], f"({time.time() - t0:.0f} s)", flush=True)
print("done ->", path)
