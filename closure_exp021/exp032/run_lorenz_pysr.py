"""EXP-032, AI-Lorenz closure: PySR with the settings of run_pysr_closure.py on the targets of prep_lorenz.py.
One CSV per route (whole front, best flagged). Restartable. Usage: python run_lorenz_pysr.py INPUTS.npz OUT_DIR [--quick]"""
import sys, os, time, numpy as np, pandas as pd
from pysr import PySRRegressor
QUICK = "--quick" in sys.argv
D = np.load(sys.argv[1]); OUTD = sys.argv[2]
keys = sorted({k.rsplit("__", 1)[0] for k in D.files if "__" in k}); seeds = [0, 1, 2]
if QUICK: keys, seeds = keys[:1], [0]
for key in keys:
    rt, task = key.split("__"); X, y = D[key + "__X"], D[key + "__y"]
    OUT = os.path.join(OUTD, f"ailorenz_{rt}{'_quick' if QUICK else ''}.csv")
    done = set(zip(*[pd.read_csv(OUT)[c] for c in ("task", "seed")])) if os.path.exists(OUT) else set()
    for seed in seeds:
        if (task, seed) in done: continue
        t0 = time.time()
        m = PySRRegressor(binary_operators=["+", "-", "*", "/"], unary_operators=["exp"], maxsize=25, niterations=1000,
                          timeout_in_seconds=20 if QUICK else 120, populations=15, model_selection="best", random_state=seed,
                          progress=False, verbosity=0, temp_equation_file=True)
        m.fit(X, y, variable_names=["T", "SW", "WS", "VPD"])
        eq = m.equations_.copy(); best = m.get_best()
        pd.DataFrame({"task": task, "seed": seed, "complexity": eq.complexity, "loss": eq.loss, "equation": eq.sympy_format.astype(str),
                      "is_best": eq.equation == best.equation, "seconds": round(time.time() - t0, 1)}).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(rt, task, seed, "best:", best.sympy_format, f"({time.time() - t0:.0f} s)", flush=True)
print("done ->", OUTD)
