"""EXP-021, PySR part: fit the closure s(T, SW, WS, VPD) of the base model B on the pointwise
residuals in closure_inputs.npz (built by prep_inputs.py). Ten tasks (5 local, 5 leave-one-site-out)
x 3 seeds, about 2 min each. Writes every Pareto front to pysr_closure_fronts.csv, one row per
equation; the run can be stopped and restarted (finished task/seed pairs are skipped).

Usage (from this folder):  python run_pysr_closure.py            # full run (~1 h)
                           python run_pysr_closure.py --quick    # 1 task, 1 seed, 20 s
"""
import sys, os, time, numpy as np, pandas as pd
from pysr import PySRRegressor

QUICK = "--quick" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "closure_inputs.npz"))
OUT = os.path.join(HERE, "pysr_closure_fronts.csv" if not QUICK else "pysr_closure_quick.csv")
tasks = sorted({k.rsplit("_", 1)[0] for k in D.files})
seeds = [0, 1, 2]
if QUICK: tasks, seeds = tasks[:1], [0]
done = set()
if os.path.exists(OUT):
    old = pd.read_csv(OUT); done = set(zip(old.task, old.seed))

for task in tasks:
    X, y = D[task + "_X"], D[task + "_y"]
    for seed in seeds:
        if (task, seed) in done: print("skip", task, seed); continue
        t0 = time.time()
        m = PySRRegressor(
            binary_operators=["+", "-", "*", "/"], unary_operators=["exp"],
            maxsize=25, niterations=1000, timeout_in_seconds=20 if QUICK else 120,
            populations=15, model_selection="best", random_state=seed,
            progress=False, verbosity=0, temp_equation_file=True,
        )
        m.fit(X, y, variable_names=["T", "SW", "WS", "VPD"])
        eq = m.equations_.copy()
        best = m.get_best()
        rows = pd.DataFrame({"task": task, "seed": seed, "complexity": eq.complexity, "loss": eq.loss,
                             "score": eq.score, "equation": eq.sympy_format.astype(str),
                             "is_best": eq.equation == best.equation, "seconds": round(time.time() - t0, 1)})
        rows.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(task, seed, "best:", best.sympy_format, f"({time.time() - t0:.0f} s)", flush=True)
print("done ->", OUT)
