"""EXP-022, PySR part, with the settings of pysr_clean_rn_rh.py (Article 1): 120 s, operators + - * / exp log
sqrt square, maxsize 25, deterministic serial, model chosen by RMSE on the 'val' rows (last 20 % of training
days), never on test data. Writes pysr_exp022_fronts.csv (whole front, chosen row flagged). Restartable.
Usage: python run_pysr_exp022.py [--quick]"""
import sys, os, numpy as np, pandas as pd
from pysr import PySRRegressor
QUICK = "--quick" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "exp022_pysr_inputs.npz"))
OUT = os.path.join(HERE, "pysr_exp022_quick.csv" if QUICK else "pysr_exp022_fronts.csv")
tasks = sorted({k.rsplit("__", 1)[0] for k in D.files})
if QUICK: tasks = tasks[:1]
done = set(pd.read_csv(OUT).task) if os.path.exists(OUT) else set()
for t in tasks:
    if t in done: continue
    target, site, mode, seed = t.split("__"); X, y, v = D[t + "__X"], D[t + "__y"], D[t + "__val"]
    names = ["SW"] if target == "Rn" else ["T", "Tmin"]
    reg = PySRRegressor(niterations=100000, populations=30, maxsize=25, binary_operators=["+", "-", "*", "/"],
                        unary_operators=["exp", "log", "sqrt", "square"], timeout_in_seconds=20 if QUICK else 120,
                        model_selection="accuracy", random_state=int(seed), deterministic=True, parallelism="serial",
                        progress=False, verbosity=0, temp_equation_file=True)
    reg.fit(X[~v], y[~v], variable_names=names)
    eq = reg.equations_[["complexity", "loss", "sympy_format"]].copy()
    with np.errstate(all="ignore"):
        eq["rmse_val"] = [float(np.sqrt(np.mean((y[v] - reg.predict(X[v], index=i)) ** 2))) for i in range(len(eq))]
    eq["rmse_val"] = eq.rmse_val.fillna(np.inf); k = int(eq.rmse_val.idxmin())
    eq = eq.assign(task=t, chosen=[i == k for i in range(len(eq))], equation=eq.sympy_format.astype(str)).drop(columns="sympy_format")
    eq.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(t, "chosen:", eq.equation.iat[k], flush=True)
print("done ->", OUT)
