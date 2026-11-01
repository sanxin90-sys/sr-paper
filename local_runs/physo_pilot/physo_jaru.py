"""PhySO on the Rebio Jaru targets of real_discovery.py (train 2006-07, 3000 pts; test 2008).
Temperatures converted to K in both modes. Units vector: [m, s, kg, K, mol]."""
import csv, os, sys, time
import numpy as np, torch, physo
from methods import _sympy_predict, complexity

D = [0, 0, 0, 0, 0]
CFG = {
    "net_radiation": dict(names=["PAR"], X=[[-2, -1, 0, 0, 1]], y=[0, -3, 1, 0, 0],
                          free=[[2, -2, 1, 0, -1], [0, -3, 1, 0, 0], D, D], kelvin=[]),
    "relative_humidity": dict(names=["T22", "Tmin_day"], X=[[0, 0, 0, 1, 0]] * 2, y=D,
                              free=[[0, 0, 0, -1, 0], [0, 0, 0, 1, 0], D, D], kelvin=[0, 1]),
    "temperature_tendency": dict(names=["Rn", "T22", "WS"], X=[[0, -3, 1, 0, 0], [0, 0, 0, 1, 0], [1, -1, 0, 0, 0]],
                                 y=[0, -1, 0, 1, 0],
                                 free=[[0, 2, -1, 1, 0], [0, -1, 0, 0, 0], [0, 0, 0, 1, 0], [1, -1, 0, 0, 0]], kelvin=[1]),
}
target, seed, mode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
c = CFG[target]
d = np.load(os.environ.get("JARU_NPZ", "jaru.npz"))  # arrays prepared from the real_discovery.py split (not included)
X, y, Xt, yt = (d[f"{target}_{k}"].astype(float).copy() for k in ("Xtr", "ytr", "Xte", "yte"))
for j in c["kelvin"]:
    X[:, j] += 273.15; Xt[:, j] += 273.15
torch.manual_seed(seed); np.random.seed(seed)
kw = dict(X_names=c["names"], y_name="y", fixed_consts=[1.0], free_consts_names=[f"c{i}" for i in range(len(c["free"]))],
          epochs=5, max_n_evaluations=None, stop_after_n_epochs=5, use_protected_ops=False,
          parallel_mode=False, device="cpu")
if mode == "units":
    kw.update(X_units=c["X"], y_units=c["y"], fixed_consts_units=[D], free_consts_units=c["free"])
t0 = time.perf_counter()
try:
    prog, _ = physo.SR(X.T.copy(), y.copy(), **kw)
    sym = prog.get_infix_sympy(evaluate_consts=True)
    if isinstance(sym, (list, tuple, np.ndarray)):
        sym = np.ravel(sym)[0]
    expr = str(sym)
    p = _sympy_predict(expr, c["names"])(Xt)
    ok = np.isfinite(p)
    rmse = float(np.sqrt(np.mean((yt - p) ** 2))) if ok.all() else float("nan")
    r2 = 1 - rmse ** 2 / np.var(yt)
    status = "ok"
except Exception as e:
    expr, rmse, r2, status = "", float("nan"), float("nan"), f"error: {type(e).__name__}: {str(e)[:150]}"
row = dict(target=target, seed=seed, mode=mode, status=status, rmse_2008=rmse, r2_2008=r2,
           complexity=complexity(expr) if expr else -1, seconds=round(time.perf_counter() - t0, 1), expression=expr)
with open(out, "a", newline="") as f:
    csv.DictWriter(f, fieldnames=list(row)).writerow(row)
print(row)
