"""PhySO pilot: same 5-epoch budget as the benchmark, with vs without unit constraints.
Units vector: [m, s, kg, K]."""
import csv, sys, time
import numpy as np, torch, physo
from tasks import STATIC_TASKS, sample_static, add_noise
from methods import _sympy_predict, complexity

D = [0, 0, 0, 0]
UNITS = {
    "stefan_boltzmann": dict(X={"eps": D, "Tk": [0, 0, 0, 1]}, y=[0, -3, 1, 0],
                             free=[[0, -3, 1, -4], D, D]),
    "monin_obukhov_length": dict(X={"ustar": [1, -1, 0, 0], "theta": [0, 0, 0, 1], "wth": [1, -1, 0, 1]},
                                 y=[1, 0, 0, 0], free=[[-1, 2, 0, 0], D, D]),
    "log_wind_profile": dict(X={"ustar": [1, -1, 0, 0], "z": [1, 0, 0, 0]}, y=[1, -1, 0, 0],
                             free=[[1, 0, 0, 0], [1, 0, 0, 0], D]),
    "bulk_sensible_heat": dict(X={"u": [1, -1, 0, 0], "dT": [0, 0, 0, 1], "z": [1, 0, 0, 0]}, y=[0, -3, 1, 0],
                               free=[[-1, -2, 1, -1], [1, 0, 0, 0], D]),
}

def r2(yt, p):
    p = np.asarray(p, float)
    return float("nan") if not np.all(np.isfinite(p)) else float(1 - np.mean((yt - p) ** 2) / np.var(yt))

task_name, noise, seed, mode, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
task = next(t for t in STATIC_TASKS if t.name == task_name)
rng = np.random.default_rng(0)                      # same data as the benchmark (data seed 0)
X, yc = sample_static(task, 1000, rng)
Xt, yt = sample_static(task, 2000, rng, test=True)
y = add_noise(yc, noise, np.random.default_rng(1))
torch.manual_seed(seed); np.random.seed(seed)
kw = dict(X_names=task.variables, y_name="y", fixed_consts=[1.0], free_consts_names=["c0", "c1", "c2"],
          epochs=5, max_n_evaluations=None, stop_after_n_epochs=5, use_protected_ops=False,
          parallel_mode=False, device="cpu")
if mode == "units":
    u = UNITS[task_name]
    kw.update(X_units=[u["X"][v] for v in task.variables], y_units=u["y"],
              fixed_consts_units=[D], free_consts_units=u["free"])
t0 = time.perf_counter()
try:
    prog, _ = physo.SR(X.T.copy(), y.copy(), **kw)
    sym = prog.get_infix_sympy(evaluate_consts=True)
    if isinstance(sym, (list, tuple, np.ndarray)):
        sym = np.ravel(sym)[0]
    expr = str(sym); score = r2(yt, _sympy_predict(expr, task.variables)(Xt)); status = "ok"
except Exception as e:
    expr, score, status = "", float("nan"), f"error: {type(e).__name__}: {str(e)[:150]}"
row = dict(task=task_name, noise=noise, seed=seed, mode=mode, status=status, r2_extrap=score,
           complexity=complexity(expr) if expr else -1, seconds=round(time.perf_counter() - t0, 1), expression=expr)
with open(out, "a", newline="") as f:
    csv.DictWriter(f, fieldnames=list(row)).writerow(row)
print(row)
