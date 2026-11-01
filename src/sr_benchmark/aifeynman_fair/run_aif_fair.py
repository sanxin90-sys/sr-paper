"""Fair AI Feynman re-run: full pipeline (long cap), NN_epochs configurable, seeded.
Data generation identical to run_benchmark.static_suite (data seed 0, 1000 train, 2000 extrapolation)."""
import argparse, csv, os, re, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, sympy as sp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/sr_benchmark
from tasks import STATIC_TASKS, sample_static
from methods import _sympy_predict, complexity
from run_benchmark import r2, nrmse

AIF = Path(os.environ.get("AIFEYNMAN_HOME", Path(__file__).resolve().parent.parent / "_aifeynman"))
ap = argparse.ArgumentParser()
ap.add_argument("--task"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=4000); ap.add_argument("--bf", type=int, default=60)
ap.add_argument("--cap", type=int, default=7200); ap.add_argument("--out", default="fair.csv")
a = ap.parse_args()
task = next(t for t in STATIC_TASKS if t.name == a.task)
rng = np.random.default_rng(0)
X, y = sample_static(task, 1000, rng); Xt, yt = sample_static(task, 2000, rng, test=True)
work = Path(tempfile.mkdtemp(prefix=f"aif_{a.task}_s{a.seed}_", dir=os.environ.get("AIF_WORKDIR") or None))
(work / "data").mkdir(); np.savetxt(work / "data" / "task.txt", np.column_stack([X, y]))
code = (f"import sys, numpy as np, torch, random; sys.path.insert(0, {str(AIF)!r})\n"
        f"np.random.seed({a.seed}); torch.manual_seed({a.seed}); random.seed({a.seed})\n"
        "from aifeynman import run_aifeynman\n"
        f"run_aifeynman('data/', 'task.txt', {a.bf}, '14ops.txt', polyfit_deg=3, NN_epochs={a.epochs})\n")
t0 = time.perf_counter()
with (work / "log.txt").open("w") as fh:
    p = subprocess.Popen([sys.executable, "-c", code], cwd=work, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True, env={**os.environ, "OMP_NUM_THREADS": "1",
                         "PATH": f"{AIF/'bin'}{os.pathsep}{os.environ['PATH']}"})
    try: p.wait(timeout=a.cap); timed_out = False
    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait(); timed_out = True
secs = time.perf_counter() - t0
sol = work / "results" / "solution_task.txt"
finished = sol.exists()
if finished:
    exprs = [" ".join(l.split()[5:]) for l in sol.read_text().splitlines() if len(l.split()) >= 6]
else:
    exprs, block = [], []
    for l in (work / "log.txt").read_text(errors="ignore").splitlines():
        if l.startswith("Complexity #"): block = []
        elif re.match(r"^\s*[\d.]+\s+[\d.]+\s+\S", l): block.append(" ".join(l.split()[2:]))
        if block: exprs = block[:]
names = task.variables; av = [f"x{i}" for i in range(len(names))]
rows = []
for e in exprs:
    try:
        en = str(sp.sympify(e).subs({sp.Symbol(x): sp.Symbol(v) for x, v in zip(av, names)}))
        f = _sympy_predict(en, names); tr = float(np.mean((f(X) - y) ** 2))
        if np.isfinite(tr): rows.append((tr, en, r2(yt, f(Xt)), nrmse(yt, f(Xt)), complexity(en)))
    except Exception: pass
best = min(rows) if rows else (np.nan, "", np.nan, np.nan, -1)
# also best-by-extrapolation on the whole Pareto front (oracle, diagnostic only)
orc = max(rows, key=lambda r: r[2] if np.isfinite(r[2]) else -9) if rows else best
new = not Path(a.out).exists()
with open(a.out, "a", newline="") as fh:
    w = csv.writer(fh)
    if new: w.writerow(["task","seed","epochs","bf","cap","finished","timed_out","seconds","n_pareto",
                        "r2_extrap","nrmse_extrap","complexity","expression","oracle_r2","oracle_expression"])
    w.writerow([a.task,a.seed,a.epochs,a.bf,a.cap,finished,timed_out,round(secs,1),len(rows),
                best[2],best[3],best[4],best[1],orc[2],orc[1]])
print(a.task, a.seed, finished, round(secs), best[2], best[1])
