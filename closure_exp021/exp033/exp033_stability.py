"""EXP-033(a): which inputs each engine's selected closure uses, across sites x modes x seeds (descriptive)."""
import re, glob, pandas as pd
R = "/home/claude/sr-paper/closure_exp021"; A = "/mnt/project-files/code/article1_closure/exp030c_out"
rows = []
def add(engine, task, seed, eq):
    eq = str(eq)
    if engine == "gplearn": eq = eq.replace("X0", "T").replace("X1", "SW").replace("X2", "WS").replace("X3", "VPD")
    has = {v: bool(re.search(rf"\b{p}\b", eq)) for v, p in [("T", "T"), ("SW", r"SWs?"), ("WS", "WS"), ("VPD", "VPD")]}
    rows.append(dict(engine=engine, task=task, mode=task.split("_")[0], seed=seed, **has, expr=eq[:200]))
p = pd.read_csv(f"{R}/pysr_closure_fronts.csv"); p = p[p.is_best.astype(str) == "True"]
for r in p.itertuples(): add("PySR", r.task, r.seed, r.equation)
k = pd.read_csv(f"{R}/kan_closure.csv")
for r in k.itertuples(): add("KAN", r.task, r.seed, r.equation)
for f, eng in [(f"{R}/exp032/ailorenz_blackbox.csv", "AI-Lorenz BB"), (f"{R}/exp032/ailorenz_graybox.csv", "AI-Lorenz GB")]:
    d = pd.read_csv(f); d = d[d.is_best.astype(str) == "True"]
    for r in d.itertuples(): add(eng, r.task, r.seed, r.equation)
for pat, eng in [("physo", "PhySO"), ("dso", "DSO")]:
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/exp032/{pat}_shard*.csv")]); d = d[d.task.str.match(r"^(local|loso)_") & (d.status == "ok")]
    for r in d.itertuples(): add(eng, r.task, r.seed, r.equation)
for f in [f"{A}/exp021_expressions.csv", f"{A}/exp021b_expressions.csv"]:
    d = pd.read_csv(f)
    for r in d.itertuples():
        m = r.model; mode = "loso" if "LOSO" in m else "local"; base = m.replace(" LOSO", "")
        eng = base.split("_s")[0]; seed = int(base.split("_s")[1]) if "_s" in base else 0
        add(eng, f"{mode}_{r.site}", seed, r.expression)
D = pd.DataFrame(rows); D["WS_only"] = D.WS & ~D[["T", "SW", "VPD"]].any(axis=1)
D.to_csv("/mnt/project-files/code/article1_closure/exp033/exp033_stability_fits.csv", index=False)
S = D.groupby("engine")[["T", "SW", "WS", "VPD", "WS_only"]].mean().round(2); S.insert(0, "n_fits", D.groupby("engine").size())
S.to_csv("/mnt/project-files/code/article1_closure/exp033/exp033_stability.csv"); print(S.to_string())
