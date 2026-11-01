"""Consolidate the synthetic-benchmark result files into the counts reported in Article 1 (Tables 2-3).

Criteria are those of summarize.py: accuracy solution R2_extrap >= 0.999; parsimonious = accuracy
solution with node count <= 2 x the true law. Counts are per (method, noise, seed), then summarised
over seeds as mean and range.

Usage: python consolidate_benchmark.py RESULTS_DIR [SEEDS_DIR] > tables.md
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "sr_benchmark"))
sys.path.insert(0, ".")  # fallback: run from the sr_benchmark folder
from summarize import load  # noqa: E402

R = Path(sys.argv[1])
S = Path(sys.argv[2]) if len(sys.argv) > 2 else None

STATIC = {
    "pysr": ["static_pysr_s0.csv", "static_pysr_s1.csv", "static_pysr_s2.csv"],
    "operon": ["static.csv"], "gplearn": ["static.csv"],
    "sindy": ["static_sindy.csv"], "dso": ["static_dso.csv"], "dso_const": ["static_dso.csv"],
    "physo": ["static_physo.csv"], "kan": ["static_kan.csv"],
    "aifeynman": ["static_aifeynman_a.csv", "static_aifeynman_b.csv"],
}
DYNAMIC = {
    "pysr": ["dynamic_pysr_s0.csv", "dynamic_pysr_s1.csv", "dynamic_pysr_s2.csv"],
    "operon": ["dynamic.csv"], "gplearn": ["dynamic.csv"], "sindy": ["dynamic_sindy.csv"],
    "kan": ["dynamic_kan.csv"],
}


def gather(spec, prefix):
    frames = []
    for m, files in spec.items():
        for f in files:
            p = R / f
            if p.exists():
                d = pd.read_csv(p)
                frames.append(d[d.method.str.startswith(m) if m == "sindy" else d.method == m])
    if S is not None:
        for p in sorted(S.glob(f"{prefix}_*_s*.csv")):
            if p.stat().st_size > 0:  # skip runs that have not written yet
                frames.append(pd.read_csv(p))
    raw = pd.concat(frames, ignore_index=True)
    # seed-0 rows of the add-on runs duplicate the originals: keep the first occurrence
    raw = raw.drop_duplicates(["suite", "task", "target", "noise", "estimator", "method", "seed"], keep="first")
    tmp = Path(tempfile.mkdtemp()) / f"consolidated_{prefix}.csv"
    raw.to_csv(tmp, index=False)
    return load([str(tmp)]), raw


def per_seed(df, keys):
    return df.groupby(keys + ["seed"]).agg(acc=("accuracy_solution", "sum"), pars=("parsimonious", "sum"),
                                           n=("task", "size")).reset_index()


def table(df, keys):
    per = per_seed(df, keys)
    out = per.groupby(keys).agg(seeds=("seed", "nunique"), acc_mean=("acc", "mean"), acc_min=("acc", "min"),
                                acc_max=("acc", "max"), pars_mean=("pars", "mean"), pars_min=("pars", "min"),
                                pars_max=("pars", "max"), cells=("n", "max"))
    return out.round(2).to_markdown()


st, st_raw = gather(STATIC, "static")
print("## Static suite (10 laws)\n")
print("Rows that failed (status != ok):", (st_raw.status != "ok").groupby(st_raw.method).sum().to_dict(), "\n")
print(table(st, ["method", "noise"]), "\n")
dy, dy_raw = gather(DYNAMIC, "dynamic")
print("## Dynamic suite (Jaru relaxation + Lorenz-63 = 4 targets x estimators)\n")
print(table(dy, ["method", "noise"]), "\n")
print(table(dy, ["method", "estimator"]), "\n")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
per_seed(st, ["method", "noise"]).to_csv(OUT / "static_counts_per_seed.csv", index=False)
per_seed(dy, ["method", "noise"]).to_csv(OUT / "dynamic_counts_per_seed.csv", index=False)
