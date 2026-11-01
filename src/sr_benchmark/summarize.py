"""
Aggregate benchmark CSVs into compact Markdown tables.

Scores
------
* R2_extrap      : coefficient of determination on a noise-free extrapolation set.
* accuracy sol.  : R2_extrap >= 0.999 (SRBench "accuracy solution" threshold).
* form recovered : NRMSE_extrap < 1e-3 when trained on noise-free data, i.e. the
                   expression is numerically indistinguishable from the true law
                   (up to fitted constants) even outside the training box.
* parsimonious   : accuracy solution AND complexity <= 2 x the complexity of the
                   true law (node count). High R2 with a bloated expression is a
                   curve fit, not a discovered law.

Usage: python summarize.py results/*.csv > results/summary.md
"""

from __future__ import annotations

import sys

import pandas as pd

from methods import complexity
from tasks import JARU_T, LORENZ, STATIC_TASKS

TRUE_COMPLEXITY = {(t.name, "y"): complexity(t.formula) for t in STATIC_TASKS}
TRUE_COMPLEXITY.update({(T.name, f"d{s}/dt"): complexity(f)
                        for T in (LORENZ, JARU_T) for s, f in zip(T.state_names, T.rhs_formulas)})


def load(paths: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    df["accuracy_solution"] = df["r2_extrap"] >= 0.999
    df["form_recovered"] = (df["nrmse_extrap"] < 1e-3) & (df["noise"] == 0)
    truth = df.apply(lambda r: TRUE_COMPLEXITY.get((r["task"], r["target"]), 0), axis=1)
    df["parsimonious"] = df["accuracy_solution"] & (df["complexity"] <= 2 * truth)
    return df


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt=".4f")


def main(paths: list[str]) -> None:
    df = load(paths)
    st = df[df["suite"] == "static"]
    dy = df[df["suite"] == "dynamic"]

    if len(st):
        print("## Static suite: R2 on the extrapolation set\n")
        piv = st.pivot_table(index="task", columns=["method", "noise"], values="r2_extrap").clip(lower=-1)
        print(md(piv), "\n")
        print("## Static suite: per-method summary\n")
        g = st.groupby(["method", "noise"]).agg(
            tasks=("task", "nunique"),
            accuracy_solutions=("accuracy_solution", "sum"),
            forms_recovered=("form_recovered", "sum"),
            parsimonious=("parsimonious", "sum"),
            median_r2=("r2_extrap", "median"),
            median_complexity=("complexity", "median"),
            median_seconds=("seconds", "median"),
        )
        print(md(g), "\n")

    if len(dy):
        print("## Dynamic suite: R2 of the recovered RHS on an independent trajectory\n")
        piv = dy.pivot_table(index=["task", "target", "noise"], columns=["estimator", "method"],
                             values="r2_extrap").clip(lower=-1)
        print(md(piv), "\n")
        print("## Dynamic suite: derivative-estimator error (mean |error| / mean |dx/dt|)\n")
        e = dy.drop_duplicates(["task", "target", "noise", "estimator"]).pivot_table(
            index=["task", "target", "noise"], columns="estimator", values="deriv_rel_err")
        print(md(e), "\n")
        print("## Dynamic suite: per-method summary (all estimators)\n")
        g = dy.groupby(["method", "estimator"]).agg(
            runs=("r2_extrap", "size"), accuracy_solutions=("accuracy_solution", "sum"),
            parsimonious=("parsimonious", "sum"), median_complexity=("complexity", "median"),
            median_r2=("r2_extrap", "median"), median_seconds=("seconds", "median"))
        print(md(g), "\n")

    print("## Expressions (noise-free runs)\n")
    for _, r in df[df["noise"] == 0].sort_values(["suite", "task", "target", "method"]).iterrows():
        est = "" if r["estimator"] == "-" else f" [{r['estimator']}]"
        print(f"* **{r['task']} {r['target']}** {r['method']}{est} (R2={r['r2_extrap']:.4f}): "
              f"`{str(r['expression'])[:160]}`")


if __name__ == "__main__":
    main(sys.argv[1:])
