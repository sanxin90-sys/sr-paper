"""
Score the PySR expressions from run_pysr_device.py with the same metrics as
e1/e2/e3. Writes results/ailorenz_pysr_scored.csv.

Usage:  python src/ailorenz/score_pysr.py "Dados Rebio Jaru 2006 a 2008.xlsx"
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import sympy as sp

from common import HERE, RESULTS, complexity, poly_terms, r2
import e1_jaru_graybox_synthetic as e1
import e2_jaru_graybox_real as e2
from tasks import simulate_lorenz


def predictor(expr, names):
    syms = sp.symbols(names)
    f = sp.lambdify(syms, sp.sympify(expr, locals=dict(zip(names, syms))), "numpy")
    return lambda X: np.broadcast_to(np.asarray(f(*X.T), dtype=float), (len(X),)).copy()


def main(xlsx):
    P = pd.read_csv(HERE / "pysr_outputs" / "ailorenz_pysr.csv")
    rows = []
    te3 = simulate_lorenz(dt=0.01, t_final=10.0, x0=(5.0, -3.0, 20.0))["states"]
    f3 = te3[:, 0] * (28 - te3[:, 2])
    tt, Tt, rnt, dTt = e1.simulate(30, 100)
    df = e2.add_features(e2.load_jaru(xlsx))
    df.loc[(df.year == 2006) & (df.month == 10), "Rn"] = np.nan
    d06 = df[df.year == 2006].dropna(subset=["dTdt", "Rn", "T22"])
    a, lam, c = np.linalg.lstsq(np.column_stack([d06.Rn, -d06.T22, np.ones(len(d06))]), d06.dTdt, rcond=None)[0]
    te = df[df.year == 2008].reset_index(drop=True)
    te = te.assign(dTdt_xtfc=e2.xtfc_tendency(te)).dropna(subset=e2.FEATS + ["dTdt", "dTdt_xtfc"]).reset_index(drop=True)
    for _, p in P.iterrows():
        row = dict(dataset=p.dataset, seed=p.seed, complexity=complexity(p.expression), expression=p.expression)
        if p.dataset.startswith("e3_"):
            names = ["x", "y", "z"]
            terms = {k: v for k, v in poly_terms(p.expression, names).items() if abs(v) > 1e-3}
            row.update(exact_structure=set(terms) == {"x", "x*z"}, coef_x=terms.get("x"), coef_xz=terms.get("x*z"),
                       r2_test=round(r2(f3, predictor(p.expression, names)(te3)), 5))
        elif p.dataset.startswith("e1_"):
            names = ["T", "Rn"]
            meta = np.load(HERE / "pysr_inputs" / f"{p.dataset}.npz")
            ka, kl, kc = float(meta["a"]), float(meta["lam"]), float(meta["c"])
            terms = poly_terms(p.expression, names)
            c2 = terms.get("Rn**2", np.nan)
            ft = predictor(p.expression, names)(np.column_stack([Tt, rnt]))
            row.update(rn2_coef=c2, rn2_rel_err=abs(c2 / -e1.K - 1) if np.isfinite(c2) else np.nan,
                       r2_f_test=round(r2(dTt - (ka * rnt - kl * Tt + kc), ft), 4),
                       r2_rhs_test=round(r2(dTt, ka * rnt - kl * Tt + kc + ft), 4))
        else:
            pr = predictor(p.expression, e2.FEATS)(te[e2.FEATS].to_numpy(float))
            known = (a * te.Rn - lam * te.T22 + c).to_numpy()
            for target in ["dTdt", "dTdt_xtfc"]:
                for season in ["all", "wet", "dry"]:
                    sel = np.ones(len(te), bool) if season == "all" else (te.season == season).to_numpy()
                    row[f"r2_{target}_{season}"] = round(r2(te[target].to_numpy()[sel], (known + pr)[sel]), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ailorenz_pysr_scored.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
