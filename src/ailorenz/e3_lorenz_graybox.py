"""
E3 - sanity check of the gray-box X-TFC port on the paper's Lorenz case
(De Florio et al. 2024, Sec. 2.2): x(rho - z) is unknown in dy/dt = f(x,y,z) - y.

Data as in the paper: t in [0, 10], dt = 0.01 (1001 points), additive Gaussian
noise sigma in {0, 0.1, 0.5}. Settings: clean -> 10 samples/sub-domain,
10 neurons; noisy -> 100 samples, 30 neurons, Savitzky-Golay(31, 5) on the
learned derivatives (paper, Sec. 3.1.4). 5 seeds of the random X-TFC weights.

Routes: black-box (w_phys = 0, f = dy/dt + y) and gray-box at w_phys in
{1, 0.1, 0.03} plus a truth-free choice of w_phys (largest weight whose data
misfit stays within 1.1x the black-box misfit). SR: SINDy poly2 on every case;
Operon (30 s) on seed 0 for black-box and the truth-free gray-box choice.
Scoring: exact structure = SINDy keeps exactly {x, x*z}; coefficient errors
vs 28 and -1; R2 of the recovered f on an independent test trajectory.

Usage:  python src/ailorenz/e3_lorenz_graybox.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS, discrepancy_choice, export_for_pysr, poly_terms, r2, run_sr, complexity
from gbxtfc import lorenz_y
from tasks import simulate_lorenz

NAMES = ["x", "y", "z"]
SEEDS = range(5)
W_GRID = [1.0, 0.1, 0.03]


def main():
    tr = simulate_lorenz(dt=0.01, t_final=10.0, x0=(-8.0, 7.0, 27.0))
    te = simulate_lorenz(dt=0.01, t_final=10.0, x0=(5.0, -3.0, 20.0))
    S, t = tr["states"], tr["t"]
    f_true = S[:, 0] * (28 - S[:, 2])
    f_te = te["states"][:, 0] * (28 - te["states"][:, 2])
    rows = []
    for sigma in [0.0, 0.1, 0.5]:
        Sn = S + np.random.default_rng(1).normal(0, sigma, S.shape)
        kw = dict(s=10, m=10) if sigma == 0 else dict(s=100, m=30, post_sg=31)
        for seed in SEEDS:
            fits = {}
            st, D, f = lorenz_y(t, Sn, w_phys=0.0, seed=seed, **kw)
            fits["blackbox"] = (st, f)
            bb_rms = float(np.sqrt(np.nanmean((st - Sn) ** 2)))
            misfit = {}
            for w in W_GRID:
                st, D, f = lorenz_y(t, Sn, w_phys=w, seed=seed, **kw)
                fits[f"graybox_w{w}"] = (st, f)
                misfit[w] = float(np.sqrt(np.nanmean((st - Sn) ** 2)))
            w_star = discrepancy_choice(misfit, bb_rms)
            fits["graybox_auto"] = fits[f"graybox_w{w_star}"]
            for route, (st, f) in fits.items():
                X, y = st[5:-5], f[5:-5]
                engines = ("sindy_poly2", "operon") if seed == 0 and route in ("blackbox", "graybox_auto") else ("sindy_poly2",)
                if seed == 0 and route in ("blackbox", "graybox_auto"):
                    export_for_pysr(f"e3_lorenz_s{sigma}_{route}", X, y, NAMES, sigma=sigma, route=route)
                for eng, fit in run_sr(X, y, NAMES, engines, budget=30, seed=seed).items():
                    terms = poly_terms(fit.expression, NAMES)
                    big = {k: v for k, v in terms.items() if abs(v) > 1e-3}
                    rows.append(dict(
                        sigma=sigma, seed=seed, route=route, w_phys=(w_star if route == "graybox_auto" else route),
                        engine=eng, f_rel_err=round(float(np.nanmean(np.abs(f - f_true)) / np.mean(np.abs(f_true))), 4),
                        state_mae=round(float(np.nanmean(np.abs(st - S))), 4),
                        exact_structure=set(big) == {"x", "x*z"},
                        coef_x=round(terms.get("x", np.nan), 4), coef_xz=round(terms.get("x*z", np.nan), 4),
                        r2_test=round(r2(f_te, fit.predict(te["states"])), 5),
                        complexity=complexity(fit.expression), expression=fit.expression))
                    print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ailorenz_e3_lorenz.csv", index=False)
    s = out[out.engine == "sindy_poly2"].groupby(["sigma", "route"]).agg(
        f_rel_err=("f_rel_err", "mean"), state_mae=("state_mae", "mean"), exact=("exact_structure", "sum"),
        coef_x=("coef_x", "mean"), coef_xz=("coef_xz", "mean"), r2_test=("r2_test", "median"))
    print(s.to_string())
    print(out[out.engine == "operon"][["sigma", "route", "r2_test", "complexity", "expression"]].to_string())


if __name__ == "__main__":
    main()
