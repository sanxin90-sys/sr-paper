"""
E1 - Gray-box vs black-box recovery of a planted extra term in the synthetic
Jaru temperature model:

    dT/dt = 0.0055*Rn - 0.30*T + 6.9  - 3e-6*Rn**2          (K h-1, Rn in W m-2)
            `------ known part -----'   `- f, to discover -'

Rn(t) is the synthetic Jaru forcing of tasks.simulate_jaru (train: 60 days,
seed 0; test: 30 days, seed 100). Observations every 30 min; Gaussian noise
of 0/1/5 % of the std on T and Rn; optional 25 % missing data in blocks of
3-48 h (each estimator then runs per gap-free run of >= 96 samples).

Known part used by the gray-box and by the subtraction routes:
  exact     - the true relaxation coefficients;
  estimated - relaxation-only OLS fit (on Savitzky-Golay derivatives) on an
              independent noisy trajectory (Rn seed 50), i.e. biased, as it will
              be on real data.
Routes: SG(7,2) derivative minus known part; black-box X-TFC derivative minus
known part; gray-box X-TFC with w_phys in {1, 0.1, 0.03} and the truth-free
choice (see common.discrepancy_choice). X-TFC: 10 samples/10 neurons when
noise-free, 48/12 otherwise (settings of run_benchmark.py); 5 weight seeds.
SR on f against (T, Rn): SINDy poly2 everywhere; Operon (20 s) on seed 0.

Success criterion (fixed before running): with the exact known part and 5 %
noise, the Rn**2 term is found with |coef error| < 10 % in >= 4/5 seeds.
The gray-box is worth its extra complexity only if it beats the black-box
route on that criterion or on test R2.

Usage:  python src/ailorenz/e1_jaru_graybox_synthetic.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

from common import RESULTS, discrepancy_choice, export_for_pysr, poly_terms, r2, run_sr, complexity
from gbxtfc import scalar_relaxation
from tasks import simulate_jaru

A, LAM, C, K = 0.0055, 0.30, 6.9, 3e-6
NAMES = ["T", "Rn"]
SEEDS = range(5)
W_GRID = [1.0, 0.1, 0.03]
MIN_RUN = 96


def simulate(n_days, seed):
    d = simulate_jaru(n_days=n_days, seed=seed)
    t, rn = d["t"], d["forcing"][:, 0]
    rhs = lambda tt, T: A * np.interp(tt, t, rn) - LAM * T + C - K * np.interp(tt, t, rn) ** 2
    sol = solve_ivp(rhs, (t[0], t[-1]), [d["states"][0, 0]], t_eval=t, rtol=1e-10, atol=1e-10, max_step=0.25)
    T = sol.y[0]
    return t, T, rn, A * rn - LAM * T + C - K * rn**2


def observe(T, rn, noise, gaps, seed):
    rng = np.random.default_rng(seed)
    To = T + rng.normal(0, noise * T.std(), T.shape)
    Ro = rn + rng.normal(0, noise * rn.std(), rn.shape)
    mask = np.ones(len(T), bool)
    while gaps and (~mask).mean() < gaps:
        L = rng.integers(6, 97)
        i = rng.integers(0, len(T) - L)
        mask[i:i + L] = False
    To[~mask] = np.nan
    return To, Ro


def runs(To):
    ok = np.isfinite(To)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    return [r for r in np.split(idx, splits) if len(r) >= MIN_RUN]


def estimate(t, To, Ro, route, known, kw, seed):
    """Returns (T_hat, f_hat) over the full grid (NaN outside usable runs)."""
    a, lam, c = known
    T_hat, f_hat = np.full(len(t), np.nan), np.full(len(t), np.nan)
    misfit = []
    for r in runs(To):
        tt, TT, RR = t[r], To[r], Ro[r]
        if route == "sg":
            Th, dT = TT, savgol_filter(TT, 7, 2, deriv=1, delta=0.5)
            f = dT - (a * RR - lam * Th + c)
        else:
            w = 0.0 if route == "blackbox" else float(route.split("_w")[1])
            Th, dT, f = scalar_relaxation(tt, TT, RR, a, lam, c, w_phys=w, seed=seed, **kw)
        Th, f = Th.copy(), f.copy()
        Th[:3] = Th[-3:] = np.nan
        T_hat[r], f_hat[r] = Th, f
        misfit.append((Th - TT)[3:-3])
    return T_hat, f_hat, float(np.sqrt(np.mean(np.concatenate(misfit) ** 2)))


def main():
    t, T, rn, dT = simulate(60, 0)
    tt, Tt, rnt, dTt = simulate(30, 100)
    rows = []
    for noise, gaps in [(0.0, 0.0), (0.01, 0.0), (0.05, 0.0), (0.0, 0.25), (0.01, 0.25), (0.05, 0.25)]:
        To, Ro = observe(T, rn, noise, gaps, seed=1)
        # biased relaxation estimate from an independent noisy trajectory
        t2, T2, rn2, _ = simulate(60, 50)
        T2o, R2o = observe(T2, rn2, noise, 0.0, seed=2)
        d2 = savgol_filter(T2o, 7, 2, deriv=1, delta=0.5)
        coef = np.linalg.lstsq(np.column_stack([R2o, -T2o, np.ones_like(T2o)]), d2, rcond=None)[0]
        knowns = {"exact": (A, LAM, C), "estimated": tuple(coef)}
        kw = dict(s=10, m=10) if noise == 0 else dict(s=48, m=12)
        for kname, known in knowns.items():
            a, lam, c = known
            f_true_tr = dT - (a * rn - lam * T + c)
            f_true_te = dTt - (a * rnt - lam * Tt + c)
            for seed in SEEDS:
                ests = {}
                for route in ["sg", "blackbox"] + [f"graybox_w{w}" for w in W_GRID]:
                    if route == "sg" and seed > 0:
                        continue  # deterministic
                    ests[route] = estimate(t, To, Ro, route, known, kw, seed)
                if seed >= 0:
                    mis = {w: ests[f"graybox_w{w}"][2] for w in W_GRID}
                    w_star = discrepancy_choice(mis, ests["blackbox"][2])
                    ests["graybox_auto"] = ests[f"graybox_w{w_star}"]
                for route, (Th, f, _) in ests.items():
                    ok = np.isfinite(Th) & np.isfinite(f)
                    X, y = np.column_stack([Th, Ro])[ok], f[ok]
                    main_route = route in ("sg", "blackbox", "graybox_auto")
                    engines = ("sindy_poly2", "operon") if seed == 0 and main_route and kname == "exact" else ("sindy_poly2",)
                    if seed == 0 and main_route:
                        export_for_pysr(f"e1_n{noise}_g{gaps}_{kname}_{route}", X, y, NAMES, a=a, lam=lam, c=c)
                    for eng, fit in run_sr(X, y, NAMES, engines, budget=20, seed=seed).items():
                        terms = poly_terms(fit.expression, NAMES)
                        f_te = fit.predict(np.column_stack([Tt, rnt]))
                        c2 = terms.get("Rn**2", np.nan)
                        rows.append(dict(
                            noise=noise, gaps=gaps, known=kname, seed=seed, route=route,
                            w_phys=(w_star if route == "graybox_auto" else ""), engine=eng, n_train=int(ok.sum()),
                            f_rel_err=round(float(np.mean(np.abs(f[ok] - f_true_tr[ok])) / np.mean(np.abs(f_true_tr[ok]))), 4),
                            rn2_coef=c2, rn2_rel_err=round(abs(c2 / -K - 1), 4) if np.isfinite(c2) else np.nan,
                            only_rn2=set(k for k, v in terms.items() if v != 0) == {"Rn**2"},
                            r2_f_test=round(r2(f_true_te, f_te), 4),
                            r2_rhs_test=round(r2(dTt, a * rnt - lam * Tt + c + f_te), 4),
                            complexity=complexity(fit.expression), expression=fit.expression))
                        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ailorenz_e1_jaru_synthetic.csv", index=False)
    sindy = out[out.engine == "sindy_poly2"]
    print(sindy.groupby(["known", "noise", "gaps", "route"]).agg(
        f_rel_err=("f_rel_err", "mean"), rn2_ok=("rn2_rel_err", lambda v: int((v < 0.1).sum())),
        n=("seed", "count"), only_rn2=("only_rn2", "sum"), r2_rhs_test=("r2_rhs_test", "median")).to_string())
    print(out[out.engine == "operon"][["noise", "gaps", "route", "r2_f_test", "r2_rhs_test", "complexity", "expression"]].to_string())


if __name__ == "__main__":
    main()
