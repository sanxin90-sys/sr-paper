"""
Inputs for the multi-seed PySR check of E1/E3 (the cases where one seed was
not enough). For every case and seed k = 0..4 the X-TFC random weights use
seed k and PySR uses random_state k (joint seed), so the spread covers both
sources of randomness. SG is deterministic: only the PySR seed varies.

Cases: E3 Lorenz sigma 0.1 and 0.5 (black-box, gray-box auto);
E1 exact known part, 5 % noise, without and with 25 % gaps (sg, black-box,
gray-box auto). Writes pysr_inputs_seeds/*.npz; run with
    python run_pysr_device.py --inputs pysr_inputs_seeds --out ailorenz_pysr_seeds.csv
"""
from __future__ import annotations

import numpy as np

from common import HERE, discrepancy_choice
import e1_jaru_graybox_synthetic as e1
from gbxtfc import lorenz_y
from tasks import simulate_lorenz

OUT = HERE / "pysr_inputs_seeds"
SEEDS = range(5)


def save(tag, X, y, names, **meta):
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{tag}.npz", X=X, y=y, names=np.array(names), **meta)


def e3():
    tr = simulate_lorenz(dt=0.01, t_final=10.0, x0=(-8.0, 7.0, 27.0))
    S, t = tr["states"], tr["t"]
    for sigma in [0.1, 0.5]:
        Sn = S + np.random.default_rng(1).normal(0, sigma, S.shape)
        kw = dict(s=100, m=30, post_sg=31)
        for k in SEEDS:
            st, _, f = lorenz_y(t, Sn, w_phys=0.0, seed=k, **kw)
            save(f"e3s_s{sigma}_blackbox_k{k}", st[5:-5], f[5:-5], ["x", "y", "z"])
            bb = float(np.sqrt(np.nanmean((st - Sn) ** 2)))
            fits, mis = {}, {}
            for w in [1.0, 0.1, 0.03]:
                st_w, _, f_w = lorenz_y(t, Sn, w_phys=w, seed=k, **kw)
                fits[w] = (st_w, f_w); mis[w] = float(np.sqrt(np.nanmean((st_w - Sn) ** 2)))
            st_w, f_w = fits[discrepancy_choice(mis, bb)]
            save(f"e3s_s{sigma}_graybox_auto_k{k}", st_w[5:-5], f_w[5:-5], ["x", "y", "z"])


def e1_cases():
    t, T, rn, _ = e1.simulate(60, 0)
    known = (e1.A, e1.LAM, e1.C)
    kw = dict(s=48, m=12)
    for gaps in [0.0, 0.25]:
        To, Ro = e1.observe(T, rn, 0.05, gaps, seed=1)
        for k in SEEDS:
            ests = {r: e1.estimate(t, To, Ro, r, known, kw, k)
                    for r in ["sg", "blackbox"] + [f"graybox_w{w}" for w in e1.W_GRID]}
            w_star = discrepancy_choice({w: ests[f"graybox_w{w}"][2] for w in e1.W_GRID}, ests["blackbox"][2])
            ests["graybox_auto"] = ests[f"graybox_w{w_star}"]
            for r in ["sg", "blackbox", "graybox_auto"]:
                Th, f, _ = ests[r]
                ok = np.isfinite(Th) & np.isfinite(f)
                save(f"e1s_n0.05_g{gaps}_exact_{r}_k{k}", np.column_stack([Th, Ro])[ok], f[ok], ["T", "Rn"],
                     a=e1.A, lam=e1.LAM, c=e1.C)


if __name__ == "__main__":
    e3(); e1_cases()
    print(len(list(OUT.glob("*.npz"))), "files in", OUT)
