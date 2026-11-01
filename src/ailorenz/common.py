"""Shared helpers for the AI-Lorenz gray-box experiments (E1-E3)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "sr_benchmark"))
from methods import fit_operon, fit_sindy_poly, complexity  # noqa: E402

RESULTS = HERE.parent / "sr_benchmark" / "results"
PYSR_IN = HERE / "pysr_inputs"


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def poly_terms(expr: str, names: list[str]) -> dict[str, float]:
    """Monomial -> coefficient of an expanded polynomial expression (non-polynomials -> {})."""
    syms = sp.symbols(names)
    e = sp.expand(sp.sympify(expr, locals=dict(zip(names, syms))))
    try:
        P = sp.Poly(e, *syms)
    except sp.PolynomialError:
        return {}
    out = {}
    for mon, c in P.terms():
        key = "*".join(f"{n}**{k}" if k > 1 else n for n, k in zip(names, mon) if k) or "1"
        out[key] = float(c)
    return out


def run_sr(X, y, names, engines=("sindy_poly2",), budget=30, seed=0):
    fits = {}
    for eng in engines:
        try:
            if eng == "sindy_poly2":
                fits[eng] = fit_sindy_poly(X, y, names)
            elif eng == "operon":
                fits[eng] = fit_operon(X, y, names, time_budget_s=budget, seed=seed)
        except Exception as exc:  # report, never hide
            print(f"{eng} failed: {type(exc).__name__}: {exc}", flush=True)
    return fits


def export_for_pysr(tag: str, X, y, names, **meta):
    PYSR_IN.mkdir(exist_ok=True)
    np.savez_compressed(PYSR_IN / f"{tag}.npz", X=X, y=y, names=np.array(names), **meta)


def discrepancy_choice(rows_by_w: dict[float, float], bb_rms: float, tol=1.1):
    """Largest physics weight whose data misfit stays within tol x the black-box misfit."""
    ok = [w for w, rms in rows_by_w.items() if rms <= tol * bb_rms]
    return max(ok) if ok else min(rows_by_w)
