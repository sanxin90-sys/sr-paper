"""
gbxtfc.py
=========

Gray-Box X-TFC (De Florio, Kevrekidis & Karniadakis, 2024, Sec. 2.2) for the
two cases used in this project, plus a black-box fit on the same basis so the
two routes differ only in the physics loss.

Basis (shared with derivatives.xtfc): on every sub-domain the state is the
constrained expression  y(x) = (h(x) - h(-1)) @ beta + y0,  x in [-1, 1],
with h = tanh(w x + b) and fixed random (w, b); y0 is the end value of the
previous sub-domain, so the learned trajectory is continuous. Unlike
derivatives.xtfc, sub-domain i spans observations [i*s, (i+1)*s] exactly, so
collocation points coincide with the observations (paper, Sec. 3.1.1) and no
spline interpolation of the data is needed. The unknown term f(t) has its own
random-feature expansion (m_f neurons + constant).

* scalar_relaxation : dT/dt = a*Rn - lam*T + c + f(t)     (linear -> one solve)
* lorenz_y          : Lorenz-63 with f(x, y, z) unknown in dy/dt = f - y
                      (nonlinear through x*y -> Gauss-Newton per sub-domain)

w_phys = 0 gives the black-box route: data-only fit, f taken as the residual
of the learned derivative.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _basis(s: int, m: int, m_f: int, seed: int, lb=-1.0, ub=1.0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, s + 1)
    w, b = rng.uniform(lb, ub, m), rng.uniform(lb, ub, m)
    h = np.tanh(np.outer(x, w) + b)
    H = h - h[0]
    Hd = w * (1.0 - h**2)
    wf, bf = rng.uniform(lb, ub, m_f), rng.uniform(lb, ub, m_f)
    F = np.column_stack([np.ones_like(x), np.tanh(np.outer(x, wf) + bf)])
    return H, Hd, F


def _subdomains(n: int, s: int):
    """Index slices of length s+1 sharing end points; the last one may be shorter."""
    out, i = [], 0
    while i < n - 1:
        j = min(i + s, n - 1)
        out.append((i, j))
        i = j
    return out


def _stitch(n, pieces):
    """pieces: list of (i, j, arrays...) -> full-length arrays (shared nodes from the later piece)."""
    k = len(pieces[0]) - 2
    outs = [np.full((n,) + pieces[0][2 + q].shape[1:], np.nan) for q in range(k)]
    for p in pieces:
        i, j = p[0], p[1]
        for q in range(k):
            outs[q][i:j + 1] = p[2 + q]
    return outs


def scalar_relaxation(t, T, Rn, a, lam, c, s=48, m=12, m_f=None, w_phys=1.0, ridge=0.0,
                      seed=0, post_sg: int | None = None):
    """Returns (T_fit, dT_fit, f_fit) at the observation times of one gap-free run."""
    n = len(t)
    dt = float(np.mean(np.diff(t)))
    m_f = m if m_f is None else m_f
    pieces, T0 = [], float(T[0])
    for i, j in _subdomains(n, s):
        k = j - i
        H, Hd, F = _basis(k, m, m_f, seed) if k != s else _basis(s, m, m_f, seed)
        Hd = Hd * (2.0 / (k * dt))
        Ts, Rs = T[i:j + 1], Rn[i:j + 1]
        A_d = np.hstack([H, np.zeros((k + 1, F.shape[1]))])
        b_d = Ts - T0
        if w_phys > 0:
            A_p = w_phys * np.hstack([Hd + lam * H, -F])
            b_p = w_phys * (a * Rs + c - lam * T0)
            A, bb = np.vstack([A_d, A_p]), np.concatenate([b_d, b_p])
        else:
            A, bb = A_d[:, :m], b_d
        A = np.vstack([A, np.sqrt(ridge) * np.eye(A.shape[1])])
        bb = np.concatenate([bb, np.zeros(A.shape[1])])
        theta = np.linalg.lstsq(A, bb, rcond=None)[0]
        beta = theta[:m]
        Ti = H @ beta + T0
        dTi = Hd @ beta
        fi = F @ theta[m:] if w_phys > 0 else dTi - (a * Rs - lam * Ti + c)
        pieces.append((i, j, Ti, dTi, fi))
        T0 = Ti[-1]
    T_fit, dT_fit, f_fit = _stitch(n, pieces)
    if post_sg and n > post_sg:
        dT_fit = savgol_filter(dT_fit, post_sg, 5)
        f_fit = savgol_filter(f_fit, post_sg, 5)
    return T_fit, dT_fit, f_fit


def lorenz_y(t, Y, sigma=10.0, beta=8.0 / 3.0, s=10, m=10, m_f=None, w_phys=1.0, ridge=0.0,
             seed=0, n_iter=20, post_sg: int | None = None):
    """Gray-box Lorenz with unknown f in dy/dt = f(x,y,z) - y. Y: (n, 3) observations.
    Returns (states_fit, derivs_fit, f_fit)."""
    n = len(t)
    dt = float(np.mean(np.diff(t)))
    m_f = m if m_f is None else m_f
    pieces, y0 = [], Y[0].astype(float).copy()
    for i, j in _subdomains(n, s):
        k = j - i
        H, Hd, F = _basis(k, m, m_f, seed)
        Hd = Hd * (2.0 / (k * dt))
        obs = Y[i:j + 1]
        pinv = np.linalg.pinv(H)
        bx, by, bz = (pinv @ (obs[:, q] - y0[q]) for q in range(3))
        if w_phys > 0:
            Z0 = np.zeros_like(H)
            ZF = np.zeros((k + 1, F.shape[1]))
            X, Yy = H @ bx + y0[0], H @ by + y0[1]
            bf = np.linalg.lstsq(F, Hd @ by + Yy, rcond=None)[0]
            th = np.concatenate([bx, by, bz, bf])
            for _ in range(n_iter):
                bx, by, bz, bf = th[:m], th[m:2 * m], th[2 * m:3 * m], th[3 * m:]
                X, Yy, Z = H @ bx + y0[0], H @ by + y0[1], H @ bz + y0[2]
                r = np.concatenate([
                    X - obs[:, 0], Yy - obs[:, 1], Z - obs[:, 2],
                    w_phys * (Hd @ bx - sigma * (Yy - X)),
                    w_phys * (Hd @ by - F @ bf + Yy),
                    w_phys * (Hd @ bz - X * Yy + beta * Z),
                ])
                J = np.block([
                    [H, Z0, Z0, ZF],
                    [Z0, H, Z0, ZF],
                    [Z0, Z0, H, ZF],
                    [w_phys * (Hd + sigma * H), -w_phys * sigma * H, Z0, ZF],
                    [Z0, w_phys * (Hd + H), Z0, -w_phys * F],
                    [-w_phys * (Yy[:, None] * H), -w_phys * (X[:, None] * H), w_phys * (Hd + beta * H), ZF],
                ])
                J = np.vstack([J, np.sqrt(ridge) * np.eye(J.shape[1])])
                r = np.concatenate([r, np.sqrt(ridge) * th])
                step = np.linalg.lstsq(J, r, rcond=None)[0]
                th = th - step
                if np.linalg.norm(step) < 1e-10 * (1 + np.linalg.norm(th)):
                    break
            bx, by, bz, bf = th[:m], th[m:2 * m], th[2 * m:3 * m], th[3 * m:]
        S = np.column_stack([H @ bx + y0[0], H @ by + y0[1], H @ bz + y0[2]])
        D = np.column_stack([Hd @ bx, Hd @ by, Hd @ bz])
        f = F @ bf if w_phys > 0 else D[:, 1] + S[:, 1]
        pieces.append((i, j, S, D, f))
        y0 = S[-1].copy()
    S, D, f = _stitch(n, pieces)
    if post_sg and n > post_sg:
        D = savgol_filter(D, post_sg, 5, axis=0)
        f = savgol_filter(f, post_sg, 5)
    return S, D, f
