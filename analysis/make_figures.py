"""Figures for Article 1. Run consolidate_benchmark.py first (it writes out/static_counts_per_seed.csv).

Usage: python make_figures.py RESULTS_DIR   (RESULTS_DIR = sr_benchmark/results)
Writes out/fig1_static.{png,pdf}, out/fig2_protocol.{png,pdf}, out/fig3_realdata.{png,pdf}, out/graphical_abstract.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(sys.argv[1])
OUT = Path(__file__).resolve().parent / "out"
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "savefig.dpi": 600})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- Fig. 1: static suite, accuracy solutions per engine ----------------
c = pd.read_csv(OUT / "static_counts_per_seed.csv")
rows = []
for (m, nz), g in c.groupby(["method", "noise"]):
    rows.append(dict(method=m, noise=nz, mean=g.acc.mean(), lo=g.acc.min(), hi=g.acc.max(), seeds=g.seed.nunique()))
aif = pd.read_csv(R.parent / "aifeynman_fair" / "fair_s0_final.csv")
rows.append(dict(method="aifeynman_full", noise=0.0, mean=int(aif.acc.sum()), lo=int(aif.acc.sum()), hi=int(aif.acc.sum()), seeds=1))
k = pd.read_csv(R / "kan_static.csv")
for cfg in ("v1", "v2"):
    for nz in (0.0, 0.05):
        g = k[(k.config == cfg) & (k.noise == nz)]
        per = g.groupby("seed").apply(lambda d: 10 * (d.r2_symbolic >= 0.999).sum() / len(d), include_groups=False)
        rows.append(dict(method=f"kan_{cfg}", noise=nz, mean=per.mean(), lo=per.min(), hi=per.max(), seeds=len(per)))
s = pd.DataFrame(rows)
s.to_csv(OUT / "fig1_data.csv", index=False)
labels = {"operon": "Operon", "pysr": "PySR", "aifeynman_full": "AI Feynman (complete)", "dso_const": "DSO + constant",
          "sindy_stlsq": "SINDy extended", "kan_v2": "KAN corrected (v2)", "sindy_poly2": "SINDy quadratic",
          "gplearn": "gplearn", "physo": "PhySO", "kan_v1": "KAN naive (v1)", "dso": "DSO, no constant"}
order = [m for m in labels if m in set(s.method)]
order = sorted(order, key=lambda m: -s[(s.method == m) & (s.noise == 0)]["mean"].iloc[0])
fig, ax = plt.subplots(figsize=(3.5, 3.2))
for i, m in enumerate(order):
    for nz, col, dy in ((0.0, BLUE, -0.15), (0.05, ORANGE, 0.15)):
        r = s[(s.method == m) & (s.noise == nz)]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.plot([r.lo, r.hi], [i + dy] * 2, color=col, lw=2, solid_capstyle="round")
        ax.plot(r["mean"], i + dy, "o", ms=5, color=col, mec="white", mew=1)
ax.set_yticks(range(len(order)), [labels[m] for m in order])
ax.invert_yaxis()
ax.set_xlim(-0.3, 10)
ax.set_xlabel("Laws with an accuracy solution (of 10)")
ax.grid(axis="x", color="#e6e5e0", lw=0.6)
ax.plot([], [], "o", color=BLUE, label="0 % noise")
ax.plot([], [], "o", color=ORANGE, label="5 % noise")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Dot = mean over seeds; bar = range. KAN scaled from fits to /10", fontsize=6.5, color=MUTED, loc="left")
save(fig, "fig1_static")

# ---------------- Fig. 2: protocol versus engine ----------------
prot = [("KAN: corrected readout and refit", 10 * 1 / 59, 10 * 18 / 60),
        ("AI Feynman: complete run", 4, 7),
        ("DSO: fitted-constant token", 0, 3),
        ("PySR: budget 45 s → 120 s", 7, 7),
        ("PhySO: unit constraints (4 laws)", 0, 0)]
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for i, (lab, a, b) in enumerate(prot):
    ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6)) if b != a else None
    ax.plot(a, i, "o", ms=5, color=GRAY, mec="white", mew=1)
    ax.plot(b, i, "o", ms=5, color=BLUE, mec="white", mew=1)
eng = s[s.noise == 0].set_index("method")["mean"]
lo, hi = eng.min(), eng.max()
j = len(prot) + 0.3
ax.plot([lo, hi], [j, j], color=ORANGE, lw=2.5, solid_capstyle="round")
ax.set_yticks(list(range(len(prot))) + [j], [p[0] for p in prot] + ["Spread between engines"])
ax.invert_yaxis()
ax.set_xlim(-0.3, 10)
ax.set_xlabel("Accuracy solutions on the static suite, 0 % noise (of 10)")
ax.grid(axis="x", color="#e6e5e0", lw=0.6)
ax.plot([], [], "o", color=GRAY, label="before")
ax.plot([], [], "o", color=BLUE, label="after")
ax.legend(frameon=False, loc="lower right", fontsize=7)
save(fig, "fig2_protocol")

# ---------------- Fig. 3: audited real data, Delta RMSE with block-bootstrap CI ----------------
d = pd.read_csv(R / "clean_rerun_rn_rh.csv")
for extra in sorted(R.glob("clean_rerun_extra_*.csv")):
    d = pd.concat([d, pd.read_csv(extra)], ignore_index=True)
pysr_file = R / "pysr_clean_rn_rh_pysr.csv"  # PySR run on a laptop; RH compared with baseline T0
if pysr_file.exists():
    p = pd.read_csv(pysr_file)
    p = p[p.method == "pysr"].copy()
    rh = p.task.str.startswith("RH")
    p.loc[~rh, "d_rmse_lo"] = p.loc[~rh, "d_rmse_vs_classical_lo"]
    p.loc[~rh, "d_rmse_hi"] = p.loc[~rh, "d_rmse_vs_classical_hi"]
    p.loc[rh, "d_rmse_lo"] = p.loc[rh, "d_rmse_vs_T0_lo"]
    p.loc[rh, "d_rmse_hi"] = p.loc[rh, "d_rmse_vs_T0_hi"]
    d = pd.concat([d, p[["task", "split", "method", "seed", "rmse", "d_rmse_lo", "d_rmse_hi"]]], ignore_index=True)
d = d[d.method != "classical"].copy()
split_lab = {"A_2006_H1toH2": "A: 2006", "B_2008_H1toH2": "B: 2008", "C_2006to2008_acrossstep": "C: across step",
             "train2006_test2005": "2005", "train2006_test2008": "2008", "train2006_test2009": "2009",
             "train2006_test2010": "2010"}
meth = ["sindy_poly2", "sindy_ext", "operon", "gplearn", "pysr", "kan_v2", "physo"]
mcol = {"sindy_poly2": "#1baf7a", "sindy_ext": "#008300", "operon": BLUE, "gplearn": ORANGE, "pysr": "#6250d6", "kan_v2": "#8a6d3b", "physo": "#e87ba4"}
mlab = {"sindy_poly2": "SINDy quadratic", "sindy_ext": "SINDy extended", "operon": "Operon", "gplearn": "gplearn", "pysr": "PySR",
        "kan_v2": "KAN v2", "physo": "PhySO"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, task, lim, unit in ((axes[0], "Rn~PAR", 12, "W m$^{-2}$"), (axes[1], "RH~T,Tmin", 4, "%")):
    t = d[d.task == task]
    splits = list(dict.fromkeys(t.split))
    present = [m for m in meth if m in set(t.method)]
    for i, sp in enumerate(splits):
        for jm, m in enumerate(present):
            g = t[(t.split == sp) & (t.method == m)].sort_values("seed")
            for ks, (_, r) in enumerate(g.iterrows()):
                y = i + (jm - (len(present) - 1) / 2) * 0.22 + (ks - (len(g) - 1) / 2) * 0.035
                if not np.isfinite(r.d_rmse_lo):
                    ax.plot(lim * 0.97, y, "x", color=mcol[m], ms=3)
                    continue
                lo_, hi_ = np.clip([r.d_rmse_lo, r.d_rmse_hi], -lim, lim)
                ax.plot([lo_, hi_], [y, y], color=mcol[m], lw=1.2, solid_capstyle="round")
                if abs(r.d_rmse_hi - r.d_rmse_lo) < 1e-3:  # identical prediction (e.g. SINDy reduced to the line)
                    ax.plot(lo_, y, "D", color=mcol[m], ms=3)
                if r.d_rmse_hi > lim:
                    ax.plot(lim, y, ">", color=mcol[m], ms=3)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(splits)), [split_lab[x] for x in splits])
    ax.invert_yaxis()
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(f"RMSE(engine) − RMSE(classical), 95 % CI ({unit})")
    ax.set_title("Net radiation from PAR" if task.startswith("Rn") else "Relative humidity (Tetens baseline)",
                 loc="left", fontsize=8)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
h = [plt.Line2D([], [], color=mcol[m], lw=2, label=mlab[m]) for m in meth if m in set(d.method)]
h += [plt.Line2D([], [], ls="", marker=">", color=GRAY, ms=4, label="CI beyond axis"),
      plt.Line2D([], [], ls="", marker="x", color=GRAY, ms=4, label="non-finite prediction"),
      plt.Line2D([], [], ls="", marker="D", color=GRAY, ms=3, label="identical to classical")]
fig.legend(handles=h, frameon=False, fontsize=6.5, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, 0.0))
save(fig, "fig3_realdata")
print("figures written to", OUT)

# ---------------- Graphical abstract (EMS: 531 x 1328 px minimum, landscape) ----------------
fig = plt.figure(figsize=(13.28 / 2.54 * 2, 5.31 / 2.54 * 2))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 3); ax.set_ylim(0, 1)
boxes = [
    (0.05, "Synthetic benchmark", "10 micrometeorological laws\n+ relaxation and Lorenz-63\n8 engines, 3 noise levels",
     "Best engines recover 7 of 10 laws\n(Operon, PySR, AI Feynman)", BLUE),
    (1.05, "Audited flux-tower record", "Rebio Jaru, Amazon\nclassical relations re-fitted\nblock bootstrap, rule fixed in advance",
     "Rn ~ PAR: no better valid form\nthan a re-fitted line\nRH: gains in some years only,\nno physical form", ORANGE),
    (2.05, "What decided the outcome", "KAN readout, AI Feynman run length,\nDSO constant token,\nsensor audit, fair baseline",
     "Protocol choices changed\nengine rankings and\nreal-data verdicts", "#1baf7a"),
]
for x, title, body, result, col in boxes:
    ax.add_patch(plt.Rectangle((x, 0.08), 0.9, 0.84, fc="white", ec=col, lw=2))
    ax.text(x + 0.45, 0.83, title, ha="center", va="center", fontsize=13, weight="bold", color=INK)
    ax.text(x + 0.45, 0.6, body, ha="center", va="center", fontsize=10.5, color=MUTED, linespacing=1.4)
    ax.add_patch(plt.Rectangle((x + 0.05, 0.13), 0.8, 0.27, fc=col, ec="none", alpha=0.12))
    ax.text(x + 0.45, 0.265, result, ha="center", va="center", fontsize=11, color=INK, linespacing=1.4)
for x in (0.97, 1.97):
    ax.annotate("", xy=(x + 0.06, 0.5), xytext=(x - 0.01, 0.5), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))
fig.savefig(OUT / "graphical_abstract.png", dpi=300, facecolor="white")
plt.close(fig)
