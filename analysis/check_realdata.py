"""Recount Table 3 (audited real-data tests) from the result CSVs.

An engine beats the re-fitted classical relation in a fit only if the upper
limit of the 95 % block-bootstrap interval of RMSE(engine) - RMSE(classical)
is below zero (the rule registered before the runs).

Usage: python check_realdata.py <results_dir>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

PYSR_FILE = "pysr_clean_rn_rh_pysr.csv"  # PySR run on a laptop (pysr_clean_rn_rh.py)
FILES = ["clean_rerun_rn_rh.csv", "clean_rerun_extra_gplearn.csv", "clean_rerun_extra_sindy_ext.csv", PYSR_FILE]


def pysr_row(r: dict) -> dict | None:
    """Map the PySR file to the common columns; RH is compared with baseline T0 (Tetens, T_d = a*T_min + b)."""
    if r["method"] != "pysr":
        return None
    ref = "T0" if r["task"].startswith("RH") else "classical"
    return dict(r, d_rmse_lo=r[f"d_rmse_vs_{ref}_lo"], d_rmse_hi=r[f"d_rmse_vs_{ref}_hi"],
                rmse_classical_same_seed=r[f"rmse_{ref}_same_seed"])


def main(res_dir: Path) -> None:
    fits = defaultdict(lambda: {"n": 0, "wins": 0, "worse20": 0, "invalid": 0})
    classical = {}
    for name in FILES:
        path = res_dir / name
        if not path.exists():
            print(f"missing: {path}")
            continue
        for r in csv.DictReader(open(path)):
            if name == PYSR_FILE:
                r = pysr_row(r)
                if r is None:
                    continue
            key = (r["task"], r["split"])
            if r["method"] == "classical":
                if r["seed"] == "0":
                    classical[key] = (float(r["rmse"]), float(r["r2"]))
                continue
            f = fits[key + (r["method"],)]
            f["n"] += 1
            try:
                hi = float(r["d_rmse_hi"])
                rmse = float(r["rmse"])
            except ValueError:
                f["invalid"] += 1
                continue
            if hi < 0:
                f["wins"] += 1
            rc = r.get("rmse_classical_same_seed") or ""
            ref = float(rc) if rc else classical.get(key, (float("nan"),))[0]
            if rmse > 1.2 * ref:
                f["worse20"] += 1
    print("task | split | classical RMSE (R2)")
    for key, (rmse, r2) in sorted(classical.items()):
        print(f"{key[0]} | {key[1]} | {rmse:.2f} ({r2:.3f})")
    print("\ntask | split | method | wins/fits | >20 % worse | invalid")
    for key, f in sorted(fits.items()):
        print(f"{key[0]} | {key[1]} | {key[2]} | {f['wins']}/{f['n']} | {f['worse20']} | {f['invalid']}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
