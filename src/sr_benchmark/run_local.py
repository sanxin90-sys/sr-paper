"""
One-click local run of the PySR part of the benchmark (the part that could not
be executed in the cloud container), followed by the merged summary.

    python src/sr_benchmark/run_local.py            # full PySR run (static + dynamic)
    python src/sr_benchmark/run_local.py --quick    # 2 static tasks only, to test the setup

The first `import pysr` downloads and installs Julia and the SymbolicRegression.jl
package automatically; this happens once and can take several minutes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def check_pysr() -> bool:
    print("Checking PySR/Julia (the first run installs Julia; this can take several minutes)...", flush=True)
    t0 = time.perf_counter()
    try:
        from pysr import PySRRegressor  # noqa: F401
    except Exception as exc:
        print(f"\nPySR could not be loaded: {type(exc).__name__}: {exc}")
        print("Check your internet connection and run `python -c \"import pysr\"` to see the full error.")
        return False
    print(f"PySR ready ({time.perf_counter() - t0:.0f} s).", flush=True)
    return True


def run(args: list[str]) -> None:
    cmd = [sys.executable, "run_benchmark.py", *args]
    print("\n>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=HERE, env=ENV, check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 static tasks, noise-free, 30 s each")
    ap.add_argument("--budget", type=int, default=45, help="seconds per static task (dynamic uses 2/3)")
    args = ap.parse_args()

    os.chdir(HERE)
    if not check_pysr():
        sys.exit(1)

    for old in ("results/quick_pysr.csv",) if args.quick else ("results/static_pysr.csv", "results/dynamic_pysr.csv"):
        (HERE / old).unlink(missing_ok=True)  # re-runs replace earlier PySR results

    if args.quick:
        run(["--suite", "static", "--methods", "pysr", "--noise", "0", "--budget", "30",
             "--tasks", "stefan_boltzmann", "bulk_sensible_heat", "--out", "results/quick_pysr.csv"])
        csvs = ["results/quick_pysr.csv"]
    else:
        run(["--suite", "static", "--methods", "pysr", "--noise", "0", "0.05",
             "--budget", str(args.budget), "--out", "results/static_pysr.csv"])
        run(["--suite", "dynamic", "--methods", "pysr", "--noise", "0", "0.01", "0.05",
             "--budget", str(max(10, args.budget * 2 // 3)), "--out", "results/dynamic_pysr.csv"])
        csvs = sorted(str(p.relative_to(HERE)) for p in (HERE / "results").glob("*.csv")
                      if not p.name.startswith("quick"))

    summary = HERE / "results" / ("summary_quick.md" if args.quick else "summary.md")
    with summary.open("w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "summarize.py", *csvs], cwd=HERE, env=ENV, stdout=fh, check=False)
    print(f"\nSummary written to {summary}")


if __name__ == "__main__":
    main()
