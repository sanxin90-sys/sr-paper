"""
audit_jaru.py
=============

Data audit of the Rebio Jaru record (DQ-001 to DQ-003 in the manuscript,
Supplementary Table S2). Prints the diagnostics behind each defect. It does
not change any value: the scripts that use the data drop the flagged periods
themselves (see SUSPECT_RN in dtdt_robustness.py and the split definitions in
clean_rerun_rn_rh.py).

Checks
------
DQ-001  Monthly median of the half-hourly ratio Rn/PAR (PAR > 50 W m-2).
DQ-002  22 m air temperature: hourly-mean diurnal range per year, and its
        correlation with the 62 m FLUXNET temperature (TA_F_MDS, QC = 0).
DQ-003  Monthly median of Rn/SW_IN against FLUXNET incoming shortwave
        radiation (SW_IN_F_MDS, QC = 0, SW_IN > 100 W m-2), also with PAR > 50
        W m-2 to drop the half-hours where PAR and Rn both fell to zero
        (the levels quoted in the manuscript use this filter); per month, the
        share of these sunlit half-hours with PAR < 50 W m-2 and the
        correlation r(Rn, SW_IN).

Usage
-----
    python audit_jaru.py "Dados Rebio Jaru 2006 a 2008.xlsx" \
        data/fluxnet_BR-Ji3/AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_discovery import load_jaru  # noqa: E402


def load_fluxnet(path: str | Path) -> pd.DataFrame:
    cols = ["TIMESTAMP_END", "TA_F_MDS", "TA_F_MDS_QC", "SW_IN_F_MDS", "SW_IN_F_MDS_QC"]
    f = pd.read_csv(path, na_values=[-9999], usecols=cols)
    # FLUXNET stamps the end of the half hour, like the spreadsheet loader
    f["timestamp"] = pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")
    f["TA62"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0)
    f["SW_IN"] = f.SW_IN_F_MDS.where(f.SW_IN_F_MDS_QC == 0)
    return f[["timestamp", "TA62", "SW_IN"]]


def main(xlsx: str, flx: str) -> None:
    j = load_jaru(xlsx)
    d = j.merge(load_fluxnet(flx), on="timestamp", how="left")
    d["ym"] = d.timestamp.dt.to_period("M")
    pd.set_option("display.width", 120)

    print("DQ-001  median Rn/PAR by month (PAR > 50 W m-2)")
    day = d[d.PAR > 50].dropna(subset=["Rn", "PAR"])
    print((day.Rn / day.PAR).groupby(day.ym).median().round(2).to_string(), "\n")

    print("DQ-002  22 m temperature: range of hourly means, and r with 62 m TA")
    for y, s in d.groupby("year"):
        hourly = s.groupby(s.timestamp.dt.hour).T22.mean()
        ok = s.dropna(subset=["T22", "TA62"])
        r = np.corrcoef(ok.T22, ok.TA62)[0, 1] if len(ok) > 10 else np.nan
        print(f"  {y}: hourly means {hourly.min():.1f}-{hourly.max():.1f} degC, "
              f"r(T22, TA62) = {r:.3f}, mean(T22 - TA62) = {(ok.T22 - ok.TA62).mean():+.2f} K")
    print()

    print("DQ-003  by month, half-hours with SW_IN > 100 W m-2")
    sun = d[d.SW_IN > 100].dropna(subset=["Rn", "PAR"])
    table = sun.groupby("ym").apply(lambda s: pd.Series({
        "median Rn/SW_IN": (s.Rn / s.SW_IN).median(),
        "same, PAR > 50": (s.Rn / s.SW_IN)[s.PAR > 50].median(),
        "% PAR < 50": (s.PAR < 50).mean() * 100,
        "r(Rn, SW_IN)": np.corrcoef(s.Rn, s.SW_IN)[0, 1],
    }))
    print(table.round(2).to_string())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
