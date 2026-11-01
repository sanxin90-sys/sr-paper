"""
rh_fair_baseline.py — EXP-017b. Is the SR gain on RH just a correction of the dew-point proxy?

Classical baselines, fitted on 2006 (FLUXNET RH, TA 62 m, QC = 0), tested on 2005, 2008, 2009, 2010:
  T0: Tetens, Td = a*Tmin + b                 (paper/03 baseline, 2 constants)
  T1: Tetens, Td = a*Tmin + c*T + b           (dew point drifts with T within the day, 3 constants)
Compared with the SINDy quadratic result of EXP-017 (results/clean_rerun_rn_rh.csv), same data and split.
Decision (fixed before running): the SR gain is "a proxy correction" if T1 matches or beats SINDy
(CI of RMSE(SINDy) - RMSE(T1) includes or exceeds zero) in the years where SINDy beat T0.
Usage: python src/rh_fair_baseline.py data/fluxnet_BR-Ji3/AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "sr_benchmark"))
from real_discovery import _es  # noqa: E402
from methods import fit_sindy_poly  # noqa: E402
from clean_rerun_rn_rh import rmse, r2, boot_diff  # noqa: E402

f = pd.read_csv(sys.argv[1], na_values=[-9999], usecols=["TIMESTAMP_END", "TA_F_MDS", "TA_F_MDS_QC", "RH"])
f["timestamp"] = pd.to_datetime(f.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M")
f["T"] = f.TA_F_MDS.where(f.TA_F_MDS_QC == 0)
f["date"] = (f.timestamp - pd.Timedelta("30min")).dt.date
full = f.groupby("date")["T"].transform("count") == 48
f["Tmin_day"] = f.groupby("date")["T"].transform("min").where(full)
f["year"] = f.timestamp.dt.year
f = f.dropna(subset=["RH", "T", "Tmin_day"])
T0 = lambda X, a, b: 100 * _es(a * X[:, 1] + b) / _es(X[:, 0])            # noqa: E731
T1 = lambda X, a, c, b: 100 * _es(a * X[:, 1] + c * X[:, 0] + b) / _es(X[:, 0])  # noqa: E731
sub = f[f.year == 2006].sample(3000, random_state=0)
Xtr, ytr = sub[["T", "Tmin_day"]].to_numpy(), sub.RH.to_numpy()
p0, _ = curve_fit(T0, Xtr, ytr, p0=[1, -1]); p1, _ = curve_fit(T1, Xtr, ytr, p0=[1, 0, -1])
sindy = fit_sindy_poly(Xtr, ytr, ["T", "Tmin_day"])
print("T0", p0.round(4), " T1", p1.round(4))
rows = []
for y in (2005, 2008, 2009, 2010):
    te = f[f.year == y]; X = te[["T", "Tmin_day"]].to_numpy(); yy = te.RH.to_numpy(); d = te.date.to_numpy()
    a, b, s = T0(X, *p0), T1(X, *p1), sindy.predict(X)
    lo, hi = boot_diff(yy, s, b, d)
    lo1, hi1 = boot_diff(yy, b, a, d)
    rows.append(dict(year=y, rmse_T0=rmse(yy, a), rmse_T1=rmse(yy, b), rmse_sindy=rmse(yy, s), r2_T1=r2(yy, b),
                     T1_minus_T0_lo=lo1, T1_minus_T0_hi=hi1, sindy_minus_T1_lo=lo, sindy_minus_T1_hi=hi))
res = pd.DataFrame(rows); print(res.round(3).to_string(index=False))
res.to_csv(HERE / "sr_benchmark" / "results" / "rh_fair_baseline.csv", index=False)
