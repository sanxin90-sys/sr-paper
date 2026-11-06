"""Baselines on the same splits as exp022b_pysr.py (run from local_runs/energy_closure)."""
import sys, numpy as np, pandas as pd
sys.argv = [sys.argv[0], 'fluxnet', 'data/blh_hourly.csv', '.']
src = open('exp022.py').read(); src = src[:src.index('rows, exprs, preds')]
ns = {}; exec(src, ns)
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
r2, deroo, F = ns['r2'], ns['deroo'], ns['FEAT']
out = []
for s in ns['SITES']:
    tr, te = ns['load'](s); eps, AE = te.eps.values, te.AE.values
    last = max(ns['SITES'][s][3]); itr = tr[tr.year != last].sample(frac=1, random_state=0).iloc[:4000]
    X = lambda d: [d.ustar_wstar.values, d.z_zi.values]
    row = dict(site=s, n_test=len(te))
    row['B0_mean_eps'] = r2(eps, np.full(len(te), tr.eps.mean()))
    row['B0_mean_I'] = r2(eps, tr.I.mean()*AE)
    row['B1_deroo'] = r2(eps, deroo(X(te), 0.197, -17.0, 0.156, 0.21, 10.69)*AE)
    try:
        p, _ = curve_fit(deroo, X(tr), tr.I.values, p0=[0.2, -10, 0.1, 1, 1], maxfev=20000)
        row['B2_deroo_fit'] = r2(eps, deroo(X(te), *p)*AE)
    except Exception: row['B2_deroo_fit'] = np.nan
    row['LIN_eps_4k'] = r2(eps, LinearRegression().fit(itr[F], itr.eps).predict(te[F]))
    row['GBM_eps_4k'] = r2(eps, HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0).fit(itr[F], itr.eps).predict(te[F]))
    row['GBM_eps_all'] = r2(eps, HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0).fit(tr[F], tr.eps).predict(te[F]))
    out.append(row); print(row, flush=True)
pd.DataFrame(out).round(3).to_csv(sys.argv[0].replace('.py', '.csv'), index=False)
print(pd.DataFrame(out).round(3).to_string())
