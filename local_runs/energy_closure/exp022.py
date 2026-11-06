"""EXP-022: symbolic regression vs the De Roo/Wanner form for the energy-balance imbalance ratio.
Usage: python exp022.py <fluxnet_dir> <blh_hourly.csv> <out_dir> [sr_seconds]"""
import sys, glob, warnings
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
FD, BLH, OUT = sys.argv[1:4]; SRT = int(sys.argv[4]) if len(sys.argv) > 4 else 60
SITES = {  # utc offset (h), z (m), d (m), train years, test years
    'Sa3': (-3, 64, 18, [2001, 2002], [2000, 2003]),
    'Sa1': (-3, 63, 27, [2002, 2003, 2004], [2009, 2010, 2011]),
    'Ma2': (-4, 50, 23, [2002, 2003, 2004, 2005], [2006, 2007, 2008, 2009]),
    'Ma3': (-4, 10, 0, [2009, 2011, 2012], [2013, 2018])}
FEAT = ['ustar_wstar', 'z_zi', 'USTAR', 'NETRAD', 'SW_IN_F', 'WS_F', 'TA_F', 'VPD_F', 'dTA', 'hour']
rng = np.random.default_rng(0)
blh = pd.read_csv(BLH, parse_dates=['time_utc'])
def r2(y, p): return 1 - np.sum((y-p)**2)/np.sum((y-y.mean())**2)
def deroo(X, a, b, c, q, m): return (a*np.exp(b*X[0]) + c)*(q + m*X[1])

def load(s):
    off, z, d0, tr, te = SITES[s]
    d = pd.read_csv(glob.glob(f'{FD}/AMF_BR-{s}_*FLUXMET_HH*.csv')[0], na_values=[-9999])
    t = pd.to_datetime(d.TIMESTAMP_START.astype(str), format='%Y%m%d%H%M') + pd.Timedelta(minutes=15)
    d['hour'] = t.dt.hour + t.dt.minute/60
    tutc = t - pd.Timedelta(hours=off)
    b = blh[blh.site == s].set_index('time_utc').blh.sort_index()
    bt = b.index.values.astype('datetime64[ns]').astype(np.int64)
    x = tutc.values.astype('datetime64[ns]').astype(np.int64)
    zi = np.interp(x, bt, b.values, left=np.nan, right=np.nan)
    # only interpolate inside the same UTC day's fetched window (11-21 UTC)
    hu = tutc.dt.hour + tutc.dt.minute/60
    zi[(hu < 11) | (hu > 21)] = np.nan
    d['zi'] = zi
    d['dTA'] = d.TA_F.shift(-1) - d.TA_F.shift(1)
    G = d.G_F_MDS.where(d.G_F_MDS_QC == 0) if s == 'Sa3' else 0*d.NETRAD
    d['AE'] = d.NETRAD - G
    d['eps'] = d.AE - d.H_F_MDS - d.LE_F_MDS
    d['I'] = d.eps/d.AE
    Tk = d.TA_F + 273.15
    d['wstar'] = (9.81/Tk*d.zi*d.H_F_MDS/1200).clip(lower=0)**(1/3)
    d['ustar_wstar'] = d.USTAR/d.wstar
    d['z_zi'] = (z - d0)/d.zi
    d['year'] = t.dt.year; d['day'] = t.dt.floor('D')
    m = ((d.H_F_MDS_QC == 0) & (d.LE_F_MDS_QC == 0) & (d.NETRAD > 50) & (d.H_F_MDS > 0) & (d.hour >= 7) & (d.hour < 17)
         & d.AE.notna() & d[FEAT].notna().all(1) & np.isfinite(d.ustar_wstar))
    D = d[m]
    return D[D.year.isin(tr)], D[D.year.isin(te)]

def fit_sr(Xtr, ytr, seed, names):
    from pyoperon.sklearn import SymbolicRegressor
    reg = SymbolicRegressor(allowed_symbols="add,sub,mul,div,exp,log,constant,variable", max_length=25, max_depth=8,
                            population_size=1000, generations=100000, max_evaluations=10**8, optimizer_iterations=10,
                            objectives=['r2', 'length'], max_time=SRT, n_threads=4, random_state=seed)
    reg.fit(np.ascontiguousarray(Xtr, float), np.ascontiguousarray(ytr, float))
    return reg

rows, exprs, preds = [], [], []
for s in SITES:
    tr, te = load(s)
    ytr, yte = tr.I.values, te.I.values
    AEte, eps = te.AE.values, te.eps.values
    P = {}
    P['B0'] = np.full(len(te), ytr.mean())
    P['B1'] = deroo([te.ustar_wstar.values, te.z_zi.values], 0.197, -17.0, 0.156, 0.21, 10.69)
    try:
        popt, _ = curve_fit(deroo, [tr.ustar_wstar.values, tr.z_zi.values], ytr, p0=[0.2, -10, 0.1, 1, 1], maxfev=20000)
    except Exception as e:
        popt = [np.nan]*5; print(s, 'B2 fit failed', e)
    P['B2'] = deroo([te.ustar_wstar.values, te.z_zi.values], *popt)
    g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0).fit(tr[FEAT], ytr)
    P['GBM'] = g.predict(te[FEAT])
    # SR: inner validation = last training year
    last = max(SITES[s][3]); itr = tr[tr.year != last]; iva = tr[tr.year == last]
    sub = itr.sample(min(len(itr), 6000), random_state=0)
    srp = []
    for seed in range(5):
        reg = fit_sr(sub[FEAT].values, sub.I.values, seed, FEAT)
        best = None
        for c in reg.pareto_front_:
            pv = reg.evaluate_model(c['tree'], np.ascontiguousarray(iva[FEAT].values, float))
            sc = r2(iva.I.values, pv)
            if np.isfinite(sc) and (best is None or sc > best[0]): best = (sc, c)
        pt = reg.evaluate_model(best[1]['tree'], np.ascontiguousarray(te[FEAT].values, float))
        srp.append(pt)
        ex = reg.get_model_string(best[1]['tree'], 4, FEAT)
        exprs.append(dict(site=s, seed=seed, length=best[1]['length'], r2_innerval=best[0], r2_test_eps=r2(eps, pt*AEte), expr=ex))
        print(s, seed, round(best[0], 3), round(r2(eps, pt*AEte), 3), ex, flush=True)
    srp = np.array(srp)
    sr_scores = [r2(eps, p*AEte) for p in srp]
    med = int(np.argsort(sr_scores)[len(sr_scores)//2]); P['SR'] = srp[med]
    E = {k: v*AEte for k, v in P.items()}
    days = te.day.values; ud = np.unique(days); idx = {u: np.where(days == u)[0] for u in ud}
    bs = {k: [] for k in ['B2-B0', 'SR-B2', 'SR-B0', 'GBM-B2']}
    for _ in range(1000):
        ii = np.concatenate([idx[u] for u in rng.choice(ud, len(ud))])
        for k in bs:
            a, b = k.split('-'); bs[k].append(r2(eps[ii], E[a][ii]) - r2(eps[ii], E[b][ii]))
    row = dict(site=s, n_train=len(tr), n_test=len(te), med_ustar_wstar=te.ustar_wstar.median(), med_z_zi=te.z_zi.median(),
               I_mean_test=yte.mean(), B2_params=';'.join(f'{p:.3g}' for p in popt))
    for k in P: row[f'R2_{k}'] = r2(eps, E[k]); row[f'bias_{k}'] = np.mean(E[k] - eps)
    for k, v in bs.items():
        a, b = k.split('-'); row[f'd_{k}'] = row[f'R2_{a}'] - row[f'R2_{b}']
        row[f'd_{k}_lo'], row[f'd_{k}_hi'] = np.percentile(v, [2.5, 97.5])
    rows.append(row)
    preds.append(pd.DataFrame({'site': s, 'time': te.index, 'eps': eps, **{k: E[k] for k in E}}))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/exp022.csv', index=False)
pd.DataFrame(exprs).to_csv(f'{OUT}/exp022_expressions.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60)
print(R.round(3).T.to_string())
