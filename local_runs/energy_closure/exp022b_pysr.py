"""EXP-022b: PySR on the energy-balance gap eps (W m-2), same data/splits as EXP-022.
Run from this folder: python exp022b_pysr.py <fluxnet_dir_with_unzipped_FLUXMET_HH_csvs>
Needs: pip install pysr scikit-learn scipy pandas (first run installs Julia)."""
import sys, numpy as np, pandas as pd
sys.argv = [sys.argv[0], sys.argv[1], 'data/blh_hourly.csv', '.']
import importlib.util
spec = importlib.util.spec_from_file_location('e', 'exp022.py'); src = open('exp022.py').read()
src = src[:src.index('rows, exprs, preds')]  # reuse load(), FEAT, SITES only
ns = {}; exec(src, ns)
from pysr import PySRRegressor
def r2(y, p): return 1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
import os
OUTF = 'exp022b_pysr.csv'
out = pd.read_csv(OUTF).to_dict('records') if os.path.exists(OUTF) else []
done = {(r['site'], r['seed']) for r in out}
for s in ns['SITES']:
    tr, te = ns['load'](s); F = ns['FEAT']
    last = max(ns['SITES'][s][3]); itr = tr[tr.year != last].sample(frac=1, random_state=0).iloc[:4000]; iva = tr[tr.year == last]
    for seed in range(3):
        if (s, seed) in done: continue
        m = PySRRegressor(niterations=100, binary_operators=['+', '-', '*', '/'], unary_operators=['exp', 'log'],
                          maxsize=25, populations=30, random_state=seed, deterministic=True, parallelism='serial',
                          timeout_in_seconds=900, procs=4, progress=False)
        m.fit(itr[F].values, itr.eps.values, variable_names=F)
        eq = m.equations_
        sc = [r2(iva.eps.values, m.predict(iva[F].values, index=i)) for i in eq.index]
        b = int(np.nanargmax(sc)); p = m.predict(te[F].values, index=b)
        out.append(dict(site=s, seed=seed, complexity=eq.complexity[b], r2_val=sc[b], r2_test=r2(te.eps.values, p), expr=eq.equation[b]))
        print(out[-1], flush=True)
        pd.DataFrame(out).to_csv(OUTF, index=False)
