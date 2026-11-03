import pandas as pd, numpy as np, glob, sys, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
rng=np.random.default_rng(0)
def r2(y,p): return 1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
rows=[];imp=[]
for s in ['Sa3','Sa1','Ma2','Ma3']:
    f=glob.glob(f'{sys.argv[1]}/AMF_BR-{s}_*FLUXMET_HH*.csv')[0]
    d=pd.read_csv(f,na_values=[-9999])
    t=pd.to_datetime(d.TIMESTAMP_START.astype(str),format='%Y%m%d%H%M')
    d['dTA']=(d.TA_F.shift(-1)-d.TA_F.shift(1))  # K per hour (2 half-hours)
    d['dVPD']=(d.VPD_F.shift(-1)-d.VPD_F.shift(1))
    hr=t.dt.hour+t.dt.minute/60
    for k in (1,2,3):
        d[f'c{k}']=np.cos(2*np.pi*k*hr/24); d[f's{k}']=np.sin(2*np.pi*k*hr/24)
    G=d.G_F_MDS.where(d.G_F_MDS_QC==0) if s=='Sa3' else 0*d.NETRAD
    d['eps']=d.NETRAD-G-d.H_F_MDS-d.LE_F_MDS
    d['noise']=d.H_RANDUNC**2+d.LE_RANDUNC**2
    d['year']=t.dt.year; d['day']=t.dt.floor('D'); d['hr']=hr
    X=['NETRAD','SW_IN_F','USTAR','WS_F','TA_F','VPD_F','dTA','dVPD']+[f'{a}{k}' for k in (1,2,3) for a in 'cs']
    m=(d.H_F_MDS_QC==0)&(d.LE_F_MDS_QC==0)&(d.NETRAD>50)&d.eps.notna()&d[X].notna().all(1)
    D=d[m].copy()
    yrs=[y for y,n in D.year.value_counts().items() if n>=500]
    D=D[D.year.isin(yrs)]
    P={k:np.full(len(D),np.nan) for k in ['M0','M1','M2']}
    for y in sorted(yrs):
        tr=(D.year!=y).values; te=~tr
        P['M0'][te]=LinearRegression().fit(D.loc[tr,['NETRAD']],D.eps[tr]).predict(D.loc[te,['NETRAD']])
        P['M1'][te]=LinearRegression().fit(D.loc[tr,X],D.eps[tr]).predict(D.loc[te,X])
        g=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=50,random_state=0).fit(D.loc[tr,X],D.eps[tr])
        P['M2'][te]=g.predict(D.loc[te,X])
    y=D.eps.values
    ceil=1-np.nanmean(D.noise)/np.var(y)
    # day-block bootstrap of R2 differences
    days=D.day.values; ud=np.unique(days); idx={u:np.where(days==u)[0] for u in ud}
    bs=[]
    for b in range(500):
        ii=np.concatenate([idx[u] for u in rng.choice(ud,len(ud))])
        bs.append([r2(y[ii],P['M2'][ii])-r2(y[ii],P['M0'][ii]), r2(y[ii],P['M1'][ii])-r2(y[ii],P['M0'][ii])])
    bs=np.array(bs)
    mo=D[(D.hr>=8)&(D.hr<10)]; af=D[(D.hr>=14)&(D.hr<16)]
    bins=[50,300,500,700,1000]
    mob=mo.groupby(pd.cut(mo.NETRAD,bins)).eps.mean(); afb=af.groupby(pd.cut(af.NETRAD,bins)).eps.mean()
    rows.append(dict(site=s,n=len(D),years=len(yrs),var_eps=np.var(y).round(0),R2_M0=r2(y,P['M0']),R2_M1=r2(y,P['M1']),R2_M2=r2(y,P['M2']),
        d21=r2(y,P['M2'])-r2(y,P['M0']),d21_lo=np.percentile(bs[:,0],2.5),d21_hi=np.percentile(bs[:,0],97.5),
        d10=r2(y,P['M1'])-r2(y,P['M0']),d10_lo=np.percentile(bs[:,1],2.5),d10_hi=np.percentile(bs[:,1],97.5),ceiling=ceil,
        morn_minus_aft=';'.join(f'{a:.0f}' for a in (mob-afb).values)))
    # permutation importance on pooled test predictions of a model fitted on all years (descriptive)
    g=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,min_samples_leaf=50,random_state=0).fit(D[X],y)
    base=r2(y,g.predict(D[X]))
    groups={'Rn/SW':['NETRAD','SW_IN_F'],'ustar':['USTAR'],'WS':['WS_F'],'TA/VPD':['TA_F','VPD_F'],'storage(dTA,dVPD)':['dTA','dVPD'],'hour':[c for c in X if c[0] in 'cs' and c[1].isdigit()]}
    for gname,cols in groups.items():
        Z=D[X].copy(); perm=rng.permutation(len(Z)); Z[cols]=Z[cols].values[perm]
        imp.append(dict(site=s,group=gname,drop=base-r2(y,g.predict(Z))))
R=pd.DataFrame(rows); pd.set_option('display.width',250); print(R.round(3).to_string())
I=pd.DataFrame(imp).pivot(index='group',columns='site',values='drop'); print(I.round(3))
R.to_csv(sys.argv[2]+'/exp021.csv',index=False); I.to_csv(sys.argv[2]+'/exp021_importance.csv')
