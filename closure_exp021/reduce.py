import sys, zipfile, pandas as pd, re
z, site, out = sys.argv[1:4]
zf = zipfile.ZipFile(z)
name = [n for n in zf.namelist() if 'FLUXMET_HH' in n][0]
hdr = zf.open(name).readline().decode().strip().split(',')
want = re.compile(r'^(TIMESTAMP_START|TIMESTAMP_END|TA_F_MDS|TA_F_MDS_QC|TA_F|SW_IN_F_MDS|SW_IN_F_MDS_QC|SW_IN_F|NETRAD|LW_IN|LW_IN_F|LW_IN_F_MDS|LW_OUT|SW_OUT|G_F_MDS|G_F_MDS_QC|H_F_MDS|H_F_MDS_QC|LE_F_MDS|LE_F_MDS_QC|WS|WS_F|WS_F_QC|P|P_F|RH|VPD_F_MDS|SWC_F_MDS_1|TS_F_MDS_1|PPFD_IN)$')
cols = [c for c in hdr if want.match(c)]
d = pd.read_csv(zf.open(name), usecols=cols, na_values=[-9999])
d.to_pickle(out)
print(site, d.shape, cols)
