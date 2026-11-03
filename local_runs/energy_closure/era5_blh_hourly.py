"""ERA5 boundary-layer height (blh, m), hourly 11-21 UTC (07-17 local, UTC-4), nearest 0.25 deg cell,
for the four energy-closure towers (Sa3 added). Source: ARCO-ERA5 public Zarr on Google Cloud
(ERA5 licence: Copernicus, attribution required). Adapted from article2_saturation/src/era5_blh_fetch.py.
Usage: python era5_blh_hourly.py out.csv"""
import sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, numcodecs, pandas as pd
B = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
SITES = {"Sa3": (-3.018, -54.971), "Sa1": (-2.857, -54.959), "Ma2": (-2.609, -60.209), "Ma3": (-2.391, -59.884)}
YEARS = list(range(2000, 2014)) + [2018]
bl = numcodecs.Blosc(); lat = 90 - 0.25*np.arange(721); lon = 0.25*np.arange(1440)
IDX = {k: (int(np.argmin(abs(lat-la))), int(np.argmin(abs(lon-lo % 360)))) for k, (la, lo) in SITES.items()}
def get(ts):
    t = int((ts - pd.Timestamp("1900-01-01")).total_seconds()//3600)
    for _ in range(4):
        try:
            a = np.frombuffer(bl.decode(urllib.request.urlopen(B+f"boundary_layer_height/{t}.0.0", timeout=90).read()), dtype="<f4").reshape(721, 1440)
            return [(ts, k, float(a[i, j])) for k, (i, j) in IDX.items()]
        except Exception: pass
    return [(ts, k, np.nan) for k in IDX]
if __name__ == "__main__":
    times = [pd.Timestamp(f"{y}-01-01")+pd.Timedelta(days=d, hours=h) for y in YEARS for d in range(366 if y % 4 == 0 else 365) for h in range(11, 22)]
    with ThreadPoolExecutor(48) as ex: rows = [r for c in ex.map(get, times) for r in c]
    pd.DataFrame(rows, columns=["time_utc", "site", "blh"]).to_csv(sys.argv[1], index=False)
    print(len(rows), "rows; missing", sum(np.isnan(r[2]) for r in rows))
