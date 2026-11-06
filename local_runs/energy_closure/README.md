# Energy-balance closure (EXP-021/022/022b)
Run PySR locally (EXP-022b):
1. Unzip each `AMF_BR-{Sa3,Sa1,Ma2,Ma3}_FLUXNET_*_v1.3_r1.zip` FLUXMET_HH csv into `fluxnet/` here.
2. `pip install pysr scikit-learn scipy pandas numcodecs`
3. `python exp022b_pysr.py fluxnet` -> `exp022b_pysr.csv`
`data/blh_hourly.csv`: ERA5 boundary-layer height (Copernicus licence, attribution required), from `era5_blh_hourly.py`.
