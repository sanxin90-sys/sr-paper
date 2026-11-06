Pantanal (Npw) as a sixth tower for the temperature closure (EXP-030 / EXP-030b with the corrected VPD, EXP-030c).
Needs raw/Npw.pkl plus the other five .pkl files (made by ../reduce.py from the FLUXNET zips).
Packages: numpy pandas scipy scikit-learn pyoperon gplearn sympy

Run (two jobs, can run in parallel), from a local folder, with VPD_SIM=recalc:
  Windows PowerShell:
    $env:VPD_SIM="recalc"
    python npw\exp030_closure.py RAW_DIR OUT_DIR\c
    python npw\exp030b_more_engines.py RAW_DIR OUT_DIR\b
Outputs: exp021_apriori.csv, exp021_sim.csv (c) and exp021b_*.csv (b). Only the Npw rows are new.
