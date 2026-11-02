# When does symbolic regression add to tower micrometeorology?

Code, result tables and figure scripts for the article:

> \[Authors\] (2026). *When does symbolic regression add to tower micrometeorology? Eight equation-discovery methods tested on known laws and on an Amazonian flux-tower record.* Submitted to *Environmental Modelling & Software*.

**In one sentence:** eight equation-discovery engines (PySR, Operon, gplearn, SINDy, DSO, PhySO, AI Feynman, KAN) and the AI-Lorenz pipeline are tested on ten known micrometeorological laws, two dynamical systems and an audited flux-tower record from Rebio Jaru (Rondônia, Brazil).

---

## Start here

Pick the level you need. Each one is independent.

| Level | What you get | Time | Needs data? |
|---|---|---|---|
| **1. Check the numbers** | Every table and figure rebuilt from the saved results | ~5 min | No |
| **2. Check the data audit** | The DQ-001 to DQ-003 diagnostics (Table S2) | ~5 min | Yes |
| **3. Rerun the engines** | New result files from scratch | hours | For the real-data part |

Most readers only need **Level 1**.

---

## Level 1: rebuild the tables and figures

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-minimal.txt

cd src/sr_benchmark
python ../../analysis/consolidate_benchmark.py results results_seeds > ../../analysis/out/tables.md
python ../../analysis/make_figures.py results
python ../../analysis/check_realdata.py results
```

You should see:

- [ ] `analysis/out/tables.md`: the static suite (Table 2) and the dynamic suite (Table S3).
- [ ] `analysis/out/fig1_static`, `fig2_protocol` and `fig3_realdata` (`.png` and `.pdf`), plus `graphical_abstract.png`.
- [ ] Printed on screen: Table 3 (wins per split and engine on the tower data).

No engine is run at this level. The scripts only read the CSV files in `src/sr_benchmark/results/` and `results_seeds/`.

---

## Level 2: check the data audit

1. Get the two data files (see [Data](#data)).
2. Run:

```bash
cd src
python audit_jaru.py "path/to/Dados Rebio Jaru 2006 a 2008.xlsx" \
    path/to/AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv
```

The output is saved for reference in `src/sr_benchmark/results/audit_jaru.log`. The script only reports. It changes no value; the analysis scripts drop the flagged months themselves.

---

## Level 3: rerun the engines

Install everything with `pip install -r requirements.txt`. Neural engines (KAN, PhySO, the AI Feynman network) need PyTorch, and PySR installs Julia on first use. A separate environment for those is easier.

| What | Command (from `src/sr_benchmark/` unless noted) | Time |
|---|---|---|
| Operon, gplearn, SINDy, 3 seeds | `bash run_seeds.sh` | ~2 h |
| PySR, 3 seeds (Windows) | `run_pysr_seeds.bat` | ~4 h |
| PySR, one-click local run | `python run_local.py` (`--quick` to test) | ~1 h |
| One engine, one suite | `python run_benchmark.py --suite static --methods operon --noise 0 0.05 --budget 45 --seed 0 --out results/my_run.csv` | minutes |
| AI Feynman, complete run | `bash setup_aifeynman.sh`, then `aifeynman_fair/wave.sh SEED CAP OUT task1 task2 ...` | up to 2 h per law |
| DSO (Python 3.8, TF 2.13) | see `dso_tf2/setup_dso_tf2.sh` | ~4 min per law |
| KAN v1 and v2 | `cd ..` then `python kan_eval.py static --seeds 0 1 2 --out sr_benchmark/results/kan_static.csv` | ~2 h |
| Audited tower tests: SINDy, Operon | `cd ..` then `python clean_rerun_rn_rh.py XLSX FLUXNET_CSV` | ~30 min |
| Audited tower tests: gplearn, extended SINDy | `cd ..` then `python clean_rerun_extra.py XLSX FLUXNET_CSV` | ~30 min |
| Audited tower tests: PySR | `cd ..` then `python pysr_clean_rn_rh.py XLSX FLUXNET_CSV` | ~40 min |

Every real-data script states its decision rule in its docstring. The rule was written before the script was run. Other options are in each script's docstring (`python script.py --help` where available).

---

## Where each result comes from

Table and figure numbers refer to the submitted manuscript.

| Manuscript item | Script | Saved results |
|---|---|---|
| Table 2, Fig. 1 (static benchmark) | `sr_benchmark/run_benchmark.py`, `run_seeds.sh`, `run_pysr_seeds.bat` | `results/static*.csv`, `results_seeds/` |
| Table S3 (dynamic benchmark, derivative estimators) | same, `--suite dynamic`; `sr_benchmark/derivatives.py` | `results/dynamic*.csv`, `results_seeds/` |
| Fig. 2, Table S4 (protocol variants) | `kan_eval.py`, `aifeynman_fair/run_aif_fair.py`, `sr_benchmark/dso_check.py`, `local_runs/physo_pilot/physo_units.py`, `pysr_pareto_seeds.py` | `kan_static.csv`, `aifeynman_fair/*.csv`, `static_dso.csv`, `local_runs/physo_pilot/synthetic_units_pilot.csv`, `static_pysr_b45_seed0.csv`, `pysr_pareto_seeds.csv` |
| Table S2 (data audit DQ-001 to DQ-003) | `audit_jaru.py` | `audit_jaru.log`, `rn_par_by_year.log` |
| Table 3, Fig. 3 (audited Rn and RH tests) | `clean_rerun_rn_rh.py`, `clean_rerun_extra.py`, `pysr_clean_rn_rh.py`, `rh_fair_baseline.py` | `clean_rerun_*.csv`, `pysr_clean_rn_rh_*`, `rh_fair_baseline.csv` |
| Section 3.3, "the audit as part of the result" (pre-audit runs) | `real_discovery.py`, `notebooks/jaru_sr_pipeline.ipynb`, `kan_eval.py`, `local_runs/physo_pilot/physo_jaru.py` | `real_discovery_*.csv`, `kan_real*.csv`, `local_runs/physo_pilot/jaru_physo.csv` |
| Section 3.4, tendency: diurnal baseline, weak form, filter windows | `sindy_jaru_check.py`, `sindy_weak_loyo.py`, `dtdt_robustness.py`, `dtdt_null_sg.py` | `sindy_jaru_*`, `sindy_weak_loyo*`, `dtdt_robustness*.csv`, `dtdt_null_sg.csv` |
| Section 3.4, AI Feynman separability | `aif_separability.py` | `aif_separability_*` |
| Section 3.4, AI-Lorenz gray box (E1, E2b, E3) | `ailorenz/e1_*.py`, `e2b_*.py`, `e3_*.py`, `run_pysr_device.py`, `score_pysr.py` | `ailorenz_*.csv`, `ailorenz/pysr_outputs/`, `local_runs/ailorenz_pysr/` |
| Section S3, KAN on tower data | `kan_eval.py` | `kan_real_no2007*.csv` |

`results/` means `src/sr_benchmark/results/`. Scripts are in `src/` unless a folder is given.

---

## Folder map

```
src/
  sr_benchmark/        synthetic benchmark: tasks, engine wrappers, runners, results
    tasks.py           the 10 laws and 2 dynamical systems (ground truth, ranges)
    methods.py         one wrapper per engine, same interface for all
    derivatives.py     finite differences, Savitzky-Golay, X-TFC
    results/           every result table used in the paper
    results_seeds/     extra seeds (1 and 2) for the static and dynamic suites
    aifeynman_fair/    complete AI Feynman run + 3 patches to upstream commit a05bc4a
    dso_tf2/           TensorFlow 2 patch for DSO
  ailorenz/            AI-Lorenz port (X-TFC) and gray-box experiments E1-E3
  audit_jaru.py        data audit DQ-001 to DQ-003
  clean_rerun_*.py     audited real-data tests (Table 3)
  real_discovery.py    data loader for the station spreadsheet; pre-audit runs
  ...                  other scripts, see the table above
analysis/              rebuild tables and figures from saved results (Level 1)
local_runs/            runs made outside the main pipeline (PhySO pilot, AI-Lorenz PySR log)
notebooks/             notebook used for the PySR runs on a laptop
```

---

## Data

The data are **not** in this repository. You need them only for Level 2 and for the real-data part of Level 3.

| File | Where to get it |
|---|---|
| `AMF_BR-Ji3_FLUXNET_FLUXMET_HH_2004-2010_v1.3_r1.csv` | AmeriFlux FLUXNET, site BR-Ji3: https://doi.org/10.17190/AMF/3027314 (AmeriFlux data policy) |
| `Dados Rebio Jaru 2006 a 2008.xlsx` | Rebio Jaru station spreadsheet. \[Availability to be stated by the authors.\] |

The synthetic benchmark needs no external data. It regenerates its inputs from fixed seeds.

---

## Known gaps

Please read these before you compare numbers.

- **PySR on the audited tower data** was run on a laptop (`pysr_clean_rn_rh.py`). Its outputs are in `results/pysr_clean_rn_rh_*` and are included in `check_realdata.py` and Fig. 3. For humidity they are compared with baseline T0, the Tetens relation used in Table 3.
- **PhySO on tower data** (`local_runs/physo_pilot/physo_jaru.py`) read a prepared array file (`jaru.npz`) that is not included. These runs used the pre-audit window and are not reported as evidence in the paper.
- **Seeds and budgets differ between engines.** AI Feynman, DSO and PhySO were run with one seed. PySR ran on an Intel i7-9750H laptop and the other engines in a 4-core Linux container, so timings compare within a machine only.
- **KAN v2 and PhySO** have options in `clean_rerun_extra.py` but were not run on the audited splits.
- **Shared scripts.** `sindy_weak_loyo.py`, `sindy_exp008.py`, `dtdt_robustness.py` and `real_discovery.py` are also used in a companion study of the air-temperature tendency. Here they are used only for the method checks in Section 3.4 and as data loaders.

---

## Citing

Use the "Cite this repository" button on GitHub, or the Zenodo DOI of the release you used. The metadata are in `CITATION.cff` and `.zenodo.json`.

## Licence

MIT (see `LICENSE`). The patches in `src/sr_benchmark/aifeynman_fair/` and `src/sr_benchmark/dso_tf2/` modify third-party code that keeps its upstream licence.
