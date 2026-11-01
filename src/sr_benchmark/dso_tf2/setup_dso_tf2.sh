#!/usr/bin/env bash
# Linux/WSL setup used in the cloud run of 2026-10-27: DSO on Python 3.8 + TensorFlow 2.13
# in tf.compat.v1 mode (the reference install needs Python 3.7 + TF 1.14).
# Run from src/sr_benchmark.  Then: export PYTHONPATH=$PWD/deep-symbolic-optimization/dso
set -e
uv python install 3.8
uv venv -p 3.8 dsoenv
for p in "numpy==1.24.3" "tensorflow-cpu==2.13.1" "numba==0.58.1" "sympy==1.9" pandas scikit-learn click tqdm commentjson PyYAML prettytable pathos scipy openpyxl "cython<3" setuptools; do
  VIRTUAL_ENV=$PWD/dsoenv uv pip install "$p"
done
git clone https://github.com/dso-org/deep-symbolic-optimization.git
cd deep-symbolic-optimization && git checkout 8348d5b08d1eef6170fdbfd222a492ded990ea12 && git apply ../dso_tf2/dso_tf2_compat.patch
cd dso && ../../dsoenv/bin/python setup.py build_ext --inplace   # builds cyfunc (required: the pure-Python fallback is broken)
