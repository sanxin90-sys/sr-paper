#!/usr/bin/env bash
# Builds AI Feynman's Fortran kernels and Windows launchers. Run in Git Bash: bash build_aif.sh > build_aif.log 2>&1
E=/c/Users/ianba/Downloads/sr-engines; export PATH="$E/aifenv/Scripts:$E/mingw/mingw64/bin:$PATH" CC=gcc FC=gfortran
PY=$E/aifenv/Scripts/python.exe; cd $E/aif-src/aifeynman || exit 1; rm -f _symbolic*.pyd; cp $E/mingw/mingw64/bin/libgfortran-5.dll $E/mingw/mingw64/bin/libquadmath-0.dll $E/mingw/mingw64/bin/libgcc_s_seh-1.dll $E/mingw/mingw64/bin/libwinpthread-1.dll .
$PY -c "import numpy;print('numpy',numpy.__version__)"
for m in symbolic_regress1 symbolic_regress2 symbolic_regress3 symbolic_regress_mdl2 symbolic_regress_mdl3 symbolic_regress_mdl4 symbolic_regress_mdl5; do
  echo "=== $m"; ls _$m*.pyd >/dev/null 2>&1 || $PY -m numpy.f2py -c $m.f90 -m _$m -I"$(pwd)" --backend meson 2>&1 | tail -n 25; ls _$m*.pyd || echo "FAILED $m"
done
mkdir -p $E/aif-src/bin; cd $E/aif-src/bin
for pair in sr1:symbolic_regress1 sr2:symbolic_regress2 sr3:symbolic_regress3 sr_mdl_plus:symbolic_regress_mdl2 sr_mdl_mult:symbolic_regress_mdl3 sr_mdl4:symbolic_regress_mdl4 sr_mdl5:symbolic_regress_mdl5; do
  n=feynman_${pair%%:*}; gcc -O1 -o $n.exe $E/feynman_launcher.c || echo "FAILED launcher $n"
  { echo 'import sys'; echo 'sys.path.insert(0, "C:/Users/ianba/Downloads/sr-engines/aif-src")'; echo 'import importlib'; echo "importlib.import_module('aifeynman._${pair##*:}').go()"; } > $n.py
done
for m in symbolic_regress1 symbolic_regress2 symbolic_regress3 symbolic_regress_mdl2 symbolic_regress_mdl3 symbolic_regress_mdl4 symbolic_regress_mdl5; do AIFEYNMAN_HOME=$E/aif-src $PY -c "import sys;sys.path.insert(0,\"C:/Users/ianba/Downloads/sr-engines/aif-src\");import importlib;m=importlib.import_module(\"aifeynman._$m\");print(\"import ok $m\",hasattr(m,\"go\"))" 2>&1 | tail -n 1; done
echo BUILD_END
