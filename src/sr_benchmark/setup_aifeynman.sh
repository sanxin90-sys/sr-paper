#!/usr/bin/env bash
# Fetches AI Feynman 2.0 (MIT licence) and compiles its Fortran brute-force
# kernels with f2py/meson. The upstream setup.py relies on numpy.distutils,
# which no longer exists in NumPy >= 2 / Python >= 3.12, so it is bypassed.
# Requirements: git, gfortran, python packages numpy meson ninja torch sympy sortedcontainers seaborn
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${AIFEYNMAN_HOME:-$HERE/_aifeynman}"
[ -d "$DEST/.git" ] || git clone --depth 1 https://github.com/SJ001/AI-Feynman.git "$DEST"
cd "$DEST/aifeynman"
for m in symbolic_regress1 symbolic_regress2 symbolic_regress3 symbolic_regress_mdl2 \
         symbolic_regress_mdl3 symbolic_regress_mdl4 symbolic_regress_mdl5; do
  ls _"$m"*.so >/dev/null 2>&1 || python3 -m numpy.f2py -c "$m.f90" -m "_$m" -I"$(pwd)" >/dev/null 2>&1
done
# Console entry points that the brute-force stage calls through subprocess
mkdir -p "$DEST/bin"
for pair in sr1:symbolic_regress1 sr2:symbolic_regress2 sr3:symbolic_regress3 \
            sr_mdl_plus:symbolic_regress_mdl2 sr_mdl_mult:symbolic_regress_mdl3 \
            sr_mdl4:symbolic_regress_mdl4 sr_mdl5:symbolic_regress_mdl5; do
  name="feynman_${pair%%:*}"; mod="_${pair##*:}"
  printf '#!/usr/bin/env python3\nimport sys\nsys.path.insert(0, "%s")\nfrom aifeynman.%s import go\ngo()\n' "$DEST" "$mod" > "$DEST/bin/$name"
  chmod +x "$DEST/bin/$name"
done
echo "AI Feynman ready in $DEST (add $DEST/bin to PATH)"
