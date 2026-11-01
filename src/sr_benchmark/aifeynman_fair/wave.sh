#!/usr/bin/env bash
# usage: wave.sh SEED CAP OUT task1 task2 ...
cd "$(dirname "$0")"; s=$1; c=$2; o=$3; shift 3
printf '%s\n' "$@" | xargs -P 4 -I{} python3 run_aif_fair.py --task {} --seed "$s" --cap "$c" --out "$o"
echo WAVE_DONE
