#!/usr/bin/env bash
# 2x2 factorial: MK_UNet_T / ClinicDB / 352x352 / 200 epochs / seed 42, run SEQUENTIALLY.
#
#   A  baseline    --deep_supervision False --ca_min_squeeze 1
#   B  +DS         --deep_supervision True  --ca_min_squeeze 1
#   C  +CA         --deep_supervision False --ca_min_squeeze 4
#   D  +DS +CA     --deep_supervision True  --ca_min_squeeze 4
#
# Sequential on purpose. Each arm peaks near 7 GB, and two concurrent jobs will OOM on
# a 32 GB machine that is also running anything else. Sequential execution also keeps
# the wall-clock timings usable as reported efficiency numbers.
#
# Usage:  bash scripts/run_2x2.sh
#         MKUNET_PY=/path/to/python bash scripts/run_2x2.sh
# Resume: each arm writes <run_id>-resume.pth every epoch; pass it to --resume.

set -u
cd "$(dirname "$0")/.."

PY="${MKUNET_PY:-C:/Users/USER/.conda/envs/mkunetenv/python.exe}"
OUT="experiments/results"
mkdir -p "$OUT"

COMMON="--network MK_UNet_T --dataset ClinicDB --device cpu --epoch 200 --runs 1 --seed 42"

run_arm () {
    local tag="$1"; shift
    local log="$OUT/run_${tag}.log"
    echo "=== [$(date '+%F %T')] starting arm ${tag} -> ${log}"
    "$PY" -W ignore train_polyp.py $COMMON "$@" > "$log" 2>&1
    echo "=== [$(date '+%F %T')] arm ${tag} exited with $?"
}

run_arm A_baseline --deep_supervision False --ca_min_squeeze 1
run_arm B_ds       --deep_supervision True  --ca_min_squeeze 1
run_arm C_ca       --deep_supervision False --ca_min_squeeze 4
run_arm D_ds_ca    --deep_supervision True  --ca_min_squeeze 4

echo "=== [$(date '+%F %T')] all arms complete"
