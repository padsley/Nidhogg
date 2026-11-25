#!/bin/bash
# Train an ensemble on this desktop: build the memory-mapped cache once, then
# run K members in parallel at T threads each. Benchmarked 2026-11-24 on
# 16 cores: 4 x 4 threads gives ~1.3x the total throughput of 1 x 16
# (2 x 8 is worse). Members differ only in --seed; validation shards are shared.
#   scripts/train_ensemble.sh SHARD_DIR OUT_PREFIX [K=4] [T=4] [EPOCHS=12]
set -e
SHARDS=$1; OUT=$2; K=${3:-4}; T=${4:-4}; EP=${5:-12}
cd "$(dirname "$0")/.."
python3 -W ignore -c "
import sys; sys.path.insert(0,'scripts'); sys.argv=['x']
from train import load_cached; load_cached('$SHARDS', 0.05)
print('cache ready')"
for s in $(seq 0 $((K-1))); do
  mkdir -p ${OUT}_m$s
  OMP_NUM_THREADS=$T nohup nice -n 10 python3 -W ignore scripts/train.py "$SHARDS" ${OUT}_m$s --epochs $EP \
      --threads $T --seed $s > ${OUT}_m$s/train.log 2>&1 &
done
echo "launched $K members"
