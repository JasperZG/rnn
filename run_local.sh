#!/usr/bin/env bash
# Full-scale run on a single local GPU (e.g. RTX 5070). No SLURM.
# Runs everything serially; total time is a few hours on a 5070.
# Scale down --seeds / --iters for a fast dry run first.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
mkdir -p results

DEV=cuda
SEEDS=${SEEDS:-50}
ITERS=${ITERS:-3000}

echo "=========================================================="
echo " STEP 0 -- known-answer regression test (must pass first)"
echo "=========================================================="
python tests/test_known_answer.py || { echo "KNOWN-ANSWER FAILED -- aborting"; exit 1; }

echo "=========================================================="
echo " STEP 1 -- structure prediction (seeds=$SEEDS, rnn+gru+lstm)"
echo "=========================================================="
for task in memory accumulation gated oscillation; do
  python experiments/exp1_structure_prediction.py --task $task \
      --seeds $SEEDS --archs rnn gru lstm --N 128 --iters $ITERS \
      --device $DEV --out results/exp1_${task}.json
done

echo "=========================================================="
echo " STEP 2 -- frequency from eigenvalue angle"
echo "=========================================================="
python experiments/exp2_frequency_from_angle.py \
    --freqs 0.05 0.08 0.10 0.12 0.15 0.20 --seeds $SEEDS --iters $ITERS \
    --device $DEV --out results/exp2_frequency.json

echo "=========================================================="
echo " STEP 3 -- capacity bound"
echo "=========================================================="
python experiments/exp3_capacity_bound.py \
    --sizes 1 2 3 4 6 8 12 16 24 32 48 64 --seeds $SEEDS --iters $ITERS \
    --device $DEV --out results/exp3_capacity.json

echo "=========================================================="
echo " STEP 4 -- structural x-ray (the demonstrable payoff)"
echo "=========================================================="
python experiments/exp4_xray.py --seeds $SEEDS --N 64 --iters $ITERS \
    --device $DEV --out results/exp4_xray.json

echo "=========================================================="
echo " STEP 5 -- build summary + figures"
echo "=========================================================="
python experiments/analyze.py
python experiments/figure_xray.py

echo "=== ALL DONE. Results in results/ ==="
