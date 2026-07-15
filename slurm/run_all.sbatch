#!/bin/bash
#SBATCH --job-name=rnnphase
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_%j.out
# Rockefeller HPC -- adjust partition/account to your allocation.

module load anaconda3 2>/dev/null || true
source activate rnnphase 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "=== known-answer regression test (must pass first) ==="
python tests/test_known_answer.py || { echo "KNOWN-ANSWER FAILED -- aborting"; exit 1; }

echo "=== exp1: structure prediction (full seed x arch grid) ==="
for task in memory accumulation gated oscillation; do
  python experiments/exp1_structure_prediction.py --task $task \
      --seeds 50 --archs rnn gru lstm --N 128 --iters 3000 \
      --out results/exp1_${task}.json
done

echo "=== exp2: frequency from angle ==="
python experiments/exp2_frequency_from_angle.py \
    --freqs 0.05 0.08 0.10 0.12 0.15 0.20 --seeds 30 --iters 3000 \
    --out results/exp2_frequency.json

echo "=== exp3: capacity bound ==="
python experiments/exp3_capacity_bound.py \
    --sizes 1 2 3 4 6 8 12 16 24 32 48 64 --seeds 30 --iters 3000 \
    --out results/exp3_capacity.json

echo "=== exp4: structural x-ray ==="
python experiments/exp4_xray.py --seeds 50 --N 64 --iters 3000 \
    --out results/exp4_xray.json

echo "=== exp5: causal perturbation (targeted vs control dissociation) ==="
python experiments/exp5_perturbation.py \
    --tasks accumulation memory gated --seeds 50 --N 128 --iters 3000 \
    --strengths 0.0 0.25 0.5 0.75 1.0 \
    --out results/exp5_perturbation.json

echo "=== done ==="
