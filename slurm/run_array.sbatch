#!/bin/bash
#SBATCH --job-name=rnnphase_arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=results/arr_%A_%a.out
# One task per array index -- parallelize the structure grid across GPUs.

module load anaconda3 2>/dev/null || true
source activate rnnphase 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"; mkdir -p results

TASKS=(memory accumulation gated oscillation)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
python experiments/exp1_structure_prediction.py --task $T \
    --seeds 50 --archs rnn gru lstm --N 128 --iters 3000 \
    --out results/exp1_${T}.json
