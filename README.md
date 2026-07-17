# Predicting the Phase-Space Structure of Recurrent Neural Networks

A method that predicts, **from a task alone and before any training**, the
dynamical structure a recurrent network must build to solve it -- then confirms
that structure in the trained network's own dynamics, shows it is causally
responsible for the computation, and turns it into a diagnostic that predicts
where a network will fail.

## The claims this code tests

1. **Structure prediction (categorical).** Each canonical task requires one type
   of dynamical object, predictable a priori: memory -> discrete fixed points;
   accumulation -> line attractor; gated selection -> context-gated fixed points;
   oscillation -> limit cycle. Verified across seeds and architectures.
2. **Frequency from eigenvalue angle (quantitative).** The frequency of an
   emergent oscillation equals the angle of the leading complex Jacobian
   eigenvalue, `f = angle/(2*pi)`. (The eigenvalue *magnitude* does **not**
   cleanly mark the behavioral onset -- only the angle carries the frequency.)
3. **Capacity as a two-sided bound.** The required structure bounds the smallest
   network that can solve the task, stated as a range, not an exact count.
4. **The structural X-ray (payoff).** Read a trained network's line-attractor
   extent from its weights alone, and predict the input regime where it must
   fail -- before those inputs are ever run.

## Honesty notes (what this project does *not* claim)

- No clean integer "count law" for K>=3 memories: trained networks build more,
  messier structure than the task strictly requires. The count is reported as a
  quantity under test, not an assumed law.
- No eigenvalue-crossing marker of the oscillation onset: trained networks
  implement a *soft* switch (amplitude ramps from ~0), not a textbook Hopf
  bifurcation. Only the frequency-from-angle relation survives.
- The X-ray is demonstrated on synthetic networks. The forward structure
  prediction is independently confirmed against real neural recordings in the
  published literature (head-direction ring: Zhang 1996; grid-cell torus: Burak
  & Fiete 2009; hypothalamic line attractor: Nair et al. 2023).

## Layout

```
src/rnnphase/     core library (tasks, models, training, fixed-point analysis,
                  structure classifier, diagnostics, causal perturbation)
experiments/      exp1 structure | exp2 frequency | exp3 capacity | exp4 xray
tests/            known-answer regression test (MUST pass before any result)
slurm/            Rockefeller A100 batch scripts (run_all, run_array)
configs/          full-scale run configuration
results/          output JSON (gitignored except final)
```

## Reproduce

### Local GPU (e.g. RTX 5070 / Blackwell) -- recommended

```bash
bash setup_5070.sh            # venv + CUDA 12.8 PyTorch + GPU sanity check
source .venv/bin/activate
bash run_local.sh             # full run: known-answer test -> 4 experiments -> summary + figures
# fast dry run first:  SEEDS=4 ITERS=500 bash run_local.sh
```

The 5070 has ample headroom for these small networks (N=128). Note it is
Blackwell (sm_120) and needs the CUDA 12.8+ PyTorch wheel that `setup_5070.sh`
installs; the default wheel raises "no kernel image available". The full grid
runs serially in a few hours; scale `SEEDS`/`ITERS` down for a quick check.

### Rockefeller A100 (SLURM)

```bash
pip install -r requirements.txt
python tests/test_known_answer.py          # gate: 2 stable FPs + saddle for 1-bit memory
sbatch slurm/run_all.sbatch                # full grid, one node
sbatch slurm/run_array.sbatch              # or parallelize the structure grid across GPUs
```

Results land in `results/*.json`; `experiments/analyze.py` consolidates them into
`results/summary.json` (the numbers to report), and `figure_xray.py` builds the
payoff figure.

Every experiment is fully determined by its random seed. The known-answer test
must pass on every revision before any downstream result is trusted.

## Figures

All figures read from `results/` and write PNGs back into `results/`. Run the
experiments first (or use the committed result JSONs), then:

```bash
# quantitative panels (read committed result JSONs -- no training needed)
python experiments/figure_frequency.py      # -> results/fig_frequency.png
python experiments/figure_xray.py           # -> results/fig_xray.png
python experiments/figure_capacity.py       # -> results/fig_capacity.png
python experiments/figure_perturbation.py   # -> results/fig_perturbation.png

# signature figure -- the Weight-Space Competence Read-out (reads results/structures/*.npz)
python experiments/figure_competence_manifold.py   # -> results/fig_competence_manifold.png

# phase-portrait gallery (trains one representative net per task; GPU recommended)
python experiments/figure_portraits.py --seeds 6 --iters 3000
# -> results/fig_portraits.png  and  results/fig_portraits.npz (the projected data)
```

`figure_portraits.py` trains up to `--seeds` networks per task, keeps the one whose
settled states are cleanest in the top-2 PC plane, recovers its stable fixed points
(or autonomous limit-cycle orbit), and colors each continuous attractor by the task
variable it stores. A task that never reaches its convergence gate is skipped with a
warning rather than plotted, so no panel misrepresents a network that did not learn.
The fixed-point search is the compute-heavy step; on a GPU it finishes in minutes.


## Cross-architecture structure recovery

`experiments/exp1_structure_prediction.py` now analyzes all three architectures.
The GRU exposes `.step()` on its hidden state; the LSTM exposes `.step()` on the
joint (h, c) state (dimension 2N) via `set_joint()`/`joint_states()`, so the same
slow-point search and Jacobian analysis apply. Among networks that pass the
convergence gate, the GRU and LSTM recover the SAME predicted dynamical structure
as the vanilla RNN on every task (`results/xarch_results.json`) -- structure is a
property of the task, not the architecture (Maheswaranathan et al., 2019).

    python experiments/exp1_structure_prediction.py --task accumulation --archs rnn gru lstm --seeds 8

## Competence Atlas

`experiments/figure_competence_atlas.py` renders the dense matrix figure
(rows = tasks, columns = network size) from the committed sweep bundle
(`results/atlas_data.npz`, `results/atlas_meta.json`); no training required.
Each cell shows the occupied state cloud with the recovered structure overlaid,
colored by whether it matches the a-priori prediction.

    python experiments/figure_competence_atlas.py
