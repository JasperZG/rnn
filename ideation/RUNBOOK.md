# RUNBOOK: running the hidden-failure study on the home computer (RTX 5070)

This file is the step-by-step plan for picking the project up on the home
machine. Read sections 0–2 before running anything.

All commands assume you are in `ideation/stress_test/` with the virtual
environment active, unless stated otherwise.

---

## 0. Where things stand (September 27, 2026)

| stage | status | where |
|---|---|---|
| Stage 1–2 stress test (regression estimator) | failed its preregistered rule; failures diagnosed | `stress_test/RESULTS.md` |
| Stage 3: exact invariant-manifold estimator | **PASS** on untouched seeds 10–14 | `PROTOCOL.md` amendments 3–5 |
| Decision test with a fixed 10% margin (Amendment 6, seeds 15–19) | **FAIL** (P2: too conservative); P1, P3–P5 pass | `PROTOCOL.md` |
| Validity-aware rule: Λ₂ gate plus γ = 0.03 error margin (Amendment 7, seeds 20–24) | **PASS**, all 4 criteria | `PROTOCOL.md`, `RESULTS.md` |
| Scale-up (this runbook) | not started | — |

Key code (in `stress_test/`):

| file | purpose |
|---|---|
| `hf_tasks.py` | tasks with a declared update F |
| `hf_core.py` | models, training (now GPU-capable), probes, regression estimator, baselines |
| `hf_exact_FROZEN.py` | **the frozen estimator.** Never edit it. |
| `hf_core_FROZEN.py` | the exact `hf_core.py` used for the frozen confirmation runs. Its hash equals the one recorded in `results/FROZEN_SHA256.txt`. `hf_core.py` has since gained a `device` argument; defaults are unchanged. |
| `prod.py` | **production runner**: parallel, GPU training, CPU float64 extraction, resumable, one result file per network |
| `decision3.py` | frozen scorer for the validity-aware rule (γ in `results/decision3_gamma_dev.json`) |
| `score_confirm.py`, `score_prereg.py` | reproduce earlier verdicts from the committed results |

---

## 1. Setup (once, ~15 min)

The setup script is bash. On Windows, use **WSL2 (Ubuntu)** with the NVIDIA
WSL driver, or translate the commands for PowerShell.

```bash
git clone https://github.com/JasperZG/rnn.git
cd rnn
git checkout hidden-failure-forecasting
bash setup_5070.sh               # venv + CUDA 12.8 PyTorch (required for RTX 50-series) + GPU check
source .venv/bin/activate
pip install -r requirements.txt  # numpy, scipy, scikit-learn, matplotlib
nvidia-smi                       # confirm the GPU is visible
```

If PyTorch reports "no kernel image is available", the wrong wheel was
installed. Rerun the cu128 line in `setup_5070.sh`.

**Record the machine**, since it goes into the methods section later:

```bash
python -c "import torch,platform,os;print(torch.__version__, torch.cuda.get_device_name(0), platform.processor(), os.cpu_count())"
```

---

## 2. Sanity check: reproduce the existing verdicts (~5 min, no training)

```bash
cd ideation/stress_test
python score_confirm.py                       # expect: OVERALL: PASS
python score_prereg.py                        # expect: OVERALL: FAIL (P2), as recorded
python decision3.py score --src stage1_prereg2  # expect: OVERALL: PASS
shasum -a 256 -c results/FROZEN_DECISION3_SHA256.txt   # use sha256sum on Linux
```

The `shasum` check will report `hf_core.py: FAILED`. That is expected, because
`hf_core.py` gained GPU support. Verify the frozen copy instead:

```bash
sha256sum hf_core_FROZEN.py   # must start with 944b62fdc5ab
```

If any verdict differs from the above, **stop** and investigate before
running anything new.

---

## 3. Timing benchmark (~30–60 min)

One network per architecture × width, 256 trials per scenario:

```bash
python prod.py bench --widths 32 128 512 --workers 3 --threads 4
```

It prints a table of seconds for training (GPU), extraction (CPU float64),
naive-baseline extraction, brute-force rollout and reduced forecast.

**Choosing `--workers`.** Extraction is CPU-bound. A reasonable start is
`workers × threads ≈ number of CPU cores` (e.g. 16 cores → `--workers 8
--threads 2`). Watch `htop` and `nvidia-smi`. The GPU will be mostly idle;
that is normal, because these networks are tiny.

**Expected**, from laptop measurements:

| width | per network |
|---|---|
| 32 / 128 | 1–3 min |
| 512 | 4–10 min (LSTM is slowest: its state has 1024 dimensions) |

Write the measured numbers into section 9 of this file. They replace these
estimates.

Benchmark outputs are in `results/prod/bench/`. They are development data
and must not be used in any confirmatory analysis.

---

## 4. Development runs at width 512 (development only, seeds 500–509)

Width 512 has **never been run**. Any fixing happens here, before the
production protocol is frozen.

```bash
python prod.py run --name dev512 --tasks accumulation --archs rnn gru lstm \
    --widths 512 --seeds 500 501 502 503 504 505 506 507 508 509 --trials 1024 --workers 8
python prod.py aggregate --name dev512
```

What to check in `results/prod/dev512/*.json`:

- **Training gate** (`gate_pass`). Does width 512 train with 4000 iterations
  at lr 2e-3? If the pass rate is low, adjust iterations or learning rate
  *here* and record the choice.
- **Extraction diagnostics** (`diag`):
  - `invariance_resid_after` should be small (it was 1e-7–1e-8 at N ≤ 128);
  - `lam2_max`;
  - `n_manifold`;
  - `flow_monotone` should be `True`.
- **Accuracy** (`summary_*`): median |log error| of E. It should be ~0.01,
  as at N ≤ 128.
- **Errors** (`status: "error"`): read the `error` traceback.

If extraction breaks at 512, fix it and **document the change as a new
amendment**. Any change to the estimator means copying it to a new frozen
file (e.g. `hf_exact_v2_FROZEN.py`) and re-running development checks at
N = 32 and 128 too.

---

## 5. Write and freeze the production preregistration (before any production run)

Add a new amendment to `stress_test/PROTOCOL.md` fixing at least the items
below. You can use AI assistance here (this is research, not the STS report),
but disclose it.

1. **Cohorts and seed ranges** (all unseen). Suggested:

   | cohort | architectures | widths | seeds | networks |
   |---|---|---|---|---|
   | factorial | rnn/gru/lstm | 32/128/512 | 1000–1059 (60 attempts per cell) | 450 target, 50 gated per cell |
   | deep reliability | rnn/gru/lstm | 128 | 2000–2059 | +150 |
   | oscillation | rnn/gru/lstm | 32/128 | 3000–3029 | 180 |

2. **Top-up rule** if a cell has fewer than 50 gated networks: more seeds, up
   to 80 attempts, decided from gate results only.
3. **Trials:** 1024 per scenario (hold, x1, x0.5); T_max = 5000. Censored
   means "no failure by T_max".
4. **Horizons:** H ∈ {250, 500, 1000, 2000}.
5. **Predictors:**
   - E (frozen);
   - NS, the naive slow-point defect (isolates the invariance correction);
   - R (regression);
   - B1, B2 (extrapolation);
   - validation-only.
6. **Policies:**
   - A: the old 10% margin, kept as recorded;
   - B: Λ₂ gate plus γ = 0.03 (already frozen). If you want a new policy,
     define it now from development data only.
7. **Metrics:**
   - P(F|A), P(A|F), P(A|¬F), coverage, AUC;
   - matched-coverage comparison;
   - dangerous over-prediction rate;
   - network-level distribution of P(F|A): median, 90th/95th percentile, max,
     and the fraction of networks above 2%.
8. **Pass criteria and denominators**, stated exactly.
9. **The analysis script.** Write it now (extend `decision3.py`, e.g. to
   include NS and the network-level analysis), run it on the `dev512` pooled
   file to check it works, then **hash it together with `prod.py`,
   `hf_core.py`, `hf_exact_FROZEN.py` and `hf_tasks.py`**:

   ```bash
   sha256sum prod.py hf_core.py hf_exact_FROZEN.py hf_tasks.py decision3.py <analysis script> > results/FROZEN_PROD_SHA256.txt
   ```

   Commit and push *before* starting production, so the timestamp proves the
   order.

**Decision to make here: hold as a scenario vs. a separately trained task.**
In everything validated so far, "hold" is a *test scenario* on
accumulation-trained networks (load a value, then zero input). The external
plan treats hold as a separately trained task, which doubles the network
count.

Recommendation: **keep hold as a scenario.** It is what was validated, it is
the clean within-range drift test, and it halves compute. Each network then
contributes hold, x1 and x0.5 cases. If you want separately trained hold
networks, a new task must be added to `hf_tasks.py` and developed first.

---

## 6. Production runs

Run inside `tmux` (or `nohup`) so they survive disconnects. The runner is
**resumable**: rerunning the same command skips finished networks.

```bash
tmux new -s prod
source ../../.venv/bin/activate

# factorial (60 attempts per cell)
python prod.py run --name factorial --tasks accumulation --archs rnn gru lstm \
    --widths 32 128 512 --seeds $(seq 1000 1059) --trials 1024 --workers 8

# deep reliability cohort (width 128)
python prod.py run --name deep128 --tasks accumulation --archs rnn gru lstm \
    --widths 128 --seeds $(seq 2000 2059) --trials 1024 --workers 8

# oscillation cohort
python prod.py run --name osc --tasks oscillation --archs rnn gru lstm \
    --widths 32 128 --seeds $(seq 3000 3029) --workers 8
```

Detach with `Ctrl-b d` and reattach with `tmux attach -t prod`.

**Rules during production:**
- do not edit any hashed file;
- do not delete or replace networks that fail extraction, since that is an
  outcome;
- do not look at pooled decision results until every run in the cohort is
  finished.

---

## 7. Analysis

```bash
python prod.py aggregate --name factorial     # -> results/cases3_prod_factorial.npz, plus the funnel counts
python prod.py aggregate --name deep128
python decision3.py score --src prod_factorial   # policy B, frozen criteria
python <your frozen analysis script> ...         # everything preregistered in step 5
```

Report the funnel (attempted → gated → extracted → forecast), every
preregistered criterion as written, and the failures. Then commit and push
`results/prod/**` JSON and NPZ files and `results/cases3_prod_*.npz`. Weights
(`*.pt`) are gitignored, so back up `nets/prod/` separately (external drive
or cloud).

---

## 8. Still open (do only if time allows; each needs its own dev → freeze → test)

- **Numerical convergence.** Re-extract a preselected random 10% of networks
  with a finer manifold grid and tighter L-BFGS tolerance. Would T̂ or the
  decisions change?
- **Paired-pulse closure test.** Two histories with the same reduced state
  but different recent inputs, then the same next input. Does the 1-D model
  predict both?
- **Coordinate-transform control.** Same computation in non-orthogonal
  coordinates. The naive slow-point drift should change; the invariant
  estimate should not.
- **Λ₂-gate sensitivity.** Needs the large cohorts above; one bad network in
  30 is too rare to test otherwise.
- **External navigation study** (DeepMind grid-cell LSTM). Their code is
  TensorFlow 1 and will need a PyTorch reimplementation for RTX 50-series.
  This is weeks of work. **Plan it for after STS.**

---

## 9. Measured timings (fill in after step 3)

| network | train (s) | extract (s) | brute (s) | forecast (s) |
|---|---|---|---|---|
| | | | | |

---

## 10. STS constraints: read before writing anything

- STS 2027 applications close **November 5, 2026**. Leave **at least 2–3
  weeks** for writing.
- Society for Science FAQ: AI tools may be used in research **with
  disclosure**. You may **not** use generative AI to write the application
  questions, draft the Research Report, or generate citations. Check the
  Accepted AI Usage Chart in the 2027 rules book.
- **Everything in `ideation/` was written with AI assistance** (Claude):
  - code;
  - experimental design;
  - `PROJECT.md`, `RESULTS.md`, `PROTOCOL.md`;
  - this runbook.

  Treat these as lab notes. Your report must be your own writing, and you
  should be able to explain every step yourself. That especially means why
  the first estimator failed and why the invariance fix works.
- Keep the chat transcripts as your record of AI-assisted research, and
  disclose that use as the rules require.
- Verify every citation yourself. `literature_notes.md` marks some as
  unverified.
- Suggested priority if time is short: factorial at widths 32 and 128 first,
  then 512, then the deep cohort, then oscillation. Navigation waits until
  after STS.
