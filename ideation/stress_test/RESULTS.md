# Stress-test results: the conjugacy-defect claim

September 26, 2026. The protocol and its two amendments are in `PROTOCOL.md`.
Everything ran on a laptop CPU: 160 networks in stage 1, plus stage-2 stress
tests on the gated ones.

**Claim tested.** A network that solves a task with declared update `F` is
approximately conjugate to `F`. Its hidden failure is the per-step conjugacy
defect `δ = F̃ − F`, propagated by the task dynamics. `δ` is read from
one-step maps of the weights at states reached by probe runs no longer than the
training horizon. No long rollouts are used for prediction.

## Verdict

**The pre-registered stage-1 pass rule failed.** The failure is informative,
not random. Two things hold: the equation is exact when `δ` is known, and the
forecast works well for integration tasks. The claim breaks in four specific,
explainable places.

## 1. What held

| evidence | result |
|---|---|
| Exact anchor (task-conjugate cell, all 4 tasks, 40 networks) | per-trial agreement **1.00** and Spearman **1.00** wherever failures occur. The equation is exact when `δ` is known. |
| Accumulation, GRU / vanilla RNN | per-trial agreement **0.94 / 0.84**, Spearman **0.98 / 0.84**. Baselines (power-law and linear extrapolation of training-horizon error): **0.09–0.31**. |
| Beats both baselines | in **every** task and architecture: per-task means 0.79 vs 0.16–0.17 (accumulation) and 0.45 vs 0.22–0.32 (context integration). |
| Ablation: which part of `δ` carries the prediction (accumulation, 39 networks) | full state-dependent `δ`: 0.61–1.00; linear `δ`: 0.02–0.11; constant `δ`: 0.26–0.32; zero `δ`: 0.00. **The forecast lives in the nonlinear shape of the defect along the task manifold.** |
| Input shift beyond the probed range (accumulation ×4) | agreement **0.84–0.98** (baselines ≤ 0.69). The defect read from short probes extrapolates to harder inputs. |
| Form L (first-order equation) vs form N (nonlinear) | L is close to N on integration tasks but weaker on flip-flop (0.86–0.90 vs 1.00), as pre-stated. **The claim must be stated in the nonlinear form N:** the defect is evaluated at the network's own state. |

## 2. Where the claim broke

**B1. Context-dependent integration fails in every standard architecture**
(agreement 0.39–0.52; pass bar 0.75).

- The one-step defect fit explains only R² ≈ 0.54–0.80 of the defect's
  variance, against 0.83–0.999 for plain accumulation.
- The network is **not Markov in the task coordinate**: its hidden state
  carries extra variables, such as partially processed irrelevant input.
- Adding an input-history memory term (exploratory, one network) raised R² but
  did not reliably improve forecasts. On accumulation it made them *worse*
  (0.68 → 0.35–0.50).
- One-step fit quality is the wrong target for long-horizon forecasting.

**B2. LSTM accumulators fail despite excellent fits** (agreement 0.61; R²
0.975–0.994).

- Fit quality does not certify the forecast within a task. The R²–accuracy
  correlation across tasks is ρ = 0.69, but it misses the LSTMs.

**B3. Forecast accuracy decays with the failure horizon** (116 accumulation
networks, stage 1 plus shift):

| measured median T* | mean per-trial agreement |
|---|---|
| < 50 | 0.92 |
| 50–150 | 0.94 |
| 150–400 | 0.74 |
| 400–1000 | 0.76 |
| > 1000 | 0.55 |

Spearman(T*, agreement) = −0.52. The shift test shows the same pattern
directly: input scale ×0.5 (T* ≈ 200–1100) gives 0.41–0.74, while ×4 (T* ≈
13–39) gives 0.84–0.98.

**Interpretation: an estimation floor.** Forecasting a failure `T*` steps ahead
requires the *systematic* part of `δ` to be known to about `ε/T*`. The
one-step fit's bias sits around 10⁻⁴–10⁻³.

**B4. Oscillators mode-lock.**

- Trained discrete-time oscillators with period 16 learn an attracting
  16-point orbit. Their phase is restored, not neutral, so they essentially
  never drift (native error ~4×10⁻⁴ after 5000 steps).
- A smooth defect fit cannot resolve 16-fold locking. The reduced model then
  leaks phase (error 0.29 by step 5000). Lowering the ridge only halves this.
- The stage-1 metric scored this as agreement because a crossing near step
  4,900 is within 1.5× of "never". **The oscillation "pass" is not evidence
  for the claim.**

**B5. Noise-driven escapes from discrete attractors are not captured.**

- In flip-flop networks at σ = 0.1, the only regime where failure is truly
  hidden (median failure 50–400 steps, beyond `T_train = 80`), the reduced
  model's failure-time distributions are poor: median KS **0.52** with
  one-step noise projection.
- At σ ≥ 0.2 failures occur within ~20 steps, which is not hidden. Those are
  predicted well (KS ≤ 0.1).
- The pre-registered "adjoint" diffusion estimate (paired k-step probes) was
  **wrong** for contracting systems (KS ≈ 1.0). Variance saturates inside a
  basin, so dividing by k underestimates the noise.
- Rare escapes depend on transverse directions and the barrier's shape, which
  a 1-D Markov reduction does not see.

**B6. Invalid test, reported for completeness.** Planting an off-manifold slow
mode broke the networks' own training-horizon performance (gate p95 up to
0.76). Those networks no longer have *hidden* failures, so S5 says nothing
about the claim.

**B7. Flip-flop without noise** never fails (0% of trials over 8000 steps). The
claim correctly predicts "no failure", but that is a weak test.

## 3. What this means for the claim

The stress test separates two things the original claim bundled together.

1. **The equation.** Failure = conjugacy defect propagated by the dynamics,
   evaluated at the network's own state (form N). This is exact. It held
   perfectly on 40 exact-anchor networks, and ablations show the forecast comes
   from the defect's state-dependent shape.
2. **Reading the defect from a trained network.** This is where every failure
   occurred: non-Markov hidden variables (B1), precision (B2–B4), and rare-event
   stochastics (B5).

The pattern in B3 suggests a reformulation that is sharper and still unifying.
**It is a hypothesis generated by this stress test and must be confirmed on
fresh networks:**

> *A failure is hidden when the systematic defect is below what the training
> horizon can resolve (`|δ| ≲ ε/T_train`). It is forecastable when the defect
> is above what the probes can resolve (`|δ| ≳ ε/T*`, estimated with
> uncertainty). Hiddenness and forecastability are two resolution conditions
> on the same quantity.*

This turns the observed decay of accuracy with T* from a weakness into the
predicted behaviour. It also requires every forecast to carry an uncertainty
band derived from the probe data.

## 4. What would be needed to re-run the claim fairly (not yet done)

- **Uncertainty-aware forecasts.** Bootstrap the defect fit over probe trials,
  forecast an interval for T*, and score coverage. This is the direct test of
  the resolution hypothesis.
- **Better estimators, chosen on development networks and confirmed on fresh
  seeds.** Nonparametric or local defect fits, which could resolve mode
  locking, and fitting criteria based on multi-step consistency rather than
  one-step R².
- **Stochastic reduction.** Include transverse directions for rare escapes,
  or restrict the claim explicitly to drift-dominated failures.
- **A stricter metric for "no failure".** Score the predicted error curve
  against the measured one, not just the threshold crossing, so near-misses
  cannot pass as agreement.

## Files

- `PROTOCOL.md`: pre-registration and amendments
- `hf_tasks.py`, `hf_core.py`: tasks, models, estimator, predictors
- `run_stage1.py`, `run_stage2.py`, `summarize.py`, `diag_lags.py`
- `results/stage1/*.json` (160 networks), `results/stage2/*.json`,
  and `logs/`
- `nets/*.pt`: trained weights, gitignored

---

# Stage 3 (added later the same day): the claim holds once the defect is read from the invariant manifold

## What changed

Every stage-1/2 failure on the neutral tasks turned out to be a failure of
**how the defect was measured**, not of the claim itself. The fix went through
four steps, each registered in `PROTOCOL.md` before it was run:

1. **Exact evaluation (3/3b).** Evaluate the weights' own map in float64 on
   the slow manifold instead of regressing noisy one-step samples.
2. **Seeding fix (3c).** Initial guesses were being driven into saturation.
3. **Adjoint projection (3d).** Input transients decay slowly (λ₂ ≈ 0.95), so
   relaxation alone cannot resolve them. Project them with the left
   eigenvector instead.
4. **Invariance, not minimum speed (5).** This was the decisive step. In
   non-normal networks, the state that moves least at a given readout is
   *not* on the invariant manifold: the fast transient leaks into the readout
   and inflated the drift 5–9×. Solving the invariance equation
   `f(h(s)) = h(s + v(s))` (the parameterization method) removed that
   (residual 1e-3 → 1e-7).

Development used seeds 0–9. The estimator was then frozen and hash-recorded
(`results/FROZEN_SHA256.txt`). Confirmation used **seeds 10–14, trained
afterwards and never inspected before scoring**.

## Confirmatory result (seeds 10–14; 57 gated networks; all criteria PASS)

| test | RNN | GRU | LSTM |
|---|---|---|---|
| driven accumulation ×1: per-trial agreement (regression estimator) | 0.98 (0.85) | 0.99 (0.91) | 0.99 (0.61) |
| driven accumulation ×0.5, long horizons | 1.00 (0.63) | 1.00 (0.74) | 1.00 (0.52) |
| hold: load a value, then no input | 1.00 (0.47) | 1.00 (0.58) | 1.00 (0.30) |
| per-trial rank correlation | 0.99–1.00 | 0.98–1.00 | 0.99–1.00 |
| median \|log(T̂/T)\|, driven | 0.007–0.017 | 0.002–0.010 | 0.001–0.004 |

Far-future failures (measured median T* > 1000; n = 7 driven, n = 2 hold):
agreement **1.00**.

**Oscillation.** Fail vs no-fail is correct on **29/29** networks. All 11
failing networks are forecast within 1.5×, almost all within ~1%: 2171 vs
2174, 4067 vs 4067, 4076 vs 4117, 2666 vs 2683, 1315 vs 1314. The forecast
comes from a single 50-step window, up to ~80× beyond it.

## What is now supported

For tasks whose memory lives on a neutral structure (a line attractor for
accumulation and hold, a limit cycle for oscillation), in vanilla RNNs, GRUs
and LSTMs:

- the **hidden failure time of every individual trial** follows from the
  conjugacy defect **measured on the network's invariant manifold from the
  weights**, propagated by the task dynamics;
- it is predicted to within about 1–2% in the median;
- failures arrive up to 100× the training horizon;
- no run longer than the training horizon is used.

Two things this rules out:
- the far-future forecasts are **not** estimation-limited;
- the "resolution window" reformulation proposed earlier in this file is
  unnecessary.

## Not yet re-tested with the final estimator (scope limits)

- **Context-dependent integration (ctxint).** Failed in stage 1 with the old
  estimator. It needs the manifold method extended to multi-channel,
  context-gated inputs. Untested.
- **Noise-driven escapes from discrete attractors (flip-flop, σ = 0.1).**
  Failed in stage 2. The deterministic manifold method does not address rare
  stochastic events. Untested.
- **Sample sizes.** Far-future bins are small (n = 7 driven, 2 hold), and 3 of
  15 vanilla-RNN confirmation networks failed the training gate.
- **Architectures.** Only these three, at widths 32 and 128.

---

# Post-confirmation audit and exploratory decision analysis (September 26, 2026)

*Not preregistered. Both analyses were designed after the confirmation outcomes
were known. Data: `results/audit_confirm.json`, `results/decision_exploratory.json`.*

## Corrections to earlier claims (from external review, verified)

- **The 5–9× slow-point bias is not shown to be caused by non-normality.** A
  normal 2×2 system with a readout aligned to a fast direction reproduces a
  7.76× inflation. Correct description: geometric bias in constrained
  slow-point measurement (readout alignment plus the Euclidean metric).
- **The tangent invariance equation is a first-order approximation** of
  `f(h(s)) = h(s+v(s))`, not an equivalence. Exact discrete residual at
  held-out points:
  - median 1e-7 to 4e-6;
  - maximum up to 1e-3 (RNN/GRU) and 4.5e-2 (LSTM), likely at the manifold
    ends.

  Convergence check: still pending.
- **"Medians within 1–2%" compared distribution medians.** Per-trial median
  |log error|:
  - hold: 0.000–0.030, 90th percentile ≤ 0.09;
  - ×0.5: 0.000–0.009;
  - ×1: 0.000–0.031, 90th percentile ≤ 0.23.
- **State range.**
  - Driven ×1: 3–80% of measured failures (typically about half) occur with
    the true state outside the calibrated manifold range.
  - Driven ×0.5: 0–20%.
  - **Hold: 0% (one network 2%).** Hold is the clean test of slow drift within
    the trained range.
- **Task-conjugate "before training".** The stress-test tc networks were
  forecast *after* g was trained. The before-training claim is untested.

## Decision analysis

Each trial is a case (27 confirmation networks × 256 trials per scenario).
Rule: approve if the predicted failure time > H.

- **Wrongly approved** = fraction of approved cases that actually fail
  before H.
- **Useful approvals** = fraction of truly-OK cases that get approved.

| scenario | H | truly OK | E wrongly approved | E useful approvals | E AUC | best baseline (wrongly / useful / AUC) | validation only (wrongly approved) |
|---|---|---|---|---|---|---|---|
| hold | 500 | 46% | 0.4% | 99.8% | 1.000 | B2 19.5% / 89% / 0.94 | 54% |
| hold | 1000 | 29% | 0.2% | 99.9% | 1.000 | B1 19% / 64% / 0.87 | 70% |
| hold | 2000 | 20% | 0.1% | 99.6% | 1.000 | B1 37% / 55% / 0.81 | 80% |
| ×1 | 1000 | 10% | 1.4% | 96% | 0.989 | R 27% / 51% / 0.84 | 90% |
| ×0.5 | 1000 | 29% | 0.6% | 99.6% | 0.999 | R 30% / 54% / 0.82 | 71% |

Caveats:
- the forecast is conditional on the known future input sequence;
- cases are clustered within 27 networks;
- the rule and the H values were chosen after the outcomes were seen.

## Cost (per network, CPU)

| step | median | range |
|---|---|---|
| extraction | 9.9 s | 5.5–45 s |
| reduced forecast (256 trials × 5000 steps) | 0.3 s | — |
| brute-force full rollout (same) | 0.42 s | 0.29–1.3 s |

**At these network sizes, brute-force simulation is cheaper.** A computational
saving is not a supportable claim here. The value shown is decision quality
and mechanism, not speed.

---

# Preregistered decision-level test (Amendment 6; seeds 15–19; 30 networks, 23,040 cases)

**Verdict: FAIL on one of five preregistered criteria (P2).** Everything below
was scored once, with the rule and margins frozen on development seeds 5–9.

| cell | truly OK | validation-only wrongly approved | E wrongly approved | E useful approvals | E AUC | wrongly approved at E's approval count: R / B1 / B2 |
|---|---|---|---|---|---|---|
| hold, H = 1000 | 37% | 63% | 0.0% | 96% | 0.997 | 30% / 30% / 28% |
| hold, H = 2000 | 29% | 71% | 0.1% | 95% | 0.995 | 38% / 37% / 41% |
| ×1, H = 1000 | 9% | 91% | 1.2% | **73%** | 0.991 | 22% / 85% / 82% |
| ×0.5, H = 1000 | 25% | 75% | 0.6% | **88%** | 0.998 | 26% / 54% / 44% |
| ×0.5, H = 2000 | 8% | 92% | 0.6% | **78%** | 0.999 | 42% / 84% / 76% |

**Safety, which is the dangerous direction:**
- E predicted more than 10% too late in 0.7% of failing cases, against
  26–68% for the baselines.
- No baseline reached a 1% wrong-approval rate on development data at any
  margin up to 3×.

**The failure (P2).** The 10% margin that keeps wrong approvals near zero also
rejects many runs that would have succeeded, in long-horizon driven cells.
Useful approval there was 46–88% (the bar was 90%).

Interpretation:
- The forecast's timing error is small (median signed log error 0.000; 95th
  percentile of over-prediction +0.022).
- So the margin was *too conservative*, not the forecast inaccurate.
- The margin was chosen for the strictest development cell and applied
  uniformly.

This does not rescue the preregistered verdict. It states what a future,
separately preregistered rule would test, for example a per-scenario margin
or one calibrated from the forecast's own error distribution.

**Network level.**
- Median per-network worst-cell wrong-approval: 0%.
- 3/30 networks exceed 2% in some cell; the maximum is 22%.
- Wrong approvals are concentrated in a few networks rather than spread
  evenly. This should be investigated.

---

# Exploratory: why wrong approvals concentrate in a few networks (seeds 15–19)

*Outcomes were already known, so everything in this section is
hypothesis-generating. Data: `results/investigate_prereg.json`,
`results/grazing_prereg.json`.*

48 wrong approvals out of 31,127 approved cases (0.15%). They have two
separable causes.

**1. Invalid reduction.**
- One network (`rnn_N32_s18`) causes 21 of the 48 wrong approvals, with up to
  89% wrong in one cell.
- Its second Jacobian eigenvalue is **λ₂ = 1.003 > 1**. A second
  non-decaying direction exists, so a 1-D reduction is invalid in principle.
- Every other network has λ₂ ≤ 0.989.
- λ₂ is computed from the weights before any forecast.

**2. Grazing crossings.** For most of the remaining wrong approvals, the
forecast state closely matches the network at failure (e.g. −0.637 vs −0.647).
The error rises slowly along the tolerance line, which makes the crossing
*time* ill-conditioned.

| predicted error margin κ = ε − max_{t≤H} ê(t) | median | notes |
|---|---|---|
| wrong approvals | 0.012 | 90th percentile 0.14 |
| correct approvals | 0.167 | 10th percentile 0.052 |

**Illustrative abstention trade-off** (thresholds chosen while looking at
this data, so no claim rests on these numbers):

| rule | wrong approvals | approvals kept |
|---|---|---|
| frozen rule only | 48 / 31,127 (0.15%) | 100% |
| + abstain if λ₂ ≥ 1 | 27 / 30,439 (0.09%) | 97.8% |
| + κ ≥ 0.01 and λ₂ < 1 | 5 / 30,151 (0.02%) | 96.9% |
| + κ ≥ 0.02 and λ₂ < 1 | 3 / 29,733 (0.01%) | 95.5% |

**Hypothesis for a new preregistered test.** Replace the fixed 10% time
margin with the rule:

> approve if the predicted error stays below ε − κ through H, abstain if
> λ₂ ≥ 1.

κ would be chosen on development seeds 5–9 and tested on new seeds. This rule
is forecast-specific, so it could keep safety while recovering the good runs
that the global margin rejected (the P2 failure).

---

# Preregistered test of the validity-aware selective rule (Amendment 7; seeds 20–24) — PASS

**Rule.**
- **UNSUPPORTED** if Λ₂ ≥ 1.
- Otherwise **APPROVE** only if the forecast error stays at least γ·ε below
  tolerance through H, with γ = 0.03 calibrated on development seeds 5–9.

The *form* of the rule came from exploratory analysis of seeds 15–19. It was
frozen and hashed before seeds 20–24 were collected. 30 networks, 23,040
cases, 12 cells.

| cell | truly OK | wrongly approved | good runs approved | old 10% rule: good runs approved | matched-coverage baselines, wrongly approved (R / B1 / B2) |
|---|---|---|---|---|---|
| hold, H = 1000 | 2979 | 0.3% | 98.1% | 97.1% | 37% / 27% / 30% |
| hold, H = 2000 | 2191 | 0.1% | 97.2% | 95.5% | 50% / 40% / 47% |
| ×1, H = 1000 | 888 | 0.4% | 94.0% | **78.9%** | 45% / 83% / 81% |
| ×1, H = 2000 | 100 | 0.0% | 93.0% | **66.0%** | 73% / 99% / 97% |
| ×0.5, H = 1000 | 2756 | 0.0% | 98.1% | **89.2%** | 32% / 52% / 49% |
| ×0.5, H = 2000 | 774 | 0.0% | 95.6% | **78.2%** | 67% / 84% / 84% |

**Pooled over all cells:**
- wrongly approved: 0.08% (old rule 0.05%);
- good runs approved: 39,017 (old rule 37,290).

**What this establishes.** Replacing the fixed time margin with a
forecast-specific error margin recovers the good runs the old rule rejected.
It keeps wrong approvals below 0.5% in every cell, on networks the rule never
saw.

**What it does not establish.** No network in this cohort had Λ₂ ≥ 1, so the
structural gate was never exercised. The single observed Λ₂ = 1.003 network
(seed 18) supports the gate's rationale, but the gate's sensitivity to
pathological networks needs a much larger sample: with a prevalence of about
1/30, 30 networks meet one only ~64% of the time.

---

# Diagnostic: why do width-512 hold-trained LSTMs forecast poorly? (September 27, 2026)

*Exploratory. The networks are from the inspected repair set (Amendment 10)
plus two good references. Script: `diag_hold_lstm.py`. Data:
`results/diag_hold_lstm.json`.*

Estimator: the corrected v2. "History divergence" is the median |Δ output|
between two input histories that end at the same decoded held value (direct
load vs overshoot-and-return), followed by zero input.

| network | forecast error | near-unit modes (λ > 0.99, mid-manifold) | λ₁ / λ₂ | history divergence at t = 0 / 1000 | 1-D closure error at t = 0 / 1000 |
|---|---|---|---|---|---|
| hold LSTM-128 s530 (good) | 0.02 | 1 | 1.001 / 0.63 | 0.011 / 0.012 | 0.009 / 0.012 |
| accum LSTM-512 s1000 (good) | 0.004 | 1 | 1.000 / 0.60 | 0.011 / 0.0001 | 0.005 / 0.00002 |
| hold LSTM-512 s1002 | 0.23 | 1 | 0.999 / 0.69 | **0.088 / 0.065** | **0.080 / 0.057** |
| hold LSTM-512 s1011 | 0.23 | 1 | 1.001 / 0.65 | **0.104 / 0.001** | **0.094 / 0.001** |
| hold LSTM-512 s1007 | 0.03 | 1 | 1.002 / 0.60 | **0.054 / 0.002** | **0.038 / 0.002** |

**Classification.**

1. **Construction / design: ruled out.**
   - Identical training (4000 iterations at 2e-3 on GPU); the worst network
     trained best (loss 1e-4).
   - The full (h, c) state is used throughout, the decoder convention is the
     same, and the manifold covers the test range.
2. **Numerical reduction: not the cause.**
   - The manifold converges.
   - The on-manifold spectrum is one-dimensional, with the same gap as the
     good networks.
   - Flow folds cover only 1–4% of the manifold.
3. **Learned structure: yes, as a closure failure off the manifold.**
   - On the manifold, the bad networks are as one-dimensional as the good
     ones.
   - After loading, they carry 5–10× more history in slowly decaying
     transients *off* the manifold. The 1-D model cannot represent these
     (closure error ≈ the divergence itself).
   - In s1002 the extra history persists for 1000 steps.
   - Likely mechanism: gate-saturated regions visited during loading, which
     the local Jacobian at manifold points does not capture. This is not yet
     tested directly.

**Consequence for production.**
- The preregistered closure test runs only for driven (accumulation)
  networks, so this failure mode is not flagged for hold networks by any
  frozen diagnostic.
- Adding a hold-network closure diagnostic would need its own amendment and a
  prospective test. It is recorded here as a limitation and an open item.
