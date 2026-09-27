# Stress-test protocol for the unifying claim

Fixed before any network was trained (September 26, 2026).

## Claim under test

A recurrent network that solves a task with declared update `F` is
approximately conjugate to `F`. Its hidden failure is the per-step conjugacy
defect `δ = F̃ − F`, propagated by the task's own dynamics.

Two forms are tested:

- **N (nonlinear reduced map):** `ẑ_{t+1} = F(ẑ_t,u_t) + δ̂(ẑ_t,u_t)`. The
  defect is evaluated at the network's own decoded state.
- **L (first-order equation as originally written):**
  `e_{t+1} = J_F(z_t,u_t) e_t + δ̂(z_t,u_t)`. The defect is evaluated at the
  true task state.

`δ̂` is estimated **only** from one-step maps of the trained weights, at states
reached by probe runs no longer than `T_train`. Probe inputs come from a
broadened version of the training distribution. No run longer than `T_train`
is used for prediction.

## Tasks (declared F; outputs = full task state z)

| task | k | F | failure threshold ε |
|---|---|---|---|
| flipflop | 1 | `u` if pulse else `z` | 0.5 (masked 3 steps after each pulse) |
| accumulation | 1 | `z + u` | 0.25 |
| ctxint | 1 | `z + u_ctx` (context selects one of two streams) | 0.25 |
| oscillation | 2 | `R_ω z + u e_1` (pulse at t=0) | 0.25 (Euclidean) |

## Architectures

- vanilla tanh RNN, GRU, LSTM (the repo's classes);
- the task-conjugate cell (`tc`, only `g` trained), as the exact anchor.

Widths N ∈ {32, 128}; 3 seeds each.

## Gate

A network enters the analysis only if its **95th-percentile masked error over
the training horizon is below ε/2** on a fresh batch. Only networks that look
healthy at the training length can have a *hidden* failure. Networks that fail
the gate are reported but not analysed.

## Measurement

- 256 independent test trials, run for `T_test = 20 × T_train`.
- Failure time is the first masked step with `|e_t| > ε`.
- Trials that never cross are censored at `T_test`.

## Predictors compared

- **N:** the claim, nonlinear form.
- **L:** the claim, first-order form.
- **B1:** per-trial power-law extrapolation of `|e_t|`, `t ≤ T_train`.
- **B2:** linear growth, rate `|e(T_train)|/T_train`.

## Metrics (per network)

- `|log(T̂_med / T_med)|`: median failure time, both censored at `T_test`.
- Censoring agreement: whether the predicted and measured trial each fail
  before `T_test`.
- Spearman ρ between predicted and measured per-trial failure times.
- Fraction of trials predicted within a factor of 1.5.

## Pass/fail for the claim (stage 1)

The claim **passes** stage 1 if form N does both of the following:

- achieves median `|log ratio| < log 1.5` on at least 75% of gated networks,
  across all four tasks and all three standard architectures;
- beats both baselines B1 and B2 on median `|log ratio|` in every task.

The claim **fails** if any task or architecture class is systematically
mispredicted, meaning more than half of its gated networks lie outside a
factor of 2. That failure would be reported, not tuned away.

Pre-stated expectation: form L will fail on flipflop. The task's `F` is neutral
there, but the network's reduced map is contracting. If that happens, the claim
has to be stated in form N.

## Later stress stages (after stage 1)

- S2 input-distribution shift at test time
- S3 neuron noise at test time (stochastic defect)
- S4 training-horizon variation
- S5 deliberately small spectral gap (non-Markov off-manifold memory)
- S6 negative controls (wrong F in the propagation)

## Amendment 1 (after a 300-iteration smoke test, before any stage-1 run)

The smoke test showed that at `T_test = 20 × T_train`, well-trained
oscillation and flip-flop networks rarely fail. Worse, "both censored" scored
as perfect agreement, which made the median metric vacuous. Changes:

- `T_test = 100 × T_train`.
- **Primary metric:** per-trial agreement. A trial agrees if predicted and
  measured failure times are within a factor of 1.5, **or** both are censored.
  Also reported: the predicted vs measured fraction of trials failing by
  `T_test`.
- The median `|log ratio|` and Spearman ρ are reported only for networks where
  at least 50% of trials fail and at least 20% of trials fail, respectively.
- **Pass rule restated:** form N must have per-trial agreement ≥ 0.75 on at
  least 75% of gated networks in every task × standard-architecture cell, and
  must beat B1 and B2 on per-trial agreement in every task.

## Amendment 2 (stage 2, before any at-scale stage-2 run)

A single-seed debug run showed that fitting the defect on *noisy* probes
regresses `δ` on a noise-corrupted `ẑ_t`. That is an errors-in-variables bias:
the noise from the previous step enters `F(ẑ_t)` and correlates with the
regressor. In the debug run it produced a spurious drift, predicting failure at
step 616 for an oscillator that never failed. The estimator is therefore
fixed, following the standard decomposition:

- the mean defect `δ̂` comes from **noise-free** probes (as in stage 1);
- the stochastic part comes from the **paired k-step** diffusion estimate
  `D_eff`, which includes non-normal amplification of off-manifold noise.

Also recorded: the planted-slow-mode test (S5) breaks the networks' own gate
(training-horizon p95 rises from 0.03–0.11 to up to 0.76). A network that no
longer looks healthy at the training length has no *hidden* failure, so S5 is
reported as exploratory only and does not count toward the verdict.

## Amendment 3: exact slow-manifold estimator (stage 3). Written before any stage-3 code ran.

**Motivation.** Stage 1/2 failures concentrated in far-future failures, where
the systematic defect is below the regression estimator's bias (~1e-4 to
1e-3 per step). Hypothesis: that is an estimator limit, not a limit of the
claim.

**Estimator E (theory-determined, no tuning).** The network runs in float64.

*Accumulation:*
1. Build the slow manifold by driving the network with constant inputs for
   10 steps, then relaxing for 40 steps with zero input.
2. Take manifold coordinate `s = C h`.
3. Drift: `v(s) = C f(h,0) − s`, exact.
4. Input displacement: the asymptotic-coordinate change
   `Δs(s,u) = C f^R(f(h,u)) − C f^R(f(h,0))`, with R = 30 relaxation steps.
5. Transient readout offset: `τ(s,u)`.
6. Reduced map: `s' = s + v + Δs`. Output: `ŷ = s' + τ`.
7. Interpolation: cubic splines over the (s, u) grid.

Every run is at most 50 steps (≤ T_train).

*Oscillation* (autonomous after the pulse):
1. Run the pulse response for T_train = 50 steps.
2. Measure the per-period phase slip by comparing the decoded state with
   itself one task period (16 steps) later. This cancels orbit-shape
   harmonics.
3. Propagate that slip, with the last observed period as the orbit shape.

**Tested on:**
- stage-1 networks (development);
- **fresh seeds 5–9 (confirmatory)**, for accumulation at input scale ×1 and
  ×0.5;
- stage-1 oscillation networks.

**Pass rule, fixed now:**
- **(i)** Accumulation, fresh seeds: mean per-trial agreement ≥ 0.85 in every
  standard architecture, at ×1 and at ×0.5.
- **(ii)** Accumulation, fresh seeds: mean agreement ≥ 0.80 among networks
  whose measured median T* > 1000. **If (ii) fails, the far-future claim is
  rejected.**
- **(iii)** E beats the stage-1 regression estimator in every architecture.
- **(iv)** Oscillation:
  - correct fail vs no-fail within T_test on ≥ 90% of gated networks;
  - among networks that fail, |log(T̂/T)| < log 1.5 on ≥ 75%.

### Amendment 3b (implementation detail, before any stage-3 execution)

Manifold points obtained by relaxing for R ≤ 50 steps keep off-manifold
residue of order λ₂^R. For slow networks that is comparable to the defect
being measured. The estimator therefore changes as follows. The pass rule is
unchanged.

- **Manifold points** come from constrained slow-point optimization:
  minimize `‖f(h,0) − h‖²` subject to `C h = s`, using L-BFGS in the null
  space of `C`. This is a computation on the weights only, the same tool as
  the earlier project's fixed-point search. Initial guesses come from
  30-step runs.
- **Asymptotic coordinate** of a displaced state: relax R = 49 steps, then
  invert the exactly known 1-D drift flow over those steps
  (`s' = S_R⁻¹(C f^R(y))`). This removes the drift accumulated during
  relaxation. Every chain is at most 50 steps.
- **Validity diagnostic,** computed before any forecast: the second-largest
  Jacobian eigenvalue modulus `λ₂` at manifold points, and `λ₂^49`. It is
  reported for every network and **not** used to exclude anything from the
  pass rule.
- **Oscillation** phase-slip pairs start at step 26 rather than 18, to avoid
  the amplitude transient.

### Amendment 3c (implementation bug, found on development networks only)

The first smoke test drove the initial guesses with constant inputs up to 0.9
per step. That is 6× the training scale and put most points deep in
saturation. Worse, one joint L-BFGS over all points let those points dominate
the line search, so the in-range points did not converge (residual speed
~0.02, against a true along-manifold drift of ~2e-3).

Fix:
- drive range ±0.3;
- L-BFGS in independent chunks of 50 points.

Verified on one development network: after the fix, residual speed equals the
along-manifold drift, as theory requires. Fresh seeds remain untouched.

### Amendment 3d (theory-driven fix, found on development networks only)

The built-in diagnostic flagged a vanilla RNN with λ₂ = 0.948. After 49
relaxation steps, 7% of each input transient remains unresolved. That residue
biased the displacement by ~1e-2 per step (predicted failure at t = 10,
measured t = 170).

Replacement, the standard asymptotic-phase construction:
1. Relax R = 10 steps.
2. Project the remaining displacement onto the manifold with the adjoint
   (left) eigenvector `ℓ` of the Jacobian (eigenvalue nearest 1), normalized
   so that `ℓᵀ ∂h/∂s = 1`.
3. Invert the 10-step drift flow.

This is exact to first order in the residual. The pass rule is unchanged.
Fresh seeds remain untouched.

## Stage-3 registered outcome (recorded before any further runs)

- **Accumulation (fresh seeds 5–9), exact estimator E:** criteria (i), (ii)
  and (iii) all **FAIL**.
  - E per-trial agreement: GRU 0.55 / 0.39, LSTM 0.61 / 0.49, RNN 0.29 / 0.08
    (at ×1 / ×0.5).
  - The stage-1 regression estimator does better for GRU and RNN.
  - Far bin (T* > 1000, n = 8): E 0.50, regression 0.51.
- **Oscillation (stage-1 networks), E:** criterion (iv) **PASS**.
  - Fail vs no-fail is correct on 39/39 networks.
  - Every failing network is predicted within ~2% (e.g. 3460 vs 3469,
    2374 vs 2375, 1587 vs 1579).

## Amendment 4: dichotomy hypothesis (registered before running)

**H-auto.** Hidden failures are forecastable from the attractor defect when
the network runs **autonomously** on its attractor. They are not forecastable
from a single task coordinate when inputs keep driving it off the attractor.

**Test A.** Oscillation E on **fresh** seeds 5–9 (rnn / gru / lstm,
N = 32 / 128). Same criterion (iv).

**Test B.** Hold task on the **same fresh accumulation networks**.
- Load a value z₀ ~ U(−1.5, 1.5) as 10 equal increments.
- Then give zero input for 4990 steps.
- Failure: |e| > 0.25 at t ≥ 10.

**Pass (B):**
- E per-trial agreement ≥ 0.85 in every standard architecture;
- among networks with measured median T* > 1000, mean agreement ≥ 0.80.

**H-auto is supported** only if A and B both pass while driven accumulation
(already measured) fails.

## Amendment 4 outcome and Amendment 5 (registered before implementation)

**Test B outcome: FAIL.**
- E agreement: RNN 0.15–0.40, GRU 0.05–0.79, LSTM 0.04–0.51.
- Per-trial rank correlation for GRU/RNN was 0.97–1.00: the ordering is
  right, the timescale is ~3× too early.
- Diagnosis on two networks: the load mapping is correct, but the drift
  `v(s)` is overestimated 5–9×. Minimizing speed at fixed readout does **not**
  find the invariant manifold when the dynamics are non-normal: the fast
  transient `C J d` leaks into the readout.

**Amendment 5.** Manifold points solve the **invariance equation**
`f(h_i) − h_i − v_i t_i = 0`, with:
- `v_i = C f(h_i) − s_i`;
- `t_i` the central-difference tangent;
- `C h_i = s_i` fixed;
- joint L-BFGS over all points.

This is the parameterization method for invariant manifolds.

- Seeds 5–9 are now **development** networks.
- **Confirmation** uses new seeds 10–14 (accumulation and oscillation, all
  architectures), trained now and not inspected before the confirmatory run.
- Criteria are identical to A and B, plus driven accumulation (i)–(iii) with
  the new E.

### Amendment 5 development check (seeds 5–9 only, recorded before confirmation)

With the invariance-solved manifold, six development networks (RNN, GRU and
LSTM at N = 32 and 128) give per-trial agreement 0.99–1.00 and Spearman
0.99–1.00 on hold, driven ×1 and driven ×0.5. Medians match within ~4% (e.g.
1891 vs 1878, 1446 vs 1389). The estimator is now **frozen**
(`hf_exact.py` at this point). Confirmation runs on seeds 10–14 with no
further changes.

## Amendment 6: preregistered decision-level test

*Written before the development margins were computed and before seeds
15–19 finished training.*

**Question.** Does the frozen estimator E support correct decisions about
whether a network will stay within tolerance for a required duration H?

**Cases.** Every gated standard-architecture accumulation network (rnn, gru,
lstm; N = 32 and 128) from seeds **15–19** (`results/stage1_prereg`). The three
scenarios are hold, driven ×1 and driven ×0.5, each with 256 trials and
T = 5000. Required durations: H ∈ {250, 500, 1000, 2000}. That gives 12 cells.

**Rule.** Approve a case if the predicted first failure exceeds H·(1+m).

**Margin selection** (`decision2.py margins`, development seeds 5–9 only):
- for each method, take the smallest m on the fixed grid whose development
  wrong-approval rate is ≤ 1% in every cell with ≥ 50 approvals;
- if no m on the grid meets this, use the largest m.

The margins are then frozen, together with `decision2.py` and
`hf_exact_FROZEN.py`.

**Definitions.**
- *Wrong approval* = fraction of approved cases that fail before H.
- *Useful approval* = fraction of truly-OK cases that are approved.

**Pass criteria (all must hold):**
- **P1.** E wrong-approval ≤ 2% in every cell with ≥ 50 approvals.
- **P2.** E useful-approval ≥ 90% in every cell with ≥ 50 truly-OK cases.
- **P3.** At matched acceptance (every method approves as many cases as E),
  E's wrong-approval is ≤ each baseline's (R, B1, B2) in all 12 cells, and
  strictly lower in at least 10.
- **P4.** E AUC ≥ 0.97 in every cell where both outcomes occur.
- **P5.** Among cases that fail within T, E predicts > 10% too late in ≤ 5%
  of them. This is the dangerous direction.

**Also reported** (not part of the pass rule):
- per-network worst-cell wrong-approval;
- validation-only approval rates;
- the same analysis on seeds 10–14, which is exploratory because those
  outcomes were already seen.

### Amendment 6: frozen development margins (seeds 5–9), recorded before seeds 15–19 were scored

| method | margin m | development worst-cell wrong-approval | 1% target met |
|---|---|---|---|
| E | 0.10 | 0.64% | yes |
| R | 3.0 (grid max) | 29.8% | no |
| B1 | 3.0 (grid max) | 83.0% | no |
| B2 | 3.0 (grid max) | 80.4% | no |

Hashes: `results/FROZEN_DECISION_SHA256.txt`.

### Amendment 6: registered outcome (seeds 15–19; 30/30 networks gated; scored once)

**OVERALL: FAIL.** One of five criteria failed.

- **P1 PASS.** E wrong-approval 0.0–1.2% in every cell.
- **P2 FAIL.** Useful approval fell below 90% in five long-horizon driven
  cells: x1 at H = 500/1000/2000 (0.87 / 0.73 / 0.46) and x0.5 at
  H = 1000/2000 (0.88 / 0.78). The 10% safety margin rejects good runs whose
  true failure time falls just above H.
- **P3 PASS.** At matched acceptance, E is strictly lower than every baseline
  in 12/12 cells.
- **P4 PASS.** AUC 0.974–1.000.
- **P5 PASS.** Over-prediction by more than 10% in 0.7% of failing cases.

## Amendment 7: preregistered validity-aware selective decision rule

*Written before γ was computed and before any seed-20–24 network was trained.*

**Origin, stated honestly.** The *form* of this rule came from exploratory
analysis of seeds 15–19, whose outcomes were already known. Its single free
threshold γ is calibrated only on the development seeds 5–9. It is tested once
on new seeds **20–24** (`results/stage1_prereg2`).

**Definitions (frozen estimator `hf_exact_FROZEN.py`):**

- **Structural statistic** `Λ₂` = `est.diag["lam2_max"]`. This is the
  largest, over 7 evenly spaced invariant-manifold points, of the
  second-largest eigenvalue *modulus* of the zero-input Jacobian. It is
  computed from the weights before any forecast.
- **Forecast safety margin** `r_H = (ε − max_{t ≤ H, scored steps} ê_t) / ε`,
  where `ê_t` is the reduced-model forecast error (normalized by ε = 0.25).

**Decision states** for a case (network, input sequence, H):

| state | condition |
|---|---|
| UNSUPPORTED | `Λ₂ ≥ 1` (structural abstention) |
| WITHHOLD | supported, but `r_H < γ` |
| APPROVE | supported and `r_H ≥ γ` |

Only APPROVE counts as approval.

**γ selection** (seeds 5–9 only). Take the smallest γ on the fixed grid
{0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30} such that the
development wrong-approval rate is ≤ 1% in every cell with ≥ 50 approvals.
If none qualifies, use the largest value.

**Cells.** Scenarios hold, x1 and x0.5 × H ∈ {250, 500, 1000, 2000} = 12
cells. Standard architectures only (rnn, gru, lstm; N = 32 and 128).

**Pass criteria.** All four of C1–C4 must hold. Denominators are all cases in
the cell, including cases in UNSUPPORTED networks.

- **C1 (risk).** `P(fail before H | APPROVE)` ≤ 2% in every cell with ≥ 50
  approvals.
- **C2 (utility; the criterion the previous rule failed).**
  `P(APPROVE | no failure before H)` ≥ 90% in every cell with ≥ 50
  truly-OK cases.
- **C3 (improvement over the old rule, applied to the same seeds 20–24).**
  Both must hold:
  - pooled over all cells, the new rule approves strictly more truly-OK
    cases than the old "T̂ > 1.10·H" rule;
  - pooled wrong-approval of the new rule ≤ old + 0.5 percentage points.
- **C4 (matched coverage).** In each cell, give each baseline (R, B1, B2,
  ranked by predicted failure time over all cases) the same number of
  approvals as the new rule. The new rule's wrong-approval must be ≤ every
  baseline in all 12 cells and strictly lower in ≥ 10.

**Reported, not part of the pass rule:**
- number of UNSUPPORTED networks;
- the rule's performance on supported networks only;
- per-network worst-cell wrong-approval.

**Stated limitation.** With ~30 networks, the λ₂ gate may meet zero or one
pathological network. This test validates the overall rule, not the
sensitivity of the λ₂ diagnostic.

### Amendment 7: frozen γ (development seeds 5–9), recorded before seeds 20–24 were collected or scored

- γ = **0.03**, i.e. 3% of ε.
- Development worst-cell wrong-approval: 0.65% (target ≤ 1% met).
- All development networks have Λ₂ < 1.

Hashes: `results/FROZEN_DECISION3_SHA256.txt`.

### Amendment 7: registered outcome (seeds 20–24; 30/30 networks gated; scored once; hashes verified)

**OVERALL: PASS (C1–C4).**

- **C1.** Wrong-approval 0.0–0.4% in every cell.
- **C2.** Useful-approval 0.930–0.992 in every cell.
- **C3.** 39,017 vs 37,290 truly-OK cases approved. Pooled wrong-approval
  0.08% vs 0.05% (within +0.5 pp).
- **C4.** Lower than every baseline at matched coverage in 12/12 cells.

**UNSUPPORTED networks encountered: 0.** The Λ₂ gate therefore made no
decisions in this test. Its sensitivity remains unvalidated, as stated in
advance.

The old 10% rule, applied to the same seeds, would again have missed the 90%
useful-approval bar in five long-horizon driven cells (0.66–0.89).
