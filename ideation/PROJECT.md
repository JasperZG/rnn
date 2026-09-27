# Predicting Hidden Failures of Recurrent Neural Networks from Their Attractor Defects

*Project description, September 26, 2026. Intended for Regeneron STS and
journal submission.*

---

## 1. One-paragraph summary

A trained recurrent neural network (RNN) can pass every validation test at its
training length and still fail silently later, when the sequences run longer
than anything it was checked on. We show that these **hidden failures can be
forecast from the network's weights before they happen**, trial by trial, far
beyond the training horizon. The idea is a single mechanism: a network that
solves a task approximately reproduces the task's own dynamics on a
low-dimensional attractor, and its hidden failure is the tiny per-step mismatch
between the two (the **conjugacy defect**) accumulated by the task's dynamics.

The key technical step is measuring that defect on the network's **true
invariant manifold**, which we obtain by solving an invariance equation. The
naive choice, the network's slowest states, inflates the defect 5–9× in
non-normal networks. With the defect measured this way, preregistered
forecasts on networks that were never inspected beforehand come out as
follows:

- 98–100% of individual trials predicted within a factor of 1.5, with median
  failure times within 1–2%;
- failures up to 100× the training length, using no run longer than the
  training length itself;
- vanilla RNNs, GRUs and LSTMs, on integration, memory-holding and oscillation
  tasks.

For one designed architecture, the task-conjugate cell, the defect is known in
closed form, so the same forecast can be made **before training**.

---

## 2. The problem: hidden failures

RNNs, and the recurrent state-space models descended from them, are trained
and validated on sequences of bounded length `T_train`. Many real uses run
much longer: memory over long delays, integration of evidence, generating
rhythms, tracking state across a long input stream.

A network that is almost right can look perfect at `T_train` because its
error per step is tiny. Those small errors keep accumulating, and the network
fails at some later time `T* ≫ T_train`. Standard validation cannot see this.
The literature documents the phenomenon empirically, e.g. length-generalization
failures in RNNs and state-space models, and neural-network performance on
formal-language tasks at longer lengths. But it gives no way to **predict when
a specific trained network will fail, on which inputs, before running it that
long.**

**Research question.** Given a trained RNN and the task it was trained on, can
we predict, without running the network beyond its training length, the time
at which each individual input sequence will first push the network's error
past tolerance?

---

## 3. Core claim

> **Hidden failures of trained recurrent networks can be forecast from their
> weights alone, before they occur. The failure is the network's conjugacy
> defect, measured on its invariant attractor manifold and accumulated by the
> task's own dynamics. For architectures whose defect is analytically known,
> the same forecast can be made before training.**

The claim has three parts, and each is tested separately:

1. **Mechanism.** Hidden failure = conjugacy defect × task dynamics. This is
   one equation for every task with a declared update rule.
2. **Measurement.** The defect is readable from the weights, provided it is
   measured on the invariant manifold, with inputs projected through the
   adjoint (left) eigenvector.
3. **Prediction.** Per-trial failure times up to 100× beyond training, on
   networks the method was never tuned on.

**What the claim does not say.** It does not predict what an ordinary RNN will
learn before training. The forecast is made *after training, before failure*.
It is a pre-deployment diagnostic. The one exception is the task-conjugate
cell (§6), where the prediction is possible before training.

---

## 4. Theory

### 4.1 Tasks as dynamical systems

A task is a declared update `z_{t+1} = F(z_t, u_t)` on a low-dimensional task
state `z` (e.g. a running sum, a stored value, a phase), driven by inputs
`u_t`. The network `h_{t+1} = f(h_t, u_t)` reports `ŷ_t = C h_t`.

| task | F | attractor the network must build |
|---|---|---|
| accumulation | `z + u` | line attractor |
| hold (analog memory) | `z` (after loading) | line attractor |
| oscillation | rotation `R_ω z` | limit cycle |

The attractor types come from the earlier stage of this project
(structure-from-task prediction).

### 4.2 Conjugacy and the defect

A network that solves the task is **approximately conjugate** to `F`: on its
attractor, parameterized by a coordinate `s` that the readout aligns with the
task state, the network's own one-step map `F̃(s,u)` nearly equals `F(s,u)`.

The **conjugacy defect** is

    δ(s, u) = F̃(s, u) − F(s, u).

### 4.3 The failure equation

Write the network's reduced state as `ŝ_t` and the true task state as `z_t`,
with error `e_t = ŝ_t − z_t`. Then exactly, on the manifold,

    e_{t+1} = [F(z_t + e_t, u_t) − F(z_t, u_t)] + δ(z_t + e_t, u_t).

**The task's dynamics carry the error forward; the architecture adds the
defect each step.** The failure time of a trial is the first `t` with
`|e_t| > ε`.

- **Neutral tasks** (hold, integration, rotation): `F` neither shrinks nor
  grows errors, so defects accumulate. These are the failures that stay
  hidden.
- **Contracting tasks** erase defects; **expanding tasks** reveal them
  quickly. Neither produces a late, hidden failure.

The first-order form `e_{t+1} ≈ J_F e_t + δ(z_t)` is *not* sufficient.
Stress tests confirmed that the defect must be evaluated at the network's own
state (§7.1).

### 4.4 Why the failure is hidden

For a defect of size `|δ|`, the accumulated error at the training horizon is
about `T_train |δ|`. It stays below tolerance whenever
`|δ| ≲ ε / T_train`, yet the network fails near `T* ≈ ε / |δ|`. A defect of
10⁻³ per step passes a 50-step validation (error 0.05) and fails near step
250. A defect of 10⁻⁵ fails near step 25,000. **Forecasting far-future
failures requires measuring defects this small.** That is the technical
challenge, and it is what §5 solves.

---

## 5. Method: reading the defect from the weights

All computations use the trained weights in double precision. No run exceeds
`T_train` steps.

### 5.1 Find the invariant manifold, not the slowest states

The obvious approach finds, for each readout value `s`, the state that moves
least (slow-point search). **This is wrong for non-normal networks**, where
the readout is not perpendicular to the fast directions:

- the slowest state sits slightly *off* the attractor;
- its fast transient leaks into the readout on the next step;
- the defect is inflated 5–9× (§7.2).

We instead solve the **invariance equation**, which says the velocity on the
manifold must be tangent to it:

    f(h(s), 0) = h(s + v(s))   ⇔   f(h_i) − h_i − v_i t_i = 0,   C h_i = s_i,

for manifold points `h_i` on a grid of `s_i`. Here:

- `v_i = C f(h_i) − s_i` is the zero-input drift;
- `t_i` is the tangent, from central differences.

We solve it jointly with L-BFGS, starting from slow points. This is the
parameterization method for invariant manifolds. The residual drops from
~10⁻³ to ~10⁻⁷–10⁻⁸.

### 5.2 Input response through the adjoint

An input kicks the state off the manifold. The kick's lasting effect on the
manifold coordinate is its **asymptotic phase**. To compute it:

1. Relax the kicked state for a few steps (R = 10) with no input.
2. Project the remaining displacement onto the manifold with the **left
   eigenvector** `ℓ` of the Jacobian (eigenvalue nearest 1), normalized so
   that `ℓᵀ ∂h/∂s = 1`.
3. Invert the known drift flow over those R steps.

The result is the reduced map `s' = s + D(s,u)` together with a readout
transient `τ(s,u)`, both tabulated on an (s,u) grid and interpolated with
cubic splines.

### 5.3 Forecast

For each test input sequence, iterate the one-dimensional reduced map and
record the first crossing of `|ŷ − z| > ε`. The cost does not depend on
network width, and nothing runs longer than `T_train`.

### 5.4 Oscillation (autonomous)

After the start pulse the network runs autonomously on its limit cycle. From
one `T_train` = 50-step window, compare the decoded state with itself one task
period later. This cancels orbit-shape harmonics and yields the per-step
**phase slip**, which we then propagate.

### 5.5 Validity diagnostic

The second-largest Jacobian eigenvalue `λ₂` at manifold points tells how
quickly off-manifold transients decay. It is reported for every network.

---

## 6. The exactly solvable case: predicting before training

In the **task-conjugate cell**,

    h_{t+1} = φ(V g F(Vᵀh_t, u_t)),    V balanced / harmonic frame,

the network is *exactly* conjugate to a known distortion of the task. The
defect is closed-form:

    δ = Ψ(gF) − F,    Ψ(q) = √N φ(q/√N)   (1-D),

Its exact failure laws are:

- lifetime ∝ N for tanh;
- lifetime ∝ N² for activations with no cubic term;
- in 2-D, phase drift is set by the frame's spherical-design order.

Here the defect, the learned gain and the failure distribution can be
computed and **sealed before training**. Across all 40 task-conjugate
networks in the stress test, forecasts matched perfectly (agreement 1.00).
This case shows the mechanism in its cleanest form and connects the project's
earlier "architecture-to-limit" derivation to trained networks. Derivations
are in `appendix_derivations.md`.

---

## 7. Evidence so far (preregistered stress test)

Everything is logged in `stress_test/`:

- `PROTOCOL.md`: rules fixed before each run, and every amendment with its
  timing;
- `RESULTS.md`: all outcomes, including the failures.

Tasks: accumulation, hold, oscillation, context-dependent integration,
flip-flop. Architectures: vanilla RNN, GRU, LSTM, task-conjugate. Widths 32
and 128, multiple seeds.

**A network enters the analysis only if it passes a training-horizon gate**
(95th-percentile error < ε/2). Only a network that looks healthy can have a
*hidden* failure.

### 7.1 Stage 1–2: the mechanism is right, the first estimator was not

A regression estimator (a polynomial fit to one-step samples from short
probes):

- beat the error-extrapolation baselines everywhere (0.79 vs 0.16–0.17 on
  accumulation);
- worked well for accumulation GRUs and RNNs (0.94 and 0.84);
- **failed the preregistered rule overall**.

Its accuracy decayed as failures grew more distant: 0.92–0.94 for T* < 150,
0.55 for T* > 1000. It also failed on LSTMs and on context-dependent
integration.

**Ablations** (which part of the defect carries the forecast):

| defect used | per-trial agreement |
|---|---|
| full, state-dependent | 0.61–1.00 |
| linear | 0.02–0.11 |
| constant | 0.26–0.32 |
| zero | 0 |

The forecast lives in the defect's shape along the attractor.

The first-order equation `e' ≈ J_F e + δ(z)` underperforms, as pre-stated.
The defect must be evaluated at the network's own state.

### 7.2 Stage 3: finding the right measurement

Each step below was a theory-driven fix, registered before it was run and
developed on seeds 0–9:

1. Exact evaluation of the weights' map on the manifold.
2. Adjoint projection of input transients, needed because λ₂ ≈ 0.95 means
   relaxation alone is too slow.
3. **Invariance instead of minimum speed.** This was decisive. Diagnosis:
   - the load step was accurate, but drift was overestimated 5–9×;
   - rank order was nonetheless near-perfect (ρ ≈ 1.00), which pointed to a
     single systematic bias;
   - the cause was non-normal leakage at speed minima.

   Solving the invariance equation fixed it.

The estimator was then frozen, with its SHA-256 hashes recorded.

### 7.3 Confirmation on untouched networks (seeds 10–14)

57 gated networks, trained after the freeze and never inspected before
scoring. **All preregistered criteria passed.**

| test | vanilla RNN | GRU | LSTM |
|---|---|---|---|
| driven accumulation: trials within 1.5× (old estimator) | 0.98 (0.85) | 0.99 (0.91) | 0.99 (0.61) |
| weaker inputs (later failures) | 1.00 (0.63) | 1.00 (0.74) | 1.00 (0.52) |
| hold (load, then no input) | 1.00 (0.47) | 1.00 (0.58) | 1.00 (0.30) |
| rank correlation, predicted vs measured per trial | 0.99–1.00 | 0.98–1.00 | 0.99–1.00 |
| median \|log(T̂/T)\|, driven | 0.007–0.017 | 0.002–0.010 | 0.001–0.004 |

- **Failures beyond step 1000:** agreement 1.00 (n = 7 driven, n = 2 hold).
- **Oscillation:** fail vs no-fail correct on 29/29 networks. All 11 failing
  networks were forecast from one 50-step window, e.g. 2171 vs 2174,
  4067 vs 4067, 1315 vs 1314, 2666 vs 2683.

### 7.4 Honest record of what failed

| failure | status |
|---|---|
| context-dependent integration | failed with the first estimator; not yet re-tested with the final one |
| noise-driven escapes from discrete attractors (flip-flop, σ = 0.1) | the 1-D reduction does not capture them; out of scope for now |
| the pre-registered "adjoint" noise estimate | wrong for contracting systems; documented |
| the planted-slow-mode test | invalid, because it broke the networks' own gate |
| an interim "resolution window" reformulation | withdrawn after stage 3 |

---

## 8. Contribution and novelty

Literature map: `literature_notes.md`, compiled from five web-verified sweeps.

| existing work | what it does | what this project adds |
|---|---|---|
| Fixed / slow-point analysis (Sussillo & Barak 2013); trained-RNN universality (Maheswaranathan et al. 2019) | find attractors in trained RNNs | turns the attractor into a **quantitative per-trial failure-time forecast** |
| Approximate continuous attractors (Ságodi et al. 2024); diffusion limits (Burak & Fiete 2012) | bounds on memory degradation | **exact forecasts** rather than bounds, validated per trial |
| Non-normal dynamics (Ganguli et al. 2008) | memory capacity | shows non-normality **biases slow-point defect estimates 5–9×**, with the fix |
| Length generalization (Delétang et al. 2023; Buitrago & Gu 2025; Chen et al. 2025) | binary generalization, or empirical length laws fitted after the fact | **predicted failure time per input, before running** |
| Error-extrapolation / scaling-law forecasting | extrapolates observed curves | outperforms extrapolation (0.98–1.00 vs 0.30–0.91) |
| Neural Engineering Framework; low-rank RNN theory | compile or reduce low-dimensional dynamics | a failure-time theory for *trained* networks, plus a before-training closed form for the task-conjugate cell |

The web-verified literature search found **no prior work that forecasts
per-trial hidden failure times of trained RNNs from their weights and
validates the forecasts preregistered.**

Classical tools used: the parameterization method for invariant manifolds
(Cabré, Fontich & de la Llave 2003; Haro et al. 2016) and asymptotic
phase / adjoint reduction (e.g. Nakao 2016). *These three citations were
added from memory and still need to be checked before use.*

---

## 9. Remaining work before submission

**Required for the core claim:**

1. **Context-dependent integration with the final estimator.** Extend the
   manifold method to multi-channel, context-gated inputs, with a manifold
   per context. It either joins the claim or becomes a stated limitation.
   This must be preregistered.
2. **Larger far-future sample.**
   - more seeds;
   - weaker inputs;
   - widths 256 and 512.

   Target n ≥ 30 networks with T* > 1000.
3. **Scaling check.** Confirm that forecast accuracy holds as width grows, and
   report the estimator's cost.
4. **Baselines, stated fully.**
   - error-curve extrapolation (power law, linear);
   - the regression estimator;
   - naive slow-point drift.

   The last one isolates the invariance contribution.

**Strengthening, optional:**

5. **The non-normality result.** Show directly that the slow-point bias scales
   with a measure of non-normality (e.g. the angle between readout and fast
   eigenvectors) across networks. This is a clean secondary finding.
6. **Before-training case.** Sealed forecasts for task-conjugate cells across
   widths and activations, as the exactly solvable companion.
7. **One more task family** with a declared F, e.g. input-switched rotation
   (state tracking), to test generality.

**Stated limitations** (no work needed beyond writing them up):

- noise-driven discrete escapes;
- chaotic or expanding tasks, which fail fast and visibly, so they are not
  hidden;
- tasks without a declared update rule.

---

## 10. Figures (planned)

1. **The phenomenon.** Error vs time for several trials: flat through
   `T_train`, then crossing tolerance at different later times. Validation
   passes, the network fails.
2. **The mechanism.** The attractor in state space, the defect along it, and
   the task dynamics carrying the error. One equation.
3. **The measurement.** Slow points vs the invariant manifold in a
   non-normal network: the readout leak and a 5–9× drift error, fixed by
   invariance.
4. **Headline.** Predicted vs measured failure time for **every trial** of
   every confirmation network (RNN, GRU, LSTM; driven, hold, oscillation), on
   the diagonal across two orders of magnitude, with the baselines off it.
5. **Oscillators.** A 50-step window forecasting failures thousands of steps
   later.
6. **Before training.** Task-conjugate cells: sealed forecast vs outcome.
7. **Ablations and limits.** Which parts of the defect matter; where the
   method fails.

---

## 11. Timeline (suggested)

| weeks | work |
|---|---|
| 1 | reconcile with the lab-machine code; package the estimator; headline figure from existing data |
| 2–3 | preregistered context-dependent integration; larger far-future sample and widths |
| 4 | non-normality analysis; before-training companion |
| 5–6 | writing: STS paper (research question → hypothesis → methods → results → significance) and journal manuscript |

Compute needed: training takes seconds to minutes per network on CPU. The full
confirmation set trains in under an hour on a laptop, and the lab GPU is
optional.

---

## 12. STS framing

- **Question.** Can we tell, before it happens, when a neural network that
  looks perfect will fail?
- **Hypothesis.** Hidden failure is the network's tiny mismatch with the task
  on its attractor, accumulated over time. Measured correctly from the
  weights, it predicts when each input will fail.
- **Method.** Invariant-manifold analysis of trained networks; preregistered
  tests on networks trained after the method was frozen.
- **Result.** 98–100% of trials predicted, medians within 1–2%, failures up to
  100× beyond the tested length; 29/29 oscillators.
- **Significance.** A way to certify recurrent models before deployment, in
  settings where the costly failures are the ones testing cannot see.
- **Scientific integrity.** Preregistration, frozen code with hashes, untouched
  confirmation networks, and a complete record of failed attempts.

---

## 13. Reproducibility

- Code: `stress_test/hf_tasks.py`, `hf_core.py`, `hf_exact.py` (frozen copy
  `hf_exact_FROZEN.py`), `run_stage1.py`, `run_stage3.py`, `score_confirm.py`.
- The verdict is reproduced with `python score_confirm.py`.
- Environment: Python 3.9, torch 2.8 (CPU), numpy, scipy.
- Trained weights: `stress_test/nets/` (gitignored; to be archived with the
  paper).
