# Predictable Recurrence

### From task-conjugate cells to an engine that forecasts what recurrent networks learn and where they fail

*Research ideation, September 26, 2026. It builds on the task-conjugate
derivation of September 21, 2026.*

*Companion files:*
- `appendix_derivations.md`: proofs and formulas (A.1–A.7)
- `literature_notes.md`: annotated, verified literature map
- `checks/verify_theory.py`: numerical checks C1–C8, with output in
  `checks/verify_theory_output.txt`
- `PLAN.md`: the working plan this document follows

---

## Summary

The September 21 derivation defines a recurrent cell whose N-dimensional
hidden dynamics are *exactly* a one- or two-dimensional task map, distorted by
a closed-form function `Ψ`. Width and activation determine `Ψ`. That makes it
possible to predict, before training, both the learned gain and the step at
which the network's error first crosses a threshold. The predictions are then
sealed and audited. This is a good seed: the forecast is exact, cheap to
compute, and made before training.

This document proposes to grow that seed into a **predictable engine** for
recurrent networks. The engine is a pipeline that takes a declared task, a
noise and precision model, and a required operating horizon. It then compiles
an architecture, forecasts the learned solution and the full distribution of
failure times, seals the forecast, trains, and audits.

Checking the current derivation numerically and against the literature turned
up six problems. They need to be fixed before scaling up:

1. **Closure is generic, and it is already known.** Any encoder/decoder
   bottleneck cell closes exactly on its readout (C1, 0.0 error). For low-rank
   RNNs this is the finite-N reduction of Pals et al. (2024) and Valente et al.
   (2022). The hidden width adds no dynamical degrees of freedom.
2. **In the noise-free setting, the "lifetime ∝ N" law is not a width law.**
   With a ±1 frame and an odd activation, all N units are sign-flipped copies
   of *one* neuron read at input scale `1/√N`. The N and N² laws (confirmed to
   within one step in C5) measure *operating amplitude*.
3. **Heterogeneity removes the law entirely.** Random gains and biases plus a
   pre-solved decoder reach one-step error ~1e−6 already at N = 16 (C7).
   Precision is limited by **noise and number format**, not by nonlinearity.
4. **Training only `g` is not evidence about learning.** In an exactly closed
   cell the native and shadow losses are the same function. "The optimizer
   reached `g_pred`" is an optimization check.
5. **The four-sign 2-D frame is anisotropic.** It is a spherical 3-design, not
   a 4-design, so its cubic distortion varies by 23.6% with direction. A
   rotation task accumulates 2.55 rad of phase error in 20,000 steps. Harmonic
   frames reduce that to 1e−3 (hexagon) and 9e−9 (M = 6) (C3, C8).
6. **NEF and spike-coding networks are close prior art.** They use the same
   encoder/decoder sandwich for declared low-dimensional dynamics. The novelty
   must be located precisely (§3).

The expanded program keeps what is genuinely new, namely an **exact, cheap
shadow turned into a sealed, quantitative forecast**. It then makes width,
learning, and prediction non-trivial, on four fronts:

- **A distortion theory.** `Ψ` is written in terms of the frame's moment
  tensors. Three independent precision knobs follow: width, activation Taylor
  order, and **frame design order**.
- **An error-propagation theory.** The task's stability class (contracting,
  neutral, or expanding) converts one-step error into a lifetime law. This
  predicts a sharp **phase boundary** for contracting tasks and only
  logarithmic returns to width for expanding ones.
- **A noise and precision theory.** Homogeneous frames are capped at a
  noise-limited lifetime exponent of `p/(p+1)` (measured 0.53 against a
  predicted 0.5 for tanh). Heterogeneous encoders can exceed this, which makes
  width a genuine resource.
- **Tiered architectures.** They run from exactly closed cells (T0), through
  compositions of those cells (T1), to rank-k + random-bulk networks with all
  weights trained (T2). T2 has only an *asymptotic* mean-field shadow, so
  forecasting what gradient descent finds is a real, falsifiable claim there.
  Generic GRU/LSTM/SSM layers (T3) serve as scope controls and connect back to
  the earlier structure-prediction project.

The literature sweep found **no** prior work that seals a pre-training
forecast of an RNN's failure horizon and then tests it, and no
modified-equation analysis of recurrence *along the sequence*. Both are open.

---

## 1. Where the project stands

### 1.1 The cell (recap)

Task state `z ∈ R^k` (k = 1, 2), declared update `F(z,u)`, balanced frame
`V ∈ R^{N×k}` with `VᵀV = I`:

    q_t = g F(Vᵀh_t, u_t),    h_{t+1} = φ(V q_t),    z_{t+1} = Vᵀh_{t+1} = Ψ(g F(z_t,u_t)),
    Ψ(q) = Vᵀφ(Vq) = √N φ(q/√N)         (1-D, odd φ)

The current protocol runs in five steps:

1. Grid-and-refine `g` on the exact shadow loss.
2. Propagate the shadow on the task distribution.
3. Solve for the first error-threshold crossing.
4. Seal `g_pred`, the predicted error curve, and the predicted boundary.
5. Train the native network from `g = 1`, then audit parameter agreement,
   boundary agreement, and shadow/native equality.

The implementation (`src/dynaspec/task_conjugate.py`,
`tools/task_conjugate_development.py`) lives on the lab machine and was not
available while this was written. Nothing here assumes its contents.
Reconciling the two is Phase 0 of the roadmap (§9).

### 1.2 Self-audit and what to redo

| # | Problem | Evidence | Fix in this program |
|---|---|---|---|
| A1 | Closure holds for any encoder/decoder, and the finite-N closure of low-rank RNNs is known | C1; Pals et al. 2024 | Cite it as known. Locate novelty in forecasting and sealing, not in closure. |
| A2 | The ±1 frame is one neuron replicated, so the N-laws are amplitude laws | algebra; C5 | Relabel them as amplitude laws. Make width real through noise, precision, and heterogeneity (§4.4). |
| A3 | Heterogeneous encoders + solved decoders beat the N-laws by orders of magnitude | C7 | Add Tier T0-H. The binding constraint becomes noise, per NEF. |
| A4 | Training only scalar `g` in an exact cell makes the learning claim tautological | A.1 corollary 1 | Present the T0 agreement as an *optimizer audit*. Move learning claims to T2, where closure is only asymptotic. |
| A5 | The four-sign 2-D frame is anisotropic | C3, C8 | Use harmonic frames (M ≥ 3). Keep four-sign as an ablation with a predicted phase-drift rate. |
| A6 | The development cohort shares one evaluation stream | Sept 21 text | Keep it as an engineering check only. Every claim rests on the confirmation protocol (§8). |
| A7 | Thresholded boundaries are discontinuous metrics | Schaeffer et al. 2023 | Seal the *continuous* error curve and the full first-passage distribution, not only the crossing step. |

None of these invalidate the September 21 result. They change what it is
evidence *for*. It shows that a closed-form architecture-to-limit map exists
and can be audited. It does not yet show that learning is predictable.

---

## 2. Thesis

> For a declared task interface — a finite-dimensional update `F`, an input
> distribution, a tolerance, and a noise/precision model — the composition
> **task stability class ∘ architectural distortion ∘ noise floor** determines
> the learned solution and the full distribution of failure times. That
> composition can be computed before training at a cost independent of width.
> In exactly closed cells the forecast is an engineering certificate. In
> asymptotically closed cells it is a falsifiable scientific prediction about
> gradient descent.

Three consequences make this an *engine* rather than one more result:

1. **Forward forecasting.** Given an architecture, predict the learned
   parameters, the error curve with Monte Carlo bands, the survival curve of
   failure times, and the censoring fraction.
2. **Inverse design.** Given a required horizon `T_req` at confidence `1 − α`,
   return the cheapest width, activation, frame, encoding scale, and number
   format that meets it. Seal that too.
3. **Audit.** Compare every sealed quantity to independent measurements, with
   calibrated statistics and pre-declared kill criteria.

---

## 3. Position in the literature

Full annotations are in `literature_notes.md`. The short version:

| Line of work | What it already does | What it does not do (our opening) |
|---|---|---|
| NEF / Nengo; spike-coding networks | Compile declared low-dimensional dynamics into rank-k encoder/decoder networks. Precision grows with N. | Closed-form distortion and lifetime laws; gradient training toward a *sealed* forecast; failure-horizon forecasts |
| Low-rank RNN mean field (Mastrogiuseppe & Ostojic; Beiran; Dubreuil; Pals) | Exact or asymptotic latent reductions. Gains `⟨φ'⟩`. Fixed-point census. | Training from a declared task to a sealed learned map; horizon forecasts |
| Learning theory for RNNs (Saxe; Proca; Bordelon et al. 2025; Ger & Barak 2026) | Analytic learning trajectories, but linear, or low-rank **without bulk** | Nonlinear distortion, random bulk, finite-N error bars, and a sealed test |
| Mean-field signal propagation (Poole; Schoenholz; Chen–Pennington–Schoenholz; Gilboa) | Timescales predicted at initialization, statistically | Exact finite-N maps for designed frames; horizons *after* training |
| Continuous-attractor robustness (Seung; Koulakov; Burak & Fiete; Ságodi et al. 2024; Mo 2026) | Bounds on drift and diffusion near approximate attractors | A case where the perturbation is known *exactly*, which lets us test how tight those bounds are |
| SSM / linear-RNN design (HiPPO, LMU, S4, LRU, Mamba-3, LinOSS) | Task-derived linear recurrences with per-mode horizons | Nonlinear or selective layers; failure length forecast before training |
| Length generalization (Delétang; Buitrago & Gu; Stuffed Mamba; François et al.) | Binary generalization, or empirical length laws fitted post hoc | A sealed, pre-training failure length with an audit |
| Scaling-law forecasting (Kaplan; Chinchilla; μTransfer) | Predict loss or hyperparameters | Predict the *solution* and a *failure horizon* |
| Compilation (Tracr, RASP, reservoir programming) | Write down weights | Claim that gradient descent *recovers* them, and test it |

**Novelty, stated narrowly.** We claim no new closure identity and no new
encoding scheme. The contributions are:

- (i) closed-form distortion and lifetime laws, including frame design order
  and noise-limited exponents;
- (ii) a sealed forecasting protocol for learned solutions *and* failure
  horizons;
- (iii) extending that protocol to networks where closure is only asymptotic
  and all weights are trained;
- (iv) inverse design of architecture for a required horizon.

---

## 4. Theory

Derivations are in `appendix_derivations.md`. Here we state results and what
they imply for design.

### 4.1 Closure (A.1; known)

For `h_{t+1} = φ(E G_θ(D h_t, u_t) + b)`, the readout `z = Dh` obeys
`z_{t+1} = Dφ(E G_θ(z_t,u_t) + b)` exactly, for any `E, D, b, θ`. Native and
shadow gradients coincide in exact arithmetic. **Design implication:**
everything in an exactly closed cell can be forecast perfectly *except* the
effects of floating point, stochastic minibatches, and noise. Those are
exactly what the engine should model.

### 4.2 Distortion is a frame-moment problem (A.2–A.3)

For odd `φ = x + Σ c_{2j+1} x^{2j+1}` and a frame with rows `V_i`:

    Ψ(q) = q + Σ_j c_{2j+1} M_{2j+2}[q,…,q],     M_{2m} = Σ_i V_i^{⊗2m}.

Three regimes follow from this one formula:

- **1-D.** The cubic coefficient is `c_3 Σ v_i⁴ ≥ c_3/N`, with equality only
  for the balanced frame (C2: measured to four significant figures). The
  balanced frame is the unique optimum.
- **Spherical designs.** If the normalized rows form a 4-design, the cubic
  term is purely radial, with coefficient `c_3 · 3k/((k+2)N)` (Venkov's
  identity; C3 exact). In 2-D, harmonic frames with M directions are
  `(2M−1)`-designs, so anisotropy first appears at order `2M+1`.
- **Random frames.** `Ψ(q) ≈ q E[φ'(|q|ξ/√N)]`, a radial Bussgang gain, plus
  uncorrelated distortion of relative size `N^{-1/2}` (C4: ratio 3.79 against
  4).

**There are three independent precision knobs:**

| knob | controls | law |
|---|---|---|
| width N (via amplitude) | radial error | `∝ |q|^{2p}/N^p` |
| activation Taylor order p | radial error exponent | tanh `p=1`; no-cubic `p=2` (e.g. `x/(1+x⁴)^{1/4}`, same saturation level) |
| frame design order t | **anisotropic** error (phase or direction drift) | first anisotropic term at order `t+2` |

The third knob is new. It explains the rotation result (C8) and should be
reported as a design law of its own.

### 4.3 Error propagation: task stability class → lifetime law (A.7)

Write `e_{t+1} = J_F e_t + δ(z_t,u_t) + ζ_t`, where `δ` is the one-step
architectural error and `ζ` is noise. The task's Jacobian class, measured
along the task distribution, decides how one-step error becomes a lifetime:

| class | examples | lifetime law | design consequence |
|---|---|---|---|
| contracting (`ρ < 1`) | driven contraction, leaky integration | **no crossing** if `δ < ε(1−ρ)` | a sharp critical width `N_c`. Censored runs are the *prediction*, not missing data |
| neutral, systematic δ | hold, accumulation | `t* ≈ ε/δ` | power law in N |
| neutral, averaging δ | anisotropic error on rotations | `t* ≈ ε²/(δ²τ)` | quadratic gain from averaging |
| neutral + noise | any neutral task | `t* ≈ ε²/σ_eff²` | noise floor dominates at large N |
| expanding (`ρ > 1`) | unstable or chaotic task maps | `t* ≈ log(ε/δ)/log ρ` | **width buys a constant number of steps** |

This table is the core of the engine's generality. Adding a new task requires
only its Jacobian class along the task distribution. The architecture supplies
`δ(N)` and `σ_eff(N)`. This is the recurrence-along-the-sequence version of
backward error analysis in numerical integration: the network is an integrator
of `F`, `Ψ ∘ gF` is its modified map, and the lifetime is the time for secular
drift to reach tolerance (Hairer, Lubich & Wanner 2006). The literature sweep
found no ML work that uses this to forecast failure length.

### 4.4 Noise and precision: where width becomes real (A.5)

With per-neuron noise `σ`, the encoding amplitude `x` trades distortion
(`∝ x^{2p}/N^p`) against noise (`∝ σ/x`). Optimizing `x` gives

    t*_max ∝ ε^{(2p+1)/(p+1)} c^{−1/(p+1)} σ^{−2p/(p+1)} N^{p/(p+1)}.

- **Homogeneous frames are capped** at exponent `p/(p+1)`. C6 measures 0.53
  for tanh (predicted 0.50) and 0.58 for the no-cubic activation (predicted
  0.67, approached slowly; H2 needs `N ≥ 4096`).
- **Heterogeneous encoders** (random gains and biases, decoder solved before
  training on the shadow) are not bound by the cap in principle. With
  `σ = 0.02`, median hold lifetime grows 14 → 32 → 99 → 136 for N = 16 → 1024
  (C7), but it flattens. The one-step least-squares decoder hits a
  bias–variance floor. **Hypothesis H2b:** a decoder chosen on the *multi-step*
  shadow objective restores the diffusion-limited exponent of 1 (the
  Burak–Fiete regime).
- **Number formats are noise.** Rounding in fp32, bf16, or fp8 acts as
  `σ_eff ≈ 2^{−m}/√12` per operation, where m is the mantissa width. For
  example, the no-cubic activation at `N = 4096` has a deterministic lifetime
  of about 4×10⁶ steps (A.4 formula). Over that many steps, float32 rounding is
  no longer obviously negligible. Which effect dominates is something the
  engine should forecast; it has not been checked here. The engine
  should forecast **iso-lifetime curves in (width, bits)** and predict when
  bf16 inference breaks long-horizon memory. That result would matter directly
  in ML practice.

Burak & Fiete predict *unbiased* diffusion `∝ t/N`, while `Ψ` predicts
*systematic*, amplitude-dependent drift `∝ z³t/N`. Both scale as 1/N but
leave different fingerprints: a mean shift toward attractors versus spreading
variance. Separating them in one experiment is itself a clean result.

### 4.5 Asymptotic shadow for rank-k + random bulk (A.6)

    h_{t+1} = φ(J h_t + E q_t + b),    q_t = G_θ(D h_t, u_t),    J_ij ~ N(0, g_J²/N).

Closure is lost. In the large-N limit (the discrete-time version of
Mastrogiuseppe & Ostojic 2018), pre-activations carry Gaussian bulk input of
variance `Δ` set self-consistently, and the latent obeys
`z_{t+1} = Ψ_Δ(G_θ(z_t,u_t)) + ζ_t`. Here `ζ_t` is an `O(N^{-1/2})` colored
fluctuation with the bulk's correlation time. None of the following
predictions is true by construction:

- the calibrated gain rises with `g_J` to compensate the lower `⟨φ'⟩(Δ)`;
- the bulk acts as self-generated colored noise, so hold lifetime
  `∝ N/(C_Δ(0) τ_Δ)`, computable from mean-field autocorrelation;
- with **all** weights trained, `ΔW` is low-rank and aligned with the task
  subspace (Schuessler et al. 2020). The learned latent map converges to the
  shadow optimum within the `O(N^{-1/2})` band given by DMFT finite-width
  theory (Bordelon & Pehlevan 2023). The learning shadow is the overlap-space
  flow of Ger & Barak (2026) extended with the bulk those authors leave open.

### 4.6 Learning dynamics in the shadow

- **T0:** training trajectories are *identical* in exact arithmetic. The
  forecastable content is the stochastic part. With independent minibatch
  streams, the shadow's multi-start landscape predicts the **distribution** of
  basins that native SGD reaches (H5). This is well posed once `θ` has more
  than one parameter.
- **T2:** gradient flow in overlap space plus the bulk kernel predicts the
  trajectory of `θ(t)` and the rich/lazy regime boundary as a function of init
  scale and `g_J` (Liu et al. 2024; Bordelon et al. 2025).

---

## 5. Architecture tiers

| tier | cell | shadow | what the forecast means | scale target |
|---|---|---|---|---|
| **T0** | exact bottleneck. Frames: balanced / harmonic / design (T0-D) or heterogeneous with solved decoder (T0-H). Learned latent map `G_θ` (not just `g`) | exact | engineering certificate plus optimizer audit | N up to 10⁵ (cost O(Nk) per step) |
| **T1** | compositions of T0 modules coupled through their latents (e.g. a rotation module driving an accumulator) | exact, product of modules | system horizon from module error budgets and coupling Jacobians | many modules, 10⁵–10⁶ units |
| **T2** | rank-k + random bulk, **all weights trained** | asymptotic mean field + O(N^{-1/2}) | falsifiable prediction about what gradient descent finds and where it fails | N = 256–8192 |
| **T3** | generic RNN / GRU / LSTM / diagonal SSM / minGRU | none a priori. Effective `δ(N)` fitted on short horizons | forecast long-horizon failure from short-horizon measurement. This is the scope control and links to the earlier X-ray | N as in the earlier project |

**Why this ordering matters.** The criticism "you predicted a function you
wrote down" applies with full force to T0, less to T1, and not to T2 or T3.
The program ranks its claims in that order.

**The SSM bridge (T0 → modern sequence models).** A T0 cell with linear `F`
and harmonic 2-D frames is a nonlinear, complex-diagonal SSM mode. An
input-switched rotation module tracks `Z_m` state exactly. Placing such
modules in a sequence model and forecasting their length-generalization
horizon connects the program to state tracking (Grazzi et al. 2025; Merrill
et al. 2024) and to length failure (Buitrago & Gu 2025; Stuffed Mamba). Those
works either give binary feasibility or fit length laws after the fact.

---

## 6. The engine

```
 Spec ──► Compile ──► Forecast ──► Seal ──► Train ──► Audit
  │          │            │          │         │         │
  F, k,      N, φ, frame,  θ_pred,    sha256    native,   identities, θ agreement,
  P(u), P(z0), scale x,    error-curve of the   from      curve calibration,
  loss, T_train, decoder D, bands,     forecast declared survival comparison,
  ε, T_req, α, format      survival   + code    init;     exact-map conjugacy
  noise model              S(t),      commit    indep.    (CSA-style)
                           censoring  + seeds   streams
```

**Spec.** Declared update `F` and its dimension, input distribution `P(u)`,
initial-state distribution, training objective and horizon `T_train`,
tolerance `ε`, required horizon `T_req` and confidence `1 − α`, noise model
(per-neuron `σ`, number format, optional neuron dropout or ablation), and
compute budget.

**Compile.** Choose `(N, φ, frame, x, D, format)` by optimizing shadow
objectives. The optimization uses a fixed global grid with bounded
refinement, as now, so the optimizer's choice cannot silently define the
prediction. The inverse-design mode minimizes cost subject to
`P(t* ≥ T_req) ≥ 1 − α` on the forecast population.

**Forecast.** Monte Carlo on an *independent forecast population* gives:

- `θ_pred` (with a basin distribution when `θ` is multi-dimensional);
- the continuous error curve `E(t)` with bands;
- the Kaplan–Meier survival curve `S(t)` of first crossings;
- the censoring fraction at `T_max`;
- a parametric first-passage fit where theory gives one (inverse Gaussian for
  drift plus diffusion).

**Seal.** Hash the forecast artifact, together with the code commit, RNG
seeds, and audit code, before any native training. Blind-analysis discipline
(Klein & Roodman 2005): pass/fail thresholds are fixed in the sealed file.

**Train.** Native network from a declared init that never reads the sealed
file. The measurement population uses independent seeds.

**Audit.**

- (i) Identity audits: shadow/native equality on fixed trajectories; frame
  moment identities.
- (ii) Parameter agreement within the sealed tolerance.
- (iii) Curve calibration: PIT histograms and interval coverage for `E(t)`.
- (iv) Survival comparison: log-rank test and ratio of median lifetimes, with
  censored runs retained.
- (v) Exact-map conjugacy check using the known `Ψ` (CSA-style), with
  DSA/InputDSA as secondary audits.

**Deliverable format.** One `forecast.json` per condition and one `seal.txt`
of hashes. An **atlas figure** plots sealed-versus-measured survival curves
across task × N × φ × frame × format. This is the successor to the
Competence Atlas of the earlier project.

---

## 7. Research program

### 7.1 Identities (must hold; any failure is a bug)

- **I1** closure and gradient equality (C1).
- **I2** the moment-tensor distortion formula (C2).
- **I3** radiality for designs (C3).
- **I4** the Stein limit for random frames (C4).

These become unit tests, as the lab code's tests already are for the 1-D and
2-D cases.

### 7.2 Hypotheses (could fail; each has a kill criterion fixed in advance)

| ID | Hypothesis | Tier | Test | Kill criterion |
|---|---|---|---|---|
| **H1** | Stability class determines the lifetime law: contracting tasks show a critical width `N_c` (sharp censoring boundary), neutral tasks a power law, expanding tasks `log N` | T0 | sweep N ∈ {16…16384} × ρ for each class; independent populations | measured exponents differ from predicted by > 0.15, or `N_c` misplaced by > 25% |
| **H2** | Homogeneous frames are capped at noise-limited exponent `p/(p+1)`. **H2b:** heterogeneous encoders with a multi-step-solved decoder exceed it, approaching 1 | T0-D, T0-H | amplitude-optimized lifetimes vs N and σ | H2: exponent outside ±0.1 at N ≥ 4096. H2b: heterogeneous exponent ≤ homogeneous |
| **H3** | Frame design order controls phase drift on rotation and input-switched rotation, with drift rate predicted from the first anisotropic moment | T0 | four-sign vs M = 3, 4, 6 at matched N | harmonic frames fail to beat four-sign by the predicted factor (± 2×) |
| **H4** | Number formats act as a predictable noise source. Iso-lifetime curves in (N, bits) forecast bf16/fp8 failure lengths | T0 | fp64/fp32/bf16/fp8 emulation | median lifetime off by > 2× |
| **H5** | With multi-parameter `θ` and independent SGD streams, the shadow landscape forecasts the *distribution* of basins native training reaches | T0 | 200 seeds per condition | calibration fails (PIT KS test p < 0.01) |
| **H6** | Rank-k + bulk: `g_pred(g_J)` and lifetime `∝ N/(C_Δ τ_Δ)` from the mean-field shadow match fully trained networks within the predicted finite-N band | T2 | N ∈ {256…8192}, g_J ∈ [0, 1.5] | learned latent map outside the `O(N^{-1/2})` band at N ≥ 1024, or lifetime off by > 2× |
| **H7** | In T2, learning is low-rank and task-aligned, and the rich/lazy boundary in (init scale, g_J) is where theory puts it | T2 | rank and alignment of ΔW; boundary sweep | boundary misplaced by > 25% |
| **H8** | Generic RNNs obey the same stability-class laws, with `δ_eff(N)` fitted on short horizons forecasting long-horizon failure | T3 | GRU/LSTM/vanilla/minGRU/diagonal SSM on hold, accumulation, rotation | forecasts no better than the naive baselines (§8.4) |
| **H9** | Inverse design: compiled `N_min` meets `T_req` with coverage ≥ 1 − α, and `N_min/2` fails | T0, T1 | 20 random specs | coverage below 1 − α − 0.05 |
| **H10** | Composition: a T1 system's horizon follows from module error budgets and coupling Jacobians | T1 | rotation → accumulator; switched rotation → hold | median off by > 2× |
| **H11** | Task-conjugate `Z_m` modules inside a sequence model have forecastable length-generalization horizons; baselines do not | T0 → SSM | forecast vs measured, compared with Mamba/LRU horizons | forecast error ≥ baselines' |

**The most important "could fail" results** are H6, H7, H8, and H11. They are
where a reviewer cannot say the prediction was built in. H1–H4 are the most
*useful* results. They are close to certain in T0 but give design laws that
nobody currently writes down.

### 7.3 Experiments that would make a strong first paper

A paper built on T0 alone, titled something like *"Precision as a design
variable in recurrent networks"*:

1. **Three-knob atlas.** Lifetime vs (N, activation order, design order) on
   hold and rotation. Sealed forecasts vs measurement, with the saturation-
   matched activation control. No prior comparison of this kind exists.
2. **Stability-class phase diagram.** (N, ρ) plane for leaky integration,
   with censored runs forming the predicted boundary.
3. **Noise-limited exponents.** Homogeneous vs heterogeneous (H2/H2b),
   together with the Burak–Fiete drift-versus-diffusion separation.
4. **Format forecasts.** When bf16 breaks long memory (H4).
5. **Inverse design demonstration.** (H9)

The second paper would be T2 (H6/H7): the first sealed forecast of what
gradient descent finds in a nonlinear RNN with a random bulk. The third would
be the engine and benchmark: an "RNN-Tracr with a horizon column", with
Delétang-style tasks reporting predicted vs observed failure lengths.

---

## 8. Evaluation protocol

### 8.1 Populations

Every condition draws three independent populations:

- **training bank**: the declared training task samples;
- **forecast population**: larger; used only by the forecaster before sealing;
- **measurement population**: independent histories; used only for audit.

Deterministic tasks such as hold and rotation still sample initial states
independently. The development cohort (shared stream) stays an engineering
check and never supports a claim.

### 8.2 Statistics

- **Survival:** Kaplan–Meier curves with censoring at `T_max` retained.
  Log-rank tests between forecast and measured curves. Report the ratio of
  medians with bootstrap CIs.
- **Curves:** probability integral transform (PIT) calibration and interval
  coverage for `E(t)`. Report the continuous curve alongside the threshold
  crossing (to avoid the "mirage" objection).
- **Parameters:** sealed tolerance intervals; the basin distribution is
  tested by a χ² or KS test against the forecast.
- **Multiplicity:** hypotheses are pre-declared (§7.2). For exploratory
  sweeps, report false discovery rate.

### 8.3 Blinding and sealing

Forecast files, pass/fail thresholds, and analysis code are hashed and
committed before training. Audits run by script against the sealed file.
Any post-seal change to the analysis is logged as a deviation.

### 8.4 Null models the engine must beat

1. **Universality null.** Any trained RNN matches `F` up to DSA
   (Maheswaranathan et al. 2019). The engine must predict *quantities*, not
   only structure.
2. **Naive error multiplication.** `t* ≈ ε/δ_1` from the measured one-step
   error, ignoring the stability class.
3. **Short-horizon extrapolation.** A scaling law fit to lifetimes measured
   at `T ≤ T_train`.

The engine must beat each by a pre-declared margin on the median absolute
log-ratio.

---

## 9. Roadmap

| phase | content | compute | exit condition |
|---|---|---|---|
| **0. Reconcile** (1–2 wk) | Re-establish SSH to the lab machine; inventory the running jobs and `dynaspec` code; port I1–I4 as unit tests; swap the four-sign frame for harmonic frames with four-sign kept as an ablation; separate the development cohort from confirmation | laptop + lab machine | lab code passes I1–I4; running jobs catalogued |
| **1. T0 laws** (3–5 wk) | H1–H4 with sealed forecasts; three-knob atlas | CPU / RTX 5070 (per-step cost O(Nk)) | paper-1 figures |
| **2. Engine** (3–4 wk) | spec / compile / forecast / seal / audit tooling; inverse design (H9); basin forecasts (H5); T1 composition (H10) | 5070 | `forecast.json` / `seal.txt` pipeline; H5, H9, H10 decided |
| **3. T2 theory + runs** (6–10 wk) | discrete-time mean-field shadow with bulk; overlap learning flow; H6/H7 | A100 (N = 8192, T = 1000, batch 64: roughly 3×10¹³ FLOP per optimizer step, so tens of minutes per run) | paper 2 |
| **4. T3 + SSM bridge** (4–6 wk) | H8 on GRU/LSTM/minGRU/diagonal SSM; H11 with `Z_m` modules | 5070 / A100 | bridge results; benchmark draft |
| **5. Confirmation** | full preregistered confirmation cohort for the headline claims | A100 | registered-report style write-up |

---

## 10. Risks, and what would change the program

- **Tautology critique (T0).** Mitigation: rank claims by tier, present T0 as
  a certificate plus design laws, and put learning claims only in T2/T3.
- **The T2 mean-field shadow is too crude at practical N.** Colored bulk
  noise and trained-bulk correlations (Schuessler et al. PRR 2020) may break
  the O(N^{-1/2}) band. Then H6 fails *informatively*: we learn at what width
  gradient descent stops being forecastable. Report that width as a result.
- **Heterogeneous decoders do not beat the cap (H2b fails).** Then width is
  only valuable through noise averaging at the p/(p+1) rate. That is a clean
  negative result with consequences for NEF-style designs.
- **Declared-F scope.** The engine needs a declared `F`. For learned or
  unknown `F`, the shadow becomes a learned model and the forecast becomes a
  generalization claim. This is out of scope and should be said plainly.
- **Floating-point artifacts.** The quintic `N²` law reaches lifetimes (~10⁶
  steps) where fp32 rounding may matter. Use fp64 for identity audits and model formats
  explicitly (H4).
- **The lab-machine runs may already contradict something here.** Phase 0
  exists for this reason. This document was written without access to them.

---

## 11. Novelty ledger

| claim | status |
|---|---|
| exact finite-N closure of bottleneck / low-rank cells | **known** (Pals et al. 2024; Valente et al. 2022). Cite it |
| encoder/decoder compilation of declared dynamics | **known** (NEF; spike-coding networks) |
| distortion via frame moment tensors; design order as a precision knob | new as a design law, as far as the sweep found (the math is classical: Venkov/Delsarte) |
| stability-class lifetime table, incl. contracting phase boundary and log-N expanding regime | new as a forecasting tool (the numerical-analysis analogue is classical) |
| noise-limited exponent `p/(p+1)` for homogeneous frames | new, as far as the sweep found |
| sealed pre-training forecast of learned solution + failure horizon, with audit | **no precedent found** |
| mean-field shadow with random bulk and all weights trained, sealed | open problem (Ger & Barak 2026 leave the bulk open) |
| format-aware (bf16/fp8) horizon forecasts | no precedent found |

---

## Appendix B: numerical checks (summary)

From `checks/verify_theory_output.txt`. All run on CPU in a few minutes.

| check | claim | result |
|---|---|---|
| C1 | closure for arbitrary E, D, b | max native−shadow error 0.0 over 200 steps |
| C2 | 1-D cubic coefficient = Σv⁴/3; balanced frame minimal | matches to 4 s.f. for three frames |
| C3 | 4-design ⇒ radial, coeff k/((k+2)N) | four-sign 23.6% anisotropy; M=3, 6 isotropic; coefficient exact |
| C4 | random frame ⇒ radial Stein gain, scatter ∝ N^{−1/2} | deviation 2.8e−2 → 7.3e−3 (N 256 → 4096), ratio 3.79 (pred. 4) |
| C5 | lifetime ∝ N (tanh), ∝ N² (no-cubic, same saturation) | 42 vs 41 and 14926 vs 14925 at N=256 |
| C6 | noise-limited exponent p/(p+1) | 0.53 (pred. 0.50); 0.58 (pred. 0.67, slow approach) |
| C7 | heterogeneous encoder + solved decoder | noise-free δ ~1e−6 at N=16; noisy lifetime 14 → 136, exponent 0.58, flattening |
| C8 | design order controls rotation phase drift | 2.55 rad (four-sign) → 1.0e−3 (M=3) → 9.3e−9 (M=6) after 20k steps |

## References

See `literature_notes.md` for full annotated citations, verification status,
and links. Items marked [P] are preprints whose venue was not confirmed, and
items marked [D] contain a detail that should be checked before it is quoted.
