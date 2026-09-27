# Appendix A — Derivations

Companion to `predictable_recurrence.md`. Every numbered result is either
proved here or checked numerically in `checks/verify_theory.py` (check IDs
C1–C8; output in `checks/verify_theory_output.txt`).

Notation: task state `z ∈ R^k`, input `u`, declared task update `F(z,u)`.
Width `N`. Encoder `E ∈ R^{N×k}` (rows `e_i`), decoder `D ∈ R^{k×N}`, bias
`b ∈ R^N`. Pointwise activation `φ`. Latent map parameters `θ`.

---

## A.1 Closure lemma (C1)

**Cell.** `q_t = G_θ(D h_t, u_t)`, `h_{t+1} = φ(E q_t + b)`.

**Lemma** (known; the finite-N exact reduction of low-rank RNNs, cf. Pals et al.
2024; Valente, Ostojic & Pillow 2022). For *any* `E, D, b, θ`, the readout
`z_t := D h_t` obeys exactly

    z_{t+1} = Ψ(G_θ(z_t, u_t)),   Ψ(q) := D φ(E q + b).

*Proof.* Substitute. ∎

**Corollaries.**

1. Any loss that depends on the trajectory only through `z_t` (or through a
   linear readout `C h_t = C φ(E q_{t-1} + b)`) is the **same function** of
   `(θ, E, D, b)` in the native and shadow systems. Their gradients are
   identical in exact arithmetic.
2. After the first step, the hidden state lies on the k-dimensional manifold
   `{φ(Eq + b)}`. Width adds no dynamical degrees of freedom. It only changes
   the shape of `Ψ`.
3. The Sept 21 cell is the special case `E = V`, `D = V^T`, `b = 0`,
   `G = g F`.

Consequence for the research program: in any exactly closed cell, "the
optimizer reaches the shadow optimum" is a statement about optimization and
floating point, not about learning. Non-trivial learning predictions require
cells that are *not* exactly closed (Tier T2, §A.6).

---

## A.2 Distortion as frame moment tensors (C2, C3)

Let `b = 0` and `φ` odd with Taylor series `φ(x) = x + Σ_{j≥1} c_{2j+1} x^{2j+1}`.
Take `D = E^T = V^T` with `V^T V = I`, rows `V_i`. Then

    Ψ(q) = q + Σ_{j≥1} c_{2j+1} M_{2j+2}[q, …, q],    M_{2m} := Σ_i V_i^{⊗ 2m},

where `M_{2m}[q,…,q]` contracts `2m−1` slots with `q`. The architecture enters
the distortion **only** through the even moment tensors of the frame rows.

**1-D (C2).** `M_4 = Σ_i v_i^4 ≥ (Σ_i v_i^2)^2 / N = 1/N` (Cauchy–Schwarz),
with equality iff `|v_i| = 1/√N`. The balanced frame is therefore the *unique*
minimizer of the leading distortion, and

    Ψ(q) = q + c_3 q^3 / N + O(q^5/N^2),     (tanh: c_3 = −1/3).

Check C2 measures `(q − Ψ(q))/q^3 = Σ v_i^4 / 3` to four significant figures for
balanced, Gaussian, and heavy-tailed frames.

**k-D and spherical designs (C3).** Write `x_i = √(N/k) V_i` (unit vectors when
row norms are equal). If `{x_i}` is a spherical 4-design,
`(1/N) Σ_i (x_i·q)^4 = 3|q|^4/(k(k+2))`. Differentiating in `q` gives
`Σ_i x_i (x_i·q)^3 = 3N|q|^2 q/(k(k+2))`, hence

    M_4[q,q,q] = (k/N)^2 · 3N|q|^2 q/(k(k+2)) = 3k |q|^2 q / ((k+2) N).

For tanh:

    Ψ(q) = q − k |q|^2 q / ((k+2) N) + O(|q|^5/N^2)       (radial).

At `k = 1` this reduces to `q − q^3/(3N)`. At `k = 2`, `N = 240` the predicted
radial coefficient `1/(2N) = 2.083e−3` is measured exactly (C3).

**2-D harmonic frames.** Directions at angles `πj/M` form, with their
antipodes, a regular `2M`-gon, which is a spherical `(2M−1)`-design. Distortion
is then radial through order `2M−1`, and the first anisotropic term is order
`2M+1`:

| frame | design order | cubic anisotropy (C3) | phase error after 20 000 rotation steps (C8) |
|---|---|---|---|
| four-sign (M=2) | 3 | 23.6 % | 2.55 rad |
| harmonic M=3 | 5 | 0.0 % | 1.0e−3 rad |
| harmonic M=6 | 11 | 0.0 % | 9.3e−9 rad |

**Design order is a third precision knob**, alongside width and activation
order. It controls *anisotropic* error (phase or direction drift), while width
and activation control *radial* error.

---

## A.3 Random frames: the Stein / mean-field limit (C4)

For `V` with approximately i.i.d. `N(0, 1/N)` entries (orthonormalized),
Stein's lemma gives

    Ψ(q) = Σ_i V_i φ(V_i·q) ≈ N · E_w[w φ(w·q)] = q · E_ξ[φ'(|q| ξ / √N)] + O(|q|/√N),

with `ξ ~ N(0,1)`. The distortion is a **radial Bussgang/Stein gain**, i.e. the
first Hermite coefficient of `φ` at scale `|q|/√N` (Bussgang 1952; Stein 1981).
This is a first-moment quantity. It is related to, but not the same as, the
second-moment length map of mean-field signal propagation (Poole et al. 2016).
The residual is uncorrelated Bussgang distortion of relative size `N^{-1/2}`,
and Price's theorem gives its covariance. C4 measures relative deviations 2.76e−2 (N=256) and 7.3e−3
(N=4096), a ratio of 3.79 against the predicted 4.

With a random bulk (§A.6), `|q|^2/N` is replaced by `|q|^2/N + Δ`, where `Δ` is
the bulk variance.

---

## A.4 Deterministic lifetime laws (C5)

Hold task, `g = 1`, balanced 1-D frame, `φ(x) = x − c x^{2p+1} + …`. The latent
map `z' = z − c z^{2p+1}/N^p` has the continuum limit
`dz/dt = −c z^{2p+1}/N^p`. Integrating to the relative threshold `ε` gives

    t*(N) = N^p ((1−ε)^{−2p} − 1) / (2p c z_0^{2p}).

| activation | p | c | prediction | C5 (N = 16 / 64 / 256) |
|---|---|---|---|---|
| tanh | 1 | 1/3 | `(3N/2z_0^2)((1−ε)^{−2} − 1)` | 3/11/42 vs 3/10/41 |
| `x/(1+x^4)^{1/4}` (saturation-matched, no cubic) | 2 | 1/4 | `(N^2/z_0^4)((1−ε)^{−4} − 1)` | 59/933/14926 vs 58/933/14925 |

**Caveat (audit item A2).** With a ±1 frame and odd `φ`, every unit equals
`±φ(q/√N)`. The whole network is one neuron read at input scale `1/√N`, so
`N` enters only through `z_0/√N`. Without noise the same lifetime is available
at `N = 1` with a smaller encoding scale. The N-laws measure **operating
amplitude**, not width.

**Calibrated `g`.** With `g > 1`, the map `z' = g z − c g^3 z^3/N` has stable
fixed points at `z*^2 ≈ N(g−1)/c`. A continuum of stored values collapses
toward `±z*`, so the calibrated hold network is a bistable memory with slow
drift. `g_pred` is the best compromise over the task's amplitude distribution.
Lifetime keeps the `N^p` scaling with an improved constant.

---

## A.5 Noise-limited laws: why width becomes real (C6, C7)

Add i.i.d. per-neuron noise `σ` to `h`. With unit-norm columns, the latent noise
is `σ` per dimension per step. Let `x` be the latent amplitude (the design
variable) and hold one value:

- drift time `t_d ≈ ε N^p / (c x^{2p})`
- diffusion time `t_n ≈ ε^2 x^2 / σ^2`

Maximizing `min(t_d, t_n)` over `x`:

    x*^2 = (N^p σ^2 / (c ε))^{1/(p+1)},
    t*   ∝ ε^{(2p+1)/(p+1)} c^{−1/(p+1)} σ^{−2p/(p+1)} N^{p/(p+1)}.

**Homogeneous frames therefore cap the noise-limited exponent at `p/(p+1)`**:
`1/2` for tanh and `2/3` for the no-cubic activation. C6 measures `0.53` and
`0.58` over `N ∈ {64, 256, 1024}` with 24 amplitudes and 64 trials each. The
tanh value matches. The `p = 2` value approaches its asymptote slowly and needs
larger `N` to confirm.

**Heterogeneous encoders (C7).** Use random gains and biases, with `D` solved
before training as the noise-regularized least-squares inverse on the shadow
(ridge `= σ^2` per sample, the standard NEF decoder). Two results:

- *Noise-free*, one-step error is already ~1e−6 at `N = 16`. The homogeneous
  N-laws are artifacts of homogeneity.
- *With* `σ = 0.02`, median hold lifetime rises 14 → 32 → 99 → 136 for
  `N = 16 → 1024`, a fitted exponent of 0.58 that flattens at the largest `N`.
  The one-step-optimal decoder hits a bias floor. The lifetime-optimal decoder
  should be solved on the **multi-step** shadow objective instead (hypothesis
  H4). The diffusion-limited ideal is exponent 1, the Burak–Fiete regime.

---

## A.6 Asymptotic shadow for rank-k + random bulk (Tier T2)

Cell: `h_{t+1} = φ(J h_t + E q_t + b)`, with `q_t = G_θ(D h_t, u_t)` and
`J_{ij} ~ N(0, g_J^2/N)`. **No exact closure.** In the large-`N` mean-field
limit (discrete-time analogue of Mastrogiuseppe & Ostojic 2018 and Beiran et
al. 2021), each unit's pre-activation is `e_i·q + b_i + η_i`, with `η_i` Gaussian
of variance `Δ_t` set self-consistently by the bulk autocorrelation. Then

    z_{t+1} = Ψ_Δ(G_θ(z_t,u_t)) + ζ_t,     Ψ_Δ(q) = E_{e,b,η}[ D-weighted φ(e·q + b + η) ],

with fluctuation `ζ_t = O(N^{-1/2})` in relative units. The fluctuation is
temporally correlated with the bulk's correlation time `τ_Δ`.

Predictions that are **not** true by construction:

1. Effective gain loss: the bulk lowers `E[φ']`, so `g_pred` rises with `g_J`.
2. Self-generated noise: the bulk acts as colored noise of variance
   `∝ C_Δ(0)/N` and correlation `τ_Δ`. Hold lifetime is then
   `t* ∝ N / (C_Δ(0) τ_Δ)` (diffusion), which can be computed before training
   from the mean-field autocorrelation.
3. Learning: with *all* of `J, E, D, b` trained from this initialization, the
   learned change is predicted to be approximately low-rank and aligned with the
   task subspace (Schuessler et al. 2020, NeurIPS). Because it is correlated with
   the bulk, it shifts the outlier eigenvalues (Schuessler et al. 2020, PRR).
   The learning shadow is the overlap-space gradient flow of Ger & Barak (2026),
   extended with the bulk those authors leave open. The learned latent map should converge to the shadow
   optimum up to `O(N^{-1/2})`. This is the first place where the "optimizer
   reaches the predicted recurrence" claim carries real content.

---

## A.7 Error propagation and the stability-class lifetime table

Let `e_t = ẑ_t − z_t` (network minus ideal task). Linearizing,

    e_{t+1} = J_F(z_t,u_t) e_t + δ(z_t, u_t) + ζ_t,

where `δ` is the one-step architectural error (`Ψ ∘ G − F`) and `ζ` is noise.
The task's stability class sets how one-step error becomes a lifetime:

| class | examples | error growth | lifetime law |
|---|---|---|---|
| contracting, `‖J_F‖ ≤ ρ < 1` | driven contraction, leaky integration | bounded: `sup e ≤ δ/(1−ρ)` | **infinite (censored)** if `δ < ε(1−ρ)`. Sharp critical width `N_c` where `δ(N_c) = ε(1−ρ)` |
| neutral, systematic `δ` | hold, bounded accumulation (interior) | linear | `t* ≈ ε / δ` |
| neutral, averaging `δ` (zero mean along trajectory, correlation time `τ`) | anisotropic error on rotation | diffusive | `t* ≈ ε^2 / (δ^2 τ)` |
| neutral + noise | any neutral task | diffusive | `t* ≈ ε^2 / σ_eff^2` |
| expanding, `ρ > 1` | unstable or chaotic task maps | exponential | `t* ≈ log(ε/δ) / log ρ`: **width buys only a constant number of steps** |

This table is what makes the engine general. A new task needs only (i) its
Jacobian class along the task distribution and (ii) the architecture's `δ(N)`
and `σ_eff(N)`. The lifetime law follows.

The contracting row predicts a **phase boundary**: plotting "failure within
T_max" against `(N, ρ)` should show a sharp curve `δ(N) = ε(1−ρ)`, with
censored (no-crossing) runs on one side. The protocol must retain censored
runs rather than drop them.
