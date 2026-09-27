# Plan: from task-conjugate cells to a predictable recurrent engine

Working plan for the ideation document (`predictable_recurrence.md`). Written
before drafting. It fixes what the document must argue, in what order, and what
evidence each claim needs.

## 0. Starting point (what exists)

- The Sept 21 derivation: cell `h' = phi(V g F(V^T h, u))`, exact latent map
  `z' = Psi(g F(z,u))`. Only `g` is trained. `g_pred` and the failure boundary are
  sealed before native training.
- Code on the lab machine (`src/dynaspec/task_conjugate.py`,
  `tools/task_conjugate_development.py`). **Not reachable from this laptop, so
  this document does not assume its contents.**
- The earlier `rnnphase` project in this repo: task -> predicted attractor type
  (line attractor, limit cycle, ...), recovered in trained RNN/GRU/LSTM, and an
  "X-ray" failure prediction for vanilla RNNs.

## 1. Problems the new document must confront (self-audit)

Found while checking the derivation numerically (`checks/`):

| # | Problem | Evidence | Consequence |
|---|---|---|---|
| A1 | Closure holds for **any** encoder/decoder, not only balanced frames. The network state is a deterministic function of a k-dim variable. | C1: native = shadow to 0.0 | "N-dim network" is a k-dim map in disguise. Hidden width adds no dynamical degrees of freedom. |
| A2 | 1-D ±1 frame with odd phi: all N units are sign-flipped copies of **one** neuron. Width acts only as input down-scaling (q/√N). | algebra; C5 | Noise-free, the "lifetime ∝ N" law is an operating-amplitude law. The same effect is available with N=1 by rescaling. |
| A3 | Heterogeneous encoders + a solved decoder make one-step error ~1e-6 already at N=16. | C7 | The N / N² laws are artifacts of homogeneity. The binding resource is **noise / precision**. |
| A4 | Only scalar `g` is trained, and native loss ≡ shadow loss as functions of g. | by construction | "Optimizer reaches g_pred" tests only 1-D optimization from g=1. It is not evidence about learning. |
| A5 | Four-sign 2-D frame is anisotropic (23.6% cubic anisotropy). | C3, C8 (2.55 rad phase drift) | Rotation tasks drift in phase for a fixable, design-level reason. |
| A6 | Development cohort shares the evaluation stream. | stated in derivation | Boundary agreement is partly by construction. Only the confirmation protocol is informative. |
| A7 | Prior art: NEF (encoders/decoders/recurrent dynamics, error vs N), low-rank RNN mean-field theory, mean-field signal propagation for RNNs. | lit review | Novelty must be stated relative to these. |

## 2. Thesis of the expanded program

Keep the part that is genuinely valuable: an **exact, cheap shadow** that turns
architecture + task + training spec into a **sealed, quantitative forecast** of
the learned solution and its failure horizon. Then make width, learning, and
prediction non-trivial:

1. **Theory of distortion** (exact): frame moment tensors, spherical designs,
   Stein/mean-field limit. Precision knobs: width, activation Taylor order,
   frame design order.
2. **Theory of error propagation**: one-step error × task stability class
   (contracting / neutral / expanding; systematic vs. averaging) gives lifetime
   laws.
3. **Noise/precision as the real resource**: encoding-scale trade-off, with
   optimal lifetime ∝ N^{p/(p+1)} (C6). Heterogeneous-encoder regime (C7).
   Quantization.
4. **Tiered architectures**, ordered by how much the shadow can be trusted:
   - T0 exact bottleneck cells (current cell, generalized: learned latent map,
     design frames, solved decoders);
   - T1 compositions of T0 modules (a "compiler" for larger systems);
   - T2 rank-k + random bulk, all weights trained, with an asymptotic mean-field
     shadow plus finite-N corrections. **Here prediction stops being
     tautological.**
   - T3 generic GRU/LSTM/SSM layers, probed with a conjugacy fit (scope control
     and bridge to the earlier project).
5. **The engine**: Spec -> Compile (choose N, phi, frame, scale, decoder) ->
   Forecast -> Seal -> Train -> Audit, plus inverse design (the minimal N for a
   required horizon).
6. **Protocol**: preregistered hypotheses, independent forecast/measurement
   populations, censoring (Kaplan–Meier), Monte Carlo CIs, kill criteria.

## 3. Document outline

1. Summary (one page).
2. Where we are, and an honest audit (A1–A7) with the fix for each.
3. Literature map: what we borrow, and what we must distinguish from.
4. Theory
   4.1 Closure lemma (general)  4.2 Distortion: moment tensors, designs, Stein
   4.3 Error propagation and lifetime laws  4.4 Noise & precision laws
   4.5 Mean-field shadow for rank-k + bulk  4.6 Learning dynamics in the shadow
5. Architecture tiers T0–T3.
6. The engine: interfaces, inverse design, what is sealed.
7. Research program: hypotheses H1–H10, each with prediction, test, and kill
   criterion.
8. Evaluation protocol.
9. Risks, and what would falsify the program.
10. Roadmap (phases, compute on the 5070 / A100).
11. Novelty positioning table.
Appendix A: derivations. Appendix B: numerical checks (C1–C8 output).
References (verified by the literature agents; unverified items marked).

## 4. Evidence standards

- Every quantitative claim in §4 is either derived in Appendix A or checked in
  `checks/verify_theory.py`. Checks that came out weaker than predicted (C6
  p=2 exponent 0.58 vs 0.67; C7 saturation at N=1024) are reported as such.
- Citations come only from the verified literature notes (`literature_notes.md`).
- Nothing is claimed about the lab machine's runs.
