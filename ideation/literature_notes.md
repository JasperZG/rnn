# Literature notes: what we borrow, and what we must distinguish from

Curated from five parallel literature sweeps (September 26, 2026). Each entry
was checked by web search against arXiv, the proceedings, or the publisher.
Tags:

- **[P]** preprint, venue unconfirmed
- **[D]** a specific detail that was not confirmed in the primary source;
  check it before quoting

Entries are grouped by the role they play in the proposal.

---

## 1. Closest prior art: the claims we must position against

| Work | What it does | How we differ / what we take |
|---|---|---|
| **Eliasmith & Anderson 2003**, *Neural Engineering* (MIT Press). Stewart 2012 (tech. report). Eliasmith 2005, *Neural Comput.* 17:1276 | NEF: encode a low-dim `x` with heterogeneous tuning curves, decode it with least-squares decoders, and implement `dx/dt = A(x)` through a rank-k recurrent matrix `ω = d·e`. Builds point, line, ring, and cyclic attractors. Noise error falls as ~1/N. Static distortion falls faster **[D: the exact 1/N² exponent is unconfirmed]**. | This is the same encoder/decoder sandwich, and reviewers will say so first. We differ in four ways: (i) a closed-form Ψ and closed-form lifetime laws; (ii) φ treated as a controlled error source, not a basis; (iii) the trained parameters are predicted and **sealed** before gradient training; (iv) a failure horizon forecast with an audit. We *borrow* heterogeneous encoders and solved decoders for Tier T0-H (C7). |
| **MacNeil & Eliasmith 2011**, *PLoS ONE* 6:e22885 | Drift in NEF integrators comes from decoder error. An online rule re-tunes the recurrence. | This is prior art for "drift is set by static distortion". Our Ψ − id is the analytic version of their fitted residual. |
| **Voelker & Eliasmith 2018**, *Neural Comput.* 30:569. **Voelker, Kajić, Eliasmith 2019** (LMU), NeurIPS | Pre-compensates the substrate so the prescribed dynamics hold exactly. LMU: a fixed Legendre-delay linear ODE with trained readout, handling ~100k-step dependencies. | The LMU keeps memory **linear**, outside the nonlinearity, so it has no Ψ distortion. We route memory through φ on purpose and predict what that costs. Ψ⁻¹ precompensation is a direct analogue to test. |
| **Boerlin, Machens, Denève 2013**, *PLoS CB* 9:e1003258. **Denève & Machens 2016**, *Nat Neuro* 19:375 | Spike-coding networks implement linear dynamical systems with encoder = decoderᵀ, are robust to neuron loss, and have precision that grows with N. | This is the closest structural match to `V / Vᵀ`. We are a rate network with exact conjugacy. Terminology warning: our "balanced frame" is **not** E/I balance. |
| **Mastrogiuseppe & Ostojic 2018**, *Neuron* 99:609. **Beiran et al. 2021**, *Neural Comput.* 33:1572. **Dubreuil et al. 2022**, *Nat Neuro* 25:783 | Low-rank + random RNN mean field: latent `κ` dynamics with gain `⟨φ'⟩(Δ)`. Bulk variance enters through `Δ`. Gaussian-mixture populations give per-population gains. | Our cell is a rank-k RNN with `m = n = V` and no bulk, and Ψ is its **exact finite-N** reduction. Tier T2 is exactly their setting plus training and sealing. Beiran's mixture form is the template for heterogeneous frames. |
| **Pals, Sağtekin, Pei, Gloeckler, Macke 2024**, NeurIPS, arXiv:2406.16749. **Valente, Ostojic, Pillow 2022**, *Neural Comput.*, arXiv:2110.09804 | Exact finite-N latent closure for low-rank RNNs. For piecewise-linear φ, all fixed points can be enumerated in O(N^R) R×R solves. Exact linear-dynamical-system ↔ low-rank equivalence. | **The closure lemma (A.1) is known.** Cite it and do not claim it. Borrow the fixed-point census so the *entire fixed-point set* can be sealed. |
| **Ger & Barak 2026**, arXiv:2605.04115 [P] | Closed-form gradient-descent ODEs in overlap space for low-rank RNNs, with all weights trained. Exact for linear networks, asymptotically exact for nonlinear ones via Stein gain. Separates loss-visible from loss-invisible overlaps. Explicitly **leaves the random bulk to future work**. | This is the most direct tool, and the gap T2 fills: the bulk, finite-N corrections, and a failure horizon, all sealed. |
| **Bordelon, Cotler, Pehlevan, Zavatone-Veth 2025**, arXiv:2503.18754 [P] | Linear RNNs learning to integrate. In the rich regime, learning reduces to a 2-D ODE for one outlier eigenvalue. In the lazy regime, DMFT gives a learnability threshold `|1 − c*| < σ`. | The closest "predict what GD finds in an RNN" result, but it is linear and not sealed. We add a nonlinear Ψ, the failure horizon, and preregistration. |
| **Nartallo-Kaluarachchi, Lambiotte, Goriely 2026**, arXiv:2602.14885 [P] | Trains RNNs whose latent drift and diffusion match target SDEs. | Close in spirit (declared latent dynamics in N units) and must be cited. We *predict* drift and diffusion from architecture instead of training to match them. |

## 2. Precision and lifetime of persistent memory

- **Seung 1996**, *PNAS* 93:13339. A line attractor needs fine-tuned feedback.
  Hold at `g = 1` is a line attractor only if `Ψ = id`.
- **Koulakov et al. 2002**, *Nat Neuro* 5:775. Bistable subunits give a
  robust staircase integrator. Our calibrated `g > 1` hold is the two-state
  limit of the same trade-off (A.4).
- **Goldman 2009**, *Neuron* 61:621. Feedforward/non-normal memory, whose
  lifetime grows with chain length **[D]**. An alternative way for lifetime to
  grow with N.
- **Lim & Goldman 2013**, *Nat Neuro* 16:1306. Negative-derivative feedback
  resists drift. A candidate latent-level correction term.
- **Burak & Fiete 2012**, *PNAS* 109:17645 (erratum 2017). Diffusion in noisy
  continuous attractors scales as 1/N. **Key contrast:** Ψ predicts a
  *systematic, amplitude-dependent bias* (`dz/dt ≈ −z³/3N`), whereas they
  predict unbiased variance growing like `t/N`. Same N-scaling, separable
  signatures.
- **Compte et al. 2000**, *Cereb Cortex* 10:910. **Wimmer et al. 2014**,
  *Nat Neuro* 17:431. Bump-attractor diffusion explains behavioural
  working-memory error. Open question: does the *drift* branch appear in
  behaviour?
- **Ságodi, Martín-Sánchez, Sokół, Park 2024**, NeurIPS, arXiv:2408.00109.
  Fenichel persistence: an approximate continuous attractor keeps a slow
  manifold, and memory error is bounded by the size of the perturbation. Our
  cell is an instance where the perturbation (`Ψ − id`) is **known exactly**,
  so we can test how tight their bounds are.
- **Park, Ságodi, Sokół 2023**, arXiv:2308.12585 [P]. Working memory without
  continuous attractors.
- **Mo 2026**, arXiv:2605.03338 [P]. Equivariance gives zero Lyapunov
  exponents along the group orbit. Least-squares decoding minimizes local
  diffusion. The rotation task with harmonic frames is a clean test bed (C8).
- **White, Lee, Sompolinsky 2004**, *PRL* 92:148102. **Ganguli, Huh,
  Sompolinsky 2008**, *PNAS* 105:18970 (Fisher memory curve). **Jaeger
  2001/2002**, GMD Report 152. Memory capacity ∝ N in linear or orthogonal
  networks. The FMC of the Ψ-conjugate map should be computable in closed form.

## 3. Mean-field propagation, Hermite/Bussgang analysis, frames and designs

- **Poole et al. 2016**, NeurIPS, arXiv:1606.05340. **Schoenholz et al.
  2017**, ICLR, arXiv:1611.01232. Length and correlation maps, and depth
  scales. These are *second-moment* objects. Our random-frame Ψ is a
  *first-moment* Bussgang gain (A.3).
- **Chen, Pennington, Schoenholz 2018**, ICML, arXiv:1806.05394. **Gilboa et
  al. 2019**, arXiv:1901.08987 [P]. Timescales predicted before training for
  RNN/minimalRNN/LSTM/GRU from mean-field theory at initialization. This is
  the nearest "predict recurrent timescales from architecture" precedent. Ours
  is exact at finite N for designed frames.
- **Pennington, Schoenholz, Ganguli 2017**, NeurIPS, arXiv:1711.04735.
  **Pennington & Worah 2017**, NeurIPS. Orthogonal weights and dynamical
  isometry. For random features, `(E φ')²` is the linear part.
- **Bussgang 1952**, MIT RLE TR-216. **Stein 1981**, *Ann. Statist.* 9:1135.
  **Price 1958**, IRE Trans. IT-4:69. The Bussgang decomposition
  `Vᵀφ(Vq) = gain·q + uncorrelated distortion`, with its covariance given by
  Price.
- **Daniely, Frostig, Singer 2016**, NeurIPS, arXiv:1602.05897 (dual
  activations). **Simon, Anand, DeWeese 2022**, ICML, arXiv:2106.03186
  (activations designed by prescribing Hermite coefficients). **Martens et al.
  2021**, arXiv:2110.01765 (Deep Kernel Shaping). The tools for designing a
  "no-cubic" activation. **Gap:** no prior work cancels the cubic term for
  recurrent latent precision, and none uses saturation-matched controls.
- **Benedetto & Fickus 2003**, *Adv. Comput. Math.* 18:357 (frame potential).
  **Casazza & Kovačević 2003**, same volume, 387 (equal-norm tight frames,
  erasures). **Delsarte, Goethals, Seidel 1977**, *Geom. Dedicata* 6:363
  (spherical designs). **Bannai & Bannai 2009**, *Eur. J. Combin.* 30:1392
  (Venkov's identity, the exact statement behind A.2). **Ehler & Okoudjou
  2012**, *J. Stat. Plan. Inf.* 142:645 (p-frame potential minimizers).
  **Waldron 2018** (book).
- The literature agent independently derived what C3 measures: the four-sign
  frame is a square, hence a 3-design and **not** a 4-design. Its cubic
  distortion is D₄-anisotropic. The fix is hexagonal or higher frames.
- Quantization and erasure: **Goyal, Vetterli, Thao 1998**, *IEEE T-IT*
  44:16 (MSE ∝ 1/redundancy). **Goyal, Kovačević, Kelner 2001**, *ACHA*
  10:203. **Holmes & Paulsen 2004**, *LAA* 377:31. **Benedetto, Powell,
  Yılmaz 2006**, *IEEE T-IT* 52:1990 (ΣΔ on frames). **Zhang, Zhou, Saab
  2023**, *SIMODS* 5:373 (GPFQ). **Czaja & Na 2024**, arXiv:2404.08131 [P].
  Width buys precision under quantization, with rates that can be predicted.
- **Cisse et al. 2017** (Parseval networks), ICML. **Papyan, Han, Donoho
  2020**, *PNAS* 117:24652 (neural collapse to an ETF). **Elhage et al.
  2022**, arXiv:2209.10652 (polygon superposition). Trained networks drift
  toward frame geometries on their own. A testable question: does a learned V
  drift toward designs?

## 4. Task-derived recurrences, SSMs, state tracking, length generalization

- **Gu et al. 2020** (HiPPO), NeurIPS. **Gu, Goel, Ré 2022** (S4), ICLR.
  **Gu et al. 2022** (S4D), NeurIPS. **Gu et al. 2023** ("How to train your
  HiPPO"), ICLR. **Orvieto et al. 2023** (LRU), ICML. Linear recurrences
  derived from a task, with per-mode horizon ≈ `1/(1 − |λ|)`. Predictable
  before training *for linear time-invariant layers only*.
- **Gu & Dao 2023** (Mamba), arXiv:2312.00752. **Dao & Gu 2024** (Mamba-2),
  ICML. **Lahoti et al. 2026** (Mamba-3), ICLR, arXiv:2603.15569. Mamba-3
  frames the SSM step as exponential-Euler/trapezoidal integration and adds
  complex (rotational) state for state tracking. That is the same vocabulary
  as our "network as integrator of F".
- **Beck et al. 2024** (xLSTM), NeurIPS. **Feng et al. 2024**
  (minGRU/minLSTM), arXiv:2410.01201 [P]. **Yang et al. 2024** (DeltaNet),
  NeurIPS. **Yang, Kautz, Hatamizadeh 2025** (Gated DeltaNet), ICLR.
  **Peng et al. 2023** (RWKV), EMNLP Findings. For selective/gated layers, no
  work predicts the failure length before training.
- **Weiss, Goldberg, Yahav 2018**, ACL. Under finite precision, the
  activation decides whether counting is possible at all.
- **Merrill et al. 2020**, ACL (formal hierarchy). **Delétang et al. 2023**,
  ICLR (Chomsky hierarchy; binary generalization, **no failure length**).
  **Liu et al. 2023**, ICLR (automata shortcuts are brittle).
- **Merrill, Petty, Sabharwal 2024**, ICML (diagonal SSMs ⊂ TC⁰; no S₅).
  **Sarrof, Veitsman, Hahn 2024**, NeurIPS. **Grazzi et al. 2025**, ICLR
  (negative eigenvalues give parity; Householder products give all regular
  languages). **Siems et al. 2025** (DeltaProduct), NeurIPS. **Terzić et
  al. 2025**, AAAI. These say which group state-tracking tasks (`Z_m`
  = input-switched rotation) are feasible for a given spectrum.
- **Buitrago Ruiz & Gu 2025**, ICML, arXiv:2507.02782 ("unexplored states"
  hypothesis for length failure). **Chen et al. 2025** (Stuffed Mamba), COLM,
  arXiv:2410.07145 (empirical length laws in state size, fitted post hoc).
  **Ben-Kish et al. 2025** (DeciMamba), ICLR. **Ye et al. 2025**
  (LongMamba), ICLR. **Li, Han, E, Li 2021**, ICLR (curse of memory).
  **Wang & Li 2024** (StableSSM), ICML. **François, Orvieto, Bach 2025**,
  COLT (uncertainty principle: resolution ≈ K/S). These are the empirical and
  theoretical anchors for "failure length" in modern models.
- **Arjovsky et al. 2016** (uRNN), ICML. **Henaff, Szlam, LeCun 2016**,
  ICML (hand-built copy/add solutions). **Chang et al. 2019**
  (AntisymmetricRNN), ICLR. **Rusch & Mishra 2021** (coRNN, ICLR;
  UnICORNN, ICML). **Erichson et al. 2021** (Lipschitz RNN), ICLR. **Rusch &
  Rus 2025** (LinOSS), ICLR. **Chen et al. 2020** (Symplectic RNN), ICLR.
  Structure-preserving recurrences with provable stability. They stop at
  stability and do not predict a horizon.
- **Hairer, Lubich, Wanner 2006**, *Geometric Numerical Integration*
  (Springer). Backward error analysis / modified equations: symplectic or
  symmetric schemes turn secular energy drift into bounded error, with phase
  error growing linearly. **Pilyugin 1999**; **Palmer 2000** (shadowing).
  **Gap:** no ML paper applies modified-equation analysis to the recurrence
  *along the sequence* to predict a failure length.

## 5. Predicting training outcomes; compilation; conjugacy; methodology

- **Kaplan et al. 2020**; **Hoffmann et al. 2022** (Chinchilla: the closest
  scaling-law precedent to a sealed forecast); **OpenAI 2023** (GPT-4
  predictable scaling **[D]**); **Ruan, Maddison, Hashimoto 2024**
  (observational scaling laws); **Yang & Hu 2021** (μP); **Yang et al.
  2022** (μTransfer). These predict *loss or hyperparameters*, not the learned
  solution or a failure length. **Vankadara et al. 2024**, NeurIPS: μP fails
  for SSMs, so recurrent scaling limits need their own derivation.
- **Saxe, McClelland, Ganguli 2014** (ICLR) and 2019 (*PNAS*): exact
  learning trajectories for deep linear networks. **Proca et al. 2025**
  (ICML): learning dynamics of linear RNNs. **Emami et al. 2021** (ICML):
  implicit bias of linear RNNs. **Liu et al. 2024** (ICLR): connectivity rank
  shapes rich vs. lazy learning. **Bordelon & Pehlevan 2022/2023**
  (NeurIPS): DMFT, and O(1/√N) finite-width fluctuations. **Bauer et al.
  2026**, arXiv:2602.15593 [P]: feature-learning theory for RNNs.
  **Hazelden & Shea-Brown 2026**, arXiv:2605.12763 [P]: learning near
  bifurcations. **Dinc et al. 2025**, arXiv:2501.02378: ghost mechanism,
  abrupt learning. **Alemohammad et al. 2021** (ICLR): recurrent NTK.
- **Schaeffer, Miranda, Koyejo 2023** (NeurIPS): emergence as a metric
  mirage. **Schaeffer et al. 2025** (ICML). **Wei et al. 2022** (TMLR).
  **Lesson:** a thresholded failure length is a discontinuous metric, so
  always seal the *continuous error curve* too.
- **Weiss, Goldberg, Yahav 2021** (RASP); **Lindner et al. 2023** (Tracr);
  **Friedman, Wettig, Chen 2023** (Learning Transformer Programs); **Gupta et
  al. 2024** (InterpBench); **Kim & Bassett 2023**, *Nat Mach Intell* 5:622
  (programming reservoirs). Compilation *without* the claim that gradient
  descent recovers the compiled solution. That claim is ours.
- **Ko et al. 2019** (POPQORN, ICML); **Jacoby, Barrett, Katz 2020** (ATVA);
  **Akintunde et al. 2019** (AAAI); **Miller & Hardt 2019** (ICLR, stable
  RNNs ≈ feedforward). Certification over finite horizons. Known conjugacy
  could supply invariants for long-horizon certificates.
- **Sussillo & Barak 2013**; **Maheswaranathan et al. 2019**; **Vyas et al.
  2020**; **Driscoll, Shenoy, Sussillo 2024** (*Nat Neuro* 27:1349);
  **Huang, Singh, Martinelli, Rajan 2025** (NeurIPS, arXiv:2410.03972:
  solution degeneracy across 3,400 RNNs); **Ostrow et al. 2023** (DSA);
  **Huang et al. 2025** (InputDSA, arXiv:2510.25943 [P]); **Godara, Tay,
  Mattar 2026** (CSA, arXiv:2607.04493 [P]: orthogonal alignment is neither
  necessary nor sufficient for conjugacy); **Redman et al. 2024** (NeurIPS:
  Koopman conjugacy of training dynamics); **Bramburger, Brunton, Kutz 2021**
  (*Physica D*: learned conjugacies). Audit tools. With a known Ψ we can test
  conjugacy *exactly* (CSA-style) rather than by proxy.
- **Klein & Roodman 2005**, *Annu. Rev. Nucl. Part. Sci.* 55:141 (blind
  analysis). **Nosek et al. 2018**, *PNAS* 115:2600. **NeurIPS 2020/2021
  pre-registration workshops** (PMLR 148/181). **Hofman et al. 2023**,
  arXiv:2311.18807 [P]. **Kaplan & Meier 1958**, *JASA* 53:457. **Redner
  2001** (first-passage). **Chhikara & Folks 1989** (inverse Gaussian).
  Sealing, censoring, and first-passage distributions. **Adriaensen et al.
  2023** (LC-PFN): learning-curve extrapolation that handles censoring.

## 6. Gaps confirmed by the sweeps

These are negative search results, not proofs of absence.

1. No sealed, pre-registered prediction of an RNN's **failure horizon** from
   architecture, followed by a test.
2. No sealed prediction of the **learned solution** of a *nonlinear* RNN. The
   linear cases (Saxe; Bordelon et al.) are not sealed.
3. No **modified-equation / backward-error** analysis of the recurrence along
   the sequence as a predictor of failure length.
4. No activation designed to **cancel the cubic term** for recurrent
   precision, and no **saturation-matched** activation comparisons.
5. No horizon laws for **selective/gated** layers derived before training.
   (Stuffed Mamba's laws are fitted post hoc.)
6. The random bulk is **explicitly open** in the overlap-space learning theory
   of low-rank RNNs (Ger & Barak 2026).
