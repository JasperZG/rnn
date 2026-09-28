# Figures 1–2: raw vector panels (SVG + PDF), ready for assembly in a vector editor

Data: stage-3 confirmation networks (seeds 10–14, untouched when scored).
Estimator: `hf_exact_v2`, which is bit-identical to the frozen estimator on these
networks. Every example is chosen by a fixed rule, written in the scripts
before plotting. The chosen networks and trials are recorded in
`figure_data.json`.

Scripts:
- `make_figs_1_2.py` builds Figure 1 and Figure 2 panel C.
- `make_fig2_AB.py` builds the Figure 2 A/B candidates.

## Figure 1: hidden failure and forecast (`fig1/`)

| panel | file | content |
|---|---|---|
| A | `fig1_A_hold_observed` | Hold: 3 trials (20th/50th/90th percentile of T*), all < ε/2 through the training horizon |
| B | `fig1_B_accumulation_observed` | Accumulation (driven): same rule |
| C | `fig1_C_oscillation_observed` | Oscillation: 3 failing GRU-128 confirmation oscillators (same percentile rule over networks) |
| D | `fig1_D_hold_forecast` | Hold, 50th-percentile trial: observed vs predicted error; T* 464 vs predicted 460 |
| E | `fig1_E_accumulation_forecast` | Accumulation: T* 534 vs 537 |
| F | `fig1_F_oscillation_forecast` | Oscillation: T* 2171 vs 2174 |

Network (Figure 1, A/B and D/E): `accumulation_gru_N128_s10`, the lowest-seed
gated confirmation GRU-128.

## Figure 2: why the defect must be measured correctly (`fig2/`)

| panel | file | status |
|---|---|---|
| B | `fig2_B_effective_drift` | **Main.** The effective drift each reduced model uses, against the drift observed in network rollouts. The invariant-manifold model tracks the observed drift; the slow-point model is off by up to ~2.5×. |
| C | `fig2_C_predicted_vs_observed` | **Main.** Predicted vs observed median T*, all 27 confirmation networks. Invariant manifold on the identity line; slow points off by 2–5×. |
| supp | `fig2_supp_statespace` | State-space projection: slow points lie within ~0.002 of the invariant manifold, which is invisible at this scale. |
| candidate A | `fig2_candidate_A_mislabel_partial` | **Exploratory; partial mechanism only.** Slow points' position error along the manifold, ℓ·(h_slow − h_manifold), against the slow-point model's drift error. Same shape (r = 0.94) but ~7× smaller, so it does not fully explain the drift error. |

Representative network (Figure 2): `accumulation_rnn_N128_s11`. Rule: its
drift inflation is closest to the median of the 27 networks.

**Important for the text.** The planned panel A ("the slow point is visibly
off the manifold → its drift is inflated") is *not* supported. The two point
sets nearly coincide, and so do their raw one-step drifts. The discrepancy is
in the *effective* drift the reduced model uses. Its full mechanism is not yet
established.

Style:
- colors: black = observed, blue = our method, gray = baseline / shading;
- Arial 7 pt;
- text is left editable in the SVGs.
