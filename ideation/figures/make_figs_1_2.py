"""Raw vector panels for Figures 1 and 2 (to be assembled in a vector editor).

    python ideation/figures/make_figs_1_2.py

Data: stage-3 confirmation networks (seeds 10-14), which were untouched when
scored. Estimator: hf_exact_v2 (bit-identical to the frozen estimator on these
networks; Amendment 9 no-op check). Every example is chosen by a fixed rule
written here BEFORE plotting (no selection for appearance):

Figure 1
  network   : the lowest-seed gated confirmation GRU with N = 128
              (accumulation-trained; "hold" = load-then-zero-input scenario,
              "accumulation" = driven x1 scenario, as in the confirmation test)
  trials    : among trials whose error stayed < eps/2 for t <= T_train and that
              failed before T = 5000, the trials at the 20th / 50th / 90th
              percentile of measured failure time (top row); the 50th-percentile
              trial for the forecast (bottom row)
  oscillation: failing GRU N = 128 confirmation oscillators; same percentile rule
              over networks (each oscillator is one deterministic run)
Figure 2
  network   : among all 27 gated confirmation networks (rnn/gru/lstm, N = 32/128),
              the one whose drift inflation (median |v_NS| / |v_E| on the manifold)
              is closest to the cohort median
  panel A   : decoded value = the manifold point where the slow-point/manifold
              separation equals its median over the manifold
  panel C   : all 27 networks, hold scenario, per-network median failure time
Outputs: ideation/figures/fig1/*.svg|pdf, fig2/*.svg|pdf, preview PDFs, figure_data.json
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, "..", "stress_test")
sys.path.insert(0, ST)
os.chdir(ST)
from hf_tasks import TASKS            # noqa: E402
import hf_core as C                    # noqa: E402
import hf_exact_v2 as X                # noqa: E402
from run_stage3 import hold_inputs     # noqa: E402

torch.set_num_threads(8)
T, TTR = 5000, 50
OUT1, OUT2 = os.path.join(HERE, "fig1"), os.path.join(HERE, "fig2")
os.makedirs(OUT1, exist_ok=True)
os.makedirs(OUT2, exist_ok=True)

# ---- style: 3 colors + gray shading, editable text in SVG ----------------
OBS, OURS, BASE, SHADE = "#1a1a1a", "#1f5fae", "#9a9a9a", "#ececec"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False,
})
PW, PH = 2.25, 1.65            # panel size (inches) for a ~7 in wide figure


def save(fig, folder, name):
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"), bbox_inches="tight", transparent=True)
    plt.close(fig)


def load(task_name, arch, N, seed):
    task = TASKS[task_name]()
    m = C.build(arch, task, N, seed)
    m.load_state_dict(torch.load(os.path.join("nets", f"{task_name}_{arch}_N{N}_s{seed}.pt")))
    m.eval()
    return task, m


def pct_pick(values, qs=(20, 50, 90)):
    order = np.argsort(values)
    return [int(order[int(round(q / 100 * (len(order) - 1)))]) for q in qs]


def err_axes(ax, ymax, eps, xlabel=True):
    ax.axvspan(1, TTR, color=SHADE, lw=0, zorder=0)
    ax.axvline(TTR, color=BASE, lw=0.6, zorder=1)
    ax.axhline(eps, color=OBS, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(1, T)
    ax.set_ylim(0, ymax)
    if xlabel:
        ax.set_xlabel("Step")
    ax.text(TTR * 0.92, ymax * 0.97, "training\nhorizon", ha="right", va="top", fontsize=5.5, color="#6b6b6b")
    ax.text(T * 0.98, eps, "ε", ha="right", va="bottom", fontsize=6.5)


def confirm_nets():
    out = []
    for p in sorted(glob.glob("results/stage1_confirm/*.json")):
        r = json.load(open(p))
        if r["gate_pass"] and r["arch"] != "tc":
            out.append(r)
    return out


record = {}
nets = confirm_nets()

# ======================= FIGURE 1 ==========================================
acc = sorted([r for r in nets if r["task"] == "accumulation" and r["arch"] == "gru" and r["N"] == 128],
             key=lambda r: r["seed"])[0]
task, m = load("accumulation", "gru", 128, acc["seed"])
est = X.AccumulationExact(m, "gru")
fig1 = {}
for col, scen in (("hold", "hold"), ("accumulation", "x1")):
    if scen == "hold":
        u = hold_inputs(256, T, torch.Generator().manual_seed(9500 + acc["seed"]))
        mt = torch.ones(256, T)
        mt[:, :10] = 0
    else:
        u = task.inputs(256, T, torch.Generator().manual_seed(9000 + acc["seed"]))
        mt = task.eval_mask(u)
    et, zt = C.native(m, u, task)
    meas = C.failure_times(et, mt, task.eps)
    ep = C.err_norm(est.predict(u), zt)
    pred = C.failure_times(ep, mt, task.eps)
    e = (et * mt).numpy()
    cand = np.where((e[:, :TTR].max(1) < task.eps / 2) & (meas <= T))[0]
    picks = [int(cand[i]) for i in pct_pick(meas[cand])]
    fig1[col] = dict(err=e, perr=(ep * mt).numpy(), meas=meas, pred=pred, picks=picks)
    record[f"fig1_{col}"] = dict(network=f"accumulation_gru_N128_s{acc['seed']}", scenario=scen,
                                 n_candidates=int(len(cand)), trials=picks,
                                 T_meas=[float(meas[i]) for i in picks], T_pred=[float(pred[i]) for i in picks])

osc_ok = []
for r in sorted([r for r in nets if r["task"] == "oscillation" and r["arch"] == "gru" and r["N"] == 128],
                key=lambda r: r["seed"]):
    t_o, m_o = load("oscillation", "gru", 128, r["seed"])
    u = t_o.inputs(1, T, torch.Generator().manual_seed(9000 + r["seed"]))
    et, zt = C.native(m_o, u, t_o)
    mt = t_o.eval_mask(u)
    Tm = float(C.failure_times(et, mt, t_o.eps)[0])
    yE, _ = X.oscillation_exact(m_o, "gru", t_o, T)
    eE = C.err_norm(yE, zt)
    Tp = float(C.failure_times(eE, mt, t_o.eps)[0])
    if Tm <= T and et[0, :TTR].max() < t_o.eps / 2:
        osc_ok.append(dict(seed=r["seed"], err=et[0].numpy(), perr=eE[0].numpy(), meas=Tm, pred=Tp))
op = pct_pick([o["meas"] for o in osc_ok])
picks_osc = sorted(set(op), key=op.index)
record["fig1_oscillation"] = dict(networks=[f"oscillation_gru_N128_s{osc_ok[i]['seed']}" for i in picks_osc],
                                  T_meas=[osc_ok[i]["meas"] for i in picks_osc],
                                  T_pred=[osc_ok[i]["pred"] for i in picks_osc],
                                  n_candidates=len(osc_ok))

YMAX = 0.6
shades = [OBS, "#555555", "#8a8a8a"]
panels_top = [("A", "hold"), ("B", "accumulation"), ("C", "oscillation")]
for letter, col in panels_top:
    fig, ax = plt.subplots(figsize=(PW, PH))
    if col == "oscillation":
        for k, i in enumerate(picks_osc):
            ax.plot(np.arange(1, T + 1), osc_ok[i]["err"], color=shades[k % 3], lw=0.8)
            ax.plot(osc_ok[i]["meas"], 0.25, "o", ms=2.5, color=shades[k % 3])
    else:
        d = fig1[col]
        for k, i in enumerate(d["picks"]):
            ax.plot(np.arange(1, T + 1), d["err"][i], color=shades[k], lw=0.8)
            ax.plot(d["meas"][i], 0.25, "o", ms=2.5, color=shades[k])
    err_axes(ax, YMAX, 0.25)
    ax.set_ylabel("Absolute error" if letter == "A" else "")
    save(fig, OUT1, f"fig1_{letter}_{col}_observed")

panels_bot = [("D", "hold"), ("E", "accumulation"), ("F", "oscillation")]
for letter, col in panels_bot:
    fig, ax = plt.subplots(figsize=(PW, PH))
    if col == "oscillation":
        o = osc_ok[pct_pick([o["meas"] for o in osc_ok], (50,))[0]]
        obs, prd, Tm, Tp = o["err"], o["perr"], o["meas"], o["pred"]
    else:
        d = fig1[col]
        i = d["picks"][1]
        obs, prd, Tm, Tp = d["err"][i], d["perr"][i], d["meas"][i], d["pred"][i]
    steps = np.arange(1, T + 1)
    ax.plot(steps, obs, color=OBS, lw=0.9, label="observed")
    ax.plot(steps, prd, color=OURS, lw=0.9, ls=(0, (4, 1.5)), label="predicted")
    ax.axvline(Tm, color=OBS, lw=0.6, ymax=0.62)
    ax.axvline(Tp, color=OURS, lw=0.6, ls=":", ymax=0.62)
    ax.text(Tm, YMAX * 0.66, f"T* = {int(Tm)}\npredicted {int(Tp)}", ha="center", va="bottom", fontsize=5.5)
    err_axes(ax, YMAX, 0.25)
    ax.set_ylabel("Absolute error" if letter == "D" else "")
    if letter == "D":
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8), fontsize=5.5, handlelength=1.8)
    save(fig, OUT1, f"fig1_{letter}_{col}_forecast")
    record[f"fig1_{letter}"] = dict(T_meas=float(Tm), T_pred=float(Tp))

# ======================= FIGURE 2 ==========================================
def drifts(m, arch):
    """Invariant-manifold points (E) and slow points (NS) with their drifts."""
    cap = {}
    orig_cb, orig_im = X.converged_block, X.invariant_manifold

    def im_cap(f, dec, S, **kw):
        cap["S_slow"] = S.clone()
        return orig_im(f, dec, S, **kw)

    def cb_cap(f, dec, H, tol):
        idx, nd, cm = orig_cb(f, dec, H, tol)
        cap["H"] = H[idx]
        cap["S_slow_kept"] = cap["S_slow"][idx]
        return idx, nd, cm
    X.invariant_manifold, X.converged_block = im_cap, cb_cap
    try:
        est = X.AccumulationExact(m, arch)
    finally:
        X.invariant_manifold, X.converged_block = orig_im, orig_cb
    X.converged_block = lambda f, dec, H, tol: (torch.arange(len(H)), 0, "")
    try:
        X.invariant_manifold = lambda f, dec, S, **kw: (S, float("nan"), float("nan"))
        naive = X.AccumulationExact(m, arch)
    finally:
        X.invariant_manifold, X.converged_block = orig_im, orig_cb
    f, dec, dim = X.stepper(m, arch)
    with torch.no_grad():
        H, Ss = cap["H"], cap["S_slow_kept"]
        z = torch.zeros(len(H), 1, dtype=torch.float64)
        s = dec(H)[:, 0].numpy()
        vE = dec(f(H, z))[:, 0].numpy() - s
        vN = dec(f(Ss, z))[:, 0].numpy() - dec(Ss)[:, 0].numpy()
    m.float()
    return est, naive, H.numpy(), Ss.numpy(), s, vE, vN


rows = []
cache = {}
for r in nets:
    if r["task"] != "accumulation":
        continue
    task, m = load("accumulation", r["arch"], r["N"], r["seed"])
    est, naive, H, Ss, s, vE, vN = drifts(m, r["arch"])
    ratio = float(np.median(np.abs(vN)) / max(np.median(np.abs(vE)), 1e-12))
    u = hold_inputs(256, T, torch.Generator().manual_seed(9500 + r["seed"]))
    mt = torch.ones(256, T)
    mt[:, :10] = 0
    et, zt = C.native(m, u, task)
    meas = C.failure_times(et, mt, task.eps)
    pE = C.failure_times(C.err_norm(est.predict(u), zt), mt, task.eps)
    pN = C.failure_times(C.err_norm(naive.predict(u), zt), mt, task.eps)
    tag = f"accumulation_{r['arch']}_N{r['N']}_s{r['seed']}"
    rows.append(dict(tag=tag, arch=r["arch"], N=r["N"], inflation=ratio,
                     T_meas=float(np.median(np.minimum(meas, T + 1))),
                     T_E=float(np.median(np.minimum(pE, T + 1))),
                     T_NS=float(np.median(np.minimum(pN, T + 1)))))
    cache[tag] = (m, r["arch"], H, Ss, s, vE, vN, et, zt, u)
    print(f"{tag:30s} inflation {ratio:6.2f}  T* meas {rows[-1]['T_meas']:6.0f}  E {rows[-1]['T_E']:6.0f}  NS {rows[-1]['T_NS']:6.0f}", flush=True)
record["fig2_C_networks"] = rows
infl = np.array([r["inflation"] for r in rows])
rep = rows[int(np.argmin(np.abs(infl - np.median(infl))))]
record["fig2_representative"] = dict(tag=rep["tag"], inflation=rep["inflation"], cohort_median_inflation=float(np.median(infl)))
m, arch, H, Ss, s, vE, vN, et, zt, u = cache[rep["tag"]]

# panel A: projection (decoded value, separation direction)
Wout = m.net.Wout.weight.detach().double().numpy()[0]
dim = H.shape[1]
Cvec = np.zeros(dim)
Cvec[: len(Wout)] = Wout
D = Ss - H                                    # separation lies in null(C) by construction
uvec = D.mean(0)
uvec -= Cvec * (uvec @ Cvec) / (Cvec @ Cvec)
uvec /= np.linalg.norm(uvec)
sep = np.linalg.norm(D, axis=1)
j = int(np.argmin(np.abs(sep - np.median(sep))))
record["fig2_A_point"] = dict(s=float(s[j]), separation=float(sep[j]))
f, dec, _ = X.stepper(m, arch)
with torch.no_grad():                         # trajectories: load various values, then zero input
    B = 12
    z0 = np.linspace(s.min() * 0.8, s.max() * 0.8, B)
    uu = torch.zeros(B, 70, 1, dtype=torch.float64)
    uu[:, :10, 0] = torch.as_tensor(z0 / 10)[:, None]
    St = torch.zeros(B, dim, dtype=torch.float64)
    traj = []
    for t in range(70):
        St = f(St, uu[:, t])
        traj.append(St.numpy().copy())
m.float()
traj = np.stack(traj, 1)
fig, ax = plt.subplots(figsize=(PW, PH))
for b in range(B):
    ax.plot(traj[b] @ Cvec, traj[b] @ uvec, color=BASE, lw=0.5, alpha=0.8, zorder=1)
ax.plot(s, H @ uvec, color=OURS, lw=1.2, zorder=3, label="invariant manifold")
ax.plot(s, Ss @ uvec, color=OBS, lw=0.6, ls=(0, (2, 1.5)), zorder=2, label="slow points")
ax.plot(s[j], Ss[j] @ uvec, "o", mfc="white", mec=OBS, ms=4, mew=0.8, zorder=4)
ax.plot(s[j], H[j] @ uvec, "o", color=OURS, ms=4, zorder=4)
ax.annotate("", xy=(s[j], H[j] @ uvec), xytext=(s[j], Ss[j] @ uvec),
            arrowprops=dict(arrowstyle="->", lw=0.6, color=OBS), zorder=5)
ax.set_xlabel("Decoded value  s")
ax.set_ylabel("Transverse coordinate")
ax.legend(loc="best", fontsize=5.5, handlelength=1.8)
save(fig, OUT2, "fig2_A_slowpoint_vs_manifold")

# panel B: drift along the manifold, with observed drift from native holds
with torch.no_grad():
    yN, _ = m(u)
yN = yN[..., 0].numpy()
ds = (yN[:, 201:] - yN[:, 200:-1]).ravel()
sv = yN[:, 200:-1].ravel()
bins = np.linspace(s.min(), s.max(), 25)
idx = np.digitize(sv, bins)
bc, bm = [], []
for k in range(1, len(bins)):
    sel = idx == k
    if sel.sum() > 200:
        bc.append(0.5 * (bins[k - 1] + bins[k]))
        bm.append(float(np.mean(ds[sel])))
fig, ax = plt.subplots(figsize=(PW, PH))
ax.axhline(0, color=BASE, lw=0.5)
ax.plot(s, vN, color=BASE, lw=1.0, label="slow-point estimate")
ax.plot(s, vE, color=OURS, lw=1.2, label="invariant-manifold estimate")
ax.plot(bc, bm, "o", ms=2.2, color=OBS, label="observed (network rollouts)")
ax.set_xlabel("Decoded value  s")
ax.set_ylabel("Drift per step  δ(s, 0)")
ax.legend(loc="best", fontsize=5.5, handlelength=1.6)
save(fig, OUT2, "fig2_B_drift")
record["fig2_B_observed_bins"] = dict(centers=bc, mean_drift=bm)

# panel C: predicted vs observed median failure time, all 27 networks
fig, ax = plt.subplots(figsize=(PW, PH))
tm = np.array([r["T_meas"] for r in rows])
lo, hi = 10, T * 1.3
ax.plot([lo, hi], [lo, hi], color=BASE, lw=0.6, zorder=1)
ax.scatter(tm, [r["T_NS"] for r in rows], s=9, facecolors="white", edgecolors=BASE, linewidths=0.7,
           label="slow-point defect", zorder=2)
ax.scatter(tm, [r["T_E"] for r in rows], s=9, color=OURS, label="invariant-manifold defect", zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Observed failure time  T*")
ax.set_ylabel("Predicted  T*")
ax.legend(loc="upper left", fontsize=5.5, handlelength=1.2)
save(fig, OUT2, "fig2_C_predicted_vs_observed")

# ---- previews (rough assembly only; final layout belongs in the vector editor)
json.dump(record, open(os.path.join(HERE, "figure_data.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in record.items() if k != "fig2_C_networks"}, indent=1, default=float))
