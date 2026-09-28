"""Figure 2 panels A and B, redrawn to show the quantity that actually differs.

    python ideation/figures/make_fig2_AB.py   (run after make_figs_1_2.py)

Same representative network and decoded value as make_figs_1_2.py (rules there;
read back from figure_data.json). Finding that motivated the redraw: slow points
lie very close to the invariant manifold (median separation ~0.002) and their raw
one-step drift is nearly identical, but the small offset lies along a fast
direction whose decay leaks into the readout while the estimator relaxes the
state. The reduced model's effective drift is therefore inflated.

  A: along the manifold, the slow points' mislabel (asymptotic position minus
     decoded label, l.(h_slow - h_manifold)) vs the slow-point model's drift error
     (a first attempt plotting raw zero-input trajectories showed no visible
     difference and was discarded)
  B: effective drift used by each reduced model, disp(s, u=0), vs drift observed
     in long native hold rollouts
The earlier state-space panel is kept as fig2_supp_statespace.
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, "..", "stress_test")
sys.path.insert(0, ST)
sys.path.insert(0, HERE)
os.chdir(ST)
from hf_tasks import TASKS            # noqa: E402
import hf_core as C                    # noqa: E402
import hf_exact_v2 as X                # noqa: E402
from run_stage3 import hold_inputs     # noqa: E402

torch.set_num_threads(8)
OBS, OURS, BASE = "#1a1a1a", "#1f5fae", "#9a9a9a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
    "pdf.fonttype": 42, "legend.frameon": False,
})
PW, PH = 2.25, 1.65
OUT = os.path.join(HERE, "fig2")
rec = json.load(open(os.path.join(HERE, "figure_data.json")))
tag = rec["fig2_representative"]["tag"]
_, arch, Nw, sd = tag.split("_")
N, seed = int(Nw[1:]), int(sd[1:])
task = TASKS["accumulation"]()
m = C.build(arch, task, N, seed)
m.load_state_dict(torch.load(os.path.join("nets", tag + ".pt")))
m.eval()


def save(fig, name):
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", transparent=True)
    plt.close(fig)


# estimators + captured point sets (same capture as make_figs_1_2.py)
cap = {}
oc, oi = X.converged_block, X.invariant_manifold


def im_cap(f, dec, S, **kw):
    cap["S"] = S.clone()
    return oi(f, dec, S, **kw)


def cb_cap(f, dec, H, tol):
    idx, nd, cm = oc(f, dec, H, tol)
    cap["H"], cap["Ss"] = H[idx], cap["S"][idx]
    return idx, nd, cm
X.invariant_manifold, X.converged_block = im_cap, cb_cap
est = X.AccumulationExact(m, arch)
X.invariant_manifold, X.converged_block = oi, oc
X.converged_block = lambda f, dec, H, tol: (torch.arange(len(H)), 0, "")
X.invariant_manifold = lambda f, dec, S, **kw: (S, float("nan"), float("nan"))
naive = X.AccumulationExact(m, arch)
X.invariant_manifold, X.converged_block = oi, oc

f, dec, dim = X.stepper(m, arch)
H, Ss = cap["H"], cap["Ss"]
with torch.no_grad():
    s = dec(H)[:, 0].numpy()
    sep = (Ss - H).norm(dim=1).numpy()
    j = int(np.argmin(np.abs(sep - np.median(sep))))
    z = torch.zeros(2, 1, dtype=torch.float64)
    st = torch.stack([Ss[j], H[j]])
    traj = [dec(st)[:, 0].numpy()]
    for _ in range(40):
        st = f(st, z)
        traj.append(dec(st)[:, 0].numpy())
m.float()
traj = np.array(traj)
s0 = float(s[j])

# ---- panel A: slow points are mislabelled along the manifold; that mislabel is the drift error
f, dec, dim = X.stepper(m, arch)              # back to float64 for the Jacobians
z1 = torch.zeros(1, 1, dtype=torch.float64)
Hn = H.detach().numpy()
tang = np.gradient(Hn, s, axis=0)
mis = np.zeros(len(s))
for i in range(len(s)):
    J = torch.autograd.functional.jacobian(lambda h: f(h[None], z1)[0], H[i]).numpy()
    w, L = np.linalg.eig(J.T)
    l = np.real(L[:, np.argmin(np.abs(w - 1))])
    l = l / (l @ tang[i])
    mis[i] = l @ (Ss[i] - H[i]).detach().numpy()        # asymptotic position - decoded label
m.float()
derr = naive.disp.ev(s, np.zeros_like(s)) - est.disp.ev(s, np.zeros_like(s))
fig, ax = plt.subplots(figsize=(PW, PH))
ax.axhline(0, color=BASE, lw=0.4)
ax.plot(s, mis, color=OBS, lw=1.2, label="slow-point mislabel  ℓ·(h_slow − h_man)")
ax.plot(s, -derr, color=BASE, lw=1.0, ls=(0, (3, 1.5)), label="drift error of slow-point model (−)")
ax.set_xlabel("Decoded value  s")
ax.set_ylabel("Per step")
ax.legend(loc="best", fontsize=5.2, handlelength=1.8)
if os.path.exists(os.path.join(OUT, "fig2_A_slowpoint_vs_manifold.pdf")):
    for ext in ("svg", "pdf"):
        shutil.move(os.path.join(OUT, f"fig2_A_slowpoint_vs_manifold.{ext}"),
                    os.path.join(OUT, f"fig2_supp_statespace.{ext}"))
for old in ("fig2_B_drift", "fig2_A_readout_leak"):
    for ext in ("svg", "pdf"):
        if os.path.exists(os.path.join(OUT, f"{old}.{ext}")):
            os.remove(os.path.join(OUT, f"{old}.{ext}"))
save(fig, "fig2_candidate_A_mislabel_partial")
corr = float(np.corrcoef(mis, -derr)[0, 1])
rec["fig2_A_mechanism"] = dict(corr_mislabel_vs_drift_error=corr,
                               median_abs_mislabel=float(np.median(np.abs(mis))),
                               median_abs_drift_error=float(np.median(np.abs(derr))))

# ---- panel B: effective drift used by each reduced model vs observed
sg = np.linspace(s.min(), s.max(), 300)
dE = est.disp.ev(sg, np.zeros_like(sg))
dN = naive.disp.ev(sg, np.zeros_like(sg))
bc, bm = rec["fig2_B_observed_bins"]["centers"], rec["fig2_B_observed_bins"]["mean_drift"]
fig, ax = plt.subplots(figsize=(PW, PH))
ax.axhline(0, color=BASE, lw=0.4)
ax.plot(sg, dN, color=BASE, lw=1.0, label="slow-point model")
ax.plot(sg, dE, color=OURS, lw=1.2, label="invariant-manifold model")
ax.plot(bc, bm, "o", ms=2.2, color=OBS, label="observed (network rollouts)")
lo = min(np.min(bc), -0.8)
hi = max(np.max(bc), 0.8)
sel = (sg >= lo) & (sg <= hi)
ymin = min(dE[sel].min(), dN[sel].min(), min(bm))
ymax = max(dE[sel].max(), dN[sel].max(), max(bm))
pad = 0.15 * (ymax - ymin)
ax.set_xlim(lo, hi)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlabel("Decoded value  s")
ax.set_ylabel("Effective drift per step")
ax.legend(loc="lower left", fontsize=5.5, handlelength=1.6)
save(fig, "fig2_B_effective_drift")

rec["fig2_B_redraw"] = dict(drift_E_at_0p5=float(est.disp.ev(0.5, 0.0)), drift_NS_at_0p5=float(naive.disp.ev(0.5, 0.0)))
json.dump(rec, open(os.path.join(HERE, "figure_data.json"), "w"), indent=1, default=float)
print(rec["fig2_A_mechanism"], rec["fig2_B_redraw"])
