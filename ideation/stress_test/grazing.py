"""EXPLORATORY (seeds 15-19, outcomes already known): are wrong approvals
'grazing' cases, where the predicted error comes close to tolerance before H?

For every approved case (frozen E rule, margin 0.10) compute the predicted
error margin  kappa = eps - max_{t<=H} e_pred(t)  and compare wrong vs correct
approvals. Also evaluate two abstention diagnostics that use no outcome data:
lambda2 >= 1 (reduction invalid) and kappa below a threshold.
"""
import json
import os

import numpy as np
import torch

from hf_tasks import TASKS
import hf_core as C
import hf_exact_FROZEN as X
from run_stage2 import shifted_inputs
from run_stage3 import hold_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
task = TASKS["accumulation"]()
T, HS, SCEN, MARGIN, EPS = 5000, (250, 500, 1000, 2000), ("hold", "x1", "x0.5"), 0.10, task.eps
inv = {r["tag"]: r for r in json.load(open(os.path.join(HERE, "results", "investigate_prereg.json")))}
recs = []
for tag, info in inv.items():
    _, arch, Nw, sd = tag.split("_")
    N, seed = int(Nw[1:]), int(sd[1:])
    m = C.build(arch, task, N, seed)
    m.load_state_dict(torch.load(os.path.join(HERE, "nets", tag + ".pt")))
    m.eval()
    est = X.AccumulationExact(m, arch)
    for sc in SCEN:
        if sc == "x1":
            u = task.inputs(256, T, torch.Generator().manual_seed(9000 + seed))
            mt = task.eval_mask(u)
        elif sc == "x0.5":
            u = shifted_inputs(task, 256, T, torch.Generator().manual_seed(9100 + seed), 0.5)
            mt = task.eval_mask(u)
        else:
            u = hold_inputs(256, T, torch.Generator().manual_seed(9500 + seed))
            mt = torch.ones(256, T)
            mt[:, :10] = 0
        et, zt = C.native(m, u, task)
        meas = C.failure_times(et, mt, EPS)
        ep = C.err_norm(est.predict(u), zt).numpy() * mt.numpy()
        pred = C.failure_times(torch.as_tensor(ep), mt, EPS)
        for H in HS:
            app = pred > H * (1 + MARGIN)
            kappa = EPS - ep[:, :H].max(1)
            for i in np.where(app)[0]:
                recs.append(dict(tag=tag, lam2=info["lam2"], scen=sc, H=H, kappa=float(kappa[i]),
                                 wrong=bool(meas[i] <= H)))
    print("done", tag, flush=True)

k = np.array([r["kappa"] for r in recs])
w = np.array([r["wrong"] for r in recs])
l2 = np.array([r["lam2"] for r in recs])
print(f"\napproved cases: {len(recs)}, wrong: {w.sum()}")
print(f"kappa of wrong approvals: median {np.median(k[w]):.4f}, 90th pct {np.percentile(k[w], 90):.4f}, max {k[w].max():.4f}")
print(f"kappa of correct approvals: 10th pct {np.percentile(k[~w], 10):.4f}, median {np.median(k[~w]):.4f}")
print(f"wrong approvals in lambda2>=1 networks: {int((w & (l2 >= 1)).sum())}/{w.sum()}")
print("\nabstention trade-off (exploratory; thresholds NOT chosen on this data for any claim):")
for kap in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
    for use_l2 in (False, True):
        keep = (k >= kap) & ((l2 < 1) if use_l2 else True)
        print(f"  kappa>={kap:<5} lam2<1 filter={str(use_l2):5s} -> wrong {int((w & keep).sum()):3d}/{int(keep.sum())} "
              f"({100 * (w & keep).sum() / max(keep.sum(), 1):.2f}%), approvals kept {100 * keep.mean():.1f}%")
json.dump(recs, open(os.path.join(HERE, "results", "grazing_prereg.json"), "w"))
