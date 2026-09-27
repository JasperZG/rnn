"""EXPLORATORY: why do wrong approvals concentrate in a few networks (seeds 15-19)?

1. per-network wrong approvals under the frozen E rule (margin 0.10)
2. candidate validity diagnostics for every network, computed from the weights
   and short probes only: exact invariance residual (held-out), lambda2^R,
   manifold range, drift rms, and how far the trials' true state goes relative
   to the calibrated manifold range
3. detail of each wrong-approved case in the flagged networks
"""
import json
import os

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.stats import spearmanr

from hf_tasks import TASKS
import hf_core as C
import hf_exact_FROZEN as X
from run_stage2 import shifted_inputs
from run_stage3 import hold_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
task = TASKS["accumulation"]()
T, HS, SCEN, MARGIN = 5000, (250, 500, 1000, 2000), ("hold", "x1", "x0.5"), 0.10
d = dict(np.load(os.path.join(HERE, "results", "cases_stage1_prereg.npz")))
nets = list(d["nets"])

rows = []
for ni, tag in enumerate(nets):
    s = d["net"] == ni
    wrong = []            # (scenario, H, predicted, measured, trial index within net-scenario)
    n_app = 0
    for H in HS:
        app = s & (d["E"] > H * (1 + MARGIN))
        bad = app & (d["meas"] <= H)
        n_app += int(app.sum())
        for i in np.where(bad)[0]:
            wrong.append((SCEN[d["scen"][i]], H, float(d["E"][i]), float(d["meas"][i]), int(i)))
    worst = 0.0
    for si in range(3):
        for H in HS:
            c = s & (d["scen"] == si)
            app = c & (d["E"] > H * (1 + MARGIN))
            if app.sum():
                worst = max(worst, float(np.mean(d["meas"][app] <= H)))

    _, arch, Nw, sd = tag.split("_")
    N, seed = int(Nw[1:]), int(sd[1:])
    m = C.build(arch, task, N, seed)
    m.load_state_dict(torch.load(os.path.join(HERE, "nets", tag + ".pt")))
    m.eval()
    est = X.AccumulationExact(m, arch)
    f, dec, dim = X.stepper(m, arch)
    # held-out exact invariance residual, as in audit.py
    with torch.no_grad():
        c0 = torch.linspace(-0.3, 0.3, 401, dtype=torch.float64)[:, None]
        S = torch.zeros(401, dim, dtype=torch.float64)
        z0 = torch.zeros(401, 1, dtype=torch.float64)
        for _ in range(10):
            S = f(S, c0)
        for _ in range(20):
            S = f(S, z0)
    S = torch.cat([X.slow_points(f, dec, S[i:i + 50])[0] for i in range(0, 401, 50)])
    with torch.no_grad():
        sv = dec(S)[:, 0]
        o = torch.argsort(sv)
        S, sv = S[o], sv[o]
        keep, last = [], -1e9
        for i, x in enumerate(sv.tolist()):
            if x - last >= 5e-3:
                keep.append(i)
                last = x
        S = S[keep]
    Hm, _, _ = X.invariant_manifold(f, dec, S)
    with torch.no_grad():
        sH, Hn = dec(Hm)[:, 0].numpy(), Hm.numpy()
        fi, ho = np.arange(0, len(sH), 2), np.arange(1, len(sH) - 1, 2)
        spl = CubicSpline(sH[fi], Hn[fi], axis=0)
        Fh = f(torch.as_tensor(Hn[ho]), torch.zeros(len(ho), 1, dtype=torch.float64))
        vh = dec(Fh)[:, 0].numpy() - sH[ho]
        res = np.linalg.norm(Fh.numpy() - spl(sH[ho] + vh), axis=1)
        # residual relative to the local drift it must resolve
        rel = res / np.maximum(np.abs(vh) * np.linalg.norm(spl(sH[ho], 1), axis=1), 1e-12)
    m.float()
    lo, hi = est.lo, est.hi
    # detail for wrong-approved cases: where was the true/predicted state?
    detail = []
    if wrong:
        by_scen = {}
        for w in wrong:
            by_scen.setdefault(w[0], []).append(w)
        for sc, ws in by_scen.items():
            if sc == "x1":
                u = task.inputs(256, T, torch.Generator().manual_seed(9000 + seed))
            elif sc == "x0.5":
                u = shifted_inputs(task, 256, T, torch.Generator().manual_seed(9100 + seed), 0.5)
            else:
                u = hold_inputs(256, T, torch.Generator().manual_seed(9500 + seed))
            with torch.no_grad():
                y, _ = m(u)
            z = task.targets(u)
            yE = est.predict(u)
            base = int(np.where(s & (d["scen"] == SCEN.index(sc)))[0][0])
            for (sc_, H, pT, mT, gi) in ws[:6]:
                j = gi - base
                t = int(mT) - 1
                detail.append(dict(scen=sc, H=H, pred=pT, meas=mT,
                                   z_true=float(z[j, t, 0]), y_net=float(y[j, t, 0]), y_pred=float(yE[j, t, 0]),
                                   in_range=bool(lo <= float(z[j, t, 0]) <= hi),
                                   net_state_margin_to_end=float(min(y[j, t, 0] - lo, hi - y[j, t, 0]))))
    rows.append(dict(tag=tag, worst_cell_wrong=worst, n_wrong=len(wrong), n_app=n_app,
                     lam2_pow_R=est.diag["lam2_pow_R"], lam2=est.diag["lam2_max"],
                     inv_res_med=float(np.median(res)), inv_res_max=float(res.max()),
                     inv_rel_max=float(np.max(rel)), inv_rel_p90=float(np.percentile(rel, 90)),
                     range=[lo, hi], drift_rms=est.diag["drift_rms"], wrong_detail=detail))
    r = rows[-1]
    print(f"{tag:28s} worst={worst:.3f} nwrong={len(wrong):3d} lam2={r['lam2']:.3f} "
          f"inv med/max={r['inv_res_med']:.1e}/{r['inv_res_max']:.1e} rel p90/max={r['inv_rel_p90']:.2f}/{r['inv_rel_max']:.1f} "
          f"range=[{lo:+.2f},{hi:+.2f}]", flush=True)

json.dump(rows, open(os.path.join(HERE, "results", "investigate_prereg.json"), "w"), indent=1)
w = np.array([r["n_wrong"] for r in rows])
print("\nSpearman(diagnostic, #wrong approvals) across 30 networks:")
for k in ("lam2", "inv_res_med", "inv_res_max", "inv_rel_p90", "inv_rel_max", "drift_rms"):
    v = np.array([r[k] for r in rows])
    print(f"  {k:12s} rho={spearmanr(v, w).correlation:+.2f}")
print("\nwrong-approved cases in flagged networks:")
for r in sorted(rows, key=lambda r: -r["n_wrong"])[:5]:
    if not r["n_wrong"]:
        continue
    print(f" {r['tag']} ({r['n_wrong']} wrong, worst cell {r['worst_cell_wrong']:.2f})")
    for x in r["wrong_detail"]:
        print(f"    {x['scen']:5s} H={x['H']:4d} pred={x['pred']:6.0f} meas={x['meas']:6.0f} z_true={x['z_true']:+.3f} "
              f"y_net={x['y_net']:+.3f} y_pred={x['y_pred']:+.3f} in_range={x['in_range']} "
              f"net margin to manifold end={x['net_state_margin_to_end']:+.3f}")
