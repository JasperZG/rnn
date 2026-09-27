"""Audit of the confirmatory result against the external review's concerns.

For every gated confirmation accumulation network (seeds 10-14):
  A  exact discrete invariance residual  f(h(s),0) - h(s+v(s))  at held-out
     midpoints (not the tangent form that was optimized)
  F  per-trial error statistics: median and 90th-percentile |log(T^/T)| over
     trials where both prediction and measurement fail; censoring counted
     separately (never scored as exact matches)
  E  state-range: fraction of measured failures at which the true task state
     lies outside the calibrated manifold range
Uses the frozen estimator (hf_exact_FROZEN.py). Exploratory audit, not a new test.
"""
import glob
import json
import os

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from hf_tasks import TASKS
import hf_core as C
import hf_exact_FROZEN as X
from run_stage2 import shifted_inputs
from run_stage3 import hold_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
task = TASKS["accumulation"]()
T = 5000
rows = []
for p in sorted(glob.glob(os.path.join(HERE, "results", "stage1_confirm", "accumulation_*.json"))):
    r = json.load(open(p))
    if not r["gate_pass"]:
        continue
    tag = f"accumulation_{r['arch']}_N{r['N']}_s{r['seed']}"
    m = C.build(r["arch"], task, r["N"], r["seed"])
    m.load_state_dict(torch.load(os.path.join(HERE, "nets", tag + ".pt")))
    m.eval()
    est = X.AccumulationExact(m, r["arch"])

    # --- A: exact discrete invariance residual at held-out midpoints ---------
    f, dec, dim = X.stepper(m, r["arch"])
    with torch.no_grad():
        c = torch.linspace(-0.3, 0.3, 401, dtype=torch.float64)[:, None]
        S = torch.zeros(401, dim, dtype=torch.float64)
        z0 = torch.zeros(401, 1, dtype=torch.float64)
        for _ in range(10):
            S = f(S, c)
        for _ in range(20):
            S = f(S, z0)
    parts = [X.slow_points(f, dec, S[i:i + 50]) for i in range(0, 401, 50)]
    S = torch.cat([q[0] for q in parts])
    with torch.no_grad():
        s = dec(S)[:, 0]
        o = torch.argsort(s)
        S, s = S[o], s[o]
        keep, last = [], -1e9
        for i, si in enumerate(s.tolist()):
            if si - last >= 5e-3:
                keep.append(i)
                last = si
        S = S[keep]
    H, _, _ = X.invariant_manifold(f, dec, S)
    with torch.no_grad():
        sH = dec(H)[:, 0].numpy()
        Hn = H.numpy()
        fit_idx, held = np.arange(0, len(sH), 2), np.arange(1, len(sH) - 1, 2)
        spl = CubicSpline(sH[fit_idx], Hn[fit_idx], axis=0)
        Hh = torch.as_tensor(Hn[held])
        Fh = f(Hh, torch.zeros(len(held), 1, dtype=torch.float64))
        vh = dec(Fh)[:, 0].numpy() - sH[held]
        exact_res = np.linalg.norm(Fh.numpy() - spl(sH[held] + vh), axis=1)
        tang_res = np.linalg.norm(Fh.numpy() - Hn[held] - vh[:, None] * spl(sH[held], 1), axis=1)
        interp_err = np.linalg.norm(spl(sH[held]) - Hn[held], axis=1)
        step = np.linalg.norm(Fh.numpy() - Hn[held], axis=1)
    m.float()

    # --- F and E: per-trial statistics and state range -----------------------
    lo, hi = est.lo, est.hi
    out = {}
    for name in ("x1", "x0.5", "hold"):
        if name == "x1":
            u = task.inputs(256, T, torch.Generator().manual_seed(9000 + r["seed"]))
            mt = task.eval_mask(u)
        elif name == "x0.5":
            u = shifted_inputs(task, 256, T, torch.Generator().manual_seed(9100 + r["seed"]), 0.5)
            mt = task.eval_mask(u)
        else:
            u = hold_inputs(256, T, torch.Generator().manual_seed(9500 + r["seed"]))
            mt = torch.ones(256, T)
            mt[:, :10] = 0
        et, zt = C.native(m, u, task)
        meas = C.failure_times(et, mt, task.eps)
        pred = C.failure_times(C.err_norm(est.predict(u), zt), mt, task.eps)
        both = (meas <= T) & (pred <= T)
        le = np.abs(np.log(pred[both] / meas[both])) if both.any() else np.array([np.nan])
        idx = np.where(meas <= T)[0]
        zf = zt[idx, (meas[idx] - 1).astype(int), 0].numpy() if len(idx) else np.array([])
        out[name] = dict(
            n_trials=256, n_both_fail=int(both.sum()),
            n_meas_censored=int((meas > T).sum()), n_pred_censored=int((pred > T).sum()),
            n_censor_disagree=int(((meas > T) != (pred > T)).sum()),
            median_abs_log_err=float(np.median(le)), p90_abs_log_err=float(np.percentile(le, 90)),
            max_abs_log_err=float(np.max(le)),
            frac_fail_outside_range=float(np.mean((zf < lo) | (zf > hi))) if len(zf) else float("nan"),
            median_T_meas=float(np.median(meas)))
    rows.append(dict(tag=tag, arch=r["arch"], width=r["N"], seed=r["seed"], range=[lo, hi],
                     invariance=dict(exact_median=float(np.median(exact_res)), exact_max=float(exact_res.max()),
                                     tangent_median=float(np.median(tang_res)),
                                     interp_median=float(np.median(interp_err)),
                                     step_median=float(np.median(step))), **out))
    inv = rows[-1]["invariance"]
    print(f"{tag:28s} inv exact med/max={inv['exact_median']:.1e}/{inv['exact_max']:.1e} "
          f"(tangent {inv['tangent_median']:.1e}, interp {inv['interp_median']:.1e}, step {inv['step_median']:.1e}) | "
          + " | ".join(f"{k}: medlog={v['median_abs_log_err']:.3f} p90={v['p90_abs_log_err']:.3f} "
                       f"cens(m/p/dis)={v['n_meas_censored']}/{v['n_pred_censored']}/{v['n_censor_disagree']} "
                       f"outside={v['frac_fail_outside_range']:.2f}" for k, v in out.items()), flush=True)
json.dump(rows, open(os.path.join(HERE, "results", "audit_confirm.json"), "w"))
