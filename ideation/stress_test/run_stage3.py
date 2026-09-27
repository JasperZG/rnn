"""Stage 3: exact slow-manifold estimator vs the stage-1 regression estimator.

    python run_stage3.py accumulation --src stage1_fresh
    python run_stage3.py oscillation  --src stage1
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from hf_tasks import TASKS
import hf_core as C
import hf_exact as X
from run_stage2 import shifted_inputs

HERE = os.path.dirname(os.path.abspath(__file__))


def nets_from(src, task):
    for p in sorted(glob.glob(os.path.join(HERE, "results", src, f"{task.name}_*.json"))):
        r = json.load(open(p))
        if not r["gate_pass"]:
            continue
        tag = f"{task.name}_{r['arch']}_N{r['N']}_s{r['seed']}"
        m = C.build(r["arch"], task, r["N"], r["seed"])
        m.load_state_dict(torch.load(os.path.join(HERE, "nets", tag + ".pt")))
        m.eval()
        yield r, m, tag


def curve_err(pred_err, meas_err, eps):
    """Median over log-spaced times of |log(e_pred/e_meas)| where e_meas > eps/20."""
    T = meas_err.shape[1]
    ts = np.unique(np.geomspace(1, T, 60).astype(int)) - 1
    p = np.median(pred_err[:, ts], 0)
    mm = np.median(meas_err[:, ts], 0)
    ok = mm > eps / 20
    return float(np.median(np.abs(np.log(np.maximum(p[ok], 1e-9) / mm[ok])))) if ok.any() else float("nan")


def run_accumulation(src, out):
    task = TASKS["accumulation"]()
    T = 100 * task.T_train
    rows = []
    for r, m, tag in nets_from(src, task):
        t0 = time.time()
        est = X.AccumulationExact(m, r["arch"])
        Z, U, D, Uh = C.probe_samples(m, task, r["seed"])
        reg = C.DefectFit(task)
        reg.fit(Z, U, D, r["seed"], Uh=Uh)
        for fac in (1.0, 0.5):
            if fac == 1.0:
                u = task.inputs(256, T, torch.Generator().manual_seed(9000 + r["seed"]))
            else:
                u = shifted_inputs(task, 256, T, torch.Generator().manual_seed(9100 + r["seed"]), fac)
            et, zt = C.native(m, u, task)
            mt = task.eval_mask(u)
            meas = C.failure_times(et, mt, task.eps)
            eE = C.err_norm(est.predict(u), zt)
            with torch.no_grad():
                eR = C.err_norm(C.predict_N(reg, task, u), zt)
            tE = C.failure_times(eE, mt, task.eps)
            tR = C.failure_times(eR, mt, task.eps)
            res = dict(E=C.compare(tE, meas, T), R=C.compare(tR, meas, T))
            res["E"]["curve"] = curve_err(eE.numpy(), et.numpy(), task.eps)
            res["R"]["curve"] = curve_err(eR.numpy(), et.numpy(), task.eps)
            rows.append(dict(tag=tag, arch=r["arch"], width=r["N"], seed=r["seed"], factor=fac,
                             diag=est.diag, **res))
            print(f"{tag:30s} x{fac:<4} medT={res['E']['median_meas']:6.0f} "
                  f"agree E={res['E']['agree']:.2f} R={res['R']['agree']:.2f} | "
                  f"rho E={res['E']['spearman']:.2f} R={res['R']['spearman']:.2f} | "
                  f"curve E={res['E']['curve']:.2f} R={res['R']['curve']:.2f} | "
                  f"lam2^R={est.diag['lam2_pow_R']:.1e} speed={est.diag['max_speed']:.1e} "
                  f"{time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(out, "w"))


def hold_inputs(B, T, gen):
    z0 = (torch.rand(B, generator=gen) * 2 - 1) * 1.5
    u = torch.zeros(B, T, 1)
    u[:, :10, 0] = (z0 / 10)[:, None]
    return u


def run_hold(src, out):
    """Amendment 4, test B: load a value, then zero input (autonomous hold)."""
    task = TASKS["accumulation"]()
    T = 100 * task.T_train
    rows = []
    for r, m, tag in nets_from(src, task):
        est = X.AccumulationExact(m, r["arch"])
        Z, U, D, Uh = C.probe_samples(m, task, r["seed"])
        reg = C.DefectFit(task)
        reg.fit(Z, U, D, r["seed"], Uh=Uh)
        u = hold_inputs(256, T, torch.Generator().manual_seed(9500 + r["seed"]))
        et, zt = C.native(m, u, task)
        mt = torch.ones(256, T)
        mt[:, :10] = 0
        meas = C.failure_times(et, mt, task.eps)
        eE = C.err_norm(est.predict(u), zt)
        with torch.no_grad():
            eR = C.err_norm(C.predict_N(reg, task, u), zt)
        res = dict(E=C.compare(C.failure_times(eE, mt, task.eps), meas, T),
                   R=C.compare(C.failure_times(eR, mt, task.eps), meas, T))
        rows.append(dict(tag=tag, arch=r["arch"], width=r["N"], seed=r["seed"], diag=est.diag, **res))
        print(f"{tag:30s} hold medT={res['E']['median_meas']:6.0f} fail={res['E']['frac_fail_meas']:.2f} "
              f"agree E={res['E']['agree']:.2f} R={res['R']['agree']:.2f} | "
              f"rho E={res['E']['spearman']:.2f} R={res['R']['spearman']:.2f} | "
              f"med pred E={res['E']['median_pred']:.0f}", flush=True)
    json.dump(rows, open(out, "w"))


def run_oscillation(src, out):
    task = TASKS["oscillation"]()
    T = 100 * task.T_train
    rows = []
    for r, m, tag in nets_from(src, task):
        u = task.inputs(1, T, torch.Generator().manual_seed(9000 + r["seed"]))
        et, zt = C.native(m, u, task)
        mt = task.eval_mask(u)
        meas = C.failure_times(et, mt, task.eps)[0]
        yE, d = X.oscillation_exact(m, r["arch"], task, T)
        eE = C.err_norm(yE, zt)
        tE = C.failure_times(eE, mt, task.eps)[0]
        # stage-1 regression prediction for the same network
        tR = json.load(open(os.path.join(HERE, "results", src, tag + ".json")))["metrics"]["N"]["median_pred"]
        rows.append(dict(tag=tag, arch=r["arch"], width=r["N"], seed=r["seed"], T_meas=float(meas),
                         T_E=float(tE), T_R=float(tR), err_end_meas=float(et[0, -1]),
                         err_end_E=float(eE[0, -1]), **d))
        print(f"{tag:30s} T meas={meas:6.0f} E={tE:6.0f} R={tR:6.0f} | end err meas={float(et[0, -1]):.4f} "
              f"E={float(eE[0, -1]):.4f} | slip/step={d['phase_slip_per_step']:.2e}", flush=True)
    json.dump(rows, open(out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["accumulation", "oscillation", "hold"])
    ap.add_argument("--src", default="stage1")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.join(HERE, "results", "stage3"), exist_ok=True)
    out = os.path.join(HERE, "results", "stage3", f"{a.task}_{a.src}.json")
    dict(accumulation=run_accumulation, oscillation=run_oscillation, hold=run_hold)[a.task](a.src, out)
