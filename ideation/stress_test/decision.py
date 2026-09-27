"""EXPLORATORY decision analysis (not preregistered; designed after seeing the
confirmation outcomes). Question: does the forecast help decide whether a
network will stay within tolerance for a required duration H?

For every gated confirmation accumulation network (seeds 10-14) and scenario
(hold, driven x1, driven x0.5), each trial is a (model, scenario) case:
  truth    = measured first failure > H   (the computation completes)
  approve  = predicted failure > H        (rule: approve if forecast exceeds H)
Predictors: E (final estimator), R (stage-1 regression), B1/B2 (extrapolation
of the first T_train steps), and 'validation' (approve everything that passed
the training-horizon gate). Also reports AUC of each predicted time for the
binary outcome, and wall-clock cost of extraction vs brute-force rollout.
"""
import glob
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from hf_tasks import TASKS
import hf_core as C
import hf_exact_FROZEN as X
from run_stage2 import shifted_inputs
from run_stage3 import hold_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
task = TASKS["accumulation"]()
T = 5000
Hs = (250, 500, 1000, 2000)
cases = {k: {n: [] for n in ("meas", "E", "R", "B1", "B2", "net")} for k in ("hold", "x1", "x0.5")}
cost = []
for p in sorted(glob.glob(os.path.join(HERE, "results", "stage1_confirm", "accumulation_*.json"))):
    r = json.load(open(p))
    if not r["gate_pass"]:
        continue
    tag = f"accumulation_{r['arch']}_N{r['N']}_s{r['seed']}"
    m = C.build(r["arch"], task, r["N"], r["seed"])
    m.load_state_dict(torch.load(os.path.join(HERE, "nets", tag + ".pt")))
    m.eval()
    t0 = time.time()
    est = X.AccumulationExact(m, r["arch"])
    t_extract = time.time() - t0
    Z, U, D, Uh = C.probe_samples(m, task, r["seed"])
    reg = C.DefectFit(task)
    reg.fit(Z, U, D, r["seed"], Uh=Uh)
    for name in ("hold", "x1", "x0.5"):
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
        t0 = time.time()
        et, zt = C.native(m, u, task)
        t_brute = time.time() - t0
        meas = C.failure_times(et, mt, task.eps)
        t0 = time.time()
        pE = C.failure_times(C.err_norm(est.predict(u), zt), mt, task.eps)
        t_fore = time.time() - t0
        with torch.no_grad():
            pR = C.failure_times(C.err_norm(C.predict_N(reg, task, u), zt), mt, task.eps)
        b1, b2 = C.baseline_times(et[:, :task.T_train], mt[:, :task.T_train], task.eps, T)
        for k, v in (("meas", meas), ("E", pE), ("R", pR), ("B1", b1), ("B2", b2),
                     ("net", np.full(256, len(cost)))):
            cases[name][k].append(v)
        cost.append(dict(tag=tag, scenario=name, extract_s=t_extract, forecast_s=t_fore, brute_s=t_brute))
    print(f"{tag:28s} extract={t_extract:.1f}s  forecast(256x5000)={cost[-1]['forecast_s']:.1f}s  "
          f"brute(256x5000)={cost[-1]['brute_s']:.2f}s", flush=True)

out = {}
for name, d in cases.items():
    d = {k: np.concatenate(v) for k, v in d.items()}
    print(f"\n=== {name}: {len(d['meas'])} cases from {len(np.unique(d['net']))} networks ===")
    print(f"{'H':>5} {'frac ok':>8} | " + " | ".join(f"{k:^30s}" for k in ("E", "R", "B1", "B2", "validation")))
    print(f"{'':>5} {'':>8} | " + " | ".join(f"{'AUC  wrongApp  usefulApp':^30s}" for _ in range(5)))
    for H in Hs:
        ok = d["meas"] > H
        cells = []
        for k in ("E", "R", "B1", "B2", "validation"):
            pred = np.full(len(ok), T + 1.0) if k == "validation" else d[k]
            app = pred > H
            wrong = float(np.mean(~ok[app])) if app.any() else float("nan")   # failing among approved
            useful = float(np.mean(app[ok])) if ok.any() else float("nan")    # approved among truly ok
            auc = float(roc_auc_score(ok, pred)) if 0 < ok.mean() < 1 and k != "validation" else float("nan")
            cells.append(f"{auc:5.3f}  {wrong:6.3f}   {useful:6.3f}")
            out.setdefault(name, {}).setdefault(str(H), {})[k] = dict(auc=auc, wrong_approval=wrong,
                                                                      useful_approval=useful)
        print(f"{H:>5} {ok.mean():>8.2f} | " + " | ".join(f"{c:^30s}" for c in cells))
c = {k: np.median([x[k] for x in cost]) for k in ("extract_s", "forecast_s", "brute_s")}
print(f"\nmedian cost per network: extraction {c['extract_s']:.1f}s, reduced forecast (256 trials x 5000) "
      f"{c['forecast_s']:.1f}s, brute-force full rollout (256 x 5000) {c['brute_s']:.2f}s")
json.dump(dict(decision=out, cost=cost), open(os.path.join(HERE, "results", "decision_exploratory.json"), "w"))
