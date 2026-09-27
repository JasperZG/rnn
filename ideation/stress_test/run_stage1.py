"""Stage 1: does the claim predict hidden failure times across tasks/archs?

    python run_stage1.py --task accumulation --archs rnn gru lstm tc --Ns 32 128 --seeds 0 1 2
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from hf_tasks import TASKS
import hf_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
LAGS = 0


def run_one(task, arch, N, seed, iters, out_dir):
    tag = f"{task.name}_{arch}_N{N}_s{seed}"
    t0 = time.time()
    model = C.build(arch, task, N, seed)
    if arch == "tc":
        loss = C.train(model, task, seed, iters=min(iters, 400), lr=2e-2)
    else:
        loss = C.train(model, task, seed, iters=iters)
    model.eval()
    os.makedirs(os.path.join(HERE, "nets"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(HERE, "nets", tag + ".pt"))

    # gate on the training horizon
    ug = task.inputs(256, task.T_train, torch.Generator().manual_seed(7000 + seed))
    eg, _ = C.native(model, ug, task)
    mg = task.eval_mask(ug)
    p95 = float(np.percentile(eg[mg > 0].numpy(), 95))
    rec = dict(task=task.name, arch=arch, N=N, seed=seed, train_loss=loss,
               gate_p95=p95, gate_pass=p95 < task.eps / 2)
    if arch == "tc":
        rec["g"] = float(model.g)

    # long native measurement on independent trials
    T_test = 100 * task.T_train
    ut = task.inputs(256, T_test, torch.Generator().manual_seed(9000 + seed))
    et, zt = C.native(model, ut, task)
    mt = task.eval_mask(ut)
    meas = C.failure_times(et, mt, task.eps)
    rec["frac_fail_meas"] = float(np.mean(meas <= T_test))

    # defect from short probes only
    Z, U, D, Uh = C.probe_samples(model, task, seed, lags=LAGS)
    fit = C.DefectFit(task, lags=LAGS)
    rec["fit"] = fit.fit(Z, U, D, seed, Uh=Uh)
    rec["lags"] = LAGS

    with torch.no_grad():
        yN = C.predict_N(fit, task, ut)
        eN = C.err_norm(yN, zt)
        eL = C.predict_L(fit, task, ut, zt)
    tN = C.failure_times(eN, mt, task.eps)
    tL = C.failure_times(eL, mt, task.eps)
    Ttr = task.T_train
    b1, b2 = C.baseline_times(et[:, :Ttr], mt[:, :Ttr], task.eps, T_test)

    rec["metrics"] = {name: C.compare(p, meas, T_test)
                      for name, p in (("N", tN), ("L", tL), ("B1", b1), ("B2", b2))}
    # compact curves for figures: median error at log-spaced times
    ts = np.unique(np.geomspace(1, T_test, 40).astype(int)) - 1
    rec["curves"] = dict(t=(ts + 1).tolist(),
                         meas=np.median(et.numpy()[:, ts], 0).tolist(),
                         N=np.median(eN.numpy()[:, ts], 0).tolist(),
                         L=np.median(eL.numpy()[:, ts], 0).tolist())
    rec["per_trial"] = dict(meas=meas.tolist(), N=tN.tolist())
    rec["seconds"] = time.time() - t0
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(rec, f)
    m = rec["metrics"]
    print(f"{tag:32s} gate={'Y' if rec['gate_pass'] else 'n'} p95={p95:.3f} "
          f"fail={rec['frac_fail_meas']:.2f} medT={m['N']['median_meas']:.0f} | "
          + " ".join(f"{k}:{v['agree']:.2f}/{v['spearman']:.2f}" for k, v in m.items())
          + f" | {rec['seconds']:.0f}s", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--archs", nargs="+", default=["rnn", "gru", "lstm", "tc"])
    ap.add_argument("--Ns", nargs="+", type=int, default=[32, 128])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "stage1"))
    ap.add_argument("--lags", type=int, default=0)
    ap.add_argument("--T_train", type=int, default=0)
    a = ap.parse_args()
    LAGS = a.lags
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    task = TASKS[a.task]()
    if a.T_train:
        task.T_train = a.T_train
    for arch in a.archs:
        for N in a.Ns:
            for s in a.seeds:
                run_one(task, arch, N, s, a.iters, a.out)
