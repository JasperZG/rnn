"""DIAGNOSTIC (Amendment 9/10 follow-up; networks already inspected, not confirmatory).
Why do width-512 hold-trained LSTMs forecast poorly? Separates:
  (2) numerical reduction: is the drift flow smooth along the converged manifold?
  (3) learned structure: how many near-unit Jacobian modes along the manifold, and
      does the future depend on history beyond the decoded state?

    python diag_hold_lstm.py hold:512:1002 hold:512:1011 hold:512:1007 accumulation:512:1000 hold:128:530
"""
import json
import sys

import numpy as np
import torch

from hf_tasks import TASKS
import hf_core as C
import hf_exact_v2 as X

torch.set_num_threads(8)


def manifold_and_est(m):
    cap = {}
    orig = X.converged_block

    def cb(f, dec, H, tol):
        idx, nd, cmap = orig(f, dec, H, tol)
        cap["H"] = H[idx]
        return idx, nd, cmap
    X.converged_block = cb
    try:
        est = X.AccumulationExact(m, "lstm")
    finally:
        X.converged_block = orig
    return est, cap["H"]


def analyze(task_name, N, seed):
    task = TASKS[task_name]()
    path = f"nets/diag/{task_name}_lstm_N{N}_s{seed}.pt"
    m = C.build("lstm", task, N, seed)
    m.load_state_dict(torch.load(path))
    m.eval()
    est, H = manifold_and_est(m)
    f, dec, dim = X.stepper(m, "lstm")
    out = dict(net=f"{task_name}_lstm_N{N}_s{seed}", n_manifold=len(H),
               flow_monotone=est.diag["flow_monotone"], dropped=est.diag["n_dropped_nonconverged"])
    with torch.no_grad():
        s = dec(H)[:, 0].numpy()
        v = dec(f(H, torch.zeros(len(H), 1, dtype=torch.float64)))[:, 0].numpy() - s
    dv = np.diff(v) / np.diff(s)
    out["frac_dvds_below_-1"] = float(np.mean(dv < -1))
    out["frac_dvds_below_-0.1"] = float(np.mean(dv < -0.1))
    out["max_abs_v"] = float(np.abs(v).max())
    # (3a) Jacobian spectrum along the manifold
    idx = np.linspace(0, len(H) - 1, 15).astype(int)
    z0 = torch.zeros(1, 1, dtype=torch.float64)
    n95, n99, top = [], [], []
    for i in idx:
        J = torch.autograd.functional.jacobian(lambda h: f(h[None], z0)[0], H[i])
        ev = np.sort(np.abs(np.linalg.eigvals(J.numpy())))[::-1]
        n95.append(int((ev > 0.95).sum()))
        n99.append(int((ev > 0.99).sum()))
        top.append([round(float(x), 4) for x in ev[:4]])
    out["modes_gt_0.95_median"] = float(np.median(n95))
    out["modes_gt_0.99_median"] = float(np.median(n99))
    out["modes_gt_0.99_by_position"] = n99
    out["top4_eigs_mid"] = top[len(top) // 2]
    m.float()
    # (3b) history dependence during a hold (zero input after loading)
    rng = np.random.default_rng(0)
    B, L, Th = 128, 20, 1000
    z0v = rng.uniform(-1.2, 1.2, B)
    delta = rng.choice([-0.6, 0.6], B)
    uA = torch.zeros(B, L + Th, 1)
    uB = torch.zeros(B, L + Th, 1)
    uA[:, L - 10:L, 0] = torch.as_tensor(z0v / 10, dtype=torch.float32)[:, None]     # direct load
    uB[:, 0:10, 0] = torch.as_tensor((z0v + delta) / 10, dtype=torch.float32)[:, None]  # overshoot ...
    uB[:, 10:20, 0] = torch.as_tensor(-delta / 10, dtype=torch.float32)[:, None]        # ... and return
    with torch.no_grad():
        yA = m(uA)[0][..., 0].numpy()
        yB = m(uB)[0][..., 0].numpy()
    pA = est.predict(uA)[..., 0].numpy()
    pB = est.predict(uB)[..., 0].numpy()
    d_net = yA - yB
    d_red = pA - pB
    for t in (L, L + 100, L + 500, L + Th - 1):
        out[f"hist_net_diff_t{t - L}"] = float(np.median(np.abs(d_net[:, t])))
        out[f"hist_closure_err_t{t - L}"] = float(np.median(np.abs(d_net[:, t] - d_red[:, t])))
    # drift magnitude reference over the same window
    out["hold_drift_net_median_1000"] = float(np.median(np.abs(yA[:, L + Th - 1] - yA[:, L])))
    return out


if __name__ == "__main__":
    rows = []
    for spec in sys.argv[1:]:
        t, N, s = spec.split(":")
        r = analyze(t, int(N), int(s))
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump(rows, open("results/diag_hold_lstm.json", "w"), indent=1)
