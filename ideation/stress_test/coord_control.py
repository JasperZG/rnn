"""Coordinate-transform control (Amendment 8).

A trained vanilla RNN h' = tanh(W h + b + Win u), y = Wout h is rewritten in
non-orthogonal coordinates q = S h:
    q' = S tanh(W S^-1 q + b + Win u),   y = Wout S^-1 q.
The computation (every output, every failure time) is unchanged. The naive
slow-point estimate minimizes Euclidean speed, which depends on the coordinates;
the invariance equation does not. Prediction: E's forecasts are unchanged by S,
NS's forecasts move.

    python coord_control.py
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

from hf_tasks import TASKS
import hf_core as C
import hf_exact_FROZEN as X
from run_stage3 import hold_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)


class TNet(nn.Module):
    """Vanilla RNN in coordinates q = S h (exposes .step and .Wout like VanillaRNN)."""

    def __init__(self, base, S):
        super().__init__()
        self.base = base
        self.register_buffer("S", S)
        self.register_buffer("Si", torch.linalg.inv(S))
        self.Wout = nn.Linear(S.shape[0], base.Wout.weight.shape[0], bias=False)
        with torch.no_grad():
            self.Wout.weight.copy_(base.Wout.weight @ self.Si)

    def step(self, q, u):
        return self.base.step(q @ self.Si.T, u) @ self.S.T

    def forward(self, x):
        B = x.shape[0]
        q = torch.zeros(B, self.S.shape[0], dtype=x.dtype)
        Q = []
        for t in range(x.shape[1]):
            q = self.step(q, x[:, t])
            Q.append(q)
        Q = torch.stack(Q, 1)
        return self.Wout(Q), Q


class TWrap(nn.Module):
    def __init__(self, base_wrapped, S):
        super().__init__()
        self.net = TNet(base_wrapped.net, S)
        self.N = base_wrapped.N

    def forward(self, x):
        return self.net(x)


def naive(m):
    orig = X.invariant_manifold
    X.invariant_manifold = lambda f, dec, S: (S, float("nan"), float("nan"))
    try:
        return X.AccumulationExact(m, "rnn")
    finally:
        X.invariant_manifold = orig


task = TASKS["accumulation"]()
T = 5000
rows = []
paths = sorted(glob.glob(os.path.join(HERE, "results", "stage1_confirm", "accumulation_rnn_N*.json")))
for p in paths:
    r = json.load(open(p))
    if not r["gate_pass"]:
        continue
    tag = f"accumulation_rnn_N{r['N']}_s{r['seed']}"
    base = C.build("rnn", task, r["N"], r["seed"])
    base.load_state_dict(torch.load(os.path.join(HERE, "nets", tag + ".pt")))
    base.eval()
    u = hold_inputs(256, T, torch.Generator().manual_seed(9500 + r["seed"]))
    mt = torch.ones(256, T)
    mt[:, :10] = 0
    et, zt = C.native(base, u, task)
    meas = C.failure_times(et, mt, task.eps)
    f = meas <= T
    g = torch.Generator().manual_seed(123)
    G = torch.randn(r["N"], r["N"], generator=g) / r["N"] ** 0.5
    for alpha in (0.0, 0.5, 1.0, 2.0):
        S = torch.eye(r["N"]) + alpha * G
        m = base if alpha == 0 else TWrap(base, S)
        if alpha:
            with torch.no_grad():
                y0, _ = base(u[:8, :300])
                y1, _ = m(u[:8, :300])
            same = float((y0 - y1).abs().max())
        else:
            same = 0.0
        out = dict(tag=tag, alpha=alpha, cond_S=float(torch.linalg.cond(S)), output_mismatch=same)
        for name, est in (("E", X.AccumulationExact(m, "rnn")), ("NS", naive(m))):
            pt = C.failure_times(C.err_norm(est.predict(u), zt), mt, task.eps)
            ok = f & (pt <= T)
            out[f"{name}_median_abs_log_err"] = float(np.median(np.abs(np.log(pt[ok] / meas[ok])))) if ok.any() else float("nan")
            out[f"{name}_median_T"] = float(np.median(np.minimum(pt, T + 1)))
        out["meas_median_T"] = float(np.median(np.minimum(meas, T + 1)))
        rows.append(out)
        print(f"{tag:26s} alpha={alpha:3.1f} cond(S)={out['cond_S']:7.1f} out-mismatch={same:.1e} | "
              f"E |log err|={out['E_median_abs_log_err']:.4f} medT={out['E_median_T']:.0f} | "
              f"NS |log err|={out['NS_median_abs_log_err']:.4f} medT={out['NS_median_T']:.0f} | "
              f"measured medT={out['meas_median_T']:.0f}", flush=True)
json.dump(rows, open(os.path.join(HERE, "results", "coord_control.json"), "w"), indent=1)
