"""Stage 2 stress tests on stage-1 gated networks.

    python run_stage2.py shift    --task accumulation
    python run_stage2.py noise    --task flipflop --sigmas 0.02 0.05 0.1 0.2
    python run_stage2.py slowmode --task accumulation
    python run_stage2.py ablate   --task accumulation
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
from scipy.stats import ks_2samp

from hf_tasks import TASKS
import hf_core as C

HERE = os.path.dirname(os.path.abspath(__file__))


def load_gated(task, seeds=None):
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "stage1", f"{task.name}_*.json"))):
        r = json.load(open(p))
        if not r["gate_pass"] or (seeds is not None and r["seed"] not in seeds):
            continue
        m = C.build(r["arch"], task, r["N"], r["seed"])
        tag = f"{task.name}_{r['arch']}_N{r['N']}_s{r['seed']}"
        m.load_state_dict(torch.load(os.path.join(HERE, "nets", tag + ".pt")))
        m.eval()
        out.append((r, m, tag))
    return out


def shifted_inputs(task, B, T, gen, factor):
    if task.name in ("accumulation", "ctxint"):
        old = task.scale
        task.scale = old * factor
        u = task.inputs(B, T, gen)
        task.scale = old
        return u
    if task.name == "flipflop":
        p = 0.06 * factor
        hit = (torch.rand(B, T, 1, generator=gen) < p).float()
        sgn = torch.where(torch.rand(B, T, 1, generator=gen) < 0.5, -1.0, 1.0)
        return hit * sgn
    if task.name == "oscillation":
        u = torch.zeros(B, T, 1)
        u[:, 0, 0] = factor
        return u
    raise ValueError(task.name)


def fit_defect(m, task, seed, degrees=None, sigma=0.0, arch=None):
    if sigma > 0:
        Z, U, D, Uh = noisy_probe_samples(m, task, seed, sigma, arch)
    else:
        Z, U, D, Uh = C.probe_samples(m, task, seed)
    f = C.DefectFit(task)
    if degrees is not None:
        f.degrees = degrees
    diag = f.fit(Z, U, D, seed, Uh=Uh)
    return f, diag


# ---------------------------------------------------------------------------
# noisy dynamics
# ---------------------------------------------------------------------------
def step_fn(m, arch):
    if arch == "tc":
        return lambda h, u: torch.tanh((m.g * m.task.F(h @ m.V, u)) @ m.V.T)
    if arch == "lstm":
        return m.net.step
    return m.net.step


def zero_state(m, arch, B):
    n = 2 * m.N if arch == "lstm" else m.N
    return torch.zeros(B, n)


def noisy_states(m, arch, u, sigma, gen, S0=None):
    """States after each step with additive Gaussian state noise sigma."""
    B, T, _ = u.shape
    f = step_fn(m, arch)
    S = zero_state(m, arch, B) if S0 is None else S0
    out = []
    with torch.no_grad():
        for t in range(T):
            S = f(S, u[:, t])
            if sigma > 0:
                S = S + sigma * torch.randn(S.shape, generator=gen)
            out.append(S)
    return torch.stack(out, 1)


def noisy_probe_samples(m, task, seed, sigma, arch, B=512):
    gen = torch.Generator().manual_seed(5000 + seed)
    u = torch.cat([task.inputs(B, task.T_train, gen, broad=True),
                   task.inputs(B // 2, task.T_train, gen)], 0)
    S = noisy_states(m, arch, u, sigma, torch.Generator().manual_seed(6000 + seed))
    with torch.no_grad():
        Zn = m.decode(S)
    Z0 = torch.cat([torch.zeros_like(Zn[:, :1]), Zn[:, :-1]], 1)
    D = Zn - task.F(Z0.reshape(-1, task.k), u.reshape(-1, task.n_in)).reshape(Zn.shape)
    Uh = np.zeros((Z0.shape[0] * Z0.shape[1], 1, task.n_in), dtype=np.float32)
    return (Z0.reshape(-1, task.k).numpy(), u.reshape(-1, task.n_in).numpy(),
            D.reshape(-1, task.k).numpy(), Uh)


def diffusion(m, task, arch, seed, sigma, k=10, B=512):
    """Effective per-step diffusion of the decoded state: paired k-step runs from
    the same start states with independent noise (captures non-normal
    amplification of off-manifold noise). Also returns the naive one-step value."""
    gen = torch.Generator().manual_seed(8000 + seed)
    u0 = task.inputs(B, task.T_train, gen)
    S = noisy_states(m, arch, u0, 0.0, gen)
    tt = torch.randint(5, task.T_train - k, (B,), generator=gen)
    S0 = S[torch.arange(B), tt]
    uk = task.inputs(B, k, gen)
    a = noisy_states(m, arch, uk, sigma, torch.Generator().manual_seed(1), S0)[:, -1]
    b = noisy_states(m, arch, uk, sigma, torch.Generator().manual_seed(2), S0)[:, -1]
    with torch.no_grad():
        dz = m.decode(a) - m.decode(b)
        W = m.V.T if arch == "tc" else m.net.Wout.weight
    D_eff = (dz ** 2).mean(0).numpy() / (2 * k)
    D_naive = (sigma ** 2) * (W ** 2).sum(1).detach().numpy()
    return D_eff, D_naive


def predict_N_noisy(delta, task, u, D, gen):
    B, T, _ = u.shape
    zh = torch.zeros(B, task.k)
    sd = torch.as_tensor(np.sqrt(D), dtype=torch.float32)
    uh = torch.zeros(B, 1, task.n_in)
    Y = []
    for t in range(T):
        zh = task.F(zh, u[:, t]) + delta(zh, u[:, t], uh) + sd * torch.randn(B, task.k, generator=gen)
        Y.append(zh)
    return torch.stack(Y, 1)


def dist_compare(pred, meas, T):
    p, mm = np.minimum(pred, T + 1), np.minimum(meas, T + 1)
    return dict(ks=float(ks_2samp(p, mm).statistic),
                abs_log_median=abs(math.log(np.median(p) / np.median(mm))),
                frac_fail_pred=float(np.mean(p <= T)), frac_fail_meas=float(np.mean(mm <= T)),
                median_pred=float(np.median(p)), median_meas=float(np.median(mm)))


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def exp_shift(task, nets, factors):
    rows = []
    for r, m, tag in nets:
        f, _ = fit_defect(m, task, r["seed"])
        T = 100 * task.T_train
        for fac in factors:
            u = shifted_inputs(task, 256, T, torch.Generator().manual_seed(9100 + r["seed"]), fac)
            et, zt = C.native(m, u, task)
            mt = task.eval_mask(u)
            meas = C.failure_times(et, mt, task.eps)
            with torch.no_grad():
                tN = C.failure_times(C.err_norm(C.predict_N(f, task, u), zt), mt, task.eps)
            b1, b2 = C.baseline_times(et[:, :task.T_train], mt[:, :task.T_train], task.eps, T)
            res = {k: C.compare(p, meas, T) for k, p in (("N", tN), ("B1", b1), ("B2", b2))}
            rows.append(dict(tag=tag, arch=r["arch"], width=r["N"], seed=r["seed"], factor=fac, **{
                k: v for k, v in res.items()}))
            print(f"{tag:30s} x{fac:<4} fail={res['N']['frac_fail_meas']:.2f}/{res['N']['frac_fail_pred']:.2f} "
                  f"agree N={res['N']['agree']:.2f} B1={res['B1']['agree']:.2f} B2={res['B2']['agree']:.2f} "
                  f"rho={res['N']['spearman']:.2f}", flush=True)
    return rows


def exp_noise(task, nets, sigmas):
    rows = []
    for r, m, tag in nets:
        arch = r["arch"]
        T = 100 * task.T_train
        u = task.inputs(256, T, torch.Generator().manual_seed(9200 + r["seed"]))
        z = task.targets(u)
        mt = task.eval_mask(u)
        for s in sigmas:
            S = noisy_states(m, arch, u, s, torch.Generator().manual_seed(9300 + r["seed"]))
            with torch.no_grad():
                meas = C.failure_times(C.err_norm(m.decode(S), z), mt, task.eps)
            f, diag = fit_defect(m, task, r["seed"])  # noise-free mean defect (amendment 2)
            D_eff, D_naive = diffusion(m, task, arch, r["seed"], s)
            out = {}
            for name, D in (("adjoint", D_eff), ("naive", D_naive), ("none", 0 * D_eff)):
                with torch.no_grad():
                    y = predict_N_noisy(f, task, u, D, torch.Generator().manual_seed(9400 + r["seed"]))
                    tp = C.failure_times(C.err_norm(y, z), mt, task.eps)
                out[name] = dist_compare(tp, meas, T)
            rows.append(dict(tag=tag, arch=arch, N=r["N"], seed=r["seed"], sigma=s,
                             D_eff=D_eff.tolist(), D_naive=D_naive.tolist(), **out))
            a = out["adjoint"]
            print(f"{tag:30s} s={s:<5} fail={a['frac_fail_meas']:.2f}/{a['frac_fail_pred']:.2f} "
                  f"med={a['median_meas']:.0f}/{a['median_pred']:.0f} KS adj={a['ks']:.2f} "
                  f"naive={out['naive']['ks']:.2f} none={out['none']['ks']:.2f} "
                  f"Deff/Dnaive={float(np.mean(D_eff) / max(np.mean(D_naive), 1e-12)):.2f}", flush=True)
    return rows


def exp_slowmode(task, nets, lams):
    """Plant an off-manifold slow mode (rnn only): W += lam v v^T, v orthogonal to Wout."""
    rows = []
    for r, m, tag in nets:
        if r["arch"] != "rnn":
            continue
        W0 = m.net.W.weight.detach().clone()
        Wout = m.net.Wout.weight.detach()
        g = torch.Generator().manual_seed(77 + r["seed"])
        v = torch.randn(m.N, generator=g)
        Q, _ = torch.linalg.qr(Wout.T)
        v = v - Q @ (Q.T @ v)
        v = v / v.norm()
        T = 100 * task.T_train
        for lam in lams:
            with torch.no_grad():
                m.net.W.weight.copy_(W0 + lam * torch.outer(v, v))
            ug = task.inputs(256, task.T_train, torch.Generator().manual_seed(7000 + r["seed"]))
            eg, _ = C.native(m, ug, task)
            p95 = float(np.percentile(eg[task.eval_mask(ug) > 0].numpy(), 95))
            u = task.inputs(256, T, torch.Generator().manual_seed(9000 + r["seed"]))
            et, zt = C.native(m, u, task)
            mt = task.eval_mask(u)
            meas = C.failure_times(et, mt, task.eps)
            f, diag = fit_defect(m, task, r["seed"])
            with torch.no_grad():
                tN = C.failure_times(C.err_norm(C.predict_N(f, task, u), zt), mt, task.eps)
            res = C.compare(tN, meas, T)
            r2 = float(np.mean([d["r2"] for d in diag.values()]))
            rows.append(dict(tag=tag, N=r["N"], seed=r["seed"], lam=lam, gate_p95=p95, r2=r2, **res))
            print(f"{tag:30s} lam={lam:<5} gate_p95={p95:.3f} r2={r2:.3f} agree={res['agree']:.2f} "
                  f"rho={res['spearman']:.2f}", flush=True)
        with torch.no_grad():
            m.net.W.weight.copy_(W0)
    return rows


def exp_ablate(task, nets):
    """What part of the defect carries the prediction?"""
    rows = []
    for r, m, tag in nets:
        T = 100 * task.T_train
        u = task.inputs(256, T, torch.Generator().manual_seed(9000 + r["seed"]))
        et, zt = C.native(m, u, task)
        mt = task.eval_mask(u)
        meas = C.failure_times(et, mt, task.eps)
        res = {}
        for name, degs in (("full", None), ("linear", (1,)), ("constant", (0,))):
            f, _ = fit_defect(m, task, r["seed"], degrees=degs)
            with torch.no_grad():
                tp = C.failure_times(C.err_norm(C.predict_N(f, task, u), zt), mt, task.eps)
            res[name] = C.compare(tp, meas, T)
        with torch.no_grad():
            tz = C.failure_times(C.err_norm(task.targets(u), zt), mt, task.eps)
        res["zero"] = C.compare(tz, meas, T)
        rows.append(dict(tag=tag, arch=r["arch"], width=r["N"], seed=r["seed"], **res))
        print(f"{tag:30s} " + " ".join(f"{k}={v['agree']:.2f}" for k, v in res.items()), flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["shift", "noise", "slowmode", "ablate"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--factors", nargs="+", type=float, default=None)
    ap.add_argument("--lams", nargs="+", type=float, default=[0.0, 0.3, 0.6, 0.9, 0.97])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    task = TASKS[a.task]()
    nets = load_gated(task, a.seeds)
    default_f = dict(accumulation=[0.5, 2, 4], ctxint=[0.5, 2, 4], flipflop=[3, 6],
                     oscillation=[0.5, 1.5, 2])
    if a.exp == "shift":
        rows = exp_shift(task, nets, a.factors or default_f[task.name])
    elif a.exp == "noise":
        rows = exp_noise(task, nets, a.sigmas)
    elif a.exp == "slowmode":
        rows = exp_slowmode(task, nets, a.lams)
    else:
        rows = exp_ablate(task, nets)
    os.makedirs(os.path.join(HERE, "results", "stage2"), exist_ok=True)
    with open(os.path.join(HERE, "results", "stage2", f"{a.exp}_{task.name}.json"), "w") as fh:
        json.dump(rows, fh)
