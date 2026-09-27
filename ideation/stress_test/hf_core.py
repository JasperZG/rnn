"""Core machinery for stress-testing the hidden-failure claim.

    defect  delta = F~ - F, estimated from one-step maps of the trained weights
            at states reached by probe runs no longer than T_train
    form N  z^_{t+1} = F(z^_t, u_t) + delta(z^_t, u_t)
    form L  e_{t+1}  = J_F(z_t, u_t) e_t + delta(z_t, u_t)
"""
import itertools
import math
import os
import sys

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*encountered in matmul")
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from rnnphase.models import VanillaRNN, GRUNet, LSTMNet  # noqa: E402


# ---------------------------------------------------------------------------
# models: uniform interface  forward(x) -> (y, states)  with states[:, t] the
# full dynamical state AFTER consuming u_t, and decode(state) -> z
# ---------------------------------------------------------------------------
class TaskConjugate(nn.Module):
    """h' = tanh(V g F(V^T h, u)); only g is trained. Exact anchor."""

    def __init__(self, task, N, seed):
        super().__init__()
        self.task, self.N = task, N
        gen = torch.Generator().manual_seed(seed)
        if task.k == 1:
            s = torch.where(torch.rand(N, generator=gen) < 0.5, -1.0, 1.0)
            V = (s / math.sqrt(N))[:, None]
        else:  # harmonic frame, M = 4 directions (a 7-design with antipodes)
            M = 4
            ang = math.pi * (torch.arange(N) % M).float() / M
            V = torch.stack([torch.cos(ang), torch.sin(ang)], 1) * math.sqrt(2.0 / N)
        self.register_buffer("V", V)
        self.g = nn.Parameter(torch.ones(()))

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.N, device=x.device, dtype=x.dtype)
        H = []
        for t in range(T):
            q = self.g * self.task.F(h @ self.V, x[:, t])
            h = torch.tanh(q @ self.V.T)
            H.append(h)
        H = torch.stack(H, 1)
        return H @ self.V, H

    def decode(self, S):
        return S @ self.V


class Wrapped(nn.Module):
    """Adapter so rnn / gru / lstm expose states and decode uniformly."""

    def __init__(self, arch, task, N, seed):
        super().__init__()
        self.arch, self.N = arch, N
        if arch == "rnn":
            rot = 0.5 if task.name == "oscillation" else 0.0
            self.net = VanillaRNN(task.n_in, task.k, N, rot, seed)
        elif arch == "gru":
            self.net = GRUNet(task.n_in, task.k, N, seed)
        elif arch == "lstm":
            self.net = LSTMNet(task.n_in, task.k, N, seed)
        else:
            raise ValueError(arch)

    def forward(self, x):
        y, H = self.net(x)
        return y, H

    def states(self, x):
        with torch.no_grad():
            if self.arch == "lstm":
                return self.net.joint_states(x)
            return self.net(x)[1]

    def decode(self, S):
        if self.arch == "lstm":
            S = S[..., : self.N]
        return self.net.Wout(S)


def build(arch, task, N, seed):
    torch.manual_seed(seed)
    if arch == "tc":
        return TaskConjugate(task, N, seed)
    return Wrapped(arch, task, N, seed)


def train(model, task, seed, iters, B=64, lr=2e-3, device="cpu"):
    """Train on `device` (inputs are always generated on CPU with a seeded
    generator, so results do not depend on the device); returns the model on CPU."""
    gen = torch.Generator().manual_seed(1000 + seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(iters):
        u = task.inputs(B, task.T_train, gen)
        z = task.targets(u).to(device)
        y, _ = model(u.to(device))
        loss = ((y - z) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.to("cpu")
    return loss.item()


# ---------------------------------------------------------------------------
# native measurement
# ---------------------------------------------------------------------------
def err_norm(y, z):
    return (y - z).norm(dim=-1)


def failure_times(err, mask, eps):
    """First masked step (1-indexed) with err > eps; censored -> T+1."""
    B, T = err.shape
    bad = (err > eps) & (mask > 0)
    t = torch.where(bad.any(1), bad.float().argmax(1) + 1, torch.full((B,), T + 1))
    return t.numpy().astype(float)


def native(model, u, task, device="cpu", chunk=128):
    """Full-network rollout (optionally on GPU), in chunks of trials so the
    hidden-state history never exceeds chunk x T x N; outputs returned on CPU."""
    ys = []
    with torch.no_grad():
        model.to(device)
        for i in range(0, u.shape[0], chunk):
            y, _ = model(u[i:i + chunk].to(device))
            ys.append(y.cpu())
            del y
        model.to("cpu")
    z = task.targets(u)
    return err_norm(torch.cat(ys), z), z


# ---------------------------------------------------------------------------
# defect estimation from short probes (<= T_train)
# ---------------------------------------------------------------------------
def monomials(nz, nc, deg, cdeg=2):
    out = []
    for d in range(deg + 1):
        for combo in itertools.combinations_with_replacement(range(nz + nc), d):
            if sum(1 for v in combo if v >= nz) <= cdeg:
                out.append(combo)
    return out


def design(zc, mons):
    cols = [np.prod(zc[:, list(m)], axis=1) if m else np.ones(len(zc)) for m in mons]
    return np.stack(cols, 1)


class DefectFit:
    """delta(z, u) as a per-discrete-key polynomial ridge fit; degree chosen
    on held-out probe samples. Inputs are clipped to the probed range."""

    def __init__(self, task, lags=0, degrees=(1, 3, 5, 7)):
        self.task, self.lags, self.degrees = task, lags, degrees

    def _cont(self, U, Uh):
        """Continuous features: current continuous input + lagged inputs."""
        key, C = self.task.split_u(torch.as_tensor(U))
        cols = [C.numpy()]
        for j in range(self.lags):
            uj = torch.as_tensor(Uh[:, j])
            cj = self.task.split_u(uj)[1]
            cols.append((cj if cj.shape[1] else uj).numpy())
        return key.numpy(), np.concatenate(cols, 1)

    def fit(self, Z, U, D, seed=0, Uh=None):
        task = self.task
        key, C = self._cont(U, Uh)
        X = np.concatenate([Z, C], 1)
        self.lo, self.hi = X.min(0), X.max(0)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        self.nz, self.nc = Z.shape[1], C.shape[1]
        rng = np.random.default_rng(seed)
        self.models, diag = {}, {}
        for kv in np.unique(key):
            idx = np.where(key == kv)[0]
            rng.shuffle(idx)
            ntr = int(0.8 * len(idx))
            tr, te = idx[:ntr], idx[ntr:]
            Xs = (X - self.mu) / self.sd
            best = None
            for deg in self.degrees:
                mons = monomials(self.nz, self.nc, deg)
                if len(mons) > 0.2 * max(ntr, 1):
                    continue
                A = design(Xs[tr], mons)
                lam = 1e-6 * np.trace(A.T @ A) / A.shape[1]
                W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ D[tr])
                res = D[te] - design(Xs[te], mons) @ W
                mse = float((res ** 2).mean()) if len(te) else 0.0
                if best is None or mse < best[0]:
                    best = (mse, deg, mons, W)
            mse, deg, mons, W = best
            A = design(Xs[idx], mons)
            lam = 1e-6 * np.trace(A.T @ A) / A.shape[1]
            W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ D[idx])
            self.models[int(kv)] = (mons, W)
            var = float(((D[idx] - D[idx].mean(0)) ** 2).mean())
            diag[int(kv)] = dict(n=int(len(idx)), deg=deg, resid_rms=math.sqrt(mse),
                                 delta_rms=math.sqrt(float((D[idx] ** 2).mean())),
                                 r2=1 - mse / var if var > 0 else float("nan"))
        return diag

    def __call__(self, z, u, uh=None):
        key, C = self._cont(u, uh)
        X = np.concatenate([z.numpy(), C], 1)
        X = np.clip(X, self.lo, self.hi)
        Xs = (X - self.mu) / self.sd
        out = np.zeros((len(X), self.nz))
        for kv, (mons, W) in self.models.items():
            m = key == kv
            if m.any():
                out[m] = design(Xs[m], mons) @ W
        return torch.as_tensor(out, dtype=torch.float32)


def history(u, lags):
    """(B, T, lags, n_in): u_{t-1}, ..., u_{t-lags} (zeros before t=0)."""
    B, T, n = u.shape
    out = torch.zeros(B, T, max(lags, 1), n)
    for j in range(1, lags + 1):
        out[:, j:, j - 1] = u[:, :-j]
    return out


def probe_samples(model, task, seed, B=512, lags=0):
    """One-step (state, input, next state) triples from runs of length T_train."""
    gen = torch.Generator().manual_seed(5000 + seed)
    u = torch.cat([task.inputs(B, task.T_train, gen, broad=True),
                   task.inputs(B // 2, task.T_train, gen, broad=False)], 0)
    with torch.no_grad():
        S = model.states(u) if hasattr(model, "states") else model(u)[1]
        Zn = model.decode(S)                                 # z^ after step t
    Z0 = torch.cat([torch.zeros_like(Zn[:, :1]), Zn[:, :-1]], 1)  # z^ before step t
    D = Zn - task.F(Z0.reshape(-1, task.k), u.reshape(-1, task.n_in)).reshape(Zn.shape)
    Uh = history(u, lags).reshape(-1, max(lags, 1), task.n_in).numpy()
    return (Z0.reshape(-1, task.k).numpy(), u.reshape(-1, task.n_in).numpy(),
            D.reshape(-1, task.k).numpy(), Uh)


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------
def predict_N(delta, task, u):
    B, T, _ = u.shape
    uh = history(u, getattr(delta, "lags", 0))
    zh = torch.zeros(B, task.k)
    Y = []
    for t in range(T):
        zh = task.F(zh, u[:, t]) + delta(zh, u[:, t], uh[:, t])
        Y.append(zh)
    return torch.stack(Y, 1)


def predict_L(delta, task, u, z):
    B, T, _ = u.shape
    uh = history(u, getattr(delta, "lags", 0))
    e = torch.zeros(B, task.k)
    zt = torch.zeros(B, task.k)
    E = []
    for t in range(T):
        J = task.JF(zt, u[:, t])
        e = torch.einsum("bij,bj->bi", J, e) + delta(zt, u[:, t], uh[:, t])
        zt = z[:, t]
        E.append(e)
    return torch.stack(E, 1).norm(dim=-1)


def baseline_times(err_short, mask_short, eps, T_test):
    """B1 power-law and B2 linear extrapolation from t <= T_train errors."""
    e = err_short.numpy()
    m = mask_short.numpy() > 0
    B, T = e.shape
    t = np.arange(1, T + 1)
    b1 = np.full(B, T_test + 1.0)
    b2 = np.full(B, T_test + 1.0)
    for i in range(B):
        ok = m[i] & (e[i] > 1e-7) & (t >= 5)
        if ok.sum() >= 5:
            b, a = np.polyfit(np.log(t[ok]), np.log(e[i, ok]), 1)
            if b > 0:
                lt = (math.log(eps) - a) / b
                b1[i] = T_test + 1.0 if lt > math.log(T_test + 1) else math.exp(lt)
        last = e[i][m[i]][-5:]
        if len(last):
            r = last.mean() / T
            if r > 0:
                b2[i] = min(T_test + 1.0, eps / r)
    return b1, b2


def compare(pred, meas, T_test):
    from scipy.stats import spearmanr
    p = np.minimum(pred, T_test + 1)
    m = np.minimum(meas, T_test + 1)
    med_p, med_m = float(np.median(p)), float(np.median(m))
    lr = abs(math.log(med_p / med_m))
    both_c = (p > T_test) & (m > T_test)
    close = np.abs(np.log(p / m)) < math.log(1.5)
    agree = float(np.mean(both_c | close))
    within = float(np.mean(close))
    cens = float(np.mean((p > T_test) == (m > T_test)))
    rho = float("nan")
    if np.unique(m).size > 1 and np.unique(p).size > 1:
        rho = float(spearmanr(p, m).correlation)
    return dict(median_pred=med_p, median_meas=med_m, abs_log_ratio=lr,
                agree=agree, within_1p5=within, censor_agree=cens, spearman=rho,
                frac_fail_pred=float(np.mean(p <= T_test)),
                frac_fail_meas=float(np.mean(m <= T_test)))
