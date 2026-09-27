"""Exact slow-manifold defect estimator, v2 (Amendment 9: construction fixes).

Changes vs hf_exact_FROZEN.py: (1) manifold points where the invariance solve
did not converge are dropped (largest contiguous converged block kept);
(2) the drift flow is inverted only on its monotone range. Both are no-ops
when every point converged and the flow is monotone.

Original docstring:
Exact slow-manifold defect estimator (Amendment 3).

Instead of regressing noisy one-step samples, evaluate the trained network's
own map in float64 at points ON its slow manifold. Every network run here is at
most T_train = 50 steps long.
"""
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline


def stepper(m, arch):
    """Return (f, decode, dim) for the model in float64."""
    m.double()
    if arch == "tc":
        f = lambda S, u: torch.tanh((m.g * m.task.F(S @ m.V, u)) @ m.V.T)
        return f, (lambda S: S @ m.V), m.N
    if arch == "lstm":
        return m.net.step, (lambda S: m.net.Wout(S[:, : m.N])), 2 * m.N
    return m.net.step, (lambda S: m.net.Wout(S)), m.N


def slow_points(f, dec, H0, iters=400):
    """min ||f(h,0)-h||^2 s.t. C h = C h0, via L-BFGS in the null space of C."""
    n, dim = H0.shape
    with torch.no_grad():
        Cm = dec(torch.eye(dim, dtype=torch.float64)) - dec(torch.zeros(1, dim, dtype=torch.float64))
        Cm = Cm.T                                             # (k, dim)
        _, _, Vh = torch.linalg.svd(Cm)
        Nb = Vh[Cm.shape[0]:].T                               # (dim, dim-k) null-space basis
    w = torch.zeros(n, Nb.shape[1], dtype=torch.float64, requires_grad=True)
    u0 = torch.zeros(n, 1, dtype=torch.float64)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, tolerance_grad=1e-14,
                            tolerance_change=1e-18, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        H = H0 + w @ Nb.T
        loss = ((f(H, u0) - H) ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        H = H0 + w @ Nb.T
        speed = (f(H, u0) - H).norm(dim=1)
    return H.detach(), speed


def invariant_manifold(f, dec, H0, iters=3000):
    """Solve f(h_i) - h_i - v_i t_i = 0 with C h_i = s_i fixed (parameterization
    method): velocity on the manifold must be tangent to it."""
    n, dim = H0.shape
    with torch.no_grad():
        Cm = (dec(torch.eye(dim, dtype=torch.float64)) - dec(torch.zeros(1, dim, dtype=torch.float64))).T
        _, _, Vh = torch.linalg.svd(Cm)
        Nb = Vh[Cm.shape[0]:].T
        s = dec(H0)[:, 0]
    ds = torch.empty(n, dtype=torch.float64)
    ds[1:-1] = s[2:] - s[:-2]
    ds[0], ds[-1] = s[1] - s[0], s[-1] - s[-2]
    u0 = torch.zeros(n, 1, dtype=torch.float64)
    w = torch.zeros(n, Nb.shape[1], dtype=torch.float64, requires_grad=True)

    def resid():
        H = H0 + w @ Nb.T
        F = f(H, u0)
        v = dec(F)[:, 0] - s
        dH = torch.empty_like(H)
        dH[1:-1] = H[2:] - H[:-2]
        dH[0], dH[-1] = H[1] - H[0], H[-1] - H[-2]
        t = dH / ds[:, None]
        return F - H - v[:, None] * t
    opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, tolerance_grad=1e-15,
                            tolerance_change=1e-20, history_size=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (resid() ** 2).sum()
        loss.backward()
        return loss
    r0 = float(resid().norm(dim=1).median())
    opt.step(closure)
    with torch.no_grad():
        r1 = float(resid().norm(dim=1).median())
        H = (H0 + w @ Nb.T).detach()
    return H, r0, r1


CONV_TOL = 1e-2          # tangent-form invariance residual above which a point is not a solution
MAX_SOLVE_ROUNDS = 10    # invariance solve continued in rounds of 3000 L-BFGS iterations until converged


def converged_block(f, dec, H, tol):
    """Keep the largest contiguous run of manifold points whose invariance residual
    ||f(h)-h-v t|| <= tol (points where the solver did not converge are not on the
    invariant manifold). A no-op when every point converged."""
    with torch.no_grad():
        n = len(H)
        s = dec(H)[:, 0]
        F = f(H, torch.zeros(n, 1, dtype=torch.float64))
        v = dec(F)[:, 0] - s
        dH = torch.empty_like(H)
        dH[1:-1] = H[2:] - H[:-2]
        dH[0], dH[-1] = H[1] - H[0], H[-1] - H[-2]
        ds = torch.empty_like(s)
        ds[1:-1] = s[2:] - s[:-2]
        ds[0], ds[-1] = s[1] - s[0], s[-1] - s[-2]
        res = (F - H - v[:, None] * dH / ds[:, None]).norm(dim=1).numpy()
    ok = res <= tol
    conv_map = "".join("." if g else "X" for g in ok)
    if ok.all():
        return torch.arange(len(H)), 0, conv_map
    best, cur, start, bstart = 0, 0, 0, 0
    for i, g in enumerate(ok):
        if g:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, bstart = cur, start
        else:
            cur = 0
    return torch.arange(bstart, bstart + best), n - best, conv_map


class AccumulationExact:
    """Reduced map s' = s + D(s,u); output y = s' + tau(s,u)."""

    def __init__(self, m, arch, n_pts=401, R=10, umax=0.35, n_u=57):
        f, dec, dim = stepper(m, arch)
        with torch.no_grad():
            c = torch.linspace(-0.3, 0.3, n_pts, dtype=torch.float64)[:, None]
            S = torch.zeros(n_pts, dim, dtype=torch.float64)
            u0 = torch.zeros(n_pts, 1, dtype=torch.float64)
            for _ in range(10):
                S = f(S, c)
            for _ in range(20):
                S = f(S, u0)
        parts = [slow_points(f, dec, S[i:i + 50]) for i in range(0, n_pts, 50)]
        S = torch.cat([p[0] for p in parts])
        speed = torch.cat([p[1] for p in parts])
        with torch.no_grad():
            s = dec(S)[:, 0]
            order = torch.argsort(s)
            S, s, speed = S[order], s[order], speed[order]
            keep, last = [], -1e9
            for i, si in enumerate(s.tolist()):
                if si - last >= 5e-3:
                    keep.append(i)
                    last = si
            S, s, speed = S[keep], s[keep], speed[keep]
            n = len(s)
        S, inv_r0, inv_r1 = invariant_manifold(f, dec, S)
        keep_idx, n_dropped, conv_map = converged_block(f, dec, S, CONV_TOL)
        solve_rounds = 1
        while n_dropped and solve_rounds < MAX_SOLVE_ROUNDS:   # continue until solved (capped)
            S, _, inv_r1 = invariant_manifold(f, dec, S)
            keep_idx, n_dropped, conv_map = converged_block(f, dec, S, CONV_TOL)
            solve_rounds += 1
        S, speed = S[keep_idx], speed[keep_idx]
        with torch.no_grad():
            s = dec(S)[:, 0]
            n = len(s)
            z0 = torch.zeros(n, 1, dtype=torch.float64)
            v = dec(f(S, z0))[:, 0] - s
        sg, vg = s.numpy(), v.numpy()
        # 1-D drift flow over R steps on a fine grid, and its inverse
        grid = np.linspace(sg[0], sg[-1], 20001)
        x = grid.copy()
        for _ in range(R):
            x = x + np.interp(x, sg, vg)
        mono = np.all(np.diff(x) > 0)
        if not mono:                      # restrict to the monotone range around s = 0
            i0 = int(np.argmin(np.abs(grid)))
            inc = np.diff(x) > 0
            lo_i = i0
            while lo_i > 0 and inc[lo_i - 1]:
                lo_i -= 1
            hi_i = i0
            while hi_i < len(inc) and inc[hi_i]:
                hi_i += 1
            grid, x = grid[lo_i:hi_i + 1], x[lo_i:hi_i + 1]
        inv = lambda a: np.interp(a, x, grid)

        with torch.no_grad():
            ug = torch.linspace(-umax, umax, n_u, dtype=torch.float64)
            Sx = S.repeat_interleave(n_u, 0)
            Ux = ug.repeat(n)[:, None]
            Y = f(Sx, Ux)
            imm = dec(Y)[:, 0].view(n, n_u).numpy()
            Zr = torch.zeros_like(Ux)
            for _ in range(R):
                Y = f(Y, Zr)
            Yr = Y
        # adjoint (left-eigenvector) projection of the relaxed residual onto the manifold
        Hm = S.numpy()
        tang = np.gradient(Hm, sg, axis=0)                     # dh/ds along the manifold
        ell = np.zeros_like(Hm)
        for i in range(n):
            J = torch.autograd.functional.jacobian(lambda h: f(h[None], z0[:1])[0], S[i]).numpy()
            w, Lv = np.linalg.eig(J.T)
            j = np.argmin(np.abs(w - 1.0))
            l = np.real(Lv[:, j])
            ell[i] = l / (l @ tang[i])
        yr = Yr.detach().numpy()
        sr = dec(Yr)[:, 0].detach().numpy()
        for _ in range(3):                                     # fixed-point refinement of the base point
            idx = np.clip(np.searchsorted(sg, sr), 0, n - 1)
            sr = sg[idx] + np.einsum("ij,ij->i", ell[idx], yr - Hm[idx])
        a = sr.reshape(n, n_u)
        s_next = inv(a)
        disp = s_next - sg[:, None]
        tau = imm - s_next
        self.disp = RectBivariateSpline(sg, ug.numpy(), disp, kx=3, ky=3, s=0)
        self.tau = RectBivariateSpline(sg, ug.numpy(), tau, kx=3, ky=3, s=0)
        self.lo, self.hi, self.umax = max(sg[0], grid[0]), min(sg[-1], grid[-1]), umax

        # validity diagnostic: second-largest |eigenvalue| at a few manifold points
        lam2 = []
        for i in np.linspace(0, n - 1, 7).astype(int):
            J = torch.autograd.functional.jacobian(lambda h: f(h[None], z0[:1])[0], S[i])
            ev = torch.linalg.eigvals(J).abs().sort(descending=True).values
            lam2.append(float(ev[1]))
        self.diag = dict(n_manifold=int(n), s_range=[float(sg[0]), float(sg[-1])],
                         max_speed=float(speed.max()), median_speed=float(speed.median()),
                         drift_rms=float(np.sqrt((vg ** 2).mean())), flow_monotone=bool(mono),
                         lam2_max=float(max(lam2)), lam2_pow_R=float(max(lam2) ** R),
                         invariance_resid_before=inv_r0, invariance_resid_after=inv_r1,
                         n_dropped_nonconverged=int(n_dropped), inverted_range=[float(grid[0]), float(grid[-1])],
                         convergence_map=conv_map, solve_rounds=solve_rounds)
        m.float()

    def predict(self, u):
        U = u[..., 0].double().numpy()
        B, T = U.shape
        s = np.zeros(B)
        Y = np.zeros((B, T))
        for t in range(T):
            sc = np.clip(s, self.lo, self.hi)
            uc = np.clip(U[:, t], -self.umax, self.umax)
            d = self.disp.ev(sc, uc)
            Y[:, t] = s + d + self.tau.ev(sc, uc)
            s = s + d
        return torch.as_tensor(Y, dtype=torch.float32)[..., None]


def oscillation_exact(m, arch, task, T_test, T_train=50, P=16, start=26):
    """Per-period phase slip from a T_train-long pulse response, propagated."""
    f, dec, dim = stepper(m, arch)
    with torch.no_grad():
        S = torch.zeros(1, dim, dtype=torch.float64)
        Z = []
        for t in range(T_train):
            u = torch.tensor([[1.0 if t == 0 else 0.0]], dtype=torch.float64)
            S = f(S, u)
            Z.append(dec(S)[0].numpy())
    m.float()
    Z = np.array(Z)
    ang = np.arctan2(Z[:, 1], Z[:, 0])
    pairs = range(start, T_train - P)
    slip = np.array([(ang[t + P] - ang[t] + np.pi) % (2 * np.pi) - np.pi for t in pairs])
    dw = slip.mean() / P
    base0 = T_train - P
    Y = np.zeros((T_test, 2))
    Y[:T_train] = Z
    for t in range(T_train, T_test):
        b = base0 + (t - base0) % P
        a = dw * (t - b)
        c, s_ = np.cos(a), np.sin(a)
        Y[t] = [c * Z[b, 0] - s_ * Z[b, 1], s_ * Z[b, 0] + c * Z[b, 1]]
    return torch.as_tensor(Y, dtype=torch.float32)[None], dict(
        phase_slip_per_step=float(dw), slip_sd=float(slip.std()),
        amp=float(np.linalg.norm(Z[-P:], axis=1).mean()))
