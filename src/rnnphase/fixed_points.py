"""
Exact structure analysis: gradient-based slow-point search followed by exact
Jacobian eigendecomposition (Sussillo & Barak, 2012). The search is seeded from
states the network actually VISITS while performing the task -- the invariant
sets that carry the computation lie near the used trajectories, so seeding from
arbitrary points misses true attractors and finds spurious ones.

The pipeline must reproduce a KNOWN ANSWER (two stable fixed points + one saddle
for one-bit memory) before it is trusted on any open question. This guards
against a tool that reports confident but incorrect structure.
"""
import torch, numpy as np


def _autonomous_input(net, n_in, device):
    return torch.zeros(1, n_in, device=device)


def find_slow_points(net, visited_states, n_in, n_seed=400, steps=1500, lr=0.02,
                     speed_tol=1e-4, dedup=0.5, u_const=None, device="cpu"):
    """Recover fixed/slow points by minimizing autonomous speed |F(h)-h|^2,
    seeded from visited states. Returns (points[K,N], speeds[K], evals[K,N] complex)."""
    V = visited_states.reshape(-1, net.N).detach().cpu().numpy()
    idx = np.random.RandomState(0).choice(len(V), min(n_seed, len(V)), replace=False)
    seeds = V[idx]
    u = _autonomous_input(net, n_in, device) if u_const is None else u_const
    pts, sps = [], []
    for s in seeds:
        h = torch.tensor(s, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([h], lr=lr)
        for _ in range(steps):
            q = ((net.step(h.unsqueeze(0), u) - h.unsqueeze(0)) ** 2).sum()
            opt.zero_grad(); q.backward(); opt.step()
        h = h.detach()
        sp = float(((net.step(h.unsqueeze(0), u) - h.unsqueeze(0)) ** 2).sum())
        pts.append(h.cpu().numpy()); sps.append(sp)
    pts = np.array(pts); sps = np.array(sps)
    keep = sps < speed_tol
    pts, sps = pts[keep], sps[keep]
    # dedup by distance
    uniq, uidx = [], []
    for i, p in enumerate(pts):
        if all(np.linalg.norm(p - pts[j]) > dedup for j in uidx):
            uidx.append(i); uniq.append(p)
    uniq = np.array(uniq) if len(uniq) else np.zeros((0, net.N))
    evals = np.array([jacobian_eigs(net, p, u, device) for p in uniq]) if len(uniq) else np.zeros((0, net.N), complex)
    return uniq, sps[uidx] if len(uidx) else np.array([]), evals


def jacobian_eigs(net, point, u, device="cpu"):
    """Exact eigenvalues of the local Jacobian d step / d h at a fixed point.
    |lambda|<1 contracting (stable direction); |lambda|>1 expanding; near 1 marginal."""
    h = torch.tensor(point, dtype=torch.float32, device=device).clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(lambda z: net.step(z.unsqueeze(0), u).squeeze(0), h)
    return np.linalg.eigvals(J.detach().cpu().numpy())


def classify_point(evals, marg_lo=0.9, marg_hi=1.1):
    """Stability + marginal-direction count from the Jacobian spectrum."""
    mag = np.abs(evals)
    n_marg = int(((mag > marg_lo) & (mag < marg_hi)).sum())
    n_expand = int((mag > marg_hi).sum())
    stable = n_expand == 0 and n_marg == 0
    return {"stable": stable, "n_marginal": n_marg, "n_expanding": n_expand,
            "spectral_radius": float(mag.max())}
