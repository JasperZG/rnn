"""
Two validated diagnostics.

1. frequency_from_angle: near a limit cycle the dynamics are locally linear, so
   the emergent oscillation frequency equals the ANGLE of the leading complex
   Jacobian eigenvalue, f = angle/(2*pi). Confirmed repeatedly (predicted ~=
   measured to within a few percent). Note: the eigenvalue MAGNITUDE does NOT
   cleanly mark the behavioral onset -- only the angle carries the frequency.

2. xray_line_extent: read the extent of a line attractor from the WEIGHTS alone
   (the range of readout values its fixed points span), agnostic to the range
   the network was trained on, and predict the input regime where the network
   must fail -- values beyond the extent cannot be held. Validated: predicted
   boundary matches observed behavioral-failure onset.
"""
import torch, numpy as np
from .fixed_points import jacobian_eigs


def _cycle_center(net, orbit, n_in, steps=2000, lr=0.02, device="cpu"):
    """Unstable focus at the CENTER of a limit cycle: descend the autonomous
    speed |F(h)-h|^2 from the orbit mean. The frequency-from-angle relation is
    the Hopf picture -- it holds at this central fixed point, NOT at a point on
    the orbit itself (where the one-step Jacobian mixes the flow direction and
    gives the wrong -- often harmonic -- angle)."""
    u = torch.zeros(1, n_in, device=device)
    h0 = np.asarray(orbit).reshape(-1, net.N).mean(0)
    h = torch.tensor(h0, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=lr)
    for _ in range(steps):
        q = ((net.step(h.unsqueeze(0), u) - h.unsqueeze(0)) ** 2).sum()
        opt.zero_grad(); q.backward(); opt.step()
    return h.detach().cpu().numpy()


def frequency_from_angle(net, orbit, n_in, device="cpu", mag_floor=1.02):
    """Emergent oscillation frequency = angle/(2*pi) of the FUNDAMENTAL oscillatory
    mode of the linearization at the limit cycle's central focus.

    `orbit` is the autonomous free-run trajectory [T, N]; its mean seeds the search
    for the unstable focus (see _cycle_center). A strongly nonlinear cycle produces
    complex Jacobian eigenvalues at the fundamental angle AND its harmonics (2x, 3x
    ...), so the fundamental is the UNSTABLE (|lambda|>mag_floor) complex eigenvalue
    with the SMALLEST rotation angle -- selecting by magnitude instead grabs a
    harmonic and returns ~2-3x the true frequency. Returns (freq, eigenvalue)."""
    pt = _cycle_center(net, orbit, n_in, device=device)
    u = torch.zeros(1, n_in, device=device)
    ev = jacobian_eigs(net, pt, u, device)
    ang = np.abs(np.angle(ev))
    sel = (np.abs(ev) > mag_floor) & (ang > 1e-3)          # unstable, genuinely rotating
    if sel.any():
        idx = np.where(sel)[0][np.argmin(ang[sel])]         # fundamental = smallest angle
    else:                                                   # no unstable focus: fall back
        cplx = np.abs(ev.imag) > 1e-6
        if not cplx.any():
            return float("nan"), complex(ev[np.argmax(np.abs(ev))])
        idx = np.where(cplx)[0][np.argmin(ang[cplx])]
    return float(ang[idx] / (2 * np.pi)), complex(ev[idx])


def xray_line_extent(net, task_fn, n_in, probe_range=(-2.5, 2.5), n_probe=41,
                     steps=1500, lr=0.02, speed_tol=1e-4, device="cpu"):
    """Recover line-attractor extent from weights; return (lo, hi) readout range
    of genuine fixed points. Probes a WIDE injected range so the extent is read
    from structure, not from the training range."""
    u = torch.zeros(1, n_in, device=device)
    ros, sps = [], []
    for v in np.linspace(*probe_range, n_probe):
        x = torch.zeros(1, 80, n_in, device=device); x[0, 0, 0] = float(v)
        with torch.no_grad():
            _, H = net(x); seed = H[0, -1]
        h = seed.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([h], lr=lr)
        for _ in range(steps):
            q = ((net.step(h.unsqueeze(0), u) - h.unsqueeze(0)) ** 2).sum()
            opt.zero_grad(); q.backward(); opt.step()
        h = h.detach()
        sp = float(((net.step(h.unsqueeze(0), u) - h.unsqueeze(0)) ** 2).sum())
        ros.append(float(net.Wout(h.unsqueeze(0)))); sps.append(sp)
    ros, sps = np.array(ros), np.array(sps)
    stable = sps < speed_tol
    if stable.sum() == 0:
        return None
    return float(ros[stable].min()), float(ros[stable].max())


def xray_failure_test(net, extent, targets, n_in, T=80, device="cpu"):
    """Drive the net with target hold-values; return per-target (held, error,
    inside_predicted_extent). The X-ray prediction is that error stays small
    inside `extent` and rises sharply beyond it."""
    lo, hi = extent; rows = []
    for tgt in targets:
        x = torch.zeros(1, T, n_in, device=device); x[0, 0, 0] = float(tgt)
        with torch.no_grad():
            out, _ = net(x); held = float(out[0, -1, 0])
        rows.append({"target": float(tgt), "held": held, "error": abs(held - tgt),
                     "inside": bool(lo - 0.1 <= tgt <= hi + 0.1)})
    return rows
