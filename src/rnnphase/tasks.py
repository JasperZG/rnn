"""
Task battery for the predictive-structure study.

Each task is chosen because the dynamical object it requires is known from
first principles, so that object serves as an a-priori prediction to test
against the trained network.

    memory       -> discrete stable fixed points   (Hopfield, 1982)
    accumulation -> line attractor                 (Seung, 1996)
    gated         -> context-selected fixed points (Mante et al., 2013)
    oscillation  -> limit cycle
    angular      -> ring attractor                 (Zhang, 1996)

All inputs are synthetic and fully determined by a random seed, so every
experiment is exactly reproducible.
"""
import torch, numpy as np


def make_memory(B, T=80, p_pulse=0.06, device="cpu", g=None):
    """Flip-flop memory: hold the sign of the most recent signed pulse across
    a variable delay. Predicted structure: two stable fixed points + one saddle.
    A mask ignores the first few steps (no value defined before the first pulse)."""
    g = g or torch.Generator(device=device)
    x = torch.zeros(B, T, 1, device=device)
    pulses = (torch.rand(B, T, 1, generator=g, device=device) < p_pulse).float()
    signs = torch.where(torch.rand(B, T, 1, generator=g, device=device) < 0.5, -1.0, 1.0)
    x = pulses * signs
    y = torch.zeros(B, T, 1, device=device)
    cur = torch.zeros(B, 1, device=device)
    for t in range(T):
        hit = pulses[:, t]
        cur = torch.where(hit > 0, signs[:, t], cur)
        y[:, t] = cur
    mask = torch.ones(B, T, device=device); mask[:, :5] = 0
    return x, y, mask


def make_kbit_memory(B, K=2, T=80, p_pulse=0.06, device="cpu", g=None):
    """K independent flip-flop channels. Tests how many discrete stable states a
    network builds as independent memories are added -- the count is a quantity
    UNDER TEST, not an assumed 2**K law."""
    g = g or torch.Generator(device=device)
    xs, ys = [], []
    for _ in range(K):
        x, y, mask = make_memory(B, T, p_pulse, device, g)
        xs.append(x); ys.append(y)
    return torch.cat(xs, -1), torch.cat(ys, -1), mask


def make_accumulation(B, T=80, scale=0.15, device="cpu", g=None):
    """Analog accumulation: output the running sum of small signed increments.
    Predicted structure: a line attractor with one marginal direction.
    No per-trial normalization -- that would collapse the continuum under test."""
    g = g or torch.Generator(device=device)
    inc = (torch.rand(B, T, 1, generator=g, device=device) * 2 - 1) * scale
    y = torch.cumsum(inc, dim=1)
    mask = torch.ones(B, T, device=device)
    return inc, y, mask


def make_gated(B, T=80, p_pulse=0.08, device="cpu", g=None):
    """Two pulse streams + a context channel selecting which to report.
    Predicted structure: context-gated fixed points, one set per context."""
    g = g or torch.Generator(device=device)
    s1, y1, _ = make_memory(B, T, p_pulse, device, g)
    s2, y2, _ = make_memory(B, T, p_pulse, device, g)
    ctx = torch.where(torch.rand(B, 1, 1, generator=g, device=device) < 0.5, 0.0, 1.0).expand(B, T, 1)
    x = torch.cat([s1, s2, ctx], -1)
    y = torch.where(ctx > 0.5, y2, y1)
    mask = torch.ones(B, T, device=device); mask[:, :5] = 0
    return x, y, mask


def make_oscillation(B, T=80, freq=0.10, t0=8, device="cpu", g=None):
    """Trigger-then-free-run: a brief pulse sets the net oscillating at a target
    frequency with no further input. Predicted structure: a limit cycle.
    Forces autonomous generation rather than tracking an input."""
    g = g or torch.Generator(device=device)
    x = torch.zeros(B, T, 1, device=device); x[:, 0:3, 0] = 1.0
    t = torch.arange(T, device=device).float()
    y = torch.sin(2 * np.pi * freq * (t - t0))[None, :, None].expand(B, T, 1).clone()
    y[:, :t0] = 0.0
    mask = torch.ones(B, T, device=device); mask[:, :t0] = 0
    return x, y, mask


def make_gated_oscillation(B, T=80, freq=0.10, cstar=0.5, t0=8, device="cpu", g=None):
    """Tonic knob c in [0,1]: below c* the network holds still, above it produces
    a rhythm whose amplitude ramps from the threshold. The knob is the continuous
    control property for the oscillation-onset sweep."""
    g = g or torch.Generator(device=device)
    c = torch.rand(B, 1, generator=g, device=device)
    x = torch.zeros(B, T, 2, device=device); x[:, :, 0] = c; x[:, 0:3, 1] = 1.0
    t = torch.arange(T, device=device).float()
    amp = torch.clamp((c - cstar) / (1 - cstar), min=0.0)
    osc = amp * torch.sin(2 * np.pi * freq * (t - t0))[None, :]
    y = torch.where(t[None, :] >= t0, osc, torch.zeros_like(osc)).unsqueeze(-1)
    mask = torch.ones(B, T, device=device); mask[:, :t0] = 0
    return x, y, mask, c


TASKS = {
    "memory": (make_memory, 1, 1),
    "accumulation": (make_accumulation, 1, 1),
    "gated": (make_gated, 3, 1),
    "oscillation": (make_oscillation, 1, 1),
}
