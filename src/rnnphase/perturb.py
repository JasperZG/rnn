"""
Causal perturbation: modify the identified structure in the network's own state
space (no retraining, which would confound manipulation with new learning) and
measure the graded behavioral effect. A dose-response curve plus a matched
random-direction control establishes that the structure is causally responsible
rather than merely present (O'Shea et al., 2022).
"""
import torch, numpy as np


def project_out(net, direction, strength):
    """Return a hook that projects `strength` of `direction` out of the hidden
    state at each step, targeting the identified structure's subspace."""
    d = torch.tensor(direction, dtype=torch.float32)
    d = d / d.norm()
    def hook(h):
        return h - strength * (h @ d)[:, None] * d[None, :]
    return hook


def dose_response(net, task_fn, direction, strengths, eval_fn, n_in,
                  device="cpu", control_direction=None):
    """Measure performance vs perturbation strength along `direction`, and along
    a matched control direction. Returns dict of strength -> (targeted, control)."""
    out = {}
    for s in strengths:
        tgt = eval_fn(net, task_fn, project_out(net, direction, s), n_in, device)
        ctl = (eval_fn(net, task_fn, project_out(net, control_direction, s), n_in, device)
               if control_direction is not None else None)
        out[float(s)] = {"targeted": tgt, "control": ctl}
    return out
