"""
Training pipeline with a convergence gate. Only networks that reach a
task-specific performance threshold are analyzed: the dynamical structure of a
network that has not learned the task carries no information about what the task
requires. This gate guards against mistaking a training failure for a missing
structure (Yang & Wang, 2021; Pascanu et al., 2013 for gradient clipping).
"""
import torch


def masked_mse(out, y, mask):
    err = ((out - y) ** 2).mean(-1) * mask
    return err.sum() / mask.sum()


def train_network(net, task_fn, iters=1500, lr=2e-3, clip=5.0, B=128,
                  device="cpu", seed=0, task_kwargs=None):
    task_kwargs = task_kwargs or {}
    g = torch.Generator(device=device).manual_seed(1000 + seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    last = None
    for it in range(iters):
        batch = task_fn(B, device=device, g=g, **task_kwargs)
        x, y, mask = batch[0], batch[1], batch[2]
        out, _ = net(x)
        loss = masked_mse(out, y, mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()
        last = loss.item()
    return last


def evaluate(net, task_fn, B=256, device="cpu", seed=999, task_kwargs=None):
    task_kwargs = task_kwargs or {}
    g = torch.Generator(device=device).manual_seed(seed)
    net.eval()
    with torch.no_grad():
        batch = task_fn(B, device=device, g=g, **task_kwargs)
        x, y, mask = batch[0], batch[1], batch[2]
        out, H = net(x)
        loss = float(masked_mse(out, y, mask))
    return loss, out, y, mask, H


def passes_gate(loss, threshold):
    """Convergence gate: True iff the network reached criterion. Threshold is
    task-specific and fixed in advance."""
    return loss < threshold
