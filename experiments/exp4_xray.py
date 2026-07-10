"""
Experiment 4 -- The structural X-ray (the demonstrable payoff).
Train an analog-memory network (finite line attractor); read the attractor
extent from the WEIGHTS alone, blind to the trained range; predict the input
regime where the network must fail; confirm behavioral error stays small inside
the extent and rises sharply beyond it. Failure located BEFORE the failing
inputs are run.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, numpy as np
from rnnphase import models, train, diagnostics


def make_hold(B, T=60, rng=1.0, device="cpu", g=None):
    """Inject a value as a pulse at t=0, hold it. Trained range = +-rng."""
    g = g or torch.Generator(device=device)
    val = (torch.rand(B, 1, generator=g, device=device) * 2 - 1) * rng
    x = torch.zeros(B, T, 1, device=device); x[:, 0, 0] = val[:, 0]
    y = val[:, None, :].expand(B, T, 1)
    mask = torch.ones(B, T, device=device); mask[:, :5] = 0
    return x, y, mask


def run(seeds, N, iters, train_range, device, out):
    rows = []
    for seed in range(seeds):
        net = models.build("rnn", 1, 1, N, seed)
        loss = train.train_network(net, make_hold, iters=iters, device=device,
                                   seed=seed, task_kwargs={"rng": train_range})
        if loss > 0.01:
            rows.append({"seed": seed, "converged": False, "loss": loss}); continue
        extent = diagnostics.xray_line_extent(net, make_hold, 1, device=device)
        if extent is None:
            rows.append({"seed": seed, "converged": True, "extent": None}); continue
        targets = [0.5, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]
        test = diagnostics.xray_failure_test(net, extent, targets, 1, device=device)
        rows.append({"seed": seed, "converged": True, "loss": loss,
                     "train_range": train_range, "extent": list(extent), "test": test})
    json.dump(rows, open(out, "w"), indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--train_range", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/exp4_xray.json")
    a = ap.parse_args()
    r = run(a.seeds, a.N, a.iters, a.train_range, a.device, a.out)
    conv=[x for x in r if x.get("converged")]
    print(f"xray: {len(conv)}/{len(r)} converged -> {a.out}")
