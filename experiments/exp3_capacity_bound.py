"""
Experiment 3 -- Capacity as a two-sided bound.
For a task requiring a given structure, train across network sizes spanning the
predicted bracket and locate where performance collapses. Tests whether collapse
falls inside the predicted [lower, upper] range rather than at an arbitrary size,
and where within the bracket trained networks actually land. Reported as a bound,
NOT an exact integer count (the sharp count law failed validation for K>=3).
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from rnnphase import tasks, models, train


def run(sizes, seeds, iters, device, out):
    rows = []
    for N in sizes:
        for seed in range(seeds):
            net = models.build("rnn", 1, 1, N, seed).to(device)
            loss = train.train_network(net, tasks.make_accumulation, iters=iters,
                                       device=device, seed=seed)
            rows.append({"N": N, "seed": seed, "loss": loss,
                         "solved": train.passes_gate(loss, 0.01)})
    json.dump(rows, open(out, "w"), indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[1,2,4,8,16,32,64])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/exp3_capacity.json")
    a = ap.parse_args()
    r = run(a.sizes, a.seeds, a.iters, a.device, a.out)
    print(f"capacity bound: {len(r)} runs across sizes {a.sizes} -> {a.out}")
