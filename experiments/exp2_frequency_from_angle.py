"""
Experiment 2 -- Frequency from eigenvalue angle (the quantitative spine).
Train the oscillation task at a range of target frequencies; for each converged
net read the emergent frequency from the leading complex Jacobian eigenvalue
angle and compare to both the behavioral frequency and the target.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, numpy as np
from rnnphase import tasks, models, train, diagnostics, structure


def run(freqs, seeds, N, iters, device, out):
    rows = []
    for f in freqs:
        for seed in range(seeds):
            net = models.build("rnn", 1, 1, N, seed, rotation_init=0.3).to(device)
            loss = train.train_network(net, tasks.make_oscillation, iters=iters,
                                       device=device, seed=seed, task_kwargs={"freq": f})
            if loss > 0.02: 
                rows.append({"target": f, "seed": seed, "converged": False}); continue
            _, _, _, _, H = train.evaluate(net, tasks.make_oscillation, B=8,
                                           device=device, task_kwargs={"freq": f})
            free = H[0].detach().cpu().numpy()
            is_c, bfreq = structure.detect_limit_cycle(free)
            # frequency-from-angle at the cycle's central focus (post-transient orbit)
            afreq, lam = diagnostics.frequency_from_angle(net, free[20:], 1, device)
            rows.append({"target": f, "seed": seed, "converged": True,
                         "behavioral_freq": bfreq, "angle_freq": afreq,
                         "lambda_mag": abs(lam)})
    json.dump(rows, open(out, "w"), indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", nargs="+", type=float, default=[0.05,0.08,0.10,0.12,0.15])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/exp2_frequency.json")
    a = ap.parse_args()
    r = run(a.freqs, a.seeds, a.N, a.iters, a.device, a.out)
    conv=[x for x in r if x["converged"]]
    print(f"frequency-from-angle: {len(conv)}/{len(r)} converged -> {a.out}")
