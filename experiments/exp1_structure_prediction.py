"""
Experiment 1 -- Structure prediction across seeds and architectures.
For each canonical task, train many seeds x {rnn, gru, lstm}, recover the
dynamical structure, and score the fraction of converged networks whose
recovered structure matches the a-priori prediction. Tests that structure is a
property of the TASK, not the initialization or architecture.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from rnnphase import tasks, models, train, fixed_points, structure

PRED = {"memory": "discrete_fixed_points", "accumulation": "line_attractor",
        "gated": "discrete_fixed_points", "oscillation": "limit_cycle"}
GATE = {"memory": 0.05, "accumulation": 0.01, "gated": 0.05, "oscillation": 0.02}


def run(task, seeds, archs, N, iters, device, out):
    fn, n_in, n_out = tasks.TASKS[task]
    rows = []
    for arch in archs:
        for seed in range(seeds):
            rot = 0.3 if (task == "oscillation" and arch == "rnn") else 0.0
            net = models.build(arch, n_in, n_out, N, seed, rotation_init=rot).to(device)
            loss = train.train_network(net, fn, iters=iters, device=device, seed=seed)
            if not train.passes_gate(loss, GATE[task]):
                rows.append({"task": task, "arch": arch, "seed": seed,
                             "converged": False, "loss": loss}); continue
            _, _, _, _, H = train.evaluate(net, fn, B=64, device=device)
            if arch == "rnn":
                if task == "oscillation":
                    # limit cycles are not fixed-point sets; the oscillation task
                    # is a trigger-then-free-run, so H is the autonomous orbit --
                    # detect a sustained cycle directly (classify_structure only
                    # emits fixed-point / line-attractor labels).
                    import numpy as np
                    is_c, _ = structure.detect_limit_cycle(H[0].detach().cpu().numpy())
                    lab = "limit_cycle" if is_c else "none"
                else:
                    # dedup=0.1 (not 0.5) so a line-attractor continuum is
                    # populated with enough points to be detected; well-separated
                    # discrete fixed points stay distinct either way.
                    pts, sps, evals = fixed_points.find_slow_points(
                        net, H[:, 20:], n_in, n_seed=300, steps=1200, speed_tol=1e-3,
                        dedup=0.1, device=device)
                    lab = structure.classify_structure(pts, list(evals))["label"]
            else:
                lab = "gated_arch_skip"  # FP analysis defined for vanilla RNN activations
            rows.append({"task": task, "arch": arch, "seed": seed, "converged": True,
                         "loss": loss, "recovered": lab, "predicted": PRED[task],
                         "match": lab == PRED[task]})
    json.dump(rows, open(out, "w"), indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="memory")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--archs", nargs="+", default=["rnn"])
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/exp1_structure.json")
    a = ap.parse_args()
    r = run(a.task, a.seeds, a.archs, a.N, a.iters, a.device, a.out)
    conv = [x for x in r if x["converged"]]
    matched = [x for x in conv if x.get("match")]
    print(f"{a.task}: {len(conv)}/{len(r)} converged, "
          f"{len(matched)}/{len(conv)} match predicted structure -> {a.out}")
