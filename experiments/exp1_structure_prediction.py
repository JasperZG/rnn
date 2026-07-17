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


def _task_batch(fn, n_in, device, B=160):
    """A fixed evaluation batch (inputs only), for seeding the slow-point search."""
    g = torch.Generator(device=device).manual_seed(777)
    return fn(B, device=device, g=g)[0]


def _autonomous_orbit(net, h_last, n_in, device, steps=200):
    """Run the trained cell forward autonomously (zero input) from a settled
    state; returns the state trajectory as a numpy array."""
    import numpy as np
    h = h_last.clone(); u = torch.zeros(1, n_in, device=device); orb = []
    for _ in range(steps):
        h = net.step(h.unsqueeze(0), u).squeeze(0)
        orb.append(h.detach().cpu().numpy())
    return np.asarray(orb)


def recover_structure(net, arch, task, H, fn, n_in, device):
    """Recover the dynamical structure for any architecture. GRU exposes .step()
    on its hidden state; LSTM exposes .step() on the JOINT state (h, c) via
    set_joint()/joint_states(), so the same slow-point search and Jacobian
    analysis apply (Maheswaranathan et al., 2019). Earlier versions skipped the
    gated architectures ("gated_arch_skip"); this makes "reproducible across
    architectures" a demonstrated result rather than an assumption.
    """
    import numpy as np
    net.train()  # cuDNN's fused GRU/LSTM kernel refuses backward in eval mode
    with torch.backends.cudnn.flags(enabled=False):
        if task == "oscillation":
            if arch == "lstm":
                Z = net.joint_states(_task_batch(fn, n_in, device))
                net.set_joint(True)
                orb = _autonomous_orbit(net, Z[0, -1], n_in, device)
                net.set_joint(False)
            else:
                orb = _autonomous_orbit(net, H[0, -1], n_in, device)
            is_c, _ = structure.detect_limit_cycle(np.asarray(orb))
            return "limit_cycle" if is_c else "none"
        # dedup=0.1 (not 0.5) so a line-attractor continuum is populated with
        # enough points to be detected; discrete fixed points stay distinct.
        if arch == "lstm":
            Z = net.joint_states(_task_batch(fn, n_in, device))
            net.set_joint(True)
            pts, sps, evals = fixed_points.find_slow_points(
                net, Z[:, 20:], n_in, n_seed=200, steps=800, speed_tol=5e-3,
                dedup=0.4, device=device)
            net.set_joint(False)
        else:
            pts, sps, evals = fixed_points.find_slow_points(
                net, H[:, 20:], n_in, n_seed=300, steps=1200, speed_tol=3e-3,
                dedup=0.25 if arch == "gru" else 0.1, device=device)
        return structure.classify_structure(pts, list(evals))["label"]


def run(task, seeds, archs, N, iters, device, out):
    fn, n_in, n_out = tasks.TASKS[task]
    rows = []
    for arch in archs:
        for seed in range(seeds):
            rot = 0.3 if (task == "oscillation" and arch == "rnn") else 0.0
            it = iters
            if arch == "lstm" and iters < 3000:
                it = 3000  # LSTM needs a little longer to clear the gate
            net = models.build(arch, n_in, n_out, N, seed, rotation_init=rot).to(device)
            loss = train.train_network(net, fn, iters=it, device=device, seed=seed)
            if not train.passes_gate(loss, GATE[task]):
                rows.append({"task": task, "arch": arch, "seed": seed,
                             "converged": False, "loss": loss}); continue
            _, _, _, _, H = train.evaluate(net, fn, B=64, device=device)
            lab = recover_structure(net, arch, task, H, fn, n_in, device)
            rows.append({"task": task, "arch": arch, "seed": seed, "converged": True,
                         "loss": loss, "recovered": lab, "predicted": PRED[task],
                         "match": lab == PRED[task]})
    json.dump(rows, open(out, "w"), indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="memory")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--archs", nargs="+", default=["rnn", "gru", "lstm"])
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
