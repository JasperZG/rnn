"""
Experiment 5 -- Causal perturbation (targeted vs. control dissociation).

For each task, the dynamical structure the network built is identified from its
own settled states (the top principal component of the settled-state cloud is the
axis along which the structure stores information: the integration axis of a line
attractor, or the axis separating discrete fixed points). That axis is then
projected out of the hidden state at every step, WITHOUT retraining, so that the
manipulation cannot be confounded with new learning (O'Shea et al., 2022). A
matched random-direction control of equal strength is projected out in the same
way. If the identified structure is causally responsible for the computation,
projecting out the structure axis must degrade performance far more than the
control, and do so as a graded dose-response.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, numpy as np
from rnnphase import tasks, models, train
from rnnphase.train import masked_mse
from rnnphase import perturb


def settled_direction(net, task_fn, n_in, device, seed=123):
    """Top principal component of the settled (final-timestep) hidden states."""
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        x = task_fn(256, device=device, g=g)[0]
        _, H = net(x)
    S = H[:, -1].cpu().numpy()
    S = S - S.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(S, full_matrices=False)
    return Vt[0]  # unit vector, R^N


def eval_perturbed(net, task_fn, hook, n_in, device, seed=999):
    """Run the vanilla RNN step-by-step, applying `hook` to the hidden state at
    each step, and return masked MSE against the target (higher = worse)."""
    g = torch.Generator(device=device).manual_seed(seed)
    net.eval()
    with torch.no_grad():
        x, y, mask = task_fn(256, device=device, g=g)[:3]
        B = x.shape[0]
        h = torch.zeros(B, net.N, device=device)
        H = []
        for t in range(x.shape[1]):
            h = net.step(h, x[:, t])
            if hook is not None:
                h = hook(h)
            H.append(h)
        H = torch.stack(H, 1)
        out = net.Wout(H)
        return float(masked_mse(out, y, mask))


def run(task, seeds, N, iters, strengths, device, out):
    fn, n_in, n_out = tasks.TASKS[task]
    rows = []
    for seed in range(seeds):
        net = models.build("rnn", n_in, n_out, N, seed)
        loss = train.train_network(net, fn, iters=iters, device=device, seed=seed)
        gate = {"memory": 0.05, "accumulation": 0.01, "gated": 0.05, "oscillation": 0.02}[task]
        if not train.passes_gate(loss, gate):
            rows.append({"task": task, "seed": seed, "converged": False, "loss": loss}); continue
        d_struct = settled_direction(net, fn, n_in, device, seed=100 + seed)
        rng = np.random.default_rng(200 + seed)
        d_ctrl = rng.standard_normal(N); d_ctrl /= np.linalg.norm(d_ctrl)
        dr = perturb.dose_response(net, fn, d_struct, strengths, eval_perturbed,
                                   n_in, device=device, control_direction=d_ctrl)
        rows.append({"task": task, "seed": seed, "converged": True, "train_loss": loss,
                     "dose_response": dr})
    json.dump(rows, open(out, "w"), indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["accumulation", "memory"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/exp5_perturbation.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    all_rows = {}
    for task in a.tasks:
        r = run(task, a.seeds, a.N, a.iters, a.strengths, a.device,
                a.out.replace(".json", f"_{task}.json"))
        conv = [x for x in r if x["converged"]]
        # summarize: mean targeted vs control loss at max strength
        smax = float(a.strengths[-1])
        def _dr(x, s):
            dr = x["dose_response"]
            return dr[s] if s in dr else dr[str(s)]
        tgt = np.mean([_dr(x, smax)["targeted"] for x in conv]) if conv else float("nan")
        ctl = np.mean([_dr(x, smax)["control"] for x in conv]) if conv else float("nan")
        all_rows[task] = {"n_converged": len(conv), "mean_targeted_maxstrength": tgt,
                          "mean_control_maxstrength": ctl}
        print(f"{task}: {len(conv)}/{a.seeds} converged | "
              f"targeted loss {tgt:.4f} vs control {ctl:.4f} at strength {smax}")
    json.dump(all_rows, open(a.out, "w"), indent=1)
