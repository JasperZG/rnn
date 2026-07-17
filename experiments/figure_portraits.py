"""Phase-portrait gallery: recover the dynamical structure each task builds and
render it in the network's own state space. GPU-recommended (the fixed-point search
is the heavy step). Trains one representative network per task, projects the visited
states to their top two principal components, overlays the recovered stable fixed
points (or the autonomous limit-cycle orbit), and colors each continuous attractor
by the task variable it stores.

Run after nothing else -- self-contained. Example:
    python experiments/figure_portraits.py --seeds 6 --iters 3000 --out results/fig_portraits.png
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rnnphase import tasks, models, train, fixed_points

GATE = {"memory": 0.05, "accumulation": 0.01, "gated": 0.05, "oscillation": 0.02}
ROT  = {"oscillation": 0.3}
GREY = "0.6"; FOCAL = "#c0392b"


def _style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160})


def _proj(x, mean, comps):
    return (x - mean) @ comps.T


def train_representative(task, seeds, iters, device):
    """Train up to `seeds` nets; return the one whose settled states are cleanest
    in-plane (highest PC1+PC2 variance fraction)."""
    fn, n_in, n_out = tasks.TASKS[task]
    best = None
    for seed in range(seeds):
        net = models.build("rnn", n_in, n_out, 128, seed, rotation_init=ROT.get(task, 0.0))
        loss = train.train_network(net, fn, iters=iters, device=device, seed=seed)
        if not train.passes_gate(loss, GATE[task]):
            continue
        g = torch.Generator(device="cpu").manual_seed(777)
        x, y, mask = fn(160, device=device, g=g)[:3]
        with torch.no_grad():
            out, H = net(x)
        Hn = H.cpu().numpy(); xn = x.cpu().numpy(); on = out.cpu().numpy()
        vis = Hn[:, 20:, :].reshape(-1, Hn.shape[-1])
        mean = vis.mean(0)
        _, S, Vt = np.linalg.svd(vis - mean, full_matrices=False)
        clean = float((S[:2] ** 2).sum() / (S ** 2).sum())
        cand = dict(net=net, loss=loss, seed=seed, Hn=Hn, xn=xn, on=on,
                    mean=mean, comps=Vt[:2], vis=vis, H=H, n_in=n_in, clean=clean)
        if best is None or clean > best["clean"]:
            best = cand
    return best  # may be None if no seed passed the gate


def recover(task, e, device):
    mean, comps = e["mean"], e["comps"]
    ent = {"states2d": _proj(e["vis"], mean, comps).astype(np.float32),
           "loss": float(e["loss"]), "seed": int(e["seed"]), "clean": float(e["clean"])}
    if task in ("memory", "accumulation"):
        ent["color"] = e["on"][:, -1, 0].astype(np.float32)
    if task == "gated":
        ent["color"] = np.argmax(e["xn"].mean(1), axis=1).astype(np.float32)
    if task == "oscillation":
        h = e["H"][0, -1].detach().clone(); u = torch.zeros(1, e["n_in"], device=device)
        orb = []
        for _ in range(600):
            h = e["net"].step(h.unsqueeze(0), u).squeeze(0); orb.append(h.detach().cpu().numpy())
        ent["orbit2d"] = _proj(np.array(orb), mean, comps).astype(np.float32)
    else:
        pts, sps, evals = fixed_points.find_slow_points(
            e["net"], e["H"][:, 20:], e["n_in"], n_seed=250, steps=1000,
            speed_tol=1e-3, device=device)
        stable = [p for p, ev in zip(pts, evals) if np.all(np.abs(ev) < 1.05)]
        ent["fps2d"] = (_proj(np.array(stable), mean, comps).astype(np.float32)
                        if stable else np.zeros((0, 2), np.float32))
    return ent


def render(data, out):
    _style()
    panels = [("memory", "Memory", "discrete fixed points"),
              ("accumulation", "Accumulation", "line attractor"),
              ("gated", "Gated selection", "context-dependent fixed points"),
              ("oscillation", "Oscillation", "limit cycle")]
    panels = [p for p in panels if p[0] in data]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.75 * len(panels), 3.05))
    if len(panels) == 1: axes = [axes]
    for ax, (key, title, sub) in zip(axes, panels):
        e = data[key]; s2 = e["states2d"]
        if "color" in e:
            cmap = "viridis" if key != "gated" else "coolwarm"
            ax.scatter(s2[:, 0], s2[:, 1], s=3, c=GREY, alpha=0.12, linewidths=0, rasterized=True)
            sc = ax.scatter(e["states2d"][::48, 0], e["states2d"][::48, 1], s=2, c=GREY, alpha=0)
        else:
            ax.scatter(s2[:, 0], s2[:, 1], s=3, c=GREY, alpha=0.18, linewidths=0, rasterized=True)
        if "fps2d" in e and len(e["fps2d"]):
            ax.scatter(e["fps2d"][:, 0], e["fps2d"][:, 1], s=55, c=FOCAL, marker="o",
                       edgecolors="white", linewidths=0.8, zorder=5)
        if "orbit2d" in e:
            ax.plot(e["orbit2d"][:, 0], e["orbit2d"][:, 1], c=FOCAL, lw=1.6, zorder=5)
        ax.set_title(title)
        ax.text(0.5, -0.14, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=6, color="0.35", style="italic")
        ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.10); ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.suptitle("Each task builds the dynamical structure it needs", y=1.03, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


def main(a):
    device = a.device
    data = {}
    for task in ["memory", "accumulation", "gated", "oscillation"]:
        e = train_representative(task, a.seeds, a.iters, device)
        if e is None:
            print(f"[warn] {task}: no seed reached the gate in {a.seeds} tries "
                  f"(iters={a.iters}); skipping this panel")
            continue
        data[task] = recover(task, e, device)
        print(f"{task}: seed {data[task]['seed']} loss {data[task]['loss']:.4f} "
              f"clean {data[task]['clean']:.2f}")
    if not data:
        raise RuntimeError("no task passed its gate; increase --iters or --seeds")
    flat = {}
    for t, e in data.items():
        for k, v in e.items():
            flat[f"{t}__{k}"] = v
    npz = a.out.replace(".png", ".npz")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(npz, **flat); print("wrote", npz)
    render(data, a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6, help="candidate seeds per task; cleanest kept")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/fig_portraits.png")
    main(ap.parse_args())
