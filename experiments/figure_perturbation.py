"""Perturbation figure: dose-response of task error when the identified structure axis
is removed (targeted) vs. a random axis of equal magnitude (control), per task.
Reads results/exp5_perturbation_{task}.json files."""
import json, argparse, os, glob
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False})


def _curve(rows, strengths):
    tg, ct = [], []
    for s in strengths:
        key = s if any(s in r["dose_response"] for r in rows) else str(s)
        t = [r["dose_response"].get(str(s), r["dose_response"].get(s, {})).get("targeted") for r in rows]
        c = [r["dose_response"].get(str(s), r["dose_response"].get(s, {})).get("control") for r in rows]
        t = [v for v in t if v is not None]; c = [v for v in c if v is not None]
        tg.append(np.mean(t)); ct.append(np.mean(c))
    return tg, ct


def main(indir, out, tasks):
    _style()
    fig, axes = plt.subplots(1, len(tasks), figsize=(3 * len(tasks), 3.0), sharey=True)
    if len(tasks) == 1: axes = [axes]
    for ax, task in zip(axes, tasks):
        f = os.path.join(indir, f"exp5_perturbation_{task}.json")
        rows = [r for r in json.load(open(f)) if r["converged"]]
        strengths = sorted({float(s) for r in rows for s in
                            [k if isinstance(k, (int, float)) else float(k) for k in r["dose_response"]]})
        tg, ct = _curve(rows, strengths)
        ax.plot(strengths, tg, "-o", color="#c0392b", ms=4, lw=1.6, label="structure axis removed", zorder=3)
        ax.plot(strengths, ct, "-o", color="0.6", ms=4, lw=1.4, label="random axis (control)", zorder=2)
        ax.set_title(task.capitalize()); ax.set_xlabel("perturbation strength"); ax.margins(0.05)
    axes[0].set_ylabel("task error"); axes[0].legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.suptitle("Removing the identified structure breaks the computation; a random direction does not",
                 y=1.02, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results")
    ap.add_argument("--out", default="results/fig_perturbation.png")
    ap.add_argument("--tasks", nargs="+", default=["accumulation", "gated", "memory"])
    a = ap.parse_args(); main(a.indir, a.out, a.tasks)
