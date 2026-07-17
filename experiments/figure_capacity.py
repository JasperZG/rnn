"""Capacity figure: fraction of seeds solving the task vs. network size.
Reads results/exp3_capacity.json."""
import json, argparse
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False})


def main(inp, out):
    _style()
    cap = json.load(open(inp))
    Ns = sorted({r["N"] for r in cap})
    solve = [float(np.mean([r["solved"] for r in cap if r["N"] == n])) for n in Ns]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(Ns, solve, "-o", color="#c0392b", ms=4, lw=1.5)
    ax.set_xscale("log"); ax.set_xlabel("network size (units)")
    ax.set_ylabel("fraction of seeds solving task")
    ax.set_title("Capacity bound: smallest network that solves the task")
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.margins(0.05)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/exp3_capacity.json")
    ap.add_argument("--out", default="results/fig_capacity.png")
    a = ap.parse_args(); main(a.inp, a.out)
