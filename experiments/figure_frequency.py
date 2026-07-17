"""Frequency figure: leading-eigenvalue plane (left) + angle-predicted vs behavioral
frequency, per seed (right). Reads results/exp2_frequency.json."""
import json, argparse, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False})


def main(inp, out):
    _style()
    rows = [r for r in json.load(open(inp)) if r["converged"]]
    tgt = np.array([r["target"] for r in rows]); beh = np.array([r["behavioral_freq"] for r in rows])
    ang = np.array([r["angle_freq"] for r in rows]); mag = np.array([r["lambda_mag"] for r in rows])
    mae = float(np.mean(np.abs(ang - beh)))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    th = np.linspace(0, 2 * np.pi, 400)
    axL.plot(np.cos(th), np.sin(th), color="0.6", lw=1.0, zorder=1)
    uniq = sorted(set(tgt.tolist())); cols = cm.viridis(np.linspace(0, 0.9, len(uniq)))
    for f, c in zip(uniq, cols):
        m = tgt == f; aa = 2 * np.pi * ang[m]; rr = mag[m]
        axL.scatter(rr * np.cos(aa), rr * np.sin(aa), s=14, color=c, label=f"{f:.2f}", alpha=0.8, edgecolors="none")
    axL.set_xlabel("Re lambda"); axL.set_ylabel("Im lambda"); axL.set_aspect("equal")
    axL.axhline(0, color="#ccc", lw=0.5, zorder=0); axL.axvline(0, color="#ccc", lw=0.5, zorder=0)
    axL.set_title("Leading eigenvalue by target frequency")
    axL.legend(title="target freq", fontsize=5, title_fontsize=5, frameon=False, loc="lower left", ncol=2)
    axR.plot([0, 0.25], [0, 0.25], color="0.6", lw=1.0, ls="--", zorder=1)
    axR.scatter(beh, ang, s=12, color="#c0392b", alpha=0.55, edgecolors="none", zorder=3)
    axR.set_xlabel("behavioral frequency"); axR.set_ylabel("frequency from eigenvalue angle")
    axR.set_title("Angle predicts frequency"); axR.set_aspect("equal"); axR.margins(0.05)
    axR.text(0.05, 0.92, f"mean |error| = {mae:.4f}\nn = {len(rows)}", transform=axR.transAxes, fontsize=6, va="top")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out, "| mean|err|", round(mae, 4), "n", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/exp2_frequency.json")
    ap.add_argument("--out", default="results/fig_frequency.png")
    a = ap.parse_args(); main(a.inp, a.out)
