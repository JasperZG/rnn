"""Signature figure: the Weight-Space Competence Read-out. Each dynamical structure
is drawn in its own native coordinate system and painted by task error at every stored
location (green = competent, red = fails), with the competence boundary predicted from
the weights alone overlaid. Reads the four structure bundles in results/structures/.

    python experiments/figure_competence_manifold.py --out results/fig_competence_manifold.png

Bundles expected (produced by the make-or-break structure runs):
    results/structures/line_data.npz     keys: vv, out, targets, held, errs, extent_hi
    results/structures/ring_manifold.npz keys: settled, out, angles
    results/structures/plane_data.npz    keys: pts, out, ext_x, ext_y, tests, res
    results/structures/gated_curves.npz  keys: tg, c0_curve, c0_ext, c1_curve, c1_ext
"""
import os, re, argparse
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

COMP = mpl.colors.LinearSegmentedColormap.from_list("comp", ["#1a9850", "#f7e34a", "#d73027"])


def _style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False})


def main(indir, out):
    _style()
    L = dict(np.load(os.path.join(indir, "line_data.npz")))
    R = dict(np.load(os.path.join(indir, "ring_manifold.npz")))
    P = dict(np.load(os.path.join(indir, "plane_data.npz")))
    G = dict(np.load(os.path.join(indir, "gated_curves.npz")))
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.6), constrained_layout=True)
    (axA, axB), (axC, axD) = axs

    # A -- LINE: asked vs held, painted by error
    ext = float(L["extent_hi"]); req = np.concatenate([np.abs(L["vv"]), L["targets"]])
    ach = np.concatenate([np.abs(L["out"]), L["held"]])
    er = np.concatenate([np.abs(L["vv"] - L["out"]), L["errs"]])
    o = np.argsort(req); req, ach, er = req[o], ach[o], er[o]
    nA = mpl.colors.Normalize(0, 0.8)
    axA.plot([0, 2.5], [0, 2.5], color="0.75", lw=1.0, ls=(0, (4, 3)), zorder=1)
    Pn = np.array([req, ach]).T.reshape(-1, 1, 2); segs = np.concatenate([Pn[:-1], Pn[1:]], axis=1)
    lc = LineCollection(segs, cmap=COMP, norm=nA, lw=3.0, zorder=3); lc.set_array((er[:-1] + er[1:]) / 2); axA.add_collection(lc)
    scA = axA.scatter(req, ach, c=er, cmap=COMP, norm=nA, s=16, zorder=4, edgecolors="white", linewidths=0.3)
    axA.axvspan(0, ext, color="#1a9850", alpha=0.07); axA.axvline(ext, color="#2c3e50", lw=1.2, zorder=2)
    axA.text(ext + 0.03, 2.3, "predicted\nextent", fontsize=5.5, va="top", color="#2c3e50")
    axA.set_xlim(0, 2.55); axA.set_ylim(0, 2.5)
    axA.set_xlabel("value asked to store"); axA.set_ylabel("value held")
    axA.set_title("Line attractor  \u00b7  stored scalar")
    axA.text(-0.12, 1.02, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")
    fig.colorbar(scA, ax=axA, fraction=0.046, pad=0.02, label="storage error")

    # B -- RING: loop in state space, painted by angular error
    s = R["settled"] - R["settled"].mean(0); _, _, Vt = np.linalg.svd(s, full_matrices=False); xy = s @ Vt[:2].T
    ro = R["out"]; err_ring = np.abs(np.angle(np.exp(1j * (np.arctan2(ro[:, 1], ro[:, 0]) - R["angles"]))))
    nB = mpl.colors.Normalize(0, 0.35); oa = np.argsort(R["angles"]); loop = np.vstack([xy[oa], xy[oa][0]])
    axB.plot(loop[:, 0], loop[:, 1], color="0.8", lw=0.8, zorder=1)
    scB = axB.scatter(xy[:, 0], xy[:, 1], c=err_ring, cmap=COMP, norm=nB, s=30, zorder=3, edgecolors="white", linewidths=0.3)
    axB.set_aspect("equal"); axB.set_xticks([]); axB.set_yticks([]); axB.set_xlabel("PC1"); axB.set_ylabel("PC2")
    axB.set_title("Ring attractor  \u00b7  stored heading")
    axB.text(-0.12, 1.02, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
    fig.colorbar(scB, ax=axB, fraction=0.046, pad=0.02, label="angular error (rad)")

    # C -- PLANE: stored-position grid + far-field failing tests
    pts = P["pts"]; err_pl = np.sqrt(((pts - P["out"]) ** 2).sum(1))
    ex = float(P["ext_x"][1]); ey = float(P["ext_y"][1]); nC = mpl.colors.Normalize(0, 2.0)
    scC = axC.scatter(pts[:, 0], pts[:, 1], c=err_pl, cmap=COMP, norm=nC, s=24, marker="s", edgecolors="white", linewidths=0.25, zorder=3)
    for i, lab in enumerate(P["tests"]):
        m = re.search(r"\(([-\d.]+),\s*([-\d.]+)\)", str(lab))
        if not m:
            continue
        rx, ry = float(m.group(1)), float(m.group(2)); e = float(P["res"][i, 2])
        if abs(rx) <= 1.0 and abs(ry) <= 1.0:
            continue
        axC.scatter([rx], [ry], c=[e], cmap=COMP, norm=nC, s=95, marker="X", edgecolors="black", linewidths=0.7, zorder=5)
    axC.add_patch(Rectangle((-ex, -ey), 2 * ex, 2 * ey, fill=False, edgecolor="#2c3e50", lw=1.4, zorder=4))
    axC.text(0, ey + 0.12, "predicted extent box", fontsize=5.5, ha="center", va="bottom", color="#2c3e50")
    axC.set_aspect("equal"); axC.set_xlim(-2.7, 2.7); axC.set_ylim(-2.7, 2.7)
    axC.set_xlabel("stored x"); axC.set_ylabel("stored y")
    axC.set_title("Plane attractor  \u00b7  stored 2-D value")
    axC.text(-0.12, 1.02, "C", transform=axC.transAxes, fontsize=12, fontweight="bold")
    axC.text(0.97, 0.03, "X = far-field tests", transform=axC.transAxes, fontsize=5, ha="right", color="#444")
    fig.colorbar(scC, ax=axC, fraction=0.046, pad=0.02, label="storage error")

    # D -- GATED: two context curves, each with its predicted extent
    tg = G["tg"]
    for cv, ext_c, col, lab in [(G["c0_curve"], G["c0_ext"], "#2980b9", "context 0"),
                                (G["c1_curve"], G["c1_ext"], "#8e44ad", "context 1")]:
        axD.plot(tg, cv, "-o", color=col, ms=3.5, lw=1.4, label=lab, zorder=3)
        axD.axvline(float(ext_c[1]), color=col, lw=0.9, ls=":", zorder=2)
    axD.plot([0, tg.max()], [0, tg.max()], color="0.75", lw=1.0, ls=(0, (4, 3)), zorder=1)
    axD.set_xlabel("input drive"); axD.set_ylabel("value held")
    axD.set_title("Gated selection  \u00b7  context-dependent readout")
    axD.legend(fontsize=5.5, frameon=False, loc="lower right"); axD.margins(0.05)
    axD.text(-0.12, 1.02, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")

    fig.suptitle("The Weight-Space Competence Read-out: each structure's geometry bounds what it can do", fontsize=10)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/structures")
    ap.add_argument("--out", default="results/fig_competence_manifold.png")
    a = ap.parse_args(); main(a.indir, a.out)
