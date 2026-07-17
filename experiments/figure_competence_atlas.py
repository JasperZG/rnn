"""Competence Atlas: the dense matrix figure. Rows = the four canonical tasks,
columns = network size (a difficulty axis). Each cell shows the occupied state
cloud (grey) with the recovered dynamical structure overlaid (blue = recovered
structure matches the a-priori prediction; orange = mismatch, which happens only
when the network is too small to build the structure the task needs). Reads the
committed sweep bundle; no training required.

    python experiments/figure_competence_atlas.py \
        --data results/atlas_data.npz --meta results/atlas_meta.json \
        --out results/fig_competence_atlas.png
"""
import argparse, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PRED_LABEL = {"memory": "discrete fixed points", "accumulation": "line attractor",
              "gated": "context-dependent fixed points", "oscillation": "limit cycle"}
TASK_LABEL = {"memory": "Memory", "accumulation": "Accumulation",
              "gated": "Gated selection", "oscillation": "Oscillation"}
MATCH, MISS, CLOUD = "#2c7fb8", "#d95f0e", "#c9c9c9"


def main(a):
    D = np.load(a.data); meta = json.load(open(a.meta))
    sizes = meta["sizes"]; tasks = meta["tasks"]; cells = meta["cells"]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": True, "axes.spines.right": True})
    fig, axes = plt.subplots(len(tasks), len(sizes), figsize=(11.0, 9.0))
    for i, task in enumerate(tasks):
        for j, N in enumerate(sizes):
            ax = axes[i, j]; key = f"{task}_{N}"; m = cells[key]; ok = m["match"]
            col = MATCH if ok else MISS
            if task == "oscillation":
                loop = D[f"{key}::loop"]
                ax.plot(np.r_[loop[:, 0], loop[0, 0]], np.r_[loop[:, 1], loop[0, 1]], color=col, lw=1.3)
            else:
                proj = D[f"{key}::proj"]; slow = D[f"{key}::slow"]
                ax.scatter(proj[:, 0], proj[:, 1], s=3, c=CLOUD, alpha=0.35, linewidths=0, rasterized=True)
                if len(slow):
                    ax.scatter(slow[:, 0], slow[:, 1], s=26, c=col, edgecolors="white", linewidths=0.5, zorder=3)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(MISS if not ok else "#aaaaaa"); s.set_linewidth(1.8 if not ok else 0.8)
            ax.margins(0.12)
    for j, N in enumerate(sizes):
        axes[0, j].set_title(f"N = {N}", fontsize=9, pad=6)
    for i, task in enumerate(tasks):
        axes[i, 0].set_ylabel(f"{TASK_LABEL[task]}\n{PRED_LABEL[task]}", fontsize=8.5,
                              rotation=90, labelpad=8, va="center")
        axes[i, 0].yaxis.label.set_multialignment("center")
    fig.suptitle("Competence Atlas: predicted structure recovered across tasks and network sizes",
                 fontsize=11, y=0.985)
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=MATCH, markersize=8, label='recovered = predicted'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor=MISS, markersize=8, label='recovered \u2260 predicted (too few units)')]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, -0.02, "grey = occupied state cloud (PC1\u2013PC2)   \u2022   colored = recovered slow points / limit cycle",
             ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=[0.02, 0.07, 1, 0.965])
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/atlas_data.npz")
    ap.add_argument("--meta", default="results/atlas_meta.json")
    ap.add_argument("--out", default="results/fig_competence_atlas.png")
    main(ap.parse_args())
