"""Build the X-ray payoff figure from exp4 results: predicted extent boundary
vs. observed behavioral error. Run after exp4_xray.py."""
import sys, os, json, argparse
import numpy as np, matplotlib.pyplot as plt


def main(inp, out):
    rows = [r for r in json.load(open(inp)) if r.get("test")]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for r in rows:
        t = [d["target"] for d in r["test"]]; e = [d["error"] for d in r["test"]]
        ax.plot(t, e, "-o", color="0.55", lw=1, ms=3, alpha=0.6)
    lo = np.mean([r["extent"][0] for r in rows]); hi = np.mean([r["extent"][1] for r in rows])
    ax.axvspan(lo, hi, color="#4C72B0", alpha=0.15, label="predicted extent (from weights)")
    ax.axvline(hi, color="#4C72B0", ls="--", lw=1.2)
    ax.set_xlabel("target value to hold"); ax.set_ylabel("behavioral error")
    ax.set_title("Structural X-ray: failure predicted from weights alone")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/exp4_xray.json")
    ap.add_argument("--out", default="results/fig_xray.png")
    a = ap.parse_args(); main(a.inp, a.out)
