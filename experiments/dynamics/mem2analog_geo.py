"""Memory-to-analog alpha sweep, re-run to record geometry for EVERY network.

The original run recorded pc1/frac_marginal only for networks that classify_structure
labelled line_attractor; discrete rows got the .get(...,0) default, so pc1=0.0 on
those rows is a MISSING VALUE, not a measurement. Plotting it against alpha would
manufacture a transition from 0 to 1.

This version computes the geometry unconditionally from the recovered point set and
saves the point cloud + eigenvalues per network, so any later analysis is free.
"""
import sys, os, argparse, json
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/rnn/src"))
from rnnphase.models import build
from rnnphase.train import train_network, evaluate
from rnnphase.fixed_points import find_slow_points
from rnnphase.structure import classify_structure

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--nshards", type=int, default=4)
ap.add_argument("--out", default=None)
args = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("shard", args.shard, "device:", DEV, flush=True)


def make_mem2analog(B, T=80, scale=0.15, alpha=0.0, device="cpu", g=None):
    g = g or torch.Generator(device=device)
    inc = (torch.rand(B, T, 1, generator=g, device=device) * 2 - 1) * scale
    s = torch.cumsum(inc, dim=1)
    y = (1 - alpha) * torch.tanh(8.0 * s) + alpha * s
    mask = torch.ones(B, T, device=device)
    return inc, y, mask


def geometry(pts, evals, marg_tol=0.02):
    """Unconditional geometry: computed for every point set with >= 3 points,
    regardless of which label classify_structure assigns."""
    if len(pts) < 3:
        return dict(pc1=None, pc2=None, frac_marginal=None, extent=None)
    X = np.asarray(pts) - np.asarray(pts).mean(0)
    sv = np.linalg.svd(X, compute_uv=False)
    var = (sv ** 2) / (sv ** 2).sum()
    Vt = np.linalg.svd(X, full_matrices=False)[2]
    proj = X @ Vt[0]
    md = np.array([np.min(np.abs(np.abs(np.asarray(ev)) - 1.0)) for ev in evals])
    return dict(pc1=float(var[0]), pc2=float(var[1]) if len(var) > 1 else None,
                frac_marginal=float(np.mean(md < marg_tol)),
                extent=float(proj.max() - proj.min()))


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]
grid = [(a, sd) for a in ALPHAS for sd in SEEDS]
mine = grid[args.shard::args.nshards]
N, ITERS = 96, 2500
out = []
for a, sd in mine:
    net = build("rnn", 1, 1, N=N, seed=sd).to(DEV)
    tk = dict(alpha=a)
    loss = train_network(net, make_mem2analog, iters=ITERS, lr=2e-3, B=128,
                        device=DEV, seed=sd, task_kwargs=tk)
    _, _, _, _, H = evaluate(net, make_mem2analog, B=128, device=DEV, task_kwargs=tk)
    Hv = H.reshape(-1, N)
    pts, sps, evals = find_slow_points(net, Hv, n_in=1, n_seed=150, steps=600,
                                       speed_tol=1e-4, dedup=0.1, device=DEV)
    cls = classify_structure(pts, list(evals)) if len(pts) else {"label": "none"}
    geo = geometry(pts, evals) if len(pts) else dict(pc1=None, pc2=None, frac_marginal=None, extent=None)
    # save the raw cloud so geometry never has to be re-derived from a re-run
    tag = "a%s_s%d" % (str(a).replace(".", "p"), sd)
    np.savez_compressed("cloud_%s.npz" % tag,
                        points=np.asarray(pts, dtype=np.float32),
                        speeds=np.asarray(sps, dtype=np.float32),
                        evals=np.asarray(evals, dtype=np.complex64),
                        states=Hv[::20].detach().cpu().numpy().astype(np.float32))
    rec = dict(alpha=a, seed=sd, loss=round(float(loss), 5), n_pts=int(len(pts)),
               label=cls["label"], cloud="cloud_%s.npz" % tag, **geo)
    out.append(rec)
    print("a=%s sd=%d loss=%.4f npts=%d -> %-22s pc1=%s fmarg=%s extent=%s"
          % (a, sd, loss, len(pts), cls["label"],
             None if geo["pc1"] is None else round(geo["pc1"], 3),
             None if geo["frac_marginal"] is None else round(geo["frac_marginal"], 2),
             None if geo["extent"] is None else round(geo["extent"], 2)), flush=True)

json.dump(out, open(args.out or "mem2analog_geo_shard%d.json" % args.shard, "w"), indent=1)
print("DONE shard", args.shard, flush=True)
