"""
Competence read-out, non-circular onset measurement.

WHY THIS VERSION EXISTS
-----------------------
The previous run measured the value a network's output "saturates at" using probes placed
at boundary x (1 +- f) with f <= 0.13. Those targets correlate with the boundary at r = 1.0
BY CONSTRUCTION, and the held value tracks the target, so the reported r = 0.961 between
boundary and ceiling was circular: the measurement axis was derived from the quantity being
predicted. A direct sweep across the full input range showed held tracks target monotonically
(0.1 -> 0.104, 1.0 -> 0.781, 2.9 -> 2.106) with no hard ceiling, confirming the artifact.
That result is retracted and is not measured here.

WHAT MAKES THIS NON-CIRCULAR
----------------------------
1. FIXED GRID. Failure onset is measured on ONE dense grid spanning the full input range,
   identical for every network and defined without reference to any network's boundary.
   Prediction (from weights) and measurement (from behavior) therefore have independent
   provenance -- nothing about the grid can encode the prediction.

2. WITHIN-CONFIG TEST. Every network shares one training configuration, so train_range
   cannot drive a correlation. Networks trained identically nonetheless build boundaries
   spanning 0.72 to 3.06 (CV 18.6% at fixed config, measured in the previous run). The
   question is whether the weight read predicts WHICH of two identically-trained networks
   fails earlier -- information no configuration knowledge provides.

3. CONFIG CONTROL ARM. A second arm varies train_range so the boundary-vs-onset relation
   can be tested with train_range partialled out, separating "the read-out is informative"
   from "both quantities track a setting the experimenter already knows".

4. GRADED ONSET. Because error rises continuously rather than switching on, onset is
   recorded at three thresholds (0.05, 0.10, 0.20) instead of one, so the conclusion cannot
   rest on an arbitrary cutoff.
"""
import sys, os, argparse, json
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/rnn/src"))
from rnnphase import models, train, diagnostics

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_RANGE = 3.0          # wide, so the built structure lands inside it
CORE = 0.5                 # dense core of the deployment distribution
P_TAIL = 0.06              # probability a draw comes from the rare periphery
FRACS = [-0.40, -0.25, -0.12, -0.04, 0.05, 0.10, 0.16, 0.24]
FAIL = 0.10                # |held - target| above this counts as a behavioral failure
THRESH = [0.05, 0.10, 0.20]   # onset recorded at three cutoffs, not one
N_GRID = 61                   # fixed grid resolution, same for every network
RANGES = [1.5, 2.25, 3.0]     # config-control arm: spreads train_range
GATE = 0.015


def _draw(B, sampling, g, device="cpu", rng=TRAIN_RANGE, core=CORE, p_tail=P_TAIL):
    if sampling == "unif":
        return (torch.rand(B, 1, generator=g, device=device) * 2 - 1) * rng
    u = torch.rand(B, 1, generator=g, device=device)
    core_v = (torch.rand(B, 1, generator=g, device=device) * 2 - 1) * core
    tail_m = torch.rand(B, 1, generator=g, device=device) * (rng - core) + core
    tail_s = torch.where(torch.rand(B, 1, generator=g, device=device) < 0.5, -1.0, 1.0)
    return torch.where(u < p_tail, tail_m * tail_s, core_v)


def make_hold(B, T=60, rng=TRAIN_RANGE, sampling="mix", core=CORE, p_tail=P_TAIL,
              device="cpu", g=None):
    """Inject a value as a pulse at t=0, hold it. Same task as exp4_xray; only the
    distribution the value is drawn from differs."""
    g = g or torch.Generator(device=device)
    val = _draw(B, sampling, g, device, rng, core, p_tail)
    x = torch.zeros(B, T, 1, device=device); x[:, 0, 0] = val[:, 0]
    y = val[:, None, :].expand(B, T, 1)
    mask = torch.ones(B, T, device=device); mask[:, :5] = 0
    return x, y, mask


def density_beyond(b, sampling, rng=TRAIN_RANGE, core=CORE, p_tail=P_TAIL):
    """Exact fraction of training draws with |value| > b. Analytic, not sampled."""
    b = abs(b)
    if b >= rng:
        return 0.0
    if sampling == "unif":
        return (rng - b) / rng
    if b < core:
        return p_tail + (1 - p_tail) * (core - b) / core
    return p_tail * (rng - b) / (rng - core)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=40)       # mixture arm
    ap.add_argument("--range-seeds", type=int, default=6)  # per train_range in control arm
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()

    # main arm: one fixed config, seeds only -> within-config test
    # control arm: train_range varied -> lets train_range be partialled out
    combos = ([("fixed", TRAIN_RANGE, s) for s in range(args.seeds)] +
              [("varied", R, s) for R in RANGES for s in range(args.range_seeds)])
    mine = combos[args.shard::args.nshards]

    rows = []
    for arm, rng_i, sd in mine:
        sampling = "mix"
        tk = {"rng": rng_i, "sampling": sampling, "core": CORE, "p_tail": P_TAIL}
        net = models.build("rnn", 1, 1, args.N, sd).to(DEV)
        loss = train.train_network(net, make_hold, iters=args.iters, device=DEV,
                                   seed=sd, task_kwargs=tk)
        rec = dict(arm=arm, sampling=sampling, seed=sd, loss=float(loss), train_range=float(rng_i),
                   core=CORE, p_tail=P_TAIL, gate=GATE, fail_thresh=FAIL,
                   converged=bool(loss <= GATE))
        if loss > GATE:
            rows.append(rec); print(f"{sampling} sd={sd} NOT CONVERGED loss={loss:.4f}", flush=True)
            continue

        def probe_task(B, T=60, device="cpu", g=None, **kw):
            return make_hold(B, T=T, rng=rng_i, sampling=sampling,
                             core=CORE, p_tail=P_TAIL, device=device, g=g)

        extent = diagnostics.xray_line_extent(net, probe_task, 1, device=DEV)
        if extent is None:
            rec["extent"] = None; rows.append(rec)
            print(f"{sampling} sd={sd} no extent recovered", flush=True)
            continue
        lo, hi = extent
        rec["extent"] = [float(lo), float(hi)]

        # ---------- FIXED-GRID ONSET MEASUREMENT (independent of the prediction)
        # The grid spans the full trained range and is identical for every network. Nothing
        # about its placement references this network's boundary, so the measured onset and
        # the weight-read prediction are independently derived quantities.
        grid = [float(x) for x in np.linspace(0.0, rng_i, N_GRID)][1:]
        gres = diagnostics.xray_failure_test(net, extent, grid, 1, device=DEV)
        gerr = [float(r["error"]) for r in gres]
        rec["grid"] = dict(targets=grid, errors=gerr,
                           held=[float(r["held"]) for r in gres])

        def onset(th):
            """Smallest grid value whose error exceeds th, linearly interpolated."""
            for k in range(len(grid)):
                if gerr[k] > th:
                    if k == 0:
                        return float(grid[0])
                    x0, x1, y0, y1 = grid[k - 1], grid[k], gerr[k - 1], gerr[k]
                    return float(x0 + (th - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
            return None            # never failed anywhere on the grid

        rec["onset"] = {str(th): onset(th) for th in THRESH}

        # error at the boundary itself and at fixed absolute offsets from it, for the
        # graded-competence curve; offsets are absolute so they do not scale with boundary
        off = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
        ot = [float(hi + d) for d in off if 0.0 < hi + d <= rng_i]
        ores = diagnostics.xray_failure_test(net, extent, ot, 1, device=DEV)
        rec["offsets"] = [dict(offset=float(t - hi), target=float(t), error=float(r["error"]),
                               held=float(r["held"]))
                          for t, r in zip(ot, ores)]

        rows.append(rec)
        o10 = rec["onset"]["0.1"]
        print(f"{arm} rng={rng_i} sd={sd} loss={loss:.4f} boundary={hi:.3f} "
              f"onset@0.10={'none' if o10 is None else round(o10,3)} "
              f"gridpts={len(grid)} offsets={len(rec['offsets'])}", flush=True)

    json.dump(rows, open(f"xray_onset_shard{args.shard}.json", "w"))
    print("DONE shard", args.shard, flush=True)
