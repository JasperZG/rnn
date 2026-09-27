"""Decision-level evaluation of failure-time forecasts.

    python decision2.py collect --src stage1_fresh      # per-trial predictions -> results/cases_<src>.npz
    python decision2.py margins                          # choose margins on development set (seeds 5-9)
    python decision2.py analyze --src stage1_confirm     # full analysis with frozen margins

Rule: approve a (network, input sequence) for required duration H if the
predicted first-failure time exceeds H*(1+m); m is a per-method margin chosen on
development networks only. 'validation' approves everything that passed the
training-horizon gate.
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = 5000
HS = (250, 500, 1000, 2000)
SCEN = ("hold", "x1", "x0.5")
METHODS = ("E", "R", "B1", "B2")
GRID = (0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TARGET = 0.01          # development target: wrong-approval rate <= 1% in every (scenario, H) cell
MIN_APPROVED = 50      # cells with fewer approvals are not used to set the margin


def collect(src):
    import torch
    from hf_tasks import TASKS
    import hf_core as C
    import hf_exact_FROZEN as X
    from run_stage2 import shifted_inputs
    from run_stage3 import hold_inputs
    torch.set_num_threads(2)
    task = TASKS["accumulation"]()
    rec = {k: [] for k in ("meas", "net", "scen", "arch") + METHODS}
    nets = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", src, "accumulation_*.json"))):
        r = json.load(open(p))
        if not r["gate_pass"] or r["arch"] == "tc":
            continue
        tag = f"accumulation_{r['arch']}_N{r['N']}_s{r['seed']}"
        m = C.build(r["arch"], task, r["N"], r["seed"])
        m.load_state_dict(torch.load(os.path.join(HERE, "nets", tag + ".pt")))
        m.eval()
        est = X.AccumulationExact(m, r["arch"])
        Z, U, D, Uh = C.probe_samples(m, task, r["seed"])
        reg = C.DefectFit(task)
        reg.fit(Z, U, D, r["seed"], Uh=Uh)
        for si, name in enumerate(SCEN):
            if name == "x1":
                u = task.inputs(256, T, torch.Generator().manual_seed(9000 + r["seed"]))
                mt = task.eval_mask(u)
            elif name == "x0.5":
                u = shifted_inputs(task, 256, T, torch.Generator().manual_seed(9100 + r["seed"]), 0.5)
                mt = task.eval_mask(u)
            else:
                u = hold_inputs(256, T, torch.Generator().manual_seed(9500 + r["seed"]))
                mt = torch.ones(256, T)
                mt[:, :10] = 0
            et, zt = C.native(m, u, task)
            rec["meas"].append(C.failure_times(et, mt, task.eps))
            rec["E"].append(C.failure_times(C.err_norm(est.predict(u), zt), mt, task.eps))
            with torch.no_grad():
                rec["R"].append(C.failure_times(C.err_norm(C.predict_N(reg, task, u), zt), mt, task.eps))
            b1, b2 = C.baseline_times(et[:, :task.T_train], mt[:, :task.T_train], task.eps, T)
            rec["B1"].append(b1)
            rec["B2"].append(b2)
            rec["net"].append(np.full(256, len(nets)))
            rec["scen"].append(np.full(256, si))
            rec["arch"].append(np.full(256, ("rnn", "gru", "lstm").index(r["arch"])))
        nets.append(tag)
        print("collected", tag, flush=True)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    np.savez(os.path.join(HERE, "results", f"cases_{src}.npz"), nets=np.array(nets), **out)


def load(src):
    d = np.load(os.path.join(HERE, "results", f"cases_{src}.npz"))
    return {k: d[k] for k in d.files}


def rates(approve, ok):
    n_app = int(approve.sum())
    wrong = float(np.mean(~ok[approve])) if n_app else float("nan")
    useful = float(np.mean(approve[ok])) if ok.any() else float("nan")
    return wrong, useful, n_app


def choose_margins(dev):
    chosen = {}
    for meth in METHODS:
        pick = None
        for mg in GRID:
            worst = 0.0
            for si in range(len(SCEN)):
                sel = dev["scen"] == si
                for H in HS:
                    ok = dev["meas"][sel] > H
                    app = dev[meth][sel] > H * (1 + mg)
                    w, _, n = rates(app, ok)
                    if n >= MIN_APPROVED:
                        worst = max(worst, w)
            if worst <= TARGET:
                pick = (mg, worst)
                break
        if pick is None:
            pick = (GRID[-1], worst)
        chosen[meth] = dict(margin=pick[0], dev_worst_wrong=pick[1], target_met=pick[1] <= TARGET)
    return chosen


def auc(score, ok):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(ok, score)) if 0 < ok.mean() < 1 else float("nan")


def analyze(d, margins):
    res = {}
    for si, name in enumerate(SCEN):
        sel = d["scen"] == si
        meas = d["meas"][sel]
        for H in HS:
            ok = meas > H
            cell = dict(frac_ok=float(ok.mean()))
            app_val = np.ones_like(ok)
            cell["validation"] = dict(zip(("wrong", "useful", "n_app"), rates(app_val, ok)))
            for meth in METHODS:
                p = d[meth][sel]
                app = p > H * (1 + margins[meth]["margin"])
                w, u, n = rates(app, ok)
                cell[meth] = dict(wrong=w, useful=u, n_app=n, auc=auc(p, ok))
            # matched acceptance: every method approves the same number of cases as E
            k = cell["E"]["n_app"]
            for meth in METHODS:
                p = d[meth][sel] + 1e-6 * np.random.default_rng(0).random(len(ok))  # break ties
                top = np.argsort(-p)[:k]
                app = np.zeros(len(ok), bool)
                app[top] = True
                cell[meth]["wrong_at_E_rate"] = rates(app, ok)[0]
            res[f"{name}|{H}"] = cell
    # asymmetry (cases where the network actually fails within the horizon)
    asym = {}
    f = d["meas"] <= T
    for meth in METHODS:
        p, mm = d[meth][f], d["meas"][f]
        lr = np.log(np.minimum(p, T + 1) / mm)
        asym[meth] = dict(frac_over=float(np.mean(lr > 0)), frac_under=float(np.mean(lr < 0)),
                          frac_over_10pct=float(np.mean(lr > np.log(1.1))),
                          frac_under_10pct=float(np.mean(lr < -np.log(1.1))),
                          median_signed_log=float(np.median(lr)),
                          p95_over=float(np.percentile(lr, 95)))
    # network-level (cluster) summary: per-network wrong-approval, E rule, pooled over H
    per_net = []
    for n in np.unique(d["net"]):
        s = d["net"] == n
        w_all = []
        for H in HS:
            ok = d["meas"][s] > H
            app = d["E"][s] > H * (1 + margins["E"]["margin"])
            w, _, na = rates(app, ok)
            if na:
                w_all.append(w)
        per_net.append(max(w_all) if w_all else 0.0)
    return res, asym, np.array(per_net)


def report(res, asym, per_net, margins, title):
    print(f"\n##### {title}")
    print("margins:", {k: v["margin"] for k, v in margins.items()})
    hdr = f"{'cell':12s}{'ok%':>5s} | {'validation':>10s} | " + " | ".join(
        f"{m + ' wrong/useful/AUC':>24s}" for m in METHODS) + " | wrong@E-rate: " + " ".join(f"{m:>6s}" for m in METHODS)
    print(hdr)
    for key, c in res.items():
        print(f"{key:12s}{100 * c['frac_ok']:5.0f} | {c['validation']['wrong']:10.3f} | " + " | ".join(
            f"{c[m]['wrong']:7.3f} {c[m]['useful']:6.3f} {c[m]['auc']:6.3f}   " for m in METHODS)
            + " |               " + " ".join(f"{c[m]['wrong_at_E_rate']:6.3f}" for m in METHODS))
    print("\nasymmetry (failing cases): over = predicted later than actual (dangerous)")
    for m, a in asym.items():
        print(f"  {m:3s} over={a['frac_over']:.3f} under={a['frac_under']:.3f} over>10%={a['frac_over_10pct']:.3f} "
              f"under>10%={a['frac_under_10pct']:.3f} median signed log={a['median_signed_log']:+.4f} "
              f"p95 over={a['p95_over']:+.3f}")
    print(f"\nnetwork-level worst-cell wrong-approval (E rule): median {np.median(per_net):.3f}, "
          f"max {per_net.max():.3f}, networks with any wrong approval >2%: {int((per_net > 0.02).sum())}/{len(per_net)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "margins", "analyze"])
    ap.add_argument("--src", default="stage1_confirm")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.src)
    elif a.cmd == "margins":
        mg = choose_margins(load("stage1_fresh"))
        json.dump(mg, open(os.path.join(HERE, "results", "decision_margins_dev.json"), "w"), indent=1)
        print(json.dumps(mg, indent=1))
    else:
        mg = json.load(open(os.path.join(HERE, "results", "decision_margins_dev.json")))
        res, asym, per_net = analyze(load(a.src), mg)
        report(res, asym, per_net, mg, a.src)
        json.dump(dict(margins=mg, cells=res, asymmetry=asym, per_net_worst=per_net.tolist()),
                  open(os.path.join(HERE, "results", f"decision_{a.src}.json"), "w"), indent=1)
