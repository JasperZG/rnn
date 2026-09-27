"""Amendment 7: validity-aware selective decision rule.

    python decision3.py collect --src stage1_fresh     # dev seeds 5-9
    python decision3.py gamma                          # choose gamma on dev (frozen procedure)
    python decision3.py score --src stage1_prereg2     # preregistered test, seeds 20-24
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
GRID = (0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
TARGET, MIN_N = 0.01, 50


def collect(src):
    import torch
    from hf_tasks import TASKS
    import hf_core as C
    import hf_exact_FROZEN as X
    from run_stage2 import shifted_inputs
    from run_stage3 import hold_inputs
    torch.set_num_threads(2)
    task = TASKS["accumulation"]()
    eps = task.eps
    rec = {k: [] for k in ("meas", "E", "R", "B1", "B2", "net", "scen", "lam2", "rH")}
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
        lam2 = est.diag["lam2_max"]
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
            ep = C.err_norm(est.predict(u), zt)
            rec["meas"].append(C.failure_times(et, mt, eps))
            rec["E"].append(C.failure_times(ep, mt, eps))
            epm = (ep * mt).numpy()
            rec["rH"].append(np.stack([(eps - epm[:, :H].max(1)) / eps for H in HS], 1))
            with torch.no_grad():
                rec["R"].append(C.failure_times(C.err_norm(C.predict_N(reg, task, u), zt), mt, eps))
            b1, b2 = C.baseline_times(et[:, :task.T_train], mt[:, :task.T_train], eps, T)
            rec["B1"].append(b1)
            rec["B2"].append(b2)
            rec["net"].append(np.full(256, len(nets)))
            rec["scen"].append(np.full(256, si))
            rec["lam2"].append(np.full(256, lam2))
        nets.append(tag)
        print(f"collected {tag} lam2={lam2:.4f}", flush=True)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    np.savez(os.path.join(HERE, "results", f"cases3_{src}.npz"), nets=np.array(nets), **out)


def load(src):
    d = np.load(os.path.join(HERE, "results", f"cases3_{src}.npz"))
    return {k: d[k] for k in d.files}


def approve_new(d, hi, gamma):
    return (d["lam2"] < 1.0) & (d["rH"][:, hi] >= gamma)


def choose_gamma(dev):
    for g in GRID:
        worst = 0.0
        for si in range(3):
            s = dev["scen"] == si
            for hi, H in enumerate(HS):
                app = approve_new(dev, hi, g) & s
                if app.sum() >= MIN_N:
                    worst = max(worst, float(np.mean(dev["meas"][app] <= H)))
        if worst <= TARGET:
            return dict(gamma=g, dev_worst_wrong=worst, target_met=True)
    return dict(gamma=GRID[-1], dev_worst_wrong=worst, target_met=False)


def score(d, gamma, verbose=True):
    rows, c1, c2, c4le, c4lt = [], [], [], [], []
    tot = dict(new_ok_app=0, old_ok_app=0, new_app=0, old_app=0, new_wrong=0, old_wrong=0)
    for si, sc in enumerate(SCEN):
        s = d["scen"] == si
        for hi, H in enumerate(HS):
            ok = (d["meas"] > H) & s
            app = approve_new(d, hi, gamma) & s
            old = (d["E"] > 1.10 * H) & s
            n_app, n_ok = int(app.sum()), int(ok.sum())
            wrong = float(np.mean(~ok[app])) if n_app else float("nan")
            useful = float(app[ok].mean()) if n_ok else float("nan")
            old_w = float(np.mean(~ok[old])) if old.any() else float("nan")
            old_u = float(old[ok].mean()) if n_ok else float("nan")
            tot["new_ok_app"] += int((app & ok).sum())
            tot["old_ok_app"] += int((old & ok).sum())
            tot["new_app"] += n_app
            tot["old_app"] += int(old.sum())
            tot["new_wrong"] += int((app & ~ok).sum())
            tot["old_wrong"] += int((old & ~ok).sum())
            if n_app >= MIN_N:
                c1.append(wrong <= 0.02)
            if n_ok >= MIN_N:
                c2.append(useful >= 0.90)
            idx = np.where(s)[0]
            base = {}
            for b in ("R", "B1", "B2"):
                sc_ = d[b][idx] + 1e-6 * np.random.default_rng(0).random(len(idx))
                top = idx[np.argsort(-sc_)[:n_app]]
                base[b] = float(np.mean(d["meas"][top] <= H)) if n_app else float("nan")
            if n_app:
                c4le.append(all(wrong <= v for v in base.values()))
                c4lt.append(all(wrong < v for v in base.values()))
            sup = s & (d["lam2"] < 1)
            ok_s = ok & sup
            useful_sup = float(app[ok_s].mean()) if ok_s.any() else float("nan")
            rows.append((f"{sc}|{H}", n_ok, n_app, wrong, useful, useful_sup, old_w, old_u, base))
    if verbose:
        print(f"gamma={gamma}")
        print(f"{'cell':10s} {'n_ok':>5s} {'n_app':>5s} {'wrong':>6s} {'useful':>6s} {'use|sup':>7s} | "
              f"{'old wrong':>9s} {'old useful':>10s} | matched baseline wrong R/B1/B2")
        for k, n_ok, n_app, w, u, us, ow, ou, b in rows:
            print(f"{k:10s} {n_ok:5d} {n_app:5d} {w:6.3f} {u:6.3f} {us:7.3f} | {ow:9.3f} {ou:10.3f} | "
                  + " ".join(f"{v:.3f}" for v in b.values()))
    pw_new = tot["new_wrong"] / max(tot["new_app"], 1)
    pw_old = tot["old_wrong"] / max(tot["old_app"], 1)
    crit = {
        "C1 P(fail|APPROVE) <= 2% every cell (>=50 approvals)": all(c1),
        "C2 P(APPROVE|ok) >= 90% every cell (>=50 ok)": all(c2),
        "C3 more ok approvals than old rule AND pooled wrong <= old + 0.5pp":
            tot["new_ok_app"] > tot["old_ok_app"] and pw_new <= pw_old + 0.005,
        "C4 matched coverage: <= all baselines 12/12, < in >=10": len(c4le) == 12 and all(c4le) and sum(c4lt) >= 10,
    }
    return crit, tot, pw_new, pw_old, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "gamma", "score"])
    ap.add_argument("--src", default="stage1_prereg2")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.src)
    elif a.cmd == "gamma":
        g = choose_gamma(load("stage1_fresh"))
        json.dump(g, open(os.path.join(HERE, "results", "decision3_gamma_dev.json"), "w"), indent=1)
        print(g)
    else:
        g = json.load(open(os.path.join(HERE, "results", "decision3_gamma_dev.json")))["gamma"]
        d = load(a.src)
        crit, tot, pw_new, pw_old, rows = score(d, g)
        nets = list(d["nets"])
        unsup = sorted({nets[i] for i in np.unique(d["net"][d["lam2"] >= 1])})
        print(f"\nUNSUPPORTED networks (lam2>=1): {len(unsup)} {unsup}")
        print(f"pooled: new approvals of ok cases {tot['new_ok_app']} vs old {tot['old_ok_app']}; "
              f"pooled wrong-approval new {100 * pw_new:.2f}% vs old {100 * pw_old:.2f}%")
        print("\nVERDICT")
        for k, v in crit.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print(f"OVERALL: {'PASS' if all(crit.values()) else 'FAIL'}")
        json.dump(dict(gamma=g, criteria=crit, pooled=tot, unsupported=unsup,
                       cells=[r[:8] for r in rows]), open(os.path.join(HERE, "results", f"decision3_{a.src}.json"), "w"),
                  indent=1, default=str)
