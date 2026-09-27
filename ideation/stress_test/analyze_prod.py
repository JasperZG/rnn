"""Preregistered production analysis (Amendment 8).

    python analyze_prod.py            # reads results/prod/{factorial,deep128,osc}/
Writes results/prod_analysis.json and prints every preregistered criterion.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
T = 5000
HS = (250, 500, 1000, 2000)
SCEN = ("hold", "x1", "x0.5")
GAMMA = json.load(open(os.path.join(HERE, "results", "decision3_gamma_dev.json")))["gamma"]
PRED = ("E", "NS", "R", "B1", "B2")
LOG11 = math.log(1.1)


def load_line(cohorts):
    """Per-network records for hold/accumulation networks that were forecast."""
    nets, funnel = [], defaultdict(lambda: dict(attempted=0, gate_pass=0, error=0, forecast=0))
    for co in cohorts:
        for p in sorted(glob.glob(os.path.join(HERE, "results", "prod", co, "*.json"))):
            r = json.load(open(p))
            if r["task"] == "oscillation":
                continue
            cell = (r["task"], r["arch"], r["N"])
            f = funnel[(co,) + cell]
            f["attempted"] += 1
            f["error"] += r["status"] == "error"
            f["gate_pass"] += bool(r.get("gate_pass"))
            npz = p[:-5] + ".npz"
            if r["status"] == "ok" and r.get("gate_pass") and os.path.exists(npz):
                f["forecast"] += 1
                r["arr"] = dict(np.load(npz))
                r["cohort"] = co
                nets.append(r)
    return nets, funnel


def log_err(pred, meas):
    f = meas <= T
    p = np.minimum(pred[f], T + 1)
    return np.log(p / meas[f])


def forecast_metrics(pred, meas):
    le = log_err(pred, meas)
    if not len(le):
        return None
    a = np.abs(le)
    rho = spearmanr(np.minimum(pred, T + 1), meas).correlation if np.unique(meas).size > 1 else float("nan")
    return dict(n_fail=int(len(le)), median_abs_log=float(np.median(a)), median_signed_log=float(np.median(le)),
                p90_abs_log=float(np.percentile(a, 90)), p95_abs_log=float(np.percentile(a, 95)),
                within_1p25=float(np.mean(a < math.log(1.25))), within_1p5=float(np.mean(a < math.log(1.5))),
                spearman=float(rho), frac_over_10pct=float(np.mean(le > LOG11)))


def confusion(app, fail):
    n = len(app)
    A, F = app, fail
    return dict(n=n, n_app=int(A.sum()), n_ok=int((~F).sum()),
                P_F_given_A=float(np.mean(F[A])) if A.any() else float("nan"),
                P_A_given_F=float(np.mean(A[F])) if F.any() else float("nan"),
                P_A_given_notF=float(np.mean(A[~F])) if (~F).any() else float("nan"),
                coverage=float(A.mean()))


def auc(score, ok):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(ok, score)) if 0 < ok.mean() < 1 else float("nan")


def cell_arrays(nets):
    out = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(nets):
        a = r["arr"]
        for si in np.unique(a["scen"]):
            s = a["scen"] == si
            key = (r["task"], r["arch"], r["N"], SCEN[si])
            for k in ("meas",) + PRED:
                out[key][k].append(a[k][s])
            out[key]["rH"].append(a["rH"][s])
            out[key]["lam2"].append(np.full(s.sum(), r["diag"]["lam2_max"]))
            out[key]["net"].append(np.full(s.sum(), i))
    return {k: {kk: np.concatenate(vv) for kk, vv in v.items()} for k, v in out.items()}


def analyze(nets, label):
    cells = cell_arrays(nets)
    res = dict(forecast={}, decision={}, network={})
    for key, d in sorted(cells.items()):
        k = "|".join(map(str, key))
        res["forecast"][k] = {p: forecast_metrics(d[p], d["meas"]) for p in PRED}
        for hi, H in enumerate(HS):
            fail = d["meas"] <= H
            appB = (d["lam2"] < 1) & (d["rH"][:, hi] >= GAMMA)
            appA = d["E"] > 1.10 * H
            c = dict(policyB=confusion(appB, fail), policyA=confusion(appA, fail),
                     validation=confusion(np.ones_like(fail), fail), auc_E=auc(d["E"], ~fail))
            nB = int(appB.sum())
            rng = np.random.default_rng(0)
            for p in ("NS", "R", "B1", "B2"):
                sc = d[p] + 1e-6 * rng.random(len(fail))
                top = np.argsort(-sc)[:nB]
                c[f"matched_{p}_P_F_given_A"] = float(np.mean(fail[top])) if nB else float("nan")
            res["decision"][f"{k}|H{H}"] = c
    # network level (policy B): worst (scenario, H) P(F|A) per network, and per-network median |log err|
    per = []
    for i, r in enumerate(nets):
        a = r["arr"]
        worst, napp = 0.0, 0
        for si in np.unique(a["scen"]):
            s = a["scen"] == si
            for hi, H in enumerate(HS):
                app = s & (r["diag"]["lam2_max"] < 1) & (a["rH"][:, hi] >= GAMMA)
                napp += int(app.sum())
                if app.any():
                    worst = max(worst, float(np.mean(a["meas"][app] <= H)))
        le = np.abs(log_err(a["E"], a["meas"]))
        per.append(dict(tag=r["tag"], cohort=r["cohort"], worst_P_F_given_A=worst, n_app=napp,
                        median_abs_log_E=float(np.median(le)) if len(le) else float("nan"),
                        lam2=r["diag"]["lam2_max"], exact_resid_max=r["diag"].get("exact_resid_max"),
                        exact_resid_median=r["diag"].get("exact_resid_median"),
                        closure_err_median=(r.get("closure") or {}).get("closure_err_median"),
                        frac_pred_outside=max([v for kk, v in r.items() if kk.startswith("frac_pred_outside")] or [0])))
    w = np.array([p["worst_P_F_given_A"] for p in per])
    res["network"] = dict(n=len(per), median=float(np.median(w)), q75=float(np.percentile(w, 75)),
                          p90=float(np.percentile(w, 90)), p95=float(np.percentile(w, 95)), max=float(w.max()),
                          frac_over_2pct=float(np.mean(w > 0.02)), frac_over_5pct=float(np.mean(w > 0.05)),
                          per_network=per)
    diag_corr = {}
    for dk in ("lam2", "exact_resid_max", "exact_resid_median", "closure_err_median", "frac_pred_outside"):
        v = np.array([p[dk] if p[dk] is not None else np.nan for p in per], float)
        ok = np.isfinite(v)
        if ok.sum() > 5:
            diag_corr[dk] = dict(vs_worst_PFA=float(spearmanr(v[ok], w[ok]).correlation),
                                 vs_median_log_err=float(spearmanr(v[ok], np.array([p["median_abs_log_E"] for p in per])[ok]).correlation))
    res["network"]["diagnostic_spearman"] = diag_corr
    return res


def convergence(nets):
    rows = []
    for r in nets:
        a = r["arr"]
        if "E_tight" not in a:
            continue
        f = (a["meas"] <= T) & (a["E"] <= T) & (a["E_tight"] <= T)
        d = np.abs(np.log(a["E_tight"][f] / a["E"][f])) if f.any() else np.array([np.nan])
        ch = []
        for hi in range(len(HS)):
            b0 = a["rH"][:, hi] >= GAMMA
            b1 = a["rH_tight"][:, hi] >= GAMMA
            ch.append(float(np.mean(b0 != b1)))
        rows.append(dict(tag=r["tag"], median_abs_log_tight_vs_std=float(np.median(d)),
                         max_frac_decisions_changed=max(ch)))
    if not rows:
        return None
    return dict(n=len(rows), median_of_medians=float(np.median([x["median_abs_log_tight_vs_std"] for x in rows])),
                max_decision_change=float(max(x["max_frac_decisions_changed"] for x in rows)),
                mean_decision_change=float(np.mean([x["max_frac_decisions_changed"] for x in rows])), rows=rows)


def oscillation(cohort="osc"):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "results", "prod", cohort, "*.json")))]
    fz = defaultdict(lambda: dict(attempted=0, gate_pass=0, error=0))
    ok = []
    for r in rows:
        c = f"{r['arch']}_N{r['N']}"
        fz[c]["attempted"] += 1
        fz[c]["gate_pass"] += bool(r.get("gate_pass"))
        fz[c]["error"] += r["status"] == "error"
        if r["status"] == "ok" and r.get("gate_pass") and "T_E" in r:
            ok.append(r)
    if not ok:
        return dict(funnel=fz)
    right = [(r["T_meas"] > T) == (r["T_E"] > T) for r in ok]
    fails = [r for r in ok if r["T_meas"] <= T]
    close = [abs(math.log(r["T_E"] / r["T_meas"])) < math.log(1.5) for r in fails]
    b = {k: [abs(math.log(min(r[k], T + 1) / r["T_meas"])) < math.log(1.5) for r in fails] for k in ("T_B1", "T_B2")}
    return dict(funnel=fz, n=len(ok), fail_nofail_correct=float(np.mean(right)), n_fail=len(fails),
                within_1p5=float(np.mean(close)) if fails else float("nan"),
                baselines_within_1p5={k: float(np.mean(v)) if v else float("nan") for k, v in b.items()},
                median_abs_log_E=float(np.median([abs(math.log(r["T_E"] / r["T_meas"])) for r in fails])) if fails else float("nan"))


def criteria(fac, deep, osc):
    crit = {}
    # PC1 accuracy: median over networks of per-network median |log err| <= 0.05 in every factorial cell
    cellnets = defaultdict(list)
    for p in fac["network"]["per_network"]:
        t = p["tag"].split("_")
        cellnets[(t[0], t[1], t[2])].append(p["median_abs_log_E"])
    pc1 = {"|".join(c): float(np.nanmedian(v)) for c, v in cellnets.items()}
    crit["PC1 per-network median |log err| (cell median) <= 0.05 in every factorial cell"] = (
        all(v <= 0.05 for v in pc1.values()), pc1)
    # PC2/PC3 pooled median |log err|: E < NS, and E < R, B1, B2, in every factorial cell x scenario
    pc2, pc3 = {}, {}
    for k, m in fac["forecast"].items():
        if m["E"] is None:
            continue
        e = m["E"]["median_abs_log"]
        pc2[k] = e < m["NS"]["median_abs_log"]
        pc3[k] = all(e < m[b]["median_abs_log"] for b in ("R", "B1", "B2"))
    crit["PC2 E beats naive slow-point NS (pooled median |log err|) in every cell"] = (all(pc2.values()), pc2)
    crit["PC3 E beats R, B1, B2 in every cell"] = (all(pc3.values()), pc3)
    pc4, pc5, pc6le, pc6lt = {}, {}, [], []
    for k, c in fac["decision"].items():
        b = c["policyB"]
        if b["n_app"] >= 50:
            pc4[k] = b["P_F_given_A"] <= 0.02
            base = [c[f"matched_{p}_P_F_given_A"] for p in ("NS", "R", "B1", "B2")]
            pc6le.append(all(b["P_F_given_A"] <= x for x in base))
            pc6lt.append(all(b["P_F_given_A"] < x for x in base))
        if b["n_ok"] >= 50:
            pc5[k] = b["P_A_given_notF"] >= 0.90
    crit["PC4 policy B P(F|A) <= 2% in every (cell,scenario,H) with >=50 approvals"] = (
        all(pc4.values()), {k: v for k, v in pc4.items() if not v})
    crit["PC5 policy B P(A|notF) >= 90% in every (cell,scenario,H) with >=50 ok"] = (
        all(pc5.values()), {k: v for k, v in pc5.items() if not v})
    crit["PC6 matched coverage: <= all baselines in >=90% and < in >=80% of eligible combos"] = (
        np.mean(pc6le) >= 0.9 and np.mean(pc6lt) >= 0.8, dict(le=float(np.mean(pc6le)), lt=float(np.mean(pc6lt)), n=len(pc6le)))
    if osc and "n" in osc:
        crit["PC7 oscillation fail/no-fail >= 90% and failing within 1.5x >= 75%"] = (
            osc["fail_nofail_correct"] >= 0.9 and (osc["n_fail"] == 0 or osc["within_1p5"] >= 0.75),
            dict(correct=osc["fail_nofail_correct"], within=osc["within_1p5"], n_fail=osc["n_fail"]))
    if deep:
        d4 = [c["policyB"]["P_F_given_A"] <= 0.02 for c in deep["decision"].values() if c["policyB"]["n_app"] >= 50]
        d5 = [c["policyB"]["P_A_given_notF"] >= 0.90 for c in deep["decision"].values() if c["policyB"]["n_ok"] >= 50]
        crit["PC8 deep cohort (hold N128, factorial+deep pooled): PC4 and PC5 hold per architecture"] = (
            all(d4) and all(d5), dict(pc4=float(np.mean(d4)) if d4 else None, pc5=float(np.mean(d5)) if d5 else None))
    return crit


if __name__ == "__main__":
    fac_nets, fac_funnel = load_line(["factorial"])
    deep_nets, deep_funnel = load_line(["deep128"])
    fac = analyze(fac_nets, "factorial") if fac_nets else None
    pooled = [r for r in fac_nets if r["task"] == "hold" and r["N"] == 128] + deep_nets
    deep = analyze(pooled, "deep") if deep_nets else None
    osc = oscillation()
    conv = convergence(fac_nets + deep_nets)
    crit = criteria(fac, deep, osc) if fac else {}
    print("FUNNEL (cohort, task, arch, N): attempted / gate_pass / error / forecast")
    for k, v in sorted({**fac_funnel, **deep_funnel}.items()):
        print(f"  {str(k):45s} {v}")
    if osc:
        print("oscillation:", {k: v for k, v in osc.items() if k != "funnel"})
    if conv:
        print("convergence subset:", {k: v for k, v in conv.items() if k != "rows"})
    if fac:
        print("network-level (factorial, policy B):", {k: v for k, v in fac["network"].items() if k not in ("per_network",)})
    print("\nVERDICT (Amendment 8)")
    for k, (v, detail) in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"OVERALL: {'PASS' if crit and all(v for v, _ in crit.values()) else 'FAIL'}")
    json.dump(dict(gamma=GAMMA, funnel={str(k): v for k, v in {**fac_funnel, **deep_funnel}.items()},
                   factorial=fac, deep=deep, oscillation=osc, convergence=conv,
                   criteria={k: dict(passed=bool(v), detail=d) for k, (v, d) in crit.items()}),
              open(os.path.join(HERE, "results", "prod_analysis.json"), "w"), indent=1, default=str)
