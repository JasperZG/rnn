"""Consolidate all experiment results into a single summary.json + printed
table -- the numbers to report back for building the submission figures."""
import json, os, glob
import numpy as np

R = "results"
summary = {}

# exp1: structure-match fraction per task
for f in sorted(glob.glob(f"{R}/exp1_*.json")):
    task = os.path.basename(f)[5:-5]
    rows = json.load(open(f))
    conv = [x for x in rows if x.get("converged")]
    rnn_conv = [x for x in conv if x.get("arch") == "rnn"]
    matched = [x for x in rnn_conv if x.get("match")]
    summary.setdefault("structure", {})[task] = {
        "n_total": len(rows), "n_converged": len(conv),
        "n_rnn_analyzed": len(rnn_conv),
        "match_fraction": round(len(matched) / len(rnn_conv), 3) if rnn_conv else None}

# exp2: angle-vs-behavioral frequency agreement
f2 = f"{R}/exp2_frequency.json"
if os.path.exists(f2):
    rows = [x for x in json.load(open(f2)) if x.get("converged")]
    if rows:
        err = [abs(x["angle_freq"] - x["behavioral_freq"]) for x in rows]
        summary["frequency"] = {
            "n": len(rows), "mean_abs_err": round(float(np.mean(err)), 5),
            "max_abs_err": round(float(np.max(err)), 5)}

# exp3: capacity collapse point
f3 = f"{R}/exp3_capacity.json"
if os.path.exists(f3):
    rows = json.load(open(f3)); sizes = sorted({x["N"] for x in rows})
    solve_rate = {N: round(np.mean([x["solved"] for x in rows if x["N"] == N]), 3) for N in sizes}
    summary["capacity"] = {"solve_rate_by_size": solve_rate}

# exp4: x-ray predicted vs observed failure boundary
f4 = f"{R}/exp4_xray.json"
if os.path.exists(f4):
    rows = [x for x in json.load(open(f4)) if x.get("test")]
    if rows:
        hi = [x["extent"][1] for x in rows]
        # observed failure onset = first target where error > 0.1
        onsets = []
        for x in rows:
            crossed = [d["target"] for d in x["test"] if d["error"] > 0.1]
            if crossed: onsets.append(min(crossed))
        summary["xray"] = {
            "n": len(rows),
            "mean_predicted_extent_hi": round(float(np.mean(hi)), 3),
            "mean_observed_failure_onset": round(float(np.mean(onsets)), 3) if onsets else None}

json.dump(summary, open(f"{R}/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print(f"\nwrote {R}/summary.json")
