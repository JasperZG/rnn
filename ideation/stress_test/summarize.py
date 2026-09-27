"""Summarize a stage's results against the pre-registered pass rule."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
stage = sys.argv[1] if len(sys.argv) > 1 else "stage1"
recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "results", stage, "*.json")))]
print(f"{len(recs)} networks in {stage}\n")

cells = defaultdict(list)
for r in recs:
    cells[(r["task"], r["arch"])].append(r)

hdr = f"{'task':13s}{'arch':6s}{'gated':>7s}{'fail%':>7s}" + "".join(
    f"{k + ' agree':>10s}" for k in ("N", "L", "B1", "B2")) + f"{'rho_N':>8s}{'cellpass':>9s}"
print(hdr)
print("-" * len(hdr))
task_pass = defaultdict(lambda: True)
task_best = defaultdict(lambda: defaultdict(list))
for (task, arch), rs in sorted(cells.items()):
    g = [r for r in rs if r["gate_pass"]]
    if not g:
        print(f"{task:13s}{arch:6s}{0:>4d}/{len(rs):<2d}  (no gated networks)")
        if arch != "tc":
            task_pass[task] = False
        continue
    ag = {k: np.array([r["metrics"][k]["agree"] for r in g]) for k in ("N", "L", "B1", "B2")}
    rho = [r["metrics"]["N"]["spearman"] for r in g
           if r["metrics"]["N"]["frac_fail_meas"] >= 0.2 and r["metrics"]["N"]["spearman"] == r["metrics"]["N"]["spearman"]]
    fail = np.mean([r["frac_fail_meas"] for r in g])
    cellpass = np.mean(ag["N"] >= 0.75) >= 0.75
    if arch != "tc":
        task_pass[task] &= bool(cellpass)
        for k in ag:
            task_best[task][k].extend(ag[k].tolist())
    print(f"{task:13s}{arch:6s}{len(g):>4d}/{len(rs):<2d}{100 * fail:>7.0f}" + "".join(
        f"{ag[k].mean():>10.2f}" for k in ag) + f"{(np.median(rho) if rho else float('nan')):>8.2f}"
        f"{'PASS' if cellpass else 'FAIL':>9s}")

print("\nPass rule per task (standard archs only):")
overall = True
for task in sorted(task_best):
    b = {k: np.mean(v) for k, v in task_best[task].items()}
    beats = b["N"] > b["B1"] and b["N"] > b["B2"]
    ok = task_pass[task] and beats
    overall &= ok
    print(f"  {task:13s} cells {'PASS' if task_pass[task] else 'FAIL'}; N={b['N']:.2f} "
          f"B1={b['B1']:.2f} B2={b['B2']:.2f} beats-baselines={'yes' if beats else 'NO'}  -> "
          f"{'PASS' if ok else 'FAIL'}")
print(f"\nSTAGE VERDICT: {'PASS' if overall else 'FAIL'}")
