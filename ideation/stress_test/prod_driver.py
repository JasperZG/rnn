"""Runs the full production plan end to end (Amendments 8-10; resumable; rerun to continue).

    python prod_driver.py --workers 12 --threads 2

Order: factorial -> deep128 -> osc (confirmatory, untouched networks only),
then v2_repair (the 65 networks whose outcomes were inspected before the
Amendment 9 correction; reported separately, never counted as confirmatory),
then aggregation and the preregistered analysis.

Amendment 10 replacement rule (factorial): every attempt in the repair set is
replaced one-for-one, in seed order within its cell, by a fresh seed from 1100
upward. The eligible-network target (50 per cell) and the top-up rule (seeds
1060-1079, blocks of 5) count only untouched networks.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
REPAIR = [tuple(x) for x in json.load(open(os.path.join(HERE, "results", "REPAIR_SET.json")))]
REPAIR_TAGS = {f"{t}_{a}_N{w}_s{s}" for t, a, w, s in REPAIR}

PLAN = [
    # cohort, tasks, archs, widths, base seeds, top-up seeds, eligible target per cell
    ("factorial", ["hold", "accumulation"], ["rnn", "gru", "lstm"], [32, 128, 512],
     list(range(1000, 1060)), list(range(1060, 1080)), 50),
    ("deep128", ["hold"], ["rnn", "gru", "lstm"], [128],
     list(range(2000, 2060)), list(range(2060, 2080)), 50),
    ("osc", ["oscillation"], ["rnn", "gru", "lstm"], [32, 128],
     list(range(3000, 3036)), list(range(3036, 3048)), 30),
]
REPLACEMENT_START = 1100


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, "logs", "prod_driver.log"), "a") as f:
        f.write(line + "\n")


def status(**kw):
    p = os.path.join(HERE, "results", "prod", "DRIVER_STATUS.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    s = json.load(open(p)) if os.path.exists(p) else {}
    s.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(s, open(p, "w"), indent=1)


def run_jobs(cohort, jobs, workers, threads):
    if not jobs:
        return
    jf = os.path.join(HERE, "results", "prod", f"_jobs_{cohort}.json")
    json.dump(jobs, open(jf, "w"))
    log(f"RUN {cohort}: {len(jobs)} jobs")
    cmd = [PY, "prod.py", "run", "--name", cohort, "--jobs-file", jf,
           "--workers", str(workers), "--threads", str(threads)]
    with open(os.path.join(HERE, "logs", f"prod_{cohort}.log"), "a") as f:
        r = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    log(f"RUN {cohort} exited with code {r.returncode}")


def cell_seeds(cohort, t, a, w, base):
    """Base attempts for a cell, with repair-set attempts replaced one-for-one."""
    if cohort != "factorial":
        return list(base)
    keep = [s for s in base if f"{t}_{a}_N{w}_s{s}" not in REPAIR_TAGS]
    n_rep = len(base) - len(keep)
    return keep + list(range(REPLACEMENT_START, REPLACEMENT_START + n_rep))


def eligible(cohort, t, a, w, seeds):
    n = 0
    for s in seeds:
        p = os.path.join(HERE, "results", "prod", cohort, f"{t}_{a}_N{w}_s{s}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            n += bool(r.get("gate_pass")) and r["status"] == "ok"
    return n


def main(workers, threads):
    keep_awake()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log(f"driver start: workers={workers} threads={threads}")
    for cohort, tasks, archs, widths, base, topup, target in PLAN:
        status(phase=cohort, step="base")
        cells = [(t, a, w) for t in tasks for a in archs for w in widths]
        seeds = {c: cell_seeds(cohort, *c, base) for c in cells}
        run_jobs(cohort, [[t, a, w, s] for (t, a, w) in cells for s in seeds[(t, a, w)]], workers, threads)
        for (t, a, w) in cells:
            used, rest = list(seeds[(t, a, w)]), list(topup)
            while eligible(cohort, t, a, w, used) < target and rest:
                block, rest = rest[:5], rest[5:]
                used += block
                status(phase=cohort, step=f"topup {t}_{a}_N{w}")
                run_jobs(cohort, [[t, a, w, s] for s in block], workers, threads)
            log(f"{cohort} {t}_{a}_N{w}: eligible={eligible(cohort, t, a, w, used)} attempted={len(used)}")
        subprocess.run([PY, "prod.py", "aggregate", "--name", cohort], cwd=HERE)
        status(phase=cohort, step="done")
    status(phase="v2_repair", step="running")
    run_jobs("v2_repair", [list(r) for r in REPAIR], workers, threads)
    status(phase="analysis", step="running")
    with open(os.path.join(HERE, "logs", "prod_analysis.log"), "w") as f:
        subprocess.run([PY, "analyze_prod.py"], cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    status(phase="complete", step="done")
    log("driver complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    main(a.workers, a.threads)
