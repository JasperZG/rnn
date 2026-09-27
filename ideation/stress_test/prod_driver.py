"""Runs the full Amendment 8 production plan end to end (resumable; rerun to continue).

    python prod_driver.py --workers 12 --threads 2

Order (plan priority): factorial -> deep128 -> osc, each with its preregistered
top-up rule, then aggregation and the preregistered analysis. Progress is written
to logs/prod_driver.log and results/prod/DRIVER_STATUS.json.
On Windows the process asks the OS not to sleep while it runs.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

PLAN = [
    # cohort, tasks, archs, widths, base seeds, top-up seeds, eligible target per cell
    ("factorial", ["hold", "accumulation"], ["rnn", "gru", "lstm"], [32, 128, 512],
     list(range(1000, 1060)), list(range(1060, 1080)), 50),
    ("deep128", ["hold"], ["rnn", "gru", "lstm"], [128],
     list(range(2000, 2060)), list(range(2060, 2080)), 50),
    ("osc", ["oscillation"], ["rnn", "gru", "lstm"], [32, 128],
     list(range(3000, 3036)), list(range(3036, 3048)), 30),
]


def keep_awake():
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


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


def run(cohort, tasks, archs, widths, seeds, workers, threads):
    cmd = [PY, "prod.py", "run", "--name", cohort, "--tasks", *tasks, "--archs", *archs,
           "--widths", *map(str, widths), "--seeds", *map(str, seeds),
           "--workers", str(workers), "--threads", str(threads)]
    log(f"RUN {cohort}: {len(tasks) * len(archs) * len(widths) * len(seeds)} jobs")
    with open(os.path.join(HERE, "logs", f"prod_{cohort}.log"), "a") as f:
        r = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    log(f"RUN {cohort} exited with code {r.returncode}")


def eligible(cohort, task, arch, N, seeds):
    n = 0
    for s in seeds:
        p = os.path.join(HERE, "results", "prod", cohort, f"{task}_{arch}_N{N}_s{s}.json")
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
        run(cohort, tasks, archs, widths, base, workers, threads)
        # preregistered top-up: per cell, add seeds in blocks of 5 until the target
        # number of eligible (gate-passing) networks is reached or the cap is hit
        for t in tasks:
            for a in archs:
                for w in widths:
                    used = list(base)
                    rest = list(topup)
                    while eligible(cohort, t, a, w, used) < target and rest:
                        block, rest = rest[:5], rest[5:]
                        used += block
                        status(phase=cohort, step=f"topup {t}_{a}_N{w}")
                        run(cohort, [t], [a], [w], block, workers, threads)
                    log(f"{cohort} {t}_{a}_N{w}: eligible={eligible(cohort, t, a, w, used)} "
                        f"attempted={len(used)}")
        subprocess.run([PY, "prod.py", "aggregate", "--name", cohort], cwd=HERE)
        status(phase=cohort, step="done")
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
