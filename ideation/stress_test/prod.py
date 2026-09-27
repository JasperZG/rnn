"""Parallel production runner: train -> gate -> extract -> forecast -> score, one
result file per network. Resumable (skips networks whose result already exists).

    # timing benchmark: one network per architecture x width
    python prod.py bench --widths 32 128 512 --workers 3

    # development / production cohort (seeds and sizes come from the protocol)
    python prod.py run --name dev512 --tasks accumulation --archs rnn gru lstm \
        --widths 512 --seeds 500 501 502 --trials 1024 --workers 8

    # oscillation cohort
    python prod.py run --name osc --tasks oscillation --archs rnn gru lstm --widths 32 128 --seeds 600 601

    # pool per-network results into one file for decision3.py
    python prod.py aggregate --name dev512

Training uses CUDA when available (--device auto). Manifold extraction always runs
in float64 on the CPU: consumer GPUs have very slow float64.
Outputs: results/prod/<name>/<tag>.json (+ .npz), weights in nets/prod/<name>/.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HS = (250, 500, 1000, 2000)
SCEN = ("hold", "x1", "x0.5")


def _setup(threads):
    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, HERE)


def naive_estimator(X, m, arch):
    """Same pipeline as the frozen estimator but WITHOUT the invariance solve
    (manifold = constrained minimum-speed points): the earlier, biased method."""
    orig = X.invariant_manifold
    X.invariant_manifold = lambda f, dec, S: (S, float("nan"), float("nan"))
    try:
        return X.AccumulationExact(m, arch)
    finally:
        X.invariant_manifold = orig


def run_accumulation(task, m, arch, seed, trials, device, C, X, rec, out_npz):
    import torch
    from run_stage2 import shifted_inputs
    from run_stage3 import hold_inputs
    t0 = time.time()
    est = X.AccumulationExact(m, arch)
    rec["time_extract_s"] = time.time() - t0
    rec["diag"] = est.diag
    t0 = time.time()
    naive = naive_estimator(X, m, arch)
    rec["time_extract_naive_s"] = time.time() - t0
    Z, U, D, Uh = C.probe_samples(m, task, seed)
    reg = C.DefectFit(task)
    reg.fit(Z, U, D, seed, Uh=Uh)
    T = 100 * task.T_train
    eps = task.eps
    out = {k: [] for k in ("meas", "E", "NS", "R", "B1", "B2", "scen", "rH")}
    tb = tf = 0.0
    for si, name in enumerate(SCEN):
        g = torch.Generator().manual_seed(9000 + 100 * si + seed)
        if name == "x1":
            u = task.inputs(trials, T, g)
            mt = task.eval_mask(u)
        elif name == "x0.5":
            u = shifted_inputs(task, trials, T, g, 0.5)
            mt = task.eval_mask(u)
        else:
            u = hold_inputs(trials, T, g)
            mt = torch.ones(trials, T)
            mt[:, :10] = 0
        t1 = time.time()
        et, zt = C.native(m, u, task, device=device)
        tb += time.time() - t1
        t1 = time.time()
        ep = C.err_norm(est.predict(u), zt)
        tf += time.time() - t1
        out["meas"].append(C.failure_times(et, mt, eps))
        out["E"].append(C.failure_times(ep, mt, eps))
        out["NS"].append(C.failure_times(C.err_norm(naive.predict(u), zt), mt, eps))
        epm = (ep * mt).numpy()
        out["rH"].append(np.stack([(eps - epm[:, :H].max(1)) / eps for H in HS], 1))
        with torch.no_grad():
            out["R"].append(C.failure_times(C.err_norm(C.predict_N(reg, task, u), zt), mt, eps))
        b1, b2 = C.baseline_times(et[:, :task.T_train], mt[:, :task.T_train], eps, T)
        out["B1"].append(b1)
        out["B2"].append(b2)
        out["scen"].append(np.full(trials, si))
    rec["time_bruteforce_s"] = tb
    rec["time_forecast_s"] = tf
    arr = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(out_npz, **arr)
    for si, name in enumerate(SCEN):
        s = arr["scen"] == si
        f = s & (arr["meas"] <= T)
        lr = np.abs(np.log(arr["E"][f] / arr["meas"][f])) if f.any() else np.array([np.nan])
        rec[f"summary_{name}"] = dict(frac_fail=float(np.mean(arr["meas"][s] <= T)),
                                      median_abs_log_err_E=float(np.median(lr)))


def run_oscillation(task, m, arch, seed, device, C, X, rec):
    import torch
    T = 100 * task.T_train
    u = task.inputs(1, T, torch.Generator().manual_seed(9000 + seed))
    et, zt = C.native(m, u, task, device=device)
    mt = task.eval_mask(u)
    t0 = time.time()
    yE, d = X.oscillation_exact(m, arch, task, T)
    rec["time_extract_s"] = time.time() - t0
    b1, b2 = C.baseline_times(et[:, :task.T_train], mt[:, :task.T_train], task.eps, T)
    rec.update(T_meas=float(C.failure_times(et, mt, task.eps)[0]),
               T_E=float(C.failure_times(C.err_norm(yE, zt), mt, task.eps)[0]),
               T_B1=float(b1[0]), T_B2=float(b2[0]),
               err_end_meas=float(et[0, -1]), err_end_E=float(C.err_norm(yE, zt)[0, -1]), **d)


def worker(job):
    name, task_name, arch, N, seed, trials, iters, device, threads = job
    _setup(threads)
    import torch
    from hf_tasks import TASKS
    import hf_core as C
    import hf_exact_FROZEN as X
    tag = f"{task_name}_{arch}_N{N}_s{seed}"
    rdir = os.path.join(HERE, "results", "prod", name)
    ndir = os.path.join(HERE, "nets", "prod", name)
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(ndir, exist_ok=True)
    out_json = os.path.join(rdir, tag + ".json")
    if os.path.exists(out_json):
        return f"skip {tag}"
    dev = device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    rec = dict(tag=tag, task=task_name, arch=arch, N=N, seed=seed, trials=trials, device=dev)
    try:
        task = TASKS[task_name]()
        m = C.build(arch, task, N, seed)
        t0 = time.time()
        rec["train_loss"] = C.train(m, task, seed, iters=iters, device=dev)
        rec["time_train_s"] = time.time() - t0
        m.eval()
        torch.save(m.state_dict(), os.path.join(ndir, tag + ".pt"))
        ug = task.inputs(256, task.T_train, torch.Generator().manual_seed(7000 + seed))
        eg, _ = C.native(m, ug, task, device=dev)
        p95 = float(np.percentile(eg[task.eval_mask(ug) > 0].numpy(), 95))
        rec.update(gate_p95=p95, gate_pass=p95 < task.eps / 2)
        if rec["gate_pass"]:
            if task_name == "accumulation":
                run_accumulation(task, m, arch, seed, trials, dev, C, X, rec, os.path.join(rdir, tag + ".npz"))
            else:
                run_oscillation(task, m, arch, seed, dev, C, X, rec)
        rec["status"] = "ok"
    except Exception:
        rec["status"] = "error"
        rec["error"] = traceback.format_exc()
    json.dump(rec, open(out_json, "w"), indent=1, default=float)
    t = sum(v for k, v in rec.items() if k.startswith("time_") and isinstance(v, float))
    return f"{rec['status']:5s} {tag:30s} gate={'Y' if rec.get('gate_pass') else 'n'} total={t:.0f}s"


def run_jobs(jobs, workers):
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(worker, jobs), 1):
            print(f"[{i}/{len(jobs)} {time.time() - t0:7.0f}s] {msg}", flush=True)


def aggregate(name):
    """Pool per-network npz files into results/cases3_prod_<name>.npz (decision3.py format)."""
    rdir = os.path.join(HERE, "results", "prod", name)
    rec = {k: [] for k in ("meas", "E", "NS", "R", "B1", "B2", "net", "scen", "lam2", "rH")}
    nets, funnel = [], dict(attempted=0, gate_pass=0, error=0, extracted=0)
    for p in sorted(glob.glob(os.path.join(rdir, "accumulation_*.json"))):
        r = json.load(open(p))
        funnel["attempted"] += 1
        funnel["error"] += r["status"] == "error"
        funnel["gate_pass"] += bool(r.get("gate_pass"))
        npz = p[:-5] + ".npz"
        if r["status"] != "ok" or not r.get("gate_pass") or not os.path.exists(npz):
            continue
        funnel["extracted"] += 1
        d = np.load(npz)
        n = len(d["meas"])
        for k in ("meas", "E", "NS", "R", "B1", "B2", "scen", "rH"):
            rec[k].append(d[k])
        rec["net"].append(np.full(n, len(nets)))
        rec["lam2"].append(np.full(n, r["diag"]["lam2_max"]))
        nets.append(r["tag"])
    out = {k: np.concatenate(v) for k, v in rec.items() if v}
    np.savez(os.path.join(HERE, "results", f"cases3_prod_{name}.npz"), nets=np.array(nets), **out)
    print("funnel:", funnel, " -> results/cases3_prod_%s.npz" % name)


def bench_table(name):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "results", "prod", name, "*.json")))]
    print(f"\n{'network':30s} {'gate':>4s} {'train':>7s} {'extract':>8s} {'naive':>7s} {'brute':>7s} {'forecast':>8s}  status")
    for r in rows:
        g = lambda k: f"{r[k]:7.1f}" if k in r else "     - "
        print(f"{r['tag']:30s} {'Y' if r.get('gate_pass') else 'n':>4s} {g('time_train_s')} {g('time_extract_s')} "
              f" {g('time_extract_naive_s')} {g('time_bruteforce_s')} {g('time_forecast_s')}  {r['status']}")
    print("(seconds; brute/forecast are for all 3 scenarios x --trials trials x 5000 steps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "run", "aggregate"])
    ap.add_argument("--name", default="bench")
    ap.add_argument("--tasks", nargs="+", default=["accumulation"])
    ap.add_argument("--archs", nargs="+", default=["rnn", "gru", "lstm"])
    ap.add_argument("--widths", nargs="+", type=int, default=[32, 128, 512])
    ap.add_argument("--seeds", nargs="+", type=int, default=[900])
    ap.add_argument("--trials", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per worker")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.cmd == "aggregate":
        aggregate(a.name)
        sys.exit()
    if a.cmd == "bench":
        a.name, a.trials = "bench", 256
    jobs = [(a.name, t, ar, w, s, a.trials, a.iters, a.device, a.threads)
            for t in a.tasks for ar in a.archs for w in a.widths for s in a.seeds]
    # largest jobs first so the long ones start early
    jobs.sort(key=lambda j: -(j[3] * (2 if j[2] == "lstm" else 1)))
    print(f"{len(jobs)} jobs, {a.workers} workers, device={a.device}")
    run_jobs(jobs, a.workers)
    if a.cmd == "bench":
        bench_table(a.name)
