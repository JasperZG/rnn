"""Production runner (Amendment 8). One result file per network; resumable.

    python prod.py bench --widths 32 128 512 --workers 3
    python prod.py run --name factorial --tasks hold accumulation --archs rnn gru lstm \
        --widths 32 128 512 --seeds 1000-1059 --workers 8
    python prod.py aggregate --name factorial

Per network (hold / accumulation):
  train (GPU) -> training-horizon gate -> frozen estimator E (CPU float64) ->
  naive slow-point estimator NS -> short-probe regression R -> extrapolation B1/B2
  -> trial banks (hold nets: 1024 hold trials; accumulation nets: 1024 normal +
  1024 weaker-input trials) -> native long rollout (T=5000) -> predictions,
  forecast safety margins r_H, validity diagnostics, closure test (driven nets),
  tight-numerics re-extraction for the preselected 10% subset (seed % 10 == 3).
Oscillation networks: E (phase slip) + B1/B2 on the autonomous pulse response.
Extraction always runs in float64 on the CPU (consumer GPUs have slow float64).
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HS = (250, 500, 1000, 2000)
SCEN = ("hold", "x1", "x0.5")          # scenario index used in all result files
T_TEST = 5000


def train_cfg(task_name, arch, N):
    """Training hyperparameters (iterations, learning rate), fixed in Amendment 8."""
    if os.environ.get("HF_ITERS_OVERRIDE"):          # smoke tests only; recorded in every result
        return int(os.environ["HF_ITERS_OVERRIDE"]), float(os.environ.get("HF_LR_OVERRIDE", 2e-3))
    if arch == "rnn" and (task_name == "hold" or N >= 512):
        return 8000, 5e-4        # vanilla RNN: hold task and width 512 (development seeds 510-527)
    return 4000, 2e-3            # GRU/LSTM everywhere; vanilla accumulation N<=128; oscillation


def conv_subset(seed):
    return seed % 10 == 3


GPU_SEM = None


def _init_pool(sem):
    """Pool initializer: share a semaphore limiting concurrent GPU work (a
    DPC_WATCHDOG bugcheck occurred on the production PC with 8 concurrent GPU jobs)."""
    global GPU_SEM
    GPU_SEM = sem


class _gpu:
    def __init__(self, device):
        self.on = device.startswith("cuda") and GPU_SEM is not None

    def __enter__(self):
        if self.on:
            GPU_SEM.acquire()

    def __exit__(self, *a):
        if self.on:
            GPU_SEM.release()


def resolve_device(device, arch, N):
    """auto: vanilla RNNs with N <= 128 train and roll out on the CPU (their per-step
    loop is launch-bound on the GPU, ~10x slower there); everything else on CUDA."""
    import torch
    if device != "auto":
        return device
    if arch == "rnn" and N <= 128:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _setup(threads):
    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, HERE)


def _patched(X, fn):
    """Temporarily replace X.invariant_manifold (the frozen module looks it up by name)."""
    orig = X.invariant_manifold

    class Ctx:
        def __enter__(self):
            X.invariant_manifold = fn(orig)

        def __exit__(self, *a):
            X.invariant_manifold = orig
    return Ctx()


def estimators(X, m, arch):
    """Estimator E (capturing its final, converged manifold points) and naive NS
    (same pipeline with neither the invariance solve nor the convergence check)."""
    cap = {}
    orig_cb = X.converged_block

    def cb_capture(fn, dec, H, tol):
        idx, nd, cmap = orig_cb(fn, dec, H, tol)
        cap["H"] = H[idx]
        return idx, nd, cmap
    X.converged_block = cb_capture
    try:
        t0 = time.time()
        est = X.AccumulationExact(m, arch)
        t_e = time.time() - t0
    finally:
        X.converged_block = orig_cb
    X.converged_block = lambda fn, dec, H, tol: (__import__("torch").arange(len(H)), 0, "")
    try:
        with _patched(X, lambda orig: (lambda fn, dec, S, **kw: (S, float("nan"), float("nan")))):
            t0 = time.time()
            naive = X.AccumulationExact(m, arch)
            t_n = time.time() - t0
    finally:
        X.converged_block = orig_cb
    return est, naive, cap["H"], t_e, t_n


def exact_residual(X, m, arch, H):
    """Exact discrete invariance residual f(h(s)) - h(s+v(s)) at held-out manifold points."""
    import torch
    from scipy.interpolate import CubicSpline
    f, dec, dim = X.stepper(m, arch)
    with torch.no_grad():
        s = dec(H)[:, 0].numpy()
        Hn = H.numpy()
        fi, ho = np.arange(0, len(s), 2), np.arange(1, len(s) - 1, 2)
        spl = CubicSpline(s[fi], Hn[fi], axis=0)
        Fh = f(torch.as_tensor(Hn[ho]), torch.zeros(len(ho), 1, dtype=torch.float64))
        v = dec(Fh)[:, 0].numpy() - s[ho]
        res = np.linalg.norm(Fh.numpy() - spl(s[ho] + v), axis=1)
    m.float()
    return dict(exact_resid_median=float(np.median(res)), exact_resid_max=float(res.max()),
                exact_resid_p90=float(np.percentile(res, 90)))


def tight_estimator(X, m, arch):
    """Tighter numerics: 2x manifold points, 2x input grid, 10000 L-BFGS iterations."""
    with _patched(X, lambda orig: (lambda fn, dec, S, **kw: orig(fn, dec, S, iters=10000))):
        return X.AccumulationExact(m, arch, n_pts=801, n_u=113)


def closure_test(task, m, est, seed, device, C):
    """Paired histories reaching (nearly) the same decoded state, then an identical
    continuation: does the network diverge as the 1-D reduced model says?"""
    import torch
    g = torch.Generator().manual_seed(4000 + seed)
    B, L1, L2 = 2048, 30, 20
    hist = task.inputs(B, L1, g)
    with torch.no_grad(), _gpu(device):
        m.to(device)
        z30 = m(hist.to(device))[0][:, -1, 0].cpu().numpy()
        m.to("cpu")
    order = np.argsort(z30)
    pairs = [(order[i], order[i + 1]) for i in range(0, B - 1, 2)
             if abs(z30[order[i + 1]] - z30[order[i]]) < 0.005]
    if len(pairs) < 20:
        return dict(closure_pairs=len(pairs))
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    cont = task.inputs(len(pairs), L2, g)
    ua = torch.cat([hist[a], cont], 1)
    ub = torch.cat([hist[b], cont], 1)
    with torch.no_grad(), _gpu(device):
        m.to(device)
        ya = m(ua.to(device))[0][:, -1, 0].cpu().numpy()
        yb = m(ub.to(device))[0][:, -1, 0].cpu().numpy()
        m.to("cpu")
    pa = est.predict(ua)[:, -1, 0].numpy()
    pb = est.predict(ub)[:, -1, 0].numpy()
    err = np.abs((ya - yb) - (pa - pb))
    drift = np.abs(ya - (z30[a] + cont.sum(1)[:, 0].numpy()))
    return dict(closure_pairs=len(pairs), closure_err_median=float(np.median(err)),
                closure_err_p95=float(np.percentile(err, 95)),
                closure_ref_20step_error_median=float(np.median(drift)))


def scenarios(task, trials, seed):
    import torch
    from run_stage2 import shifted_inputs
    out = []
    if task.name == "hold":
        g = torch.Generator().manual_seed(9000 + seed)
        u = task.inputs(trials, T_TEST, g)
        out.append((0, u, task.eval_mask(u)))
    else:
        g = torch.Generator().manual_seed(9100 + seed)
        u = task.inputs(trials, T_TEST, g)
        out.append((1, u, task.eval_mask(u)))
        g = torch.Generator().manual_seed(9200 + seed)
        u = shifted_inputs(task, trials, T_TEST, g, 0.5)
        out.append((2, u, task.eval_mask(u)))
    return out


def run_line(task, m, arch, seed, trials, device, C, X, rec, out_npz):
    import torch
    est, naive, H, t_e, t_n = estimators(X, m, arch)
    rec.update(time_extract_s=t_e, time_extract_naive_s=t_n, diag=est.diag)
    rec["diag"].update(exact_residual(X, m, arch, H))
    tight = None
    if conv_subset(seed):
        t0 = time.time()
        tight = tight_estimator(X, m, arch)
        rec["time_extract_tight_s"] = time.time() - t0
        rec["diag_tight"] = tight.diag
    Z, U, D, Uh = C.probe_samples(m, task, seed)
    reg = C.DefectFit(task)
    reg.fit(Z, U, D, seed, Uh=Uh)
    eps = task.eps
    out = {k: [] for k in ("meas", "E", "NS", "R", "B1", "B2", "scen", "rH")}
    if tight is not None:
        out.update(E_tight=[], rH_tight=[])
    tb = tf = 0.0
    rec["input_bank_sha256"] = {}
    for si, u, mt in scenarios(task, trials, seed):
        rec["input_bank_sha256"][SCEN[si]] = hashlib.sha256(u.numpy().tobytes()).hexdigest()
        t1 = time.time()
        with _gpu(device):
            et, zt = C.native(m, u, task, device=device)
        tb += time.time() - t1
        t1 = time.time()
        yE = est.predict(u)
        tf += time.time() - t1
        ep = C.err_norm(yE, zt)
        out["meas"].append(C.failure_times(et, mt, eps))
        out["E"].append(C.failure_times(ep, mt, eps))
        epm = (ep * mt).numpy()
        out["rH"].append(np.stack([(eps - epm[:, :h].max(1)) / eps for h in HS], 1))
        out["NS"].append(C.failure_times(C.err_norm(naive.predict(u), zt), mt, eps))
        with torch.no_grad():
            out["R"].append(C.failure_times(C.err_norm(C.predict_N(reg, task, u), zt), mt, eps))
        b1, b2 = C.baseline_times(et[:, :task.T_train], mt[:, :task.T_train], eps, T_TEST)
        out["B1"].append(b1)
        out["B2"].append(b2)
        out["scen"].append(np.full(trials, si))
        yv = yE[..., 0].numpy()
        rec[f"frac_pred_outside_manifold_{SCEN[si]}"] = float(np.mean(
            (yv.min(1) < est.lo) | (yv.max(1) > est.hi)))
        if tight is not None:
            ept = C.err_norm(tight.predict(u), zt)
            out["E_tight"].append(C.failure_times(ept, mt, eps))
            eptm = (ept * mt).numpy()
            out["rH_tight"].append(np.stack([(eps - eptm[:, :h].max(1)) / eps for h in HS], 1))
    rec.update(time_bruteforce_s=tb, time_forecast_s=tf)
    if task.name == "accumulation":
        rec["closure"] = closure_test(task, m, est, seed, device, C)
    arr = {k: np.concatenate(v) for k, v in out.items()}
    # failure times are integer steps (censored = T_TEST + 1); margins as float32
    store = {k: (v.astype(np.float32) if k.startswith("rH") else v.astype(np.int32)) for k, v in arr.items()}
    np.savez_compressed(out_npz, **store)
    f = arr["meas"] <= T_TEST
    lr = np.abs(np.log(arr["E"][f] / arr["meas"][f])) if f.any() else np.array([np.nan])
    rec["summary"] = dict(frac_fail=float(f.mean()), median_abs_log_err_E=float(np.median(lr)))


def run_oscillation(task, m, arch, seed, device, C, X, rec):
    import torch
    u = task.inputs(1, T_TEST, torch.Generator().manual_seed(9000 + seed))
    with _gpu(device):
        et, zt = C.native(m, u, task, device=device)
    mt = task.eval_mask(u)
    t0 = time.time()
    yE, d = X.oscillation_exact(m, arch, task, T_TEST)
    rec["time_extract_s"] = time.time() - t0
    b1, b2 = C.baseline_times(et[:, :task.T_train], mt[:, :task.T_train], task.eps, T_TEST)
    eE = C.err_norm(yE, zt)
    rec.update(T_meas=float(C.failure_times(et, mt, task.eps)[0]),
               T_E=float(C.failure_times(eE, mt, task.eps)[0]),
               T_B1=float(b1[0]), T_B2=float(b2[0]),
               err_end_meas=float(et[0, -1]), err_end_E=float(eE[0, -1]), **d)


def worker(job):
    name, task_name, arch, N, seed, trials, device, threads = job
    _setup(threads)
    import torch
    from hf_tasks import TASKS
    import hf_core as C
    import hf_exact_v2 as X            # Amendment 9: corrected construction
    tag = f"{task_name}_{arch}_N{N}_s{seed}"
    rdir = os.path.join(HERE, "results", "prod", name)
    ndir = os.path.join(HERE, "nets", "prod", name)
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(ndir, exist_ok=True)
    out_json = os.path.join(rdir, tag + ".json")
    if os.path.exists(out_json):
        return f"skip  {tag}"
    dev = resolve_device(device, arch, N)
    iters, lr = train_cfg(task_name, arch, N)
    rec = dict(tag=tag, cohort=name, task=task_name, arch=arch, N=N, seed=seed, trials=trials,
               device=dev, iters=iters, lr=lr, conv_subset=conv_subset(seed))
    try:
        task = TASKS[task_name]()
        m = C.build(arch, task, N, seed)
        t0 = time.time()
        with _gpu(dev):
            rec["train_loss"] = C.train(m, task, seed, iters=iters, lr=lr, device=dev)
        rec["time_train_s"] = time.time() - t0
        m.eval()
        torch.save(m.state_dict(), os.path.join(ndir, tag + ".pt"))
        ug = task.inputs(256, task.T_train, torch.Generator().manual_seed(7000 + seed))
        with _gpu(dev):
            eg, _ = C.native(m, ug, task, device=dev)
        p95 = float(np.percentile(eg[task.eval_mask(ug) > 0].numpy(), 95))
        rec.update(gate_p95=p95, gate_pass=p95 < task.eps / 2)
        if rec["gate_pass"] and not os.environ.get("HF_GATE_ONLY"):   # HF_GATE_ONLY: development only
            if task_name in ("hold", "accumulation"):
                run_line(task, m, arch, seed, trials, dev, C, X, rec, os.path.join(rdir, tag + ".npz"))
            else:
                run_oscillation(task, m, arch, seed, dev, C, X, rec)
        rec["status"] = "ok"
    except Exception:
        rec["status"] = "error"
        rec["error"] = traceback.format_exc()
    json.dump(rec, open(out_json, "w"), indent=1, default=float)
    t = sum(v for k, v in rec.items() if k.startswith("time_") and isinstance(v, float))
    return f"{rec['status']:5s} {tag:30s} gate={'Y' if rec.get('gate_pass') else 'n'} p95={rec.get('gate_p95', float('nan')):.3f} total={t:.0f}s"


def run_jobs(jobs, workers, gpu_slots=3):
    ctx = mp.get_context("spawn")
    t0 = time.time()
    sem = ctx.Semaphore(gpu_slots)
    with ctx.Pool(workers, initializer=_init_pool, initargs=(sem,), maxtasksperchild=1) as pool:
        for i, msg in enumerate(pool.imap_unordered(worker, jobs), 1):
            print(f"[{i}/{len(jobs)} {time.time() - t0:7.0f}s] {msg}", flush=True)


def aggregate(name):
    rdir = os.path.join(HERE, "results", "prod", name)
    rec = {k: [] for k in ("meas", "E", "NS", "R", "B1", "B2", "net", "scen", "lam2", "rH")}
    nets, funnel = [], {}
    for p in sorted(glob.glob(os.path.join(rdir, "*.json"))):
        r = json.load(open(p))
        if r["task"] == "oscillation":
            continue
        cell = f"{r['task']}_{r['arch']}_N{r['N']}"
        fz = funnel.setdefault(cell, dict(attempted=0, gate_pass=0, error=0, forecast=0))
        fz["attempted"] += 1
        fz["error"] += r["status"] == "error"
        fz["gate_pass"] += bool(r.get("gate_pass"))
        npz = p[:-5] + ".npz"
        if r["status"] != "ok" or not r.get("gate_pass") or not os.path.exists(npz):
            continue
        fz["forecast"] += 1
        d = np.load(npz)
        n = len(d["meas"])
        for k in ("meas", "E", "NS", "R", "B1", "B2", "scen", "rH"):
            rec[k].append(d[k])
        rec["net"].append(np.full(n, len(nets)))
        rec["lam2"].append(np.full(n, r["diag"]["lam2_max"]))
        nets.append(r["tag"])
    out = {k: np.concatenate(v) for k, v in rec.items() if v}
    np.savez_compressed(os.path.join(HERE, "results", f"cases3_prod_{name}.npz"), nets=np.array(nets), **out)
    json.dump(funnel, open(os.path.join(HERE, "results", f"funnel_prod_{name}.json"), "w"), indent=1)
    for c, f in sorted(funnel.items()):
        print(f"  {c:28s} {f}")


def parse_seeds(items):
    out = []
    for it in items:
        if "-" in str(it):
            a, b = map(int, str(it).split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(it))
    return out


def make_jobs(name, tasks, archs, widths, seeds, trials, device, threads):
    jobs = [(name, t, ar, w, s, trials, device, threads)
            for t in tasks for ar in archs for w in widths for s in seeds]
    jobs.sort(key=lambda j: -(j[3] * (2 if j[2] == "lstm" else 1)))   # big jobs first
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "run", "aggregate"])
    ap.add_argument("--name", default="bench")
    ap.add_argument("--tasks", nargs="+", default=["accumulation"])
    ap.add_argument("--archs", nargs="+", default=["rnn", "gru", "lstm"])
    ap.add_argument("--widths", nargs="+", type=int, default=[32, 128, 512])
    ap.add_argument("--seeds", nargs="+", default=["900"])
    ap.add_argument("--trials", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per worker")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpu-slots", type=int, default=3, help="max concurrent GPU jobs")
    ap.add_argument("--jobs-file", default=None, help="JSON list of [task, arch, N, seed]; overrides the grid")
    a = ap.parse_args()
    # pin BLAS/OpenMP threads per worker (inherited by spawned workers) to avoid oversubscription
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    if a.cmd == "aggregate":
        aggregate(a.name)
        sys.exit()
    if a.cmd == "bench":
        a.name, a.trials = "bench", 256
    if a.jobs_file:
        spec = json.load(open(a.jobs_file))
        jobs = [(a.name, t, ar, int(w), int(sd), a.trials, a.device, a.threads) for t, ar, w, sd in spec]
        jobs.sort(key=lambda j: -(j[3] * (2 if j[2] == "lstm" else 1)))
    else:
        jobs = make_jobs(a.name, a.tasks, a.archs, a.widths, parse_seeds(a.seeds), a.trials, a.device, a.threads)
    print(f"{len(jobs)} jobs, {a.workers} workers, device={a.device}", flush=True)
    run_jobs(jobs, a.workers, a.gpu_slots)
