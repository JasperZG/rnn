"""Exploratory: effect of input-history closure on a saved network (development only)."""
import sys, numpy as np, torch
from hf_tasks import TASKS
import hf_core as C
task_name, arch, N, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
task = TASKS[task_name]()
m = C.build(arch, task, N, seed); m.load_state_dict(torch.load(f"nets/{task_name}_{arch}_N{N}_s{seed}.pt")); m.eval()
T = 100 * task.T_train
ut = task.inputs(256, T, torch.Generator().manual_seed(9000 + seed))
et, zt = C.native(m, ut, task); mt = task.eval_mask(ut)
meas = C.failure_times(et, mt, task.eps)
for lags in [0, 1, 2, 3]:
    Z, U, D, Uh = C.probe_samples(m, task, seed, lags=lags)
    f = C.DefectFit(task, lags=lags); d = f.fit(Z, U, D, seed, Uh=Uh)
    with torch.no_grad():
        tN = C.failure_times(C.err_norm(C.predict_N(f, task, ut), zt), mt, task.eps)
    r = C.compare(tN, meas, T)
    print(f"lags={lags} r2={np.mean([v['r2'] for v in d.values()]):.3f} agree={r['agree']:.2f} rho={r['spearman']:.2f} logratio={r['abs_log_ratio']:.2f}")
