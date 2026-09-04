"""Time-indexed oscillator trajectories, FULL hidden states, matched frequency.

The saved geo_data.npz keeps only a 2-D projection of one network per architecture,
and the three networks were not commanded the same frequency (LSTM period 20.1 vs
10.1), so they are not a controlled comparison. This run fixes both:

  * FULL hidden state at every timestep, shape (T, N), not a projection
  * SAME commanded frequency for every architecture and seed
  * Jacobian eigenvalues along the orbit, so stability can be read per phase
  * multiple seeds per architecture

LSTM note: its recurrent state is the concatenation (h, c), so `step` carries 2*N
numbers while the Jacobian is taken with respect to that full state. The projection
and radius are computed on the h-part only, which is what the readout sees and what
the existing 2-D projection showed.
"""
import sys, os, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/rnn/src"))
from rnnphase.models import build
from rnnphase.tasks import make_oscillation
from rnnphase.train import train_network, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--freq", type=float, default=0.10)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--tfree", type=int, default=200)
ap.add_argument("--out", default="orbits.json")
args = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, "| commanded freq:", args.freq, flush=True)

GATE = 0.05
N_UNITS, ITERS = 128, 3000
ROT = {"rnn": 0.3, "gru": 0.0, "lstm": 0.0}   # only the vanilla RNN takes rotation_init


def free_run(net, arch, T):
    """Trigger with the task's own pulse, then evolve autonomously. Returns the
    recurrent state at every timestep plus the readout."""
    with torch.no_grad():
        x = torch.zeros(1, T, 1, device=DEV)
        x[:, 0:3, 0] = 1.0
        st = None
        H, Y = [], []
        for t in range(T):
            if st is None:
                z = torch.zeros(1, net.N * (2 if arch == "lstm" else 1), device=DEV)
            else:
                z = st
            st = net.step(z, x[:, t])
            H.append(st[0].clone())
            h = st[0, :net.N] if arch == "lstm" else st[0]
            Y.append(float(net.Wout(h.unsqueeze(0))))
        return torch.stack(H), np.array(Y)


def jac_evals(net, arch, z):
    """Jacobian spectrum of the autonomous map at recurrent state z.

    cuDNN's fused GRU/LSTM kernel refuses to run backward while the module is in
    eval mode, so the Jacobian must be taken with the net in TRAIN mode and the
    fused path disabled. Neither changes the computed map: these models carry no
    dropout or batch-norm, so train and eval evaluate identically.
    """
    was_training = net.training
    net.train()
    z = z.detach().clone().requires_grad_(True)
    u = torch.zeros(1, 1, device=DEV)
    try:
        with torch.backends.cudnn.flags(enabled=False):
            J = torch.autograd.functional.jacobian(
                lambda q: net.step(q.unsqueeze(0), u).squeeze(0), z)
    finally:
        net.train(was_training)
    ev = np.linalg.eigvals(J.detach().cpu().numpy())
    return ev


rows, series = [], []
for arch in ["rnn", "gru", "lstm"]:
    for s in range(args.seeds):
        net = build(arch, 1, 1, N=N_UNITS, seed=s, rotation_init=ROT[arch]).to(DEV)
        tk = dict(freq=args.freq)
        loss = train_network(net, make_oscillation, iters=ITERS, device=DEV, seed=s, task_kwargs=tk)
        ev_loss = float(evaluate(net, make_oscillation, device=DEV, task_kwargs=tk)[0])
        conv = ev_loss < GATE
        H, Y = free_run(net, arch, args.tfree)
        Hn = H.cpu().numpy().astype(np.float32)
        hpart = Hn[:, :N_UNITS]
        # settle: drop the first 100 steps, project the remainder on its own PCs
        tail = hpart[100:]
        mu = tail.mean(0)
        Vt = np.linalg.svd(tail - mu, full_matrices=False)[2]
        proj = (hpart - mu) @ Vt[:3].T
        ang = np.unwrap(np.arctan2(proj[100:, 1], proj[100:, 0]))
        rev = abs(ang[-1] - ang[0]) / (2 * np.pi)
        per = (len(ang) / rev) if rev > 1e-6 else float("nan")
        # eigenvalues at 8 phases around the settled orbit
        pspan = per if (per == per and 2 <= per <= 100) else 10.0
        idx = (100 + np.linspace(0, pspan - 1, 8)).astype(int)
        idx = idx[idx < len(Hn)]
        phase_ev = []
        for k in idx:
            ev = jac_evals(net, arch, H[k])
            mags = np.abs(ev)
            comp = ev[np.abs(ev.imag) > 1e-6]
            lead = float(mags.max())
            fund = None
            if len(comp):
                unst = comp[np.abs(comp) > 1.02]
                pick = unst if len(unst) else comp
                a = np.abs(np.angle(pick))
                fund = float(a[a > 1e-6].min()) if np.any(a > 1e-6) else None
            phase_ev.append(dict(t=int(k), lead=lead, n_unstable=int((mags > 1.0).sum()),
                                 fund_angle=fund,
                                 fund_freq=(fund / (2 * np.pi)) if fund else None))
        rows.append(dict(arch=arch, seed=s, loss=ev_loss, converged=conv,
                         commanded_freq=args.freq, period=per,
                         measured_freq=(1.0 / per) if per == per else None,
                         n_state=int(Hn.shape[1]), evr=[float(v) for v in
                            ((np.linalg.svd(tail - mu, compute_uv=False) ** 2) /
                             (np.linalg.svd(tail - mu, compute_uv=False) ** 2).sum())[:3]],
                         phase_ev=phase_ev))
        series.append(dict(arch=arch, seed=s,
                           proj=[[round(float(v), 5) for v in p] for p in proj],
                           out=[round(float(v), 5) for v in Y]))
        np.savez_compressed("orbit_%s_s%d.npz" % (arch, s),
                            states=Hn, proj=proj.astype(np.float32),
                            out=Y.astype(np.float32))
        print("%-5s s=%d loss=%.4f conv=%s | period %.2f -> freq %.4f (commanded %.3f) | state dim %d"
              % (arch, s, ev_loss, conv, per, 1 / per if per == per else float("nan"),
                 args.freq, Hn.shape[1]), flush=True)

json.dump(dict(rows=rows, series=series), open(args.out, "w"))
print("WROTE", len(rows), "networks")
