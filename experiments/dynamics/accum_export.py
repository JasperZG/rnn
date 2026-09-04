"""Single-network mechanistic export for the accumulation line attractor.

Writes ONE .npz per network carrying everything a six-panel mechanistic figure
needs, so no panel has to be approximated:

  behaviour     inc, running-sum target, network output, hidden trajectory
  geometry      recovered slow points, autonomous speeds, PC basis of the point set
  readout       Wout weight vector, and the readout value AT every recovered point
  spectrum      Jacobian eigenvalues AND eigenvectors at every recovered point
  causality     autonomous trajectories after norm-matched TANGENT vs ORTHOGONAL
                displacement, readout tracked at every timestep

The causal arm is the part no existing file has. The ablation pilot removes a
direction from the recurrent dynamics DURING the task; this instead displaces the
state once and lets it run free, which is what tests whether the manifold
direction is the one that stores the value. Both displacements use the same L2
norm, so only direction differs.

Readout note: Wout is nn.Linear(N, 1, bias=False), so represented value is exactly
w . h with no offset -- manifold position maps to readout value directly.

cuDNN note: the fused kernel refuses backward in eval mode, so Jacobians are taken
in train mode with cudnn disabled. These models carry no dropout or batch-norm, so
train and eval evaluate identically.
"""
import sys, os, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/rnn/src"))
from rnnphase.models import build
from rnnphase.tasks import make_accumulation
from rnnphase.train import train_network, evaluate
from rnnphase.fixed_points import find_slow_points

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--free", type=int, default=120)
ap.add_argument("--out", default="accum_export")
args = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_UNITS, ITERS, GATE, T_TASK = 128, 3000, 0.01, 80
print("device:", DEV, flush=True)


def jac_at(net, h):
    was = net.training
    net.train()
    z = h.detach().clone().requires_grad_(True)
    u = torch.zeros(1, 1, device=DEV)
    try:
        with torch.backends.cudnn.flags(enabled=False):
            J = torch.autograd.functional.jacobian(
                lambda q: net.step(q.unsqueeze(0), u).squeeze(0), z)
    finally:
        net.train(was)
    Jn = J.detach().cpu().numpy()
    ev, evec = np.linalg.eig(Jn)
    o = np.argsort(-np.abs(ev))
    return Jn, ev[o], evec[:, o]


def free_run_from(net, h0, T):
    with torch.no_grad():
        h = h0.clone().unsqueeze(0)
        u = torch.zeros(1, 1, device=DEV)
        H, Y = [], []
        for _ in range(T):
            h = net.step(h, u)
            H.append(h.squeeze(0).clone())
            Y.append(float(net.Wout(h)))
        return torch.stack(H).cpu().numpy().astype(np.float32), np.array(Y, dtype=np.float32)


meta = []
for s in range(args.seeds):
    net = build("rnn", 1, 1, N=N_UNITS, seed=s).to(DEV)
    train_network(net, make_accumulation, iters=ITERS, device=DEV, seed=s)
    ev_loss = float(evaluate(net, make_accumulation, device=DEV)[0])
    conv = ev_loss < GATE
    print("seed %d loss %.5f conv %s" % (s, ev_loss, conv), flush=True)

    g = torch.Generator(device=DEV).manual_seed(1000 + s)
    inc, tgt, _ = make_accumulation(8, T=T_TASK, device=DEV, g=g)
    with torch.no_grad():
        out, H_task = net(inc)

    with torch.no_grad():
        inc2, _, _ = make_accumulation(128, T=T_TASK, device=DEV,
                                       g=torch.Generator(device=DEV).manual_seed(7))
        _, Hbig = net(inc2)
    Hv = Hbig.reshape(-1, N_UNITS)
    pts, sp, _ = find_slow_points(net, Hv, n_in=1, n_seed=400, steps=1500,
                                  speed_tol=1e-4, dedup=0.5, device=DEV)
    P = np.asarray(pts.detach().cpu() if torch.is_tensor(pts) else pts, dtype=np.float64)
    SP = np.asarray(sp.detach().cpu() if torch.is_tensor(sp) else sp, dtype=np.float32)
    print("  recovered %d points" % len(P), flush=True)
    if len(P) < 3:
        meta.append(dict(seed=s, loss=ev_loss, converged=bool(conv), n_pts=int(len(P)), usable=False))
        continue

    mu = P.mean(0)
    Vt = np.linalg.svd(P - mu, full_matrices=False)[2]
    sv = np.linalg.svd(P - mu, compute_uv=False) ** 2
    evr = (sv / sv.sum()).astype(np.float32)
    tangent = Vt[0] / np.linalg.norm(Vt[0])
    coord = (P - mu) @ tangent
    order = np.argsort(coord)

    W = net.Wout.weight.detach().cpu().numpy().astype(np.float32).ravel()
    read_at_pts = (P @ W).astype(np.float32)

    EV = np.zeros((len(P), N_UNITS), dtype=np.complex64)
    EVEC = np.zeros((len(P), N_UNITS, 8), dtype=np.complex64)
    for k in range(len(P)):
        _, ev, evec = jac_at(net, torch.tensor(P[k], dtype=torch.float32, device=DEV))
        EV[k] = ev.astype(np.complex64)
        EVEC[k] = evec[:, :8].astype(np.complex64)
    mid = int(order[len(order) // 2])
    Jmid, _, _ = jac_at(net, torch.tensor(P[mid], dtype=torch.float32, device=DEV))

    h0 = torch.tensor(P[mid], dtype=torch.float32, device=DEV)
    base_H, base_Y = free_run_from(net, h0, args.free)
    span = float(np.ptp(coord))
    mags = (np.array([0.05, 0.10, 0.25], dtype=np.float32) * span)

    cand = np.real(EVEC[mid][:, 1])
    cand = cand - (cand @ tangent) * tangent
    orth = cand / np.linalg.norm(cand)
    rng = np.random.default_rng(0)
    rnd = rng.standard_normal(N_UNITS)
    rnd = rnd - (rnd @ tangent) * tangent
    rnd /= np.linalg.norm(rnd)

    pert = {}
    for mi, mag in enumerate(mags):
        for name, d in [("tangent", tangent), ("orthogonal", orth), ("random_orth", rnd)]:
            hp = h0 + torch.tensor(mag * d, dtype=torch.float32, device=DEV)
            Hh, Yy = free_run_from(net, hp, args.free)
            pert["pert_%s_m%d_states" % (name, mi)] = Hh
            pert["pert_%s_m%d_out" % (name, mi)] = Yy

    np.savez_compressed(
        "%s_s%d.npz" % (args.out, s),
        inc=inc.cpu().numpy().astype(np.float32),
        target=tgt.cpu().numpy().astype(np.float32),
        output=out.cpu().numpy().astype(np.float32),
        hidden_task=H_task.cpu().numpy().astype(np.float32),
        points=P.astype(np.float32), speeds=SP,
        pc_basis=Vt[:8].astype(np.float32), pc_evr=evr,
        point_order=order.astype(np.int32), manifold_coord=coord.astype(np.float32),
        manifold_mean=mu.astype(np.float32), tangent=tangent.astype(np.float32),
        orth_dir=orth.astype(np.float32), rand_orth_dir=rnd.astype(np.float32),
        Wout=W, readout_at_points=read_at_pts,
        evals=EV, evecs=EVEC, jac_mid=Jmid.astype(np.float32), mid_index=np.int32(mid),
        base_states=base_H, base_out=base_Y, pert_mags=mags, free_T=np.int32(args.free),
        **pert)

    dom = np.abs(EV[:, 0]); sec = np.abs(EV[:, 1])
    meta.append(dict(seed=s, loss=ev_loss, converged=bool(conv), n_pts=int(len(P)), usable=True,
                     pc1_var=float(evr[0]), extent=float(np.ptp(coord)),
                     dom_min=float(dom.min()), dom_max=float(dom.max()),
                     sec_min=float(sec.min()), sec_max=float(sec.max()),
                     n_exactly_one_marginal=int(np.sum([(np.abs(np.abs(e) - 1) < 0.03).sum() == 1 for e in EV])),
                     readout_range=[float(read_at_pts.min()), float(read_at_pts.max())],
                     target_range=[float(tgt.min()), float(tgt.max())],
                     readout_vs_coord_r=float(np.corrcoef(coord, read_at_pts)[0, 1])))
    print("  dom |lam| %.3f-%.3f | 2nd %.3f-%.3f | readout %.2f..%.2f | r(coord,readout)=%.4f"
          % (dom.min(), dom.max(), sec.min(), sec.max(), read_at_pts.min(), read_at_pts.max(),
             np.corrcoef(coord, read_at_pts)[0, 1]), flush=True)

json.dump(meta, open("%s_meta.json" % args.out, "w"), indent=1)
print("WROTE", sum(1 for m in meta if m.get("usable")), "usable exports")
