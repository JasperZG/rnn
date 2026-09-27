"""Numerical checks for the claims in ../predictable_recurrence.md.

Each check prints PASS/FAIL plus the measured quantity. Pure numpy/scipy, CPU,
runs in a few minutes (C6 dominates). Nothing here is fitted to the document: every
predicted number is computed from a closed-form expression first, then
compared with direct simulation.

    python ideation/checks/verify_theory.py
"""
import warnings
import numpy as np

# numpy 2.0 + Apple Accelerate emits spurious matmul overflow warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul")
from numpy.linalg import qr

rng = np.random.default_rng(0)
tanh = np.tanh


def quintic(x):
    # odd, saturating at +-1, phi'(0)=1, NO cubic term: x - x^5/4 + ...
    return x / (1.0 + x**4) ** 0.25


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
# C1  Closure lemma: z_{t+1} = D phi(E gF(z_t,u_t) + b) exactly, for ANY E, D, b
# ---------------------------------------------------------------------------
def c1():
    N, k, T = 257, 3, 200
    E = rng.normal(size=(N, k)) / np.sqrt(k)
    D = rng.normal(size=(k, N)) / np.sqrt(N)
    b = rng.normal(size=N) * 0.3
    A = qr(rng.normal(size=(k, k)))[0] * 0.97
    F = lambda z, u: A @ z + 0.2 * np.sin(z) + u
    u = rng.normal(size=(T, k)) * 0.1
    h = tanh(E @ np.ones(k) + b)            # arbitrary native initial state
    z = D @ h                               # shadow starts from its read-out
    err = 0.0
    for t in range(T):
        h = tanh(E @ (1.3 * F(D @ h, u[t])) + b)   # native, N-dimensional
        z = D @ tanh(E @ (1.3 * F(z, u[t])) + b)   # shadow, k-dimensional
        err = max(err, np.abs(D @ h - z).max())
    report("C1 closure for arbitrary encoder/decoder/bias", err < 1e-10,
           f"max |native - shadow| over {T} steps = {err:.1e}")


# ---------------------------------------------------------------------------
# C2  1-D: cubic distortion coefficient = (1/3) sum v_i^4 ; balanced minimizes
# ---------------------------------------------------------------------------
def c2():
    N = 64
    q = 1e-2
    rows = []
    for name, v in [("balanced", np.sign(rng.normal(size=N))),
                    ("gaussian", rng.normal(size=N)),
                    ("sparse-ish", rng.normal(size=N) ** 3)]:
        v = v / np.linalg.norm(v)
        psi = v @ tanh(v * q)
        c_meas = (q - psi) / q**3
        c_pred = np.sum(v**4) / 3
        rows.append((name, c_meas, c_pred))
    ok = all(abs(m - p) / p < 1e-3 for _, m, p in rows)
    ok &= rows[0][2] <= min(r[2] for r in rows)
    det = "; ".join(f"{n}: meas {m:.4e} pred {p:.4e}" for n, m, p in rows)
    report("C2 cubic coeff = sum v^4 / 3, balanced = 1/(3N) minimal", ok, det)


# ---------------------------------------------------------------------------
# C3  2-D anisotropy: four-sign frame vs harmonic (regular) frames
#     A frame whose normalized rows form a spherical 4-design has purely radial
#     cubic distortion with coefficient 3k/((k+2)N) * (1/3) = k/((k+2)N).
# ---------------------------------------------------------------------------
def harmonic_frame(N, M):
    # N rows, directions at angles pi*j/M (antipodal symmetry is automatic for odd phi)
    ang = np.pi * (np.arange(N) % M) / M
    V = np.stack([np.cos(ang), np.sin(ang)], 1)
    return V * np.sqrt(2.0 / N)             # V^T V = I when N divisible by M, M>=2


def c3():
    N, r = 240, 0.05
    th = np.linspace(0, 2 * np.pi, 721)[:-1]
    out = []
    for name, V in [("four-sign (M=2)", harmonic_frame(N, 2)),
                    ("harmonic M=3", harmonic_frame(N, 3)),
                    ("harmonic M=6", harmonic_frame(N, 6))]:
        assert np.allclose(V.T @ V, np.eye(2))
        Q = r * np.stack([np.cos(th), np.sin(th)], 1)
        P = tanh(Q @ V.T) @ V                  # Psi(q) for every q, rows
        radial = np.sum(P * Q, 1) / r          # component along q
        tang = (P[:, 1] * Q[:, 0] - P[:, 0] * Q[:, 1]) / r
        c_rad = (r - radial) / r**3
        out.append((name, c_rad.mean(), c_rad.std() / c_rad.mean(),
                    np.abs(tang).max() / r**3))
    pred = 2 / (4 * N)                          # k/((k+2)N) with k=2
    det = "; ".join(f"{n}: c_rad {m:.3e} (aniso {a:.1%}, max tangential/r^3 {t:.2e})"
                    for n, m, a, t in out)
    ok = out[0][2] > 0.05 and out[1][2] < 1e-3 and abs(out[1][1] - pred) / pred < 1e-2
    report(f"C3 4-design frames are radial, coeff k/((k+2)N)={pred:.3e}", ok, det)


# ---------------------------------------------------------------------------
# C4  Random (orthonormalized Gaussian) frame, large N:
#     Psi(q) ~= q * E_xi[phi'(|q| xi / sqrt(N))]  (Stein) + O(1/sqrt(N)) scatter
# ---------------------------------------------------------------------------
def c4():
    from scipy.integrate import quad
    k = 8
    res = []
    for N in [256, 4096]:
        V = qr(rng.normal(size=(N, k)))[0]
        errs = []
        for _ in range(50):
            d = rng.normal(size=k); d /= np.linalg.norm(d)
            r = 1.2 * np.sqrt(N)                  # O(1) per-neuron pre-activation
            q = r * d
            psi = V.T @ tanh(V @ q)
            s = r / np.sqrt(N)
            gain = quad(lambda x: (1 - tanh(s * x) ** 2) * np.exp(-x * x / 2)
                        / np.sqrt(2 * np.pi), -12, 12)[0]
            errs.append(np.linalg.norm(psi - gain * q) / np.linalg.norm(q))
        res.append((N, np.mean(errs)))
    ratio = res[0][1] / res[1][1]
    report("C4 Gaussian frame -> radial Stein gain, scatter ~ N^-1/2", 2.5 < ratio < 6.5,
           "; ".join(f"N={N}: rel. dev {e:.3e}" for N, e in res)
           + f"; ratio {ratio:.2f} (sqrt(16)=4 expected)")


# ---------------------------------------------------------------------------
# C5  Hold task lifetime laws (noise-free, g=1): t* ~ N (tanh), ~ N^2 (quintic)
#     integrated drift ODEs (continuum limit of the map):
#       tanh    dz/dt = -z^3/(3N)     ->  t* = (3N / 2z0^2) ((1-eps)^-2 - 1)
#       quintic dz/dt = -z^5/(4N^2)   ->  t* = (N^2 / z0^4) ((1-eps)^-4 - 1)
# ---------------------------------------------------------------------------
def lifetime(phi, N, z0, eps, g=1.0, Tmax=10**7):
    z, s = z0, np.sqrt(N)
    for t in range(1, Tmax):
        z = s * phi(g * z / s)
        if abs(z - z0) > eps * z0:
            return t
    return None


def c5():
    z0, eps = 1.0, 0.05
    rows = []
    for N in [16, 64, 256]:
        rows.append((N, lifetime(tanh, N, z0, eps),
                     1.5 * N / z0**2 * ((1 - eps) ** -2 - 1),
                     lifetime(quintic, N, z0, eps),
                     N**2 / z0**4 * ((1 - eps) ** -4 - 1)))
    ok = all(abs(a - b) <= max(1, 0.05 * b) and abs(c - d) <= max(1, 0.05 * d)
             for _, a, b, c, d in rows)
    report("C5 lifetime ~N (tanh) and ~N^2 (no-cubic, saturation-matched)", ok,
           "; ".join(f"N={N}: tanh {a} (pred {b:.0f}), quintic {c} (pred {d:.0f})"
                     for N, a, b, c, d in rows))


# ---------------------------------------------------------------------------
# C6  Width is NOT a resource without noise: +-1 balanced 1-D frame == one neuron
#     operated at input scale 1/sqrt(N). With per-neuron noise sigma, the encoding
#     scale s trades distortion (~s^2/N) against noise (~sigma/s): optimal lifetime
#     scales as N^{p/(p+1)} : N^{1/2} for tanh, N^{2/3} for the no-cubic activation.
# ---------------------------------------------------------------------------
def noisy_lifetime(phi, N, amp, sigma, eps, trials=200, Tmax=200000):
    """Hold a value of |z|=1 (task units) encoded at latent amplitude `amp`.
    Per-neuron additive noise sigma on h. Returns median first-crossing time."""
    v = np.ones(N) / np.sqrt(N)
    z = np.full(trials, amp)
    alive = np.ones(trials, bool)
    tcross = np.full(trials, Tmax)
    for t in range(1, Tmax):
        pre = np.outer(z, v)
        h = phi(pre) + sigma * rng.normal(size=pre.shape)
        z = h @ v
        bad = alive & (np.abs(z - amp) > eps * amp)
        tcross[bad] = t
        alive &= ~bad
        if not alive.any():
            break
    return np.median(tcross)


def c6():
    sigma, eps = 0.01, 0.1
    Ns = [64, 256, 1024]
    out = {}
    for name, phi, p in [("tanh", tanh, 1), ("quintic", quintic, 2)]:
        best = []
        for N in Ns:
            amps = np.sqrt(N) * np.geomspace(0.01, 1.5, 24)
            lts = [noisy_lifetime(phi, N, a, sigma, eps, trials=64) for a in amps]
            best.append(max(lts))
        slope = np.polyfit(np.log(Ns), np.log(best), 1)[0]
        out[name] = (slope, p / (p + 1), best)
    ok = all(abs(s - pr) < 0.1 for s, pr, _ in out.values())
    report("C6 noise-limited optimal lifetime exponent p/(p+1)", ok,
           "; ".join(f"{n}: slope {s:.2f} (pred {pr:.2f}), best t* {list(map(int, b))}"
                     for n, (s, pr, b) in out.items()))


# ---------------------------------------------------------------------------
# C7  Heterogeneous encoder E (random gains/biases) + decoder D solved BEFORE
#     training on the shadow. Two regimes:
#       noise-free: even N=16 approximates the identity to ~1e-6, so the
#         balanced-frame "lifetime ~ N" law is an artifact of homogeneity;
#       per-neuron noise sigma (NEF regime): D is the noise-regularized solve
#         (ridge = sigma^2 per sample), and the hold lifetime now grows with N
#         at fixed O(1) per-neuron amplitude -- width is a genuine resource.
# ---------------------------------------------------------------------------
def c7():
    grid = np.linspace(-1, 1, 801)
    z0 = np.linspace(-0.9, 0.9, 61)
    rows = []
    for N in [16, 64, 256, 1024]:
        e = rng.choice([-1, 1], N) * rng.uniform(0.5, 2.0, N)   # encoders (gains)
        b = rng.uniform(-1.5, 1.5, N)                             # biases
        Phi = tanh(np.outer(grid, e) + b)                         # 801 x N
        res = []
        for sigma in [0.0, 0.02]:
            lam = max(sigma**2, 1e-12) * grid.size
            D = np.linalg.solve(Phi.T @ Phi + lam * np.eye(N), Phi.T @ grid)
            delta = np.abs(Phi @ D - grid)[np.abs(grid) <= 0.9].max()
            trials = np.repeat(z0, 2)
            z, t_star = trials.copy(), np.full(trials.size, 8000)
            alive = np.ones(trials.size, bool)
            for t in range(1, 8000):
                h = tanh(np.outer(z, e) + b) + sigma * rng.normal(size=(z.size, N))
                z = h @ D
                bad = alive & (np.abs(z - trials) > 0.05)
                t_star[bad] = t
                alive &= ~bad
                if not alive.any():
                    break
            res.append((delta, np.median(t_star)))
        rows.append((N, res))
    free_delta = [r[0][0] for _, r in rows]
    noisy_t = [r[1][1] for _, r in rows]
    slope = np.polyfit(np.log([16, 64, 256, 1024]), np.log(noisy_t), 1)[0]
    ok = max(free_delta) < 1e-3 and slope > 0.5
    report("C7 heterogeneous encoder + shadow-solved decoder", ok,
           "; ".join(f"N={N}: noise-free delta {r[0][0]:.1e}; sigma=.02 delta "
                     f"{r[1][0]:.1e}, median t* {r[1][1]:.0f}" for N, r in rows)
           + f"; noisy lifetime exponent {slope:.2f}")


# ---------------------------------------------------------------------------
# C8  Rotation task, frame design. Four-sign frame (a 3-design) has 4-fold
#     anisotropic distortion -> amplitude-dependent phase error; a harmonic
#     M frame (a (2M-1)-design) is radial through order 2M-1 -> phase error
#     shrinks with M (residual set by order-2M anisotropy), and
#     g > 1 turns amplitude drift into an attracting limit cycle at g c(r*) = 1.
# ---------------------------------------------------------------------------
def c8():
    N, w, T = 240, 2 * np.pi / 37.3, 20000
    R = np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
    out = []
    for name, V in [("four-sign", harmonic_frame(N, 2)), ("harmonic M=3", harmonic_frame(N, 3)),
                    ("harmonic M=6", harmonic_frame(N, 6))]:
        z = np.array([6.0, 0.0])
        g = 1.02
        phase_err = 0.0
        for t in range(1, T + 1):
            z = V.T @ tanh(V @ (g * (R @ z)))
            ideal = (w * t + np.pi) % (2 * np.pi) - np.pi
            got = np.arctan2(z[1], z[0])
            phase_err = abs((got - ideal + np.pi) % (2 * np.pi) - np.pi)
        out.append((name, np.linalg.norm(z), phase_err))
    ok = out[1][2] < 0.01 * out[0][2] and out[2][2] < out[1][2]
    report("C8 harmonic frame preserves rotation phase; four-sign frame drifts", ok,
           "; ".join(f"{n}: radius after {T} steps {r:.3f}, |phase error| {p:.2e} rad"
                     for n, r, p in out))


if __name__ == "__main__":
    for c in (c1, c2, c3, c4, c5, c6, c7, c8):
        c()
