"""
Classify the geometry of a recovered slow-point set and detect continuous
structure. A line attractor is a connected 1-D continuum, requiring a detector
distinct from counting isolated points: many slow points seeded across visited
states should populate a 1-D manifold (first PC dominant) with a single
near-unity Jacobian eigenvalue (the marginal direction).
"""
import numpy as np
from sklearn.decomposition import PCA


def classify_structure(points, evals_list, marg_tol=0.02):
    """Return a qualitative label: discrete_fixed_points | line_attractor | none.

    A line attractor is a densely-sampled 1-D continuum of fixed points, each
    carrying a marginal Jacobian direction (an eigenvalue on the unit circle).
    Requiring BOTH the 1-D geometry (pc1_var>0.9 over >=8 recovered points) AND
    the marginal eigenvalue across most points separates a true continuum from a
    handful of discrete points that merely happen to be collinear -- the latter
    lack the near-unity eigenvalue and were previously mislabeled when the point
    set was sampled finely. The fixed-point search must dedup finely enough to
    populate the continuum (>=8 points); a coarse dedup collapses it and hides it."""
    if len(points) == 0:
        return {"label": "none", "n_points": 0}
    n_stable = sum(1 for ev in evals_list if np.all(np.abs(ev) < 1.05))
    if len(points) >= 8:
        pca = PCA().fit(points)
        var1 = pca.explained_variance_ratio_[0]
        min_dist_1 = np.array([np.min(np.abs(np.abs(ev) - 1.0)) for ev in evals_list])
        frac_marginal = float(np.mean(min_dist_1 < marg_tol))
        # A line attractor is BOTH a near-perfect 1-D continuum (pc1_var ~= 1) AND
        # pervasively marginal (a unit-circle eigenvalue at most points). Discrete
        # memory is collinear but not marginal; gated context structure is marginal
        # in places but genuinely higher-D. Requiring both, at tight thresholds,
        # separates the true continuum from both discrete cases.
        if var1 > 0.97 and frac_marginal > 0.7:
            return {"label": "line_attractor", "n_points": len(points),
                    "pc1_var": float(var1), "frac_marginal": frac_marginal}
    return {"label": "discrete_fixed_points", "n_points": len(points),
            "n_stable": int(n_stable)}


def detect_limit_cycle(H_free, t_settle=30):
    """A sustained autonomous orbit shows persistent variance after transients
    decay. Returns (is_cycle, dominant_freq)."""
    traj = H_free[t_settle:]
    proj = traj @ np.linalg.svd(traj - traj.mean(0), full_matrices=False)[2][0]
    proj = proj - proj.mean()
    var = float(proj.var())
    spec = np.abs(np.fft.rfft(proj)); freqs = np.fft.rfftfreq(len(proj))
    dom = float(freqs[1 + np.argmax(spec[1:])]) if len(spec) > 1 else 0.0
    return var > 1e-3, dom
