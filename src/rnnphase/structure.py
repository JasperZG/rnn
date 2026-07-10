"""
Classify the geometry of a recovered slow-point set and detect continuous
structure. A line attractor is a connected 1-D continuum, requiring a detector
distinct from counting isolated points: many slow points seeded across visited
states should populate a 1-D manifold (first PC dominant) with a single
near-unity Jacobian eigenvalue (the marginal direction).
"""
import numpy as np
from sklearn.decomposition import PCA


def classify_structure(points, evals_list):
    """Return a qualitative label: discrete_fixed_points | line_attractor | none."""
    if len(points) == 0:
        return {"label": "none", "n_points": 0}
    n_stable = sum(1 for ev in evals_list if np.all(np.abs(ev) < 1.05))
    if len(points) >= 8:
        pca = PCA().fit(points)
        var1 = pca.explained_variance_ratio_[0]
        if var1 > 0.9:
            return {"label": "line_attractor", "n_points": len(points),
                    "pc1_var": float(var1)}
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
