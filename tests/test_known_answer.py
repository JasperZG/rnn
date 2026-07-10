"""
Known-answer regression test: the fixed-point pipeline MUST recover the
one-bit-memory structure (2 stable fixed points + >=1 saddle) known from first
principles before any result on an open task is trusted. Run on every revision.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from rnnphase import tasks, models, train, fixed_points


def test_onebit_memory_known_answer():
    torch.set_num_threads(4)
    net = models.build("rnn", n_in=1, n_out=1, N=128, seed=0)
    loss = train.train_network(net, tasks.make_memory, iters=800, seed=0)
    assert train.passes_gate(loss, 0.05), f"did not converge: loss={loss}"
    _, _, _, _, H = train.evaluate(net, tasks.make_memory, B=64)
    pts, sps, evals = fixed_points.find_slow_points(
        net, H[:, 20:], n_in=1, n_seed=300, steps=1200, speed_tol=1e-3, dedup=0.5)
    classes = [fixed_points.classify_point(ev) for ev in evals]
    n_stable = sum(c["stable"] for c in classes)
    n_saddle = sum(c["n_expanding"] >= 1 for c in classes)
    assert n_stable == 2, f"expected 2 stable fixed points, got {n_stable}"
    assert n_saddle >= 1, f"expected >=1 saddle, got {n_saddle}"


if __name__ == "__main__":
    test_onebit_memory_known_answer()
    print("KNOWN-ANSWER TEST: PASS")
