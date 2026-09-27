"""Score Amendment 6 (preregistered decision test, seeds 15-19) exactly as written."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results", "decision_stage1_prereg.json")))
cases = np.load(os.path.join(HERE, "results", "cases_stage1_prereg.npz"))
cells = d["cells"]
N_per_cell = {k: int((cases["scen"] == ["hold", "x1", "x0.5"].index(k.split("|")[0])).sum()) for k in cells}

p1, p2, p3_le, p3_lt, p4 = [], [], [], [], []
print(f"{'cell':12s} {'n_app':>6s} {'wrong':>7s} {'n_ok':>6s} {'useful':>7s} {'AUC':>6s}  matched wrong (E | R B1 B2)")
for k, c in cells.items():
    e = c["E"]
    n_ok = int(round(c["frac_ok"] * N_per_cell[k]))
    if e["n_app"] >= 50:
        p1.append((k, e["wrong"] <= 0.02))
    if n_ok >= 50:
        p2.append((k, e["useful"] >= 0.90))
    base = [c[m]["wrong_at_E_rate"] for m in ("R", "B1", "B2")]
    p3_le.append(all(e["wrong_at_E_rate"] <= b for b in base))
    p3_lt.append(all(e["wrong_at_E_rate"] < b for b in base))
    if e["auc"] == e["auc"]:
        p4.append((k, e["auc"] >= 0.97))
    print(f"{k:12s} {e['n_app']:6d} {e['wrong']:7.3f} {n_ok:6d} {e['useful']:7.3f} {e['auc']:6.3f}  "
          f"{e['wrong_at_E_rate']:.3f} | " + " ".join(f"{b:.3f}" for b in base))

a = d["asymmetry"]["E"]
crit = {
    "P1 wrong-approval <= 2% (cells with >=50 approvals)": all(v for _, v in p1),
    "P2 useful-approval >= 90% (cells with >=50 OK cases)": all(v for _, v in p2),
    "P3 matched-acceptance: E <= all baselines in 12/12, strictly lower in >=10":
        all(p3_le) and sum(p3_lt) >= 10,
    "P4 AUC >= 0.97 (cells with both outcomes)": all(v for _, v in p4),
    "P5 over-prediction >10% in <= 5% of failing cases": a["frac_over_10pct"] <= 0.05,
}
print("\nfailing cells:",
      {"P1": [k for k, v in p1 if not v], "P2": [k for k, v in p2 if not v], "P4": [k for k, v in p4 if not v]})
print(f"P3: <= in {sum(p3_le)}/12, strictly < in {sum(p3_lt)}/12;  P5 over>10% = {a['frac_over_10pct']:.3f}")
print("\nVERDICT")
for k, v in crit.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"OVERALL: {'PASS' if all(crit.values()) else 'FAIL'}")
