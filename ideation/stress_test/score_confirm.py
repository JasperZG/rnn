"""Score the confirmatory run (seeds 10-14) against the registered criteria."""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(HERE, "results", "stage3", n)))
STD = ("rnn", "gru", "lstm")
verdict = {}

acc = R("accumulation_stage1_confirm.json")
print("DRIVEN ACCUMULATION (seeds 10-14)")
print("arch   factor  n   agree E  agree R   rho E   median |log T^/T| E")
ok_i = True
for a in STD + ("tc",):
    for fac in (1.0, 0.5):
        L = [r for r in acc if r["arch"] == a and r["factor"] == fac]
        if not L:
            continue
        aE = np.mean([r["E"]["agree"] for r in L])
        aR = np.mean([r["R"]["agree"] for r in L])
        rho = np.nanmedian([r["E"]["spearman"] for r in L])
        lr = np.median([r["E"]["abs_log_ratio"] for r in L])
        print(f"{a:5s}  {fac:5}  {len(L):2d}   {aE:.2f}     {aR:.2f}     {rho:.2f}    {lr:.3f}")
        if a in STD and aE < 0.85:
            ok_i = False
far = [r for r in acc if r["arch"] in STD and r["E"]["median_meas"] > 1000]
far_ok = bool(far) and np.mean([r["E"]["agree"] for r in far]) >= 0.80
beats = all(np.mean([r["E"]["agree"] for r in acc if r["arch"] == a]) >
            np.mean([r["R"]["agree"] for r in acc if r["arch"] == a]) for a in STD)
print(f"far bin T*>1000: n={len(far)}, mean agree E="
      f"{np.mean([r['E']['agree'] for r in far]) if far else float('nan'):.2f}")
verdict["(i) driven agree>=0.85 every std arch & factor"] = ok_i
verdict["(ii) far-bin agree>=0.80"] = far_ok
verdict["(iii) E beats regression in every arch"] = beats

hold = R("hold_stage1_confirm.json")
print("\nHOLD (seeds 10-14)")
okB = True
for a in STD + ("tc",):
    L = [r for r in hold if r["arch"] == a]
    aE = np.mean([r["E"]["agree"] for r in L])
    print(f"{a:5s} n={len(L)} agree E={aE:.2f} R={np.mean([r['R']['agree'] for r in L]):.2f} "
          f"rho E={np.nanmedian([r['E']['spearman'] for r in L]):.2f}")
    if a in STD and aE < 0.85:
        okB = False
farB = [r for r in hold if r["arch"] in STD and r["E"]["median_meas"] > 1000]
farB_ok = (not farB) or np.mean([r["E"]["agree"] for r in farB]) >= 0.80
print(f"far bin: n={len(farB)} mean agree E="
      f"{np.mean([r['E']['agree'] for r in farB]) if farB else float('nan'):.2f}")
verdict["(B) hold agree>=0.85 every std arch"] = okB
verdict["(B) hold far-bin agree>=0.80"] = farB_ok

osc = [r for r in R("oscillation_stage1_confirm.json")]
T = 5000
right = [((r["T_meas"] > T) == (r["T_E"] > T)) for r in osc]
fails = [r for r in osc if r["T_meas"] <= T]
close = [abs(math.log(r["T_E"] / r["T_meas"])) < math.log(1.5) for r in fails]
print(f"\nOSCILLATION (seeds 10-14): n={len(osc)} fail/no-fail correct {sum(right)}/{len(osc)}; "
      f"failing nets within 1.5x: {sum(close)}/{len(fails)}")
for r in fails:
    print(f"   {r['tag']:28s} T meas={r['T_meas']:.0f} E={r['T_E']:.0f} (regression {r['T_R']:.0f})")
verdict["(iv/A) oscillation fail/no-fail >=90%"] = np.mean(right) >= 0.9
verdict["(iv/A) failing nets within 1.5x >=75%"] = (not fails) or np.mean(close) >= 0.75

print("\nVERDICT")
for k, v in verdict.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"\nOVERALL: {'PASS' if all(verdict.values()) else 'FAIL'}")
