"""Recompute the ablation comparison from the raw trial logs and show both runs per policy x tier (independent of e3_ablation_report.py).

    python3 scripts/e3_ablation_verify.py results/e3_grid results/e3_ablation_walkaround > results/e3_ablation_walkaround/verification.md

Reads ``scenes/<scene>/trials.jsonl.gz`` of each run; counts collision events and robot-moving collisions from the events themselves.
Also compares trial by trial: blind of the ablation against fields of the main run (their totals can coincide), the totals per scene, and digests.
"""
import collections
import glob
import gzip
import json
import sys

import numpy as np


def load(root):
    d, meta = {}, collections.Counter()
    for f in sorted(glob.glob(f"{root}/scenes/P*/trials.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for ln in fh:
                r = json.loads(ln)
                ev = [e for e in r["events"] if e["type"] == "collision"]
                d[(r["scene"], r["tier"], r["policy"], r["seed"])] = (len(ev), sum(e["robot_moving"] for e in ev), r["digest"], r["outcome"])
                meta[(r.get("walkers", "fixed"), r["code_commit"], r["e3_policies"])] += 1
    return d, meta


main_root, abl_root = sys.argv[1:3]
M, mm = load(main_root)
A, am = load(abl_root)
print("# Verification of the walk-around comparison from the raw trial logs\n")
print(f"- main run: `{main_root}`, {len(M)} trials, (walkers, code commit, policies) = {dict(mm)}")
print(f"- ablation: `{abl_root}`, {len(A)} trials, (walkers, code commit, policies) = {dict(am)}\n")
print("## Both runs per policy x tier (300 trials per cell; counts from the collision events)\n")
print("| Tier | Policy | Main: moving / trial | Main: stationary / trial | Main: moving events | Walk-around: moving / trial | Walk-around: stationary / trial | Walk-around: moving events |")
print("|---|---|---|---|---|---|---|---|")
for t in ("easy", "medium", "hard"):
    for p in ("blind", "fields", "aware"):
        row = []
        for D in (M, A):
            R = [v for k, v in D.items() if k[1] == t and k[2] == p]
            row += [f"{sum(v[1] for v in R) / len(R):.3f}", f"{sum(v[0] - v[1] for v in R) / len(R):.3f}", str(sum(v[1] for v in R))]
        print(f"| {t} | {p} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |")
print("\n## Is walk-around blind the same as main fields? (moving collisions)\n")
print("The totals coincide in the medium and hard tiers (239 and 477 events) but the trials do not:\n")
print("| Tier | Events: main fields | Events: walk-around blind | Trials with equal count | Correlation of per-trial counts | Identical trial digests |")
print("|---|---|---|---|---|---|")
for t in ("easy", "medium", "hard"):
    keys = [k for k in M if k[1] == t and k[2] == "fields"]
    a = np.array([M[k][1] for k in keys])
    b = np.array([A[(k[0], t, "blind", k[3])][1] for k in keys])
    dg = sum(M[k][2] == A[(k[0], t, "blind", k[3])][2] for k in keys)
    print(f"| {t} | {a.sum()} | {b.sum()} | {int((a == b).sum())} / {len(keys)} | {np.corrcoef(a, b)[0, 1]:.2f} | {dg} |")
scenes = sorted({k[0] for k in M})
print("\nMoving-collision events per scene, hard tier:\n")
print("| Scene | " + " | ".join(scenes) + " |")
print("|---|" + "---|" * len(scenes))
print("| main fields | " + " | ".join(str(sum(v[1] for k, v in M.items() if k[0] == s and k[1] == "hard" and k[2] == "fields")) for s in scenes) + " |")
print("| walk-around blind | " + " | ".join(str(sum(v[1] for k, v in A.items() if k[0] == s and k[1] == "hard" and k[2] == "blind")) for s in scenes) + " |")
print("| main blind | " + " | ".join(str(sum(v[1] for k, v in M.items() if k[0] == s and k[1] == "hard" and k[2] == "blind")) for s in scenes) + " |")
