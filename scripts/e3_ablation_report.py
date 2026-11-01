"""Walker-model ablation (walk-around, or the withdrawn waiting variant) against the main grid: moving and stationary collisions (and success, frozen, time to goal) per policy x tier.

    uv run python scripts/e3_ablation_report.py out/e3/grid out/e3/ablation_reactive --out results/e3_ablation_reactive

Both runs use the same scenes, variants, seeds and policies; only the walkers differ (the ``walkers`` column).  Estimates and CIs are
e3_analysis's (cluster bootstrap over scenes, then seeds).  The paired test compares the per-scene means of the two runs per policy x
tier (Wilcoxon over the 10 scenes, Holm within each metric over the 9 policy x tier cells).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import e3_analysis as A

METRICS = (("Moving coll./trial", "moving_collisions", 2, False), ("Stationary coll./trial", "stationary_collisions", 2, False),
           ("Success (%)", "success", 0, True), ("Frozen (%)", "frozen", 0, True), ("Time to goal (s)", "time_to_goal", 0, False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("ablation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main_rows = A.load(sorted(glob.glob(f"{a.main}/P*/trials.csv")))
    abl_rows = A.load(sorted(glob.glob(f"{a.ablation}/P*/trials.csv")))
    em, ea = A.estimate(main_rows, a.boot, 0), A.estimate(abl_rows, a.boot, 0)
    scenes = sorted({r["scene"] for r in main_rows})

    def scene_means(rows, fn):
        return {(t, p, sc): float(np.mean([fn(r) for r in rows if r["scene"] == sc and r["tier"] == t and r["policy"] == p])) for t in A.TIERS for p in A.POLICIES for sc in scenes}

    tests = {}
    for key, fn in (("moving_collisions", lambda r: r["collisions_robot_moving"]), ("stationary_collisions", lambda r: r["collisions_robot_stationary"])):
        mm, ma = scene_means(main_rows, fn), scene_means(abl_rows, fn)
        block = []
        for t in A.TIERS:
            for p in A.POLICIES:
                x, y = np.array([mm[(t, p, s)] for s in scenes]), np.array([ma[(t, p, s)] for s in scenes])
                d = y - x
                pv = float(wilcoxon(x, y, zero_method="wilcox").pvalue) if (d != 0).any() else 1.0
                block.append({"tier": t, "policy": p, "up": int((d > 0).sum()), "down": int((d < 0).sum()), "p": pv})
        for r, ph in zip(block, A.holm(np.array([b["p"] for b in block]))):
            r["p_holm"] = float(ph)
            tests[(key, r["tier"], r["policy"])] = r

    def ws(rows, t, p, k):
        v = [r[k] for r in rows if r["tier"] == t and r["policy"] == p]
        return float(np.mean(v)) if v else float("nan")

    wcols = ([("Walker replans/trial", "walker_replans"), ("Walker detour (s)/trial", "walker_detour_s"), ("No-path events/trial", "walker_no_path")] if "walker_replans" in abl_rows[0]
             else [("Walker stops/trial", "walker_stops"), ("Walker waiting (s)/trial", "walker_held_s")])
    head = ["Tier", "Policy"] + [f"{h}: main -> ablation" for h, *_ in METRICS] + [w[0] for w in wcols] + ["Scenes up / down, moving (Holm p)", "Scenes up / down, stationary (Holm p)"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    flat = [["tier", "policy"] + [f"{m}_{k}_{v}" for _, m, _, _ in METRICS for k in ("main", "reactive") for v in ("point", "lo", "hi")] + [w[1] for w in wcols]]
    for t in A.TIERS:
        for i, p in enumerate(A.POLICIES):
            cells = [t if i == 0 else "", A.POLICY_LABEL[p]]
            row = [t, p]
            for _, m, d, pc in METRICS:
                cells.append(f"{A.fmt(*em[m][(t, p)][:3], digits=d, pct=pc)} -> {A.fmt(*ea[m][(t, p)][:3], digits=d, pct=pc)}")
                row += [round(float(v), 4) for v in (*em[m][(t, p)][:3], *ea[m][(t, p)][:3])]
            cells += [f"{ws(abl_rows, t, p, w[1]):.1f}" for w in wcols]
            for k in ("moving_collisions", "stationary_collisions"):
                r = tests[(k, t, p)]
                cells.append(f"{r['up']} / {r['down']} ({'<0.001' if r['p_holm'] < 0.001 else f'{r['p_holm']:.3f}'})")
            md.append("| " + " | ".join(cells) + " |")
            flat.append(row + [round(ws(abl_rows, t, p, w[1]), 3) for w in wcols])
    outc = {}
    for lab, rows in (("main", main_rows), ("reactive", abl_rows)):  # 'reactive' = the ablation run
        for t in A.TIERS:
            for p in A.POLICIES:
                R = [r for r in rows if r["tier"] == t and r["policy"] == p]
                outc[(lab, t, p)] = {o: sum(r["outcome"] == o for r in R) for o in ("success", "collision", "frozen", "timeout") }
    md += ["", "Main run = walkers that do not react; ablation = walkers that treat the robot as an obstacle and replan around it with A* (footprint + 0.3 m), no waiting "
           "(or, for the withdrawn variant, walkers that wait). "
           f"Same {len(scenes)} scenes, seeds and policies; per cell 300 trials. Point estimate [95 % cluster-bootstrap CI]. 'Scenes up / down': scenes whose mean rose / fell in the "
           "ablation run (Wilcoxon over scenes, Holm over the 9 cells per metric).", "", "Outcome counts per cell (of 300), main -> ablation:", "",
           "| Tier | Policy | success | collision | frozen | timeout |", "|---|---|---|---|---|---|"]
    for t in A.TIERS:
        for p in A.POLICIES:
            md.append(f"| {t} | {A.POLICY_LABEL[p]} | " + " | ".join(f"{outc[('main', t, p)][o]} -> {outc[('reactive', t, p)][o]}" for o in ("success", "collision", "frozen", "timeout")) + " |")
    (out / "ablation_reactive.md").write_text("\n".join(md) + "\n")
    with open(out / "ablation_reactive.csv", "w", newline="") as f:
        csv.writer(f).writerows(flat)
    json.dump({f"{k[0]}/{k[1]}/{k[2]}": v for k, v in tests.items()}, open(out / "ablation_tests.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
