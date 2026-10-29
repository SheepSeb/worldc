"""E3 prototype: the 2D trial grid (tiers x seeds x policies) on one accepted scene.

    uv run python scripts/e3_one_scene.py SCENE_DIR --name P18 --seeds 10 --out out/e3/P18

Builds the scene's E3 variants first (``worldc.e3.tiers``: ``base`` for easy / medium, ``hard`` with occluders; rebuilt and
statically checked) into OUT/scenes unless they are there.  Easy / medium / hard = 1 / 3 / 6 workers.  Writes trials.csv,
trials.jsonl (with drawn workers, events, trajectories and a digest), summary.json, summary.md, map_{base,hard}.png (grid,
task points, goals), and for the hard tier per policy a picture and a GIF of the seed with the most robot-moving
collisions (then the smallest gap).  Replays one trial from its seed and checks it is identical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldc.e3 import sim as S  # noqa: E402
from worldc.e3.tiers import make_variants  # noqa: E402
from worldc.e3.walkers import workers  # noqa: E402
from worldc.e3.world import load_world  # noqa: E402

TIERS = {"easy": ("base", 1), "medium": ("base", 3), "hard": ("hard", 6)}
PX = 40  # pixels per metre
POI_COLORS = {"door": (0, 140, 70), "bench": (30, 90, 200), "storage": (150, 90, 20), "cell_gate": (160, 40, 160)}
WORKER_COLORS = [(230, 120, 0), (0, 150, 150), (200, 0, 120), (110, 110, 0), (90, 60, 200), (0, 120, 220)]


def px(world, x, y):
    return (x - world.grid.x0) * PX, (world.grid.y1 - y) * PX


def base_image(world) -> Image.Image:
    g, h = world.grid, world.hgrid
    rgb = np.full(g.occ.shape + (3,), 245, np.uint8)
    rgb[~h.free & ~g.occ] = (225, 215, 215)
    rgb[g.occ] = (60, 60, 60)
    im = Image.fromarray(rgb).resize((int(g.nx * g.res * PX), int(g.ny * g.res * PX)), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    for P in world.cells:
        dr.polygon([px(world, *p) for p in P], outline=(200, 0, 0))
    for kind, pts in world.pois.items():
        for p in pts:
            cx, cy = px(world, *p)
            dr.rectangle([cx - 4, cy - 4, cx + 4, cy + 4], fill=POI_COLORS.get(kind, (0, 0, 0)))
    for gl in world.goals:
        cx, cy = px(world, *gl["pose"][:2])
        dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(0, 170, 0), width=2)
    return im


def robot_poly(world, x, y, th):
    hx, hy = world.body_half
    yaw = th - world.phi0
    c, s = math.cos(yaw), math.sin(yaw)
    return [px(world, x + c * a - s * b, y + s * a + c * b) for a, b in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]


def trial_picture(world, H, res, path):
    im = base_image(world)
    dr = ImageDraw.Draw(im)
    n = len(res["trajectory"]) * 4
    for i in range(H.shape[1]):
        dr.line([px(world, *p) for p in H[: n : 4, i]], fill=WORKER_COLORS[i % 6], width=1)
    dr.line([px(world, r[1], r[2]) for r in res["trajectory"]], fill=(220, 0, 0), width=3)
    for e in res["events"]:
        cx, cy = px(world, *e["xy"])
        col = (255, 0, 0) if e["robot_moving"] else (120, 120, 120)
        dr.line([cx - 8, cy - 8, cx + 8, cy + 8], fill=col, width=3)
        dr.line([cx - 8, cy + 8, cx + 8, cy - 8], fill=col, width=3)
    dr.text((8, 8), f"{res['policy']}  moving/all collisions {res['collisions_robot_moving']}/{res['collisions']}  "
                    f"min gap {res['min_gap_m']} m  t {res['sim_s']} s  {res['outcome']}", fill=(0, 0, 0))
    im.save(path)


def trial_gif(world, H, res, path, every=2):
    base = base_image(world)
    frames = []
    for r in res["trajectory"][::every]:
        k = int(round(r[0] / S.DT))
        im = base.copy()
        dr = ImageDraw.Draw(im)
        for i in range(H.shape[1]):
            cx, cy = px(world, *H[k, i])
            rad = 0.26 * PX
            dr.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=WORKER_COLORS[i % 6])
        gp, _ = S.gaps(r[1], r[2], r[3] - world.phi0, world.body_half, H[k])
        hit = len(gp) and gp.min() <= 0
        dr.polygon(robot_poly(world, *r[1:4]), fill=(255, 0, 0) if hit else (230, 60, 60), outline=(0, 0, 0))
        dr.text((8, 8), f"{res['policy']}  t={r[0]:5.1f} s  v={r[4]:.2f}  gap={gp.min() if len(gp) else 0:.2f} m", fill=(0, 0, 0))
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(S.DT * 4 * every * 1000), loop=0)


def digest(res) -> str:
    return hashlib.sha256(json.dumps([res["trajectory"], res["events"]]).encode()).hexdigest()[:16]


def summarize(rows) -> dict:
    out = {}
    for tier in TIERS:
        for pol in S.POLICIES:
            R = [r for r in rows if r["tier"] == tier and r["policy"] == pol]
            done = [r["time_to_goal_s"] for r in R if r["time_to_goal_s"] is not None]
            mean = lambda k: round(float(np.mean([r[k] for r in R])), 2)  # noqa: E731
            out[f"{tier}/{pol}"] = {
                "trials": len(R), "success_rate": mean("success"),
                "outcomes": {o: sum(r["outcome"] == o for r in R) for o in S.OUTCOMES if any(r["outcome"] == o for r in R)},
                "collisions_robot_moving_mean": mean("collisions_robot_moving"),
                "trials_with_robot_moving_collision": sum(r["collisions_robot_moving"] > 0 for r in R),
                "collisions_robot_moving_from_behind": sum(r["collisions_robot_moving_from_behind"] for r in R),
                "collisions_robot_moving_total": sum(r["collisions_robot_moving"] for r in R),
                "protective_stops_mean": mean("protective_stops"), "field_warning_mean_s": mean("field_warning_s"),
                "collisions_mean": mean("collisions"), "collisions_robot_stationary_mean": mean("collisions_robot_stationary"),
                "min_gap_moving_median_m": round(float(np.median([r["min_gap_moving_m"] for r in R if r["min_gap_moving_m"] is not None])), 3),
                "near_miss_moving_rate": round(float(np.mean([r["min_gap_moving_m"] is not None and 0 < r["min_gap_moving_m"] < 0.5 for r in R])), 2),
                "time_within_0.5m_moving_mean_s": mean("time_within_0.5m_moving_s"), "time_within_1m_moving_mean_s": mean("time_within_1m_moving_s"),
                "time_within_0.5m_mean_s": mean("time_within_0.5m_s"), "time_within_1m_mean_s": mean("time_within_1m_s"),
                "stationary_near_1m_mean_s": mean("stationary_near_1m_s"),
                "time_to_goal_median_s": round(float(np.median(done)), 1) if done else None,
                "human_human_overlap_mean_s": mean("human_human_overlap_s")}
    return out


def table(summary) -> str:
    cols = [("success", lambda s: f"{s['success_rate']:.1f}"),
            ("moving coll./trial", lambda s: f"**{s['collisions_robot_moving_mean']:.1f}**"),
            ("moving from behind (of total)", lambda s: f"{s['collisions_robot_moving_from_behind']} of {s['collisions_robot_moving_total']}"),
            ("all / stationary coll.", lambda s: f"{s['collisions_mean']:.1f} / {s['collisions_robot_stationary_mean']:.1f}"),
            ("<0.5 m / <1 m moving (s)", lambda s: f"{s['time_within_0.5m_moving_mean_s']:.1f} / {s['time_within_1m_moving_mean_s']:.1f}"),
            ("<0.5 m / <1 m all (s)", lambda s: f"{s['time_within_0.5m_mean_s']:.1f} / {s['time_within_1m_mean_s']:.1f}"),
            ("stationary <1 m of a person (s)", lambda s: f"{s['stationary_near_1m_mean_s']:.1f}"),
            ("time to goal (median s)", lambda s: f"{s['time_to_goal_median_s']}"),
            ("outcomes", lambda s: ", ".join(f"{k} {v}" for k, v in s["outcomes"].items()))]
    lines = ["| tier / policy | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for key, s in summary.items():
        lines.append(f"| {key.replace('/', ' / ')} | " + " | ".join(f(s) for _, f in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="an accepted E1 scene directory (with resolved_spec.json)")
    ap.add_argument("--name", required=True, help="scene label, e.g. the prompt id")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or f"out/e3/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vdir = out / "scenes"
    if not ((vdir / "base" / "scene" / "e3_variant.json").exists() and (vdir / "hard" / "scene" / "e3_variant.json").exists()):
        v = make_variants(a.scene, vdir)
        if not (v["base"]["ok"] and v["hard"]["ok"]):
            sys.exit(f"variants failed: {json.dumps(v)[:2000]}")
    worlds, legs, timeouts = {}, {}, {}
    for variant in ("base", "hard"):
        w = load_world(vdir / variant / "scene")
        w.name = f"{a.name}/{variant}"
        worlds[variant], legs[variant] = w, S.static_legs(w)
        timeouts[variant] = S.timeout_s(w, legs[variant])
        base_image(w).save(out / f"map_{variant}.png")
        print(f"{w.name}: grid {w.grid.nx}x{w.grid.ny}, task points { {k: len(p) for k, p in w.pois.items()} }, goals {len(w.goals)}, "
              f"unreachable legs {sum(p is None for p in legs[variant])}, "
              f"timeout {timeouts[variant]:.0f} s; robot {w.robot}")
    timeout = max(timeouts.values())  # one timeout for every tier

    rows, keep = [], {}
    with open(out / "trials.jsonl", "w") as jf:
        for tier, (variant, n) in TIERS.items():
            world = worlds[variant]
            for seed in range(a.seeds):
                H, info = workers(world, seed, n, timeout + 1.0, S.DT)
                for pol in S.POLICIES:
                    t2 = time.time()
                    res = S.run_trial(world, H, pol, legs[variant], timeout)
                    row = {"scene": a.name, "variant": variant, "tier": tier, "workers": n, "seed": seed, "timeout_s": round(timeout, 1),
                           "wall_s": round(time.time() - t2, 2),
                           **{k: v for k, v in res.items() if k not in ("events", "arrivals", "trajectory")}}
                    rows.append(row)
                    jf.write(json.dumps({**row, "workers_drawn": info, "events": res["events"], "arrivals": res["arrivals"],
                                         "trajectory": res["trajectory"], "digest": digest(res)}) + "\n")
                    if tier == "hard":
                        rank = (res["collisions_robot_moving"], -(res["min_gap_m"] if res["min_gap_m"] is not None else 9))
                        if pol not in keep or rank > keep[pol][0]:
                            keep[pol] = (rank, seed, res)
                print(f"{tier:6s} seed {seed:2d}: " + " | ".join(
                    f"{r['policy']:6s} {r['outcome'][:9]:9s} coll {r['collisions_robot_moving']}/{r['collisions']} t {r['sim_s']}" for r in rows[-3:]),
                    flush=True)

    with open(out / "trials.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    md = table(summary)
    (out / "summary.md").write_text(md + "\n")

    _, seed, first = keep["blind"]
    H, _ = workers(load_world(vdir / "hard" / "scene"), seed, TIERS["hard"][1], timeout + 1.0, S.DT)
    again = S.run_trial(worlds["hard"], H, "blind", legs["hard"], timeout)
    print(f"replay hard seed {seed} blind from a fresh world: {'identical' if digest(again) == digest(first) else 'DIFFERENT'} ({digest(first)})")

    for pol, (_, seed, res) in keep.items():
        H, _ = workers(worlds["hard"], seed, TIERS["hard"][1], timeout + 1.0, S.DT)
        trial_picture(worlds["hard"], H, res, out / f"hard_{pol}_seed{seed}.png")
        if not a.no_gif:
            trial_gif(worlds["hard"], H, res, out / f"hard_{pol}_seed{seed}.gif")
    print(md)
    print(f"total {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
