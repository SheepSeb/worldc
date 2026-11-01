"""Kinematic Isaac Sim replay (rendering only, no physics) of logged E3 trials.

    uv run python scripts/e3_replay.py --scene P23 --seed 14 --tier hard --policies blind,fields,aware --out results/e3_grid/replay

Rebuilds the scene's hard variant with six workers (human characters) so that the prims exist, then for each policy writes the recorded poses
(the robot's logged trajectory interpolated to the frame times, the workers' positions recomputed from the trial seed with the same code as the
trial, heading from their direction of travel) and renders them with ``worldc.render.isaac_replay`` (behind the machine-wide Isaac lock).
Frames are joined to ``<scene>_<tier>_seed<seed>_<policy>_<camera>.mp4`` with ffmpeg.  The trial itself is not re-run: its log is the source.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldc.e3 import sim as S  # noqa: E402
from worldc.e3.tiers import _build  # noqa: E402
from worldc.e3.walkers import workers  # noqa: E402
from worldc.e3.world import load_world  # noqa: E402

CHARS = ["male_adult_construction_05_new", "male_adult_construction_03", "male_adult_construction_01_new", "female_adult_police_02"]


def replay_scene(grid: Path, scene: str, tier_variant: str, H0: np.ndarray, out: Path) -> Path:
    """The tier's variant spec plus six static workers, compiled to OUT/scene (the replay moves them)."""
    d = out / "scene_build"
    if (d / "scene" / "scene.usd").exists():
        return d / "scene"
    spec = json.loads((grid / scene / "scenes" / tier_variant / "spec.json").read_text())
    spec["agents"] = [{"id": f"worker_{i}", "asset": "human", "params": {"character": CHARS[i % len(CHARS)]},
                       "path": {"waypoints": [[float(H0[i][0]), float(H0[i][1])], [float(H0[i][0]) + 0.3, float(H0[i][1])]], "mode": "pingpong", "speed": 0.1},
                       "keep_clear_width": 0.8} for i in range(len(H0))]
    res = _build(spec, d, 0)
    print("replay scene build:", "ok" if res["ok"] else f"static check failed ({res['errors'][:2]}); the USD is used anyway")
    if not (d / "scene" / "scene.usd").exists():
        sys.exit("the replay scene was not written")
    return d / "scene"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="P23")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--tier", default="hard")
    ap.add_argument("--policies", default="blind,fields,aware")
    ap.add_argument("--grid", default="out/e3/grid")
    ap.add_argument("--out", default="results/e3_grid/replay")
    ap.add_argument("--work", default="out/e3/replay")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--cameras", default="overhead,corner")
    a = ap.parse_args()
    grid, out, work = Path(a.grid), Path(a.out), Path(a.work) / f"{a.scene}_{a.tier}"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    variant, n_workers = {"easy": ("base", 1), "medium": ("base", 3), "hard": ("hard", 6)}[a.tier]
    world = load_world(grid / a.scene / "scenes" / variant / "scene")
    rows = {}
    for ln in open(grid / a.scene / "trials.jsonl"):
        r = json.loads(ln)
        if r["tier"] == a.tier and r["seed"] == a.seed:
            rows[r["policy"]] = r
    timeout = rows["blind"]["timeout_s"]
    H, _ = workers(world, a.seed, n_workers, timeout + 1.0, S.DT)
    scene_dir = replay_scene(grid, a.scene, variant, H[0], work)
    graph = json.loads((scene_dir / "scene_graph.json").read_text())
    rob_prim = graph["robot"]["prim"]
    from worldc.isaac_env import isaac_lock, require_isaac_sim, run_isaac

    sim = require_isaac_sim()
    # worker headings from the direction of travel (held while standing), smoothed over 0.5 s
    D = np.gradient(H, S.DT, axis=0)
    heading = np.zeros(H.shape[:2])
    for i in range(H.shape[1]):
        last = 0.0
        for k in range(H.shape[0]):
            lo, hi = max(0, k - 5), min(H.shape[0] - 1, k + 5)
            v = H[hi, i] - H[lo, i]
            if math.hypot(*v) > 0.15:
                last = math.atan2(v[1], v[0])
            heading[k, i] = last
    for pol in a.policies.split(","):
        r = rows[pol]
        traj = np.array(r["trajectory"], float)  # t, x, y, th, v at 0.2 s
        th = np.unwrap(traj[:, 3])
        T = np.arange(0.0, r["sim_s"] + 1e-9, 1.0 / a.fps)
        frames = []
        for t in T:
            k = min(int(round(t / S.DT)), H.shape[0] - 1)
            x, y = (float(np.interp(t, traj[:, 0], traj[:, c])) for c in (1, 2))
            thh = float(np.interp(t, traj[:, 0], th))
            frames.append({"t": round(float(t), 3), "robot": [x, y, thh - world.phi0], "workers": [[float(H[k, i, 0]), float(H[k, i, 1]), float(heading[k, i])] for i in range(H.shape[1])]})
        label = f"{a.scene} {a.tier} seed {a.seed}  {pol}  ({r['outcome']})"
        rj = work / f"replay_{pol}.json"
        rj.write_text(json.dumps({"fps": a.fps, "label": label, "robot_prim": rob_prim, "worker_prims": [f"/World/Agents/worker_{i}" for i in range(H.shape[1])],
                                  "frames": frames, "collisions": [e["t"] for e in r["events"] if e["type"] == "collision"]}))
        fdir = work / f"frames_{pol}"
        shutil.rmtree(fdir, ignore_errors=True)
        cmd = [str(sim.python), "-m", "worldc.render.isaac_replay", str(scene_dir), str(rj), str(fdir), "--cameras", a.cameras, "--res", str(a.res[0]), str(a.res[1])]
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        print(f"{pol}: {len(frames)} frames x {len(a.cameras.split(','))} cameras", flush=True)
        with isaac_lock():
            p = run_isaac(cmd, env=env, timeout=3600 + len(frames) * 12)
        (work / f"replay_{pol}.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
        if "REPLAY_DONE" not in p.stdout:
            sys.exit(f"{pol}: replay failed; see {work / f'replay_{pol}.log'}\n" + p.stdout[-1500:])
        for cam in a.cameras.split(","):
            mp4 = out / f"{a.scene}_{a.tier}_seed{a.seed}_{pol}_{cam}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(fdir / cam / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)], check=True)
            hit = [e["t"] for e in r["events"] if e["type"] == "collision"]
            still = int(round((hit[0] if hit else r["sim_s"] / 2) * a.fps))
            shutil.copy(fdir / cam / f"{min(still, len(frames) - 1):05d}.png", out / f"{a.scene}_{a.tier}_seed{a.seed}_{pol}_{cam}.png")
            print("->", mp4, flush=True)


if __name__ == "__main__":
    main()
