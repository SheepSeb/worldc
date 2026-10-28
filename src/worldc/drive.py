"""Drive the scene's real Isaac Sim robot to its goals in Isaac Sim physics, and (optionally) film it.

    worldc drive out/scene_3 --video                 # plan routes, drive in Isaac Sim, write drive_result.json + drive_*.mp4
    from worldc.drive import run_drive; run_drive("out/scene_3", n_goals=3, video=True)

The route is planned on the same navigation grid as the post-build checks (robot footprint inflation included).  The robot then
drives it by wheel velocity commands only (a differential-drive controller closes the loop on the robot's real pose), while moving
hazards (forklifts, people) run on their paths, so reaching the goals is a physical result, not an assumption.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .assets import robots
from .check import nav as NV
from .check import scene as SC
from .colliders import collect


def _los(free: np.ndarray, a, b) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    for t in np.linspace(0.0, 1.0, n):
        r, c = int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))
        if not free[r, c]:
            return False
    return True


def _smooth(free: np.ndarray, cells: list) -> list:
    """Greedy line-of-sight shortcutting of a grid path (keeps the path inside the inflated free space)."""
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _los(free, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def plan_route(scene, goal_ids: list[str] | None = None, n_goals: int = 3, min_goal_gap_m: float = 3.0) -> dict:
    """Waypoints from the spawn through ``n_goals`` goals (or the named ``goal_ids``), planned on the check grid."""
    from pxr import Usd

    d, usd, graph = SC.locate(scene)
    robot = graph.get("robot")
    if not robot or not robot.get("spawn"):
        raise ValueError("the scene has no robot")
    cols = collect(Usd.Stage.Open(str(usd)))
    cfg = SC.nav_config(graph)
    xr, yr = SC.environment_bounds(graph, cols)
    grid = NV.Grid(xr, yr, float(cfg["resolution_m"]))
    NV.build_grid(grid, NV.polygons(cols, NV.nav_colliders(cols, cfg["z_band"], set(cfg["ignore_categories"]))), SC.robot_radius(graph))
    sx, sy, syaw = robot["spawn"][:3]
    cur = grid.clamp(grid.cell(sx, sy))
    if not grid.free[cur]:
        raise ValueError("the robot spawn is not free for its footprint")

    goals = graph.get("goals") or []
    if goal_ids:
        by_id = {g.get("id") or g.get("name"): g for g in goals}
        missing = [g for g in goal_ids if g not in by_id]
        if missing:
            raise ValueError(f"unknown goals {missing}; available: {sorted(by_id)}")
        chosen = [by_id[g] for g in goal_ids]
    else:  # reachable goals, skipping near-duplicates, in scene order
        chosen, taken = [], [(sx, sy)]
        for g in goals:
            gx, gy = g["base_pose"][:2]
            cell = grid.clamp(grid.cell(gx, gy))
            if not grid.is_free(gx, gy) or any(math.hypot(gx - x, gy - y) < min_goal_gap_m for x, y in taken):
                continue
            dist, _ = NV.dijkstra(grid.free, cur, [cell])
            if dist[cell] < 1e17:
                chosen.append(g)
                taken.append((gx, gy))
            if len(chosen) >= n_goals:
                break
    if not chosen:
        raise ValueError("no reachable goals to drive to")

    waypoints, legs = [[float(sx), float(sy)]], []
    for g in chosen:
        gx, gy, gyaw = g["base_pose"][:3]
        cell = grid.clamp(grid.cell(gx, gy))
        dist, prev = NV.dijkstra(grid.free, cur, [cell])
        if dist[cell] >= 1e17:
            raise ValueError(f"goal {g.get('id')} is not reachable from {cur}")
        cells = [cur] + list(reversed(NV.trace_path(prev, cur, cell)))
        for r, c in _smooth(grid.free, cells)[1:]:
            waypoints.append([grid.x0 + (c + 0.5) * grid.res, grid.y1 - (r + 0.5) * grid.res])
        waypoints[-1] = [float(gx), float(gy)]
        legs.append({"goal": g.get("id") or g.get("name"), "end_index": len(waypoints) - 1, "pose": [float(gx), float(gy), float(gyaw)]})
        cur = cell
    length = float(sum(math.dist(a, b) for a, b in zip(waypoints, waypoints[1:])))
    return {"waypoints": waypoints, "legs": legs, "spawn": [float(sx), float(sy), float(syaw)], "length_m": round(length, 2),
            "robot": robot.get("asset"), "radius_m": SC.robot_radius(graph)}


def _wide_camera(graph: dict) -> dict:
    from .tools import auto_cameras

    return auto_cameras(graph)["auto_overview"]


def run_drive(scene, out_dir=None, *, goal_ids=None, n_goals: int = 3, video: bool = True, fps: int = 30, res=(1280, 720), speed: float = 1.0,
              timeout_s: float | None = None, rt_subframes: int = 2, cameras=("chase", "wide")) -> dict:
    """Plan, drive in Isaac Sim, return the result (and encode videos if ``video`` and ffmpeg is available)."""
    from .isaac_env import require_isaac_sim

    d, usd, graph = SC.locate(scene)
    alias = (graph.get("robot") or {}).get("asset")
    drive = robots.DIFF_DRIVE.get(alias)
    if drive is None:
        raise ValueError(f"driving is supported for differential-drive robots: {', '.join(sorted(robots.DIFF_DRIVE))} (the scene has {alias!r})")
    route = plan_route(d, goal_ids, n_goals)
    out = Path(out_dir or d / "drive")
    out.mkdir(parents=True, exist_ok=True)
    route.update(drive=drive, speed=float(speed), wide_camera=_wide_camera(graph), cameras=list(cameras) if video else [])
    (out / "route.json").write_text(json.dumps(route, indent=1))
    sim = require_isaac_sim()
    max_s = min(40.0 + 3.0 * route["length_m"] / max(speed, 0.1), 600.0)
    if timeout_s is None:  # filming is slow (RTX per frame); scale the limit with the frames it may need
        timeout_s = 900.0 + (max_s * fps * len(cameras) * 1.2 * (res[0] * res[1]) / (1920 * 1080) if video else max_s * 3.0)
    cmd = [str(sim.python), "-m", "worldc.render.isaac_drive", str(d), str(out), "--fps", str(fps), "--res", str(res[0]), str(res[1]),
           "--max-seconds", f"{max_s:.0f}", "--rt-subframes", str(rt_subframes)]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    from .isaac_env import isaac_lock

    with isaac_lock():  # one Isaac Sim on the GPU at a time
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    (out / "drive_run.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    rf = out / "drive_result.json"
    if not rf.is_file():
        raise RuntimeError("the Isaac Sim drive run produced no result; see " + str(out / "drive_run.log"))
    result = json.loads(rf.read_text())
    result["route"] = {"length_m": route["length_m"], "legs": route["legs"]}
    result["videos"] = {}
    if video and shutil.which("ffmpeg"):
        for cam in cameras:
            frames = out / f"frames_{cam}"
            if frames.is_dir() and any(frames.iterdir()):
                mp4 = out / f"drive_{cam}.mp4"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "%05d.png"), "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)], check=True)
                result["videos"][cam] = str(mp4)
    rf.write_text(json.dumps(result, indent=1))
    return result
