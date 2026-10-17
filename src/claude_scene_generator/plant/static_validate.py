"""Pre-flight checks without a physics engine: initial interpenetration (dynamic/articulated vs anything), floating loose
bodies, and the 5 cm occupancy-grid navigation check from the robot spawn to every manipulation goal.

    python -m claude_scene_generator.plant.static_validate plant_out/seed_0
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from pxr import Usd

from ..colliders import collect, shrunk
from ..geom import obb_penetration
from ..static_validate import broadphase
from . import nav as NV

PEN_TOL = 0.003


def penetrations(cols):
    pairs, lo, hi = broadphase(cols)
    bad = []
    for a, b in pairs:
        A, B = cols[a], cols[b]
        if A.is_mesh or B.is_mesh:
            continue
        if A.body is None and B.body is None:
            continue
        if A.body == B.body or (A.group and A.group == B.group):
            continue
        if A.body and B.body and "/Hoses/" in A.body and A.body.rsplit("/", 1)[0] == B.body.rsplit("/", 1)[0]:
            continue  # segments of one hose are chained by spherical joints (jointed bodies do not collide)
        pen = obb_penetration(shrunk(A), shrunk(B))
        if pen > PEN_TOL:
            bad.append({"a": A.path, "b": B.path, "depth_mm": round(pen * 1000, 1)})
    return bad


def floating_loose(cols):
    out = []
    lo = np.array([c.obb.aabb()[0] for c in cols])
    hi = np.array([c.obb.aabb()[1] for c in cols])
    by_body = {}
    for i, c in enumerate(cols):
        if c.body and "/Loose/" in c.body:
            by_body.setdefault(c.body, []).append(i)
    for body, idx in by_body.items():
        zb = lo[idx, 2].min()
        if zb > 0.02:
            out.append({"body": body, "bottom_z": round(float(zb), 3)})
    return out


def nav_check(cols, graph, out_png=None):
    drain = graph["randomization"]["drain_xy"]
    occ = NV.build_occ(cols, [(drain[0] - 0.4, drain[1] - 0.4, drain[0] + 0.4, drain[1] + 0.4)])
    free = NV.free_mask(occ)
    sx, sy, _ = graph["robot"]["spawn"]
    start = NV.w2g(sx, sy)
    res = {"grid_shape": list(occ.shape), "resolution_m": NV.RES, "robot_radius_m": NV.ROBOT_R, "spawn_free": bool(free[start]), "goals": []}
    cells = [NV.w2g(g["base_pose"][0], g["base_pose"][1]) for g in graph["goals"]]
    dist, prev = NV.dijkstra(free, start, [c for c in cells if free[c]])
    paths = {}
    for g, c in zip(graph["goals"], cells):
        ok = bool(free[c]) and dist[c] < 1e17
        res["goals"].append({"name": g["name"], "type": g["type"], "goal_cell_free": bool(free[c]), "reachable": ok, "path_length_m": round(float(dist[c] * NV.RES), 2) if ok else None})
        if ok:
            p, cur = [], c
            while cur != start:
                p.append(cur)
                cur = tuple(prev[cur])
            paths[g["name"]] = p
    res["n_goals"] = len(graph["goals"])
    res["n_reachable"] = sum(g["reachable"] for g in res["goals"])
    res["all_reachable"] = bool(res["spawn_free"] and res["n_reachable"] == res["n_goals"] and res["n_goals"] >= 10)
    # tighter clearance sanity: smallest passable aisle width along the paths (distance-transform value on the path)
    from scipy.ndimage import distance_transform_edt

    dt = distance_transform_edt(~occ) * NV.RES
    clear = [min(dt[r, c] for r, c in p) for p in paths.values() if p]
    res["min_path_clearance_m"] = round(float(min(clear)), 3) if clear else None
    if out_png is not None:
        NV.render_png(out_png, occ, free, start, [g["base_pose"] for g in graph["goals"]], paths)
    return res


def run(seed_dir: Path) -> dict:
    seed_dir = Path(seed_dir)
    stage = Usd.Stage.Open(str(seed_dir / "plant.usd"))
    graph = json.loads((seed_dir / "scene_graph.json").read_text())
    cols = collect(stage)
    pen = penetrations(cols)
    flt = floating_loose(cols)
    nav = nav_check(cols, graph, seed_dir / "nav_occupancy.png")
    return {"n_colliders": len(cols), "initial_interpenetrations": pen, "floating_loose_items": flt, "navigation": nav,
            "passed": bool(not pen and not flt and nav["all_reachable"])}


if __name__ == "__main__":
    d = Path(sys.argv[1])
    r = run(d)
    (d / "static_validation.json").write_text(json.dumps(r, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: (v if k != "navigation" else {kk: vv for kk, vv in v.items() if kk != "goals"}) for k, v in r.items()}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:3500])
