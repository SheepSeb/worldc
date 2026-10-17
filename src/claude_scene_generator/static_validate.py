"""Pre-flight checks that need no physics engine: spawn interpenetration, floating bodies,
hazard-path clearance, and the 5 cm occupancy-grid navigability check.

Usage: python -m claude_scene_generator.static_validate warehouse_out/seed_0
"""
from __future__ import annotations

import heapq
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd

from . import layout as LY
from .colliders import Collider, collect
from .geom import OBB, obb_penetration, qmat, qyaw

RES = 0.05  # occupancy grid resolution (m)
ROBOT_R = 0.38  # mobile-manipulator footprint radius
Z_BAND = (0.10, 1.80)  # heights that block a ground robot
PEN_TOL = 0.003


def broadphase(cols: list[Collider]):
    lo = np.array([c.obb.aabb()[0] for c in cols])
    hi = np.array([c.obb.aabb()[1] for c in cols])
    pairs = []
    n = len(cols)
    for s in range(0, n, 256):
        e = min(n, s + 256)
        ov = np.all((lo[s:e, None, :] < hi[None, :, :]) & (hi[s:e, None, :] > lo[None, :, :]), axis=2)
        ii, jj = np.nonzero(ov)
        for a, b in zip(ii + s, jj):
            if a < b:
                pairs.append((a, b))
    return pairs, lo, hi


def spawn_penetration(cols: list[Collider]):
    pairs, lo, hi = broadphase(cols)
    bad = []
    for a, b in pairs:
        A, B = cols[a], cols[b]
        if A.is_mesh or B.is_mesh:
            continue
        if A.body is None and B.body is None:
            continue
        if A.body == B.body:
            continue
        pen = obb_penetration(A.obb, B.obb)
        if pen > PEN_TOL:
            bad.append({"a": A.path, "b": B.path, "depth_mm": round(pen * 1000, 1)})
    return bad


ARTIC_PREFIX = ("door_", "gate_", "drawer", "hazard", "conveyor")


def floating_bodies(cols: list[Collider]):
    by_body = defaultdict(list)
    for i, c in enumerate(cols):
        if c.body:
            by_body[c.body].append(i)
    lo = np.array([c.obb.aabb()[0] for c in cols])
    hi = np.array([c.obb.aabb()[1] for c in cols])
    out = []
    for body, idx in by_body.items():
        c0 = cols[idx[0]]
        if c0.kinematic or c0.category.startswith(ARTIC_PREFIX):
            continue
        zb = lo[idx, 2].min()
        if zb < 0.02:
            continue
        mn, mx = lo[idx].min(0), hi[idx].max(0)
        supported = False
        for j, o in enumerate(cols):
            if o.body == body or o.is_mesh:
                continue
            if abs(hi[j, 2] - zb) > 0.012 and not (lo[j, 2] < zb < hi[j, 2]):
                continue
            if hi[j, 0] > mn[0] + 0.01 and lo[j, 0] < mx[0] - 0.01 and hi[j, 1] > mn[1] + 0.01 and lo[j, 1] < mx[1] - 0.01:
                supported = True
                break
        if not supported:
            out.append({"body": body, "bottom_z": round(float(zb), 3)})
    return out


def hazard_paths_clear(stage: Usd.Stage, cols: list[Collider], graph: dict):
    """Sweep each scripted hazard's collider along its path; report penetration with anything it does not own."""
    bad = []
    static = [c for c in cols if not c.is_mesh and c.category != "hazard"]
    for h in graph["hazards"]:
        hb = [c for c in cols if c.body == h["path"]]
        wp = np.array(h["waypoints"], float)
        pts = wp.tolist()
        if h["path_mode"] == "loop":
            pts = pts + [pts[0]]
        samples = []
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            L = math.hypot(bx - ax, by - ay)
            yaw = math.atan2(by - ay, bx - ax)
            for s in np.arange(0.0, L, 0.2):
                samples.append((ax + (bx - ax) * s / L, ay + (by - ay) * s / L, yaw))
        worst = 0.0
        hit = None
        for (x, y, yaw) in samples:
            for c in hb:
                # hazard colliders sit at the body origin with identity local rotation
                o = OBB(np.array([x, y, c.obb.c[2]]), c.obb.h.copy(), qmat(qyaw(yaw)))
                for s_ in static:
                    lo, hi = o.aabb()
                    slo, shi = s_.obb.aabb()
                    if np.any(lo >= shi) or np.any(hi <= slo):
                        continue
                    pen = obb_penetration(o, s_.obb)
                    if pen > worst:
                        worst, hit = pen, s_.path
        if worst > PEN_TOL:
            bad.append({"hazard": h["path"], "obstacle": hit, "depth_mm": round(worst * 1000, 1)})
    return bad


# --------------------------------------------------------------------------- occupancy grid
def build_grid(cols: list[Collider], band=Z_BAND):
    nx, ny = int(round(LY.W / RES)), int(round(LY.D / RES))
    SS = 5  # supersample, then sample the centre sub-pixel: a cell is occupied iff its centre is inside a footprint
    img = Image.new("L", (nx * SS, ny * SS), 0)
    d = ImageDraw.Draw(img)
    for c in cols:
        if c.is_mesh or c.category in ("hazard", "floor", "roof"):
            continue
        lo, hi = c.obb.aabb()
        if hi[2] < band[0] or lo[2] > band[1]:
            continue
        cs = c.obb.corners()
        pts = cs[:, :2]
        # footprint = convex hull of projected corners
        from scipy.spatial import ConvexHull

        try:
            h = ConvexHull(pts)
            poly = pts[h.vertices]
        except Exception:
            continue
        px = [((x - LY.X0) / RES * SS, (LY.Y1 - y) / RES * SS) for x, y in poly]
        d.polygon(px, fill=1)
    occ = np.asarray(img, bool)[SS // 2 :: SS, SS // 2 :: SS]
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return occ


def w2g(x, y):
    return int((LY.Y1 - y) / RES), int((x - LY.X0) / RES)  # row, col


def dijkstra(free: np.ndarray, start, goals):
    H, W = free.shape
    INF = 1e18
    dist = np.full((H, W), INF)
    prev = np.full((H, W, 2), -1, np.int32)
    dist[start] = 0.0
    pq = [(0.0, start)]
    remaining = set(goals)
    nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while pq and remaining:
        dd, (r, c) = heapq.heappop(pq)
        if dd > dist[r, c]:
            continue
        remaining.discard((r, c))
        for dr, dc, w in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and free[rr, cc]:
                if dr and dc and not (free[r, cc] and free[rr, c]):
                    continue
                nd = dd + w
                if nd < dist[rr, cc]:
                    dist[rr, cc] = nd
                    prev[rr, cc] = (r, c)
                    heapq.heappush(pq, (nd, (rr, cc)))
    return dist, prev


def exact_clearance(cols: list[Collider], ch: dict):
    """Narrowest passage through the blocked aisle section (min over scanlines of the largest free gap),
    from exact footprint/scanline intersection - independent of the 5 cm grid."""
    from .geom import footprint, largest_gap, scan_interval

    x0, x1 = ch["aisle_x_range"]
    polys = []
    for c in cols:
        if c.is_mesh or c.category in ("hazard", "floor", "roof"):
            continue
        lo, hi = c.obb.aabb()
        if hi[2] < Z_BAND[0] or lo[2] > Z_BAND[1] or hi[0] < x0 - 0.6 or lo[0] > x1 + 0.6:
            continue
        if hi[1] < ch["y_range"][0] or lo[1] > ch["y_range"][1]:
            continue
        polys.append(footprint(c.obb))
    worst, worst_y = 1e9, None
    for y in np.arange(ch["y_range"][0], ch["y_range"][1], 0.01):
        iv = [i for i in (scan_interval(p, y) for p in polys) if i]
        g = largest_gap(iv)
        if g < worst:
            worst, worst_y = g, float(y)
    return float(worst), worst_y


def nav_check(cols: list[Collider], graph: dict, out_png: Path | None = None):
    from scipy.ndimage import distance_transform_edt

    occ = build_grid(cols)
    free = distance_transform_edt(~occ) * RES >= ROBOT_R + 0.02
    sx, sy, _ = graph["robot"]["spawn"]
    start = w2g(sx, sy)
    res = {"grid_shape": list(occ.shape), "resolution_m": RES, "robot_radius_m": ROBOT_R, "spawn_free": bool(free[start]), "goals": []}
    gcells = {}
    for g in graph["goals"]:
        gx, gy, _ = g["base_pose"]
        gcells[g["name"]] = w2g(gx, gy)
    dist, prev = dijkstra(free, start, [v for v in gcells.values() if free[v]])
    paths = {}
    for g in graph["goals"]:
        cell = gcells[g["name"]]
        ok = bool(free[cell]) and dist[cell] < 1e17
        length = float(dist[cell] * RES) if ok else None
        res["goals"].append({"name": g["name"], "type": g["type"], "goal_cell_free": bool(free[cell]), "reachable": ok,
                             "path_length_m": None if length is None else round(length, 2)})
        if ok:
            p, cur = [], cell
            while cur != start:
                p.append(cur)
                cur = tuple(prev[cur])
            paths[g["name"]] = p
    res["all_reachable"] = bool(res["spawn_free"] and all(g["reachable"] for g in res["goals"]))
    res["n_goals"] = len(graph["goals"])
    # chokepoint clearance on the *un-inflated* occupancy
    ch = graph["chokepoint"]
    x0, x1 = ch["aisle_x_range"]
    c0, c1 = int((x0 - LY.X0) / RES), int((x1 - LY.X0) / RES)
    r0, r1 = int((LY.Y1 - ch["y_range"][1]) / RES), int((LY.Y1 - ch["y_range"][0]) / RES)
    widest = []
    for r in range(r0, r1):
        row = ~occ[r, c0 + 1 : c1 - 1]
        best = run = 0
        for v in row:
            run = run + 1 if v else 0
            best = max(best, run)
        widest.append(best * RES)
    res["chokepoint_min_clearance_grid_m"] = round(float(min(widest)), 3)
    ex, ey = exact_clearance(cols, ch)
    res["chokepoint_min_clearance_m"] = round(ex, 3)
    res["chokepoint_narrowest_y"] = round(ey, 2)
    gx = (ch["blocked_until_x"] + ch["free_edge_right_x"]) / 2
    res["chokepoint_centre_cell_free_for_robot"] = bool(free[w2g(gx, ch["y_center"])])
    res["chokepoint_target_m"] = ch["target_clearance_m"]
    res["chokepoint_ok"] = bool(abs(res["chokepoint_min_clearance_m"] - ch["target_clearance_m"]) <= 0.03 and res["chokepoint_centre_cell_free_for_robot"])
    if out_png is not None:
        rgb = np.zeros(occ.shape + (3,), np.uint8)
        rgb[...] = (235, 235, 235)
        rgb[~free] = (200, 160, 160)
        rgb[occ] = (40, 40, 40)
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        for name, p in paths.items():
            dr.line([(c, r) for r, c in p[::3]], fill=(30, 120, 220), width=1)
        for g in graph["goals"]:
            r, c = w2g(*g["base_pose"][:2])
            dr.ellipse([c - 5, r - 5, c + 5, r + 5], outline=(0, 160, 60) if g["type"] != "pick_high" else (230, 120, 0), width=2)
        r, c = start
        dr.ellipse([c - 6, r - 6, c + 6, r + 6], fill=(220, 0, 0))
        im.save(out_png)
    return res


def run(seed_dir: Path) -> dict:
    seed_dir = Path(seed_dir)
    stage = Usd.Stage.Open(str(seed_dir / "warehouse.usd"))
    graph = json.loads((seed_dir / "scene_graph.json").read_text())
    cols = collect(stage)
    pen = spawn_penetration(cols)
    flt = floating_bodies(cols)
    hz = hazard_paths_clear(stage, cols, graph)
    nav = nav_check(cols, graph, seed_dir / "nav_occupancy.png")
    res = {
        "n_colliders": len(cols),
        "spawn_interpenetrations": pen,
        "floating_bodies": flt,
        "hazard_path_conflicts": hz,
        "navigation": nav,
        "passed": bool(not pen and not flt and not hz and nav["all_reachable"] and nav["chokepoint_ok"]),
    }
    return res


if __name__ == "__main__":
    d = Path(sys.argv[1])
    r = run(d)
    (d / "static_validation.json").write_text(json.dumps(r, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: (v if k != "navigation" else {kk: vv for kk, vv in v.items() if kk != "goals"}) for k, v in r.items()}, indent=1)[:4000])
