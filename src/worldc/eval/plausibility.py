"""Layout plausibility: does the layout make functional sense?  Rule checks per instance, aggregated per scene.

Post-hoc only (no issues to the repair loop, no effect on acceptance).  Geometry comes from ``scene_graph.json`` (instance
poses + registry footprints) and the occupancy grid rebuilt from the scene's colliders (``Scene.occupancy``).

* bench_access: each workbench's working side has BENCH_ACCESS_DEPTH_M of free floor along it.  The working side is the
  side its ``front`` port points to when the asset has one (method ``port``: ``work_table``); otherwise the long side
  facing the most free floor (method ``free_space``: the curated SimReady benches).
* conveyor_connected (per conveyor): (a) at least one end is within CONVEYOR_LINK_M of a workbench, robot cell, pallet, door
  or other conveyor, and (b) at least one workbench or robot cell is within CONVEYOR_STATION_M of the conveyor (end or side).
* storage_reachable: every rack / shelving unit and pallet staging area is reachable on foot from a door, on the grid
  inflated by HUMAN_RADIUS_M (outdoors: from the site boundary).
* cell_not_blocking: no robot cell's footprint within CELL_DOOR_MIN_M of a door opening.

Per rule: ``{"n", "passed", "rate"}`` (rate None when the scene has no instance of it); ``plausibility`` = mean of the rates
that exist.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .adherence import CONVEYOR_ASSETS, RACK_ASSETS, ROBOT_CELL_ASSETS, WORKBENCH_ASSETS
from .scene import Inst, Scene

# ---- thresholds (all rules) ------------------------------------------------------------------------------------------
BENCH_ACCESS_DEPTH_M = 0.8      # free floor in front of a bench's working side
BENCH_EDGE_GAP_M = 0.02         # the free strip starts this far from the bench edge
BENCH_SIDE_INSET_M = 0.10       # and ignores this much at both ends of the side
CONVEYOR_LINK_M = 1.0           # (a) a conveyor end must be this close to something it feeds / is fed by
# (b) a workbench / robot cell must be this close to the conveyor (end or side). 1.5 m = the mandatory 1.2 m aisle
# (check_scene) + 0.3 m tolerance: a bench whose working side faces the belt must leave a 1.2 m walkway in between, so a
# tighter (b) would penalise layouts for obeying the aisle rule.
CONVEYOR_STATION_M = 1.5
HUMAN_RADIUS_M = 0.26           # walking grid inflation
REACH_TOL_M = 0.30              # a target counts as reached when a free cell lies this close to it
CELL_DOOR_MIN_M = 2.0           # robot cell footprint to door opening
PALLET_ASSETS = {"pallet", "pallet_load", "pallet_euro", "pallet_block"}
DOOR_ASSETS = {"rollup_door", "swing_door"}
RULES = ("bench_access", "conveyor_connected", "storage_reachable", "cell_not_blocking")


# ---- geometry helpers ------------------------------------------------------------------------------------------------
def _point_poly_dist(pt, poly: np.ndarray) -> float:
    from ..check.safety import _seg_dist

    p = np.asarray(pt, float).reshape(1, 2)
    e, r = np.roll(poly, -1, axis=0) - poly, p[0] - poly
    inside = bool(((e[:, 0] * r[:, 1] - e[:, 1] * r[:, 0]) >= 0).all())  # CCW convex polygon
    return 0.0 if inside else _seg_dist(p, poly)


def _point_seg_dist(pt, a, b) -> float:
    pt, a, b = (np.asarray(v, float) for v in (pt, a, b))
    e = b - a
    t = 0.0 if not e.any() else float(np.clip((pt - a) @ e / (e @ e), 0.0, 1.0))
    return float(np.linalg.norm(pt - (a + t * e)))


def _seg_poly_dist(a, b, poly: np.ndarray) -> float:
    from ..check.safety import poly_distance

    a, b = np.asarray(a, float), np.asarray(b, float)
    n = np.array([-(b - a)[1], (b - a)[0]])
    n = n / (np.linalg.norm(n) or 1.0) * 1e-3
    return poly_distance(np.array([a - n, b - n, b + n, a + n]), poly)


def _mask(grid, polys) -> np.ndarray:
    """Cells whose centres lie inside any of ``polys``."""
    img = Image.new("1", (grid.nx, grid.ny), 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        d.polygon([((x - grid.x0) / grid.res - 0.5, (grid.y1 - y) / grid.res - 0.5) for x, y in p], fill=1)
    return np.asarray(img, bool)


def _near(grid, polys, tol) -> np.ndarray:
    m = _mask(grid, polys)
    if not m.any():
        return m
    return ndimage.distance_transform_edt(~m) * grid.res <= tol


def _rate(results: list[bool]) -> dict:
    return {"n": len(results), "passed": int(sum(results)), "rate": round(sum(results) / len(results), 4) if results else None}


# ---- rules -----------------------------------------------------------------------------------------------------------
def bench_access(scene: Scene, occ) -> tuple[dict, list]:
    grid, polys, cols = occ
    out, details = [], []
    for b in scene.by_asset(WORKBENCH_ASSETS):
        own = _mask(grid, [polys[i] for i in polys if b.root and (cols[i].path == b.root or cols[i].path.startswith(b.root + "/"))])
        poly = b.local_polygon()
        if poly is None:
            continue
        (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
        g, inset, depth = BENCH_EDGE_GAP_M, BENCH_SIDE_INSET_M, BENCH_ACCESS_DEPTH_M
        strips = {"+y": [(x0 + inset, y1 + g), (x1 - inset, y1 + g), (x1 - inset, y1 + g + depth), (x0 + inset, y1 + g + depth)],
                  "-y": [(x0 + inset, y0 - g - depth), (x1 - inset, y0 - g - depth), (x1 - inset, y0 - g), (x0 + inset, y0 - g)],
                  "+x": [(x1 + g, y0 + inset), (x1 + g + depth, y0 + inset), (x1 + g + depth, y1 - inset), (x1 + g, y1 - inset)],
                  "-x": [(x0 - g - depth, y0 + inset), (x0 - g, y0 + inset), (x0 - g, y1 - inset), (x0 - g - depth, y1 - inset)]}
        a = b.asset_obj()
        front = next((an for an in a.anchors(a.params(b.params)) if an.name == "front"), None) if a else None
        if front is not None:
            fx, fy = front.pos[0], front.pos[1]
            sides, method = (["+y" if fy > 0 else "-y"] if abs(fy) >= abs(fx) else ["+x" if fx > 0 else "-x"]), "port"
        else:
            sides, method = (["+y", "-y"] if (x1 - x0) >= (y1 - y0) else ["+x", "-x"]), "free_space"
        best = None
        for sd in sides:
            m = _mask(grid, [b.to_world(strips[sd])])
            cells = int(m.sum())
            blocked = int((m & grid.occ & ~own).sum()) if cells else 0  # the bench's own collider boxes are not in its way
            inside = all(grid.inside(*p) for p in b.to_world(strips[sd]))
            score = (inside and cells > 0 and blocked == 0, -blocked)
            if best is None or score > best[0]:
                best = (score, sd, blocked, cells, inside)
        ok = bool(best[0][0])
        out.append(ok)
        details.append({"rule": "bench_access", "id": b.key, "pass": ok, "method": method, "side": best[1], "blocked_cells": best[2],
                        "strip_cells": best[3], "inside_room": best[4]})
    return _rate(out), details


def conveyor_connected(scene: Scene) -> tuple[dict, list]:
    """Per conveyor: (a) an end within CONVEYOR_LINK_M of a workbench, robot cell, pallet, door or other conveyor, and (b) a
    workbench or robot cell within CONVEYOR_STATION_M of the conveyor itself (at an end or along a side)."""
    from ..check.safety import poly_distance

    def polys(ids):
        return [(i.key, p) for i in scene.by_asset(ids) if (p := i.polygon()) is not None]

    end_targets = polys(WORKBENCH_ASSETS | ROBOT_CELL_ASSETS | PALLET_ASSETS | DOOR_ASSETS | CONVEYOR_ASSETS)
    stations = polys(WORKBENCH_ASSETS | ROBOT_CELL_ASSETS)
    out, details = [], []
    for c in scene.by_asset(CONVEYOR_ASSETS):
        L = float(c.params.get("length", 4.0))
        ends = []
        for end, y in (("infeed", -L / 2), ("outfeed", L / 2)):
            pt = c.to_world([(0.0, y)])[0]
            dists = [(_point_poly_dist(pt, p), k) for k, p in end_targets if k != c.key]
            dists += [(_point_seg_dist(pt, o.a, o.b), f"door:{o.id}") for o in scene.openings()]
            d, near = min(dists, default=(math.inf, None))
            ends.append({"end": end, "nearest": near, "distance_m": None if math.isinf(d) else round(d, 3), "linked": d <= CONVEYOR_LINK_M})
        body = c.polygon()
        sd, station = min(((poly_distance(body, p), k) for k, p in stations), default=(math.inf, None)) if body is not None else (math.inf, None)
        end_ok, station_ok = any(e["linked"] for e in ends), sd <= CONVEYOR_STATION_M
        ok = end_ok and station_ok
        out.append(ok)
        details.append({"rule": "conveyor_connected", "id": c.key, "pass": ok, "end_linked": end_ok, "station_near": station_ok,
                        "nearest_station": station, "station_distance_m": None if math.isinf(sd) else round(sd, 3), "ends": ends})
    return _rate(out), details


def storage_reachable(scene: Scene, occ) -> tuple[dict, list]:
    grid = occ[0]
    targets: list[tuple[str, list]] = [(r.key, [r.polygon()]) for r in scene.by_asset(RACK_ASSETS) if r.polygon() is not None]
    tags = scene.zone_tags()
    staging = {z: rect for z, rect in scene.zones().items() if "staging" in z.lower() or any("staging" in t for t in tags.get(z, []))}
    for z, (x0, y0, x1, y1) in staging.items():
        targets.append((f"zone:{z}", [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]))
    for p in scene.by_asset(PALLET_ASSETS):  # floor pallets outside a staging zone are a staging area of their own
        if p.parent is None and not any(x0 <= p.xy[0] <= x1 and y0 <= p.xy[1] <= y1 for x0, y0, x1, y1 in staging.values()):
            targets.append((p.key, [p.polygon()]))
    if not targets:
        return _rate([]), []
    free = grid.free
    src = np.zeros_like(free)
    openings = scene.openings()
    if openings:
        from ..check.safety import _disk

        for o in openings:
            src |= _disk(grid, *o.inside, REACH_TOL_M)
    elif (scene.graph.get("environment") or {}).get("kind") == "outdoor":
        band = int(math.ceil((HUMAN_RADIUS_M + REACH_TOL_M) / grid.res)) + 2
        src[:band, :] = src[-band:, :] = src[:, :band] = src[:, -band:] = True
    lab, _ = ndimage.label(free, structure=np.ones((3, 3), bool))
    ids = np.unique(lab[src & free])
    reach = np.isin(lab, ids[ids > 0])
    out, details = [], []
    for key, polys in targets:
        ok = bool((reach & _near(grid, polys, REACH_TOL_M)).any())
        out.append(ok)
        details.append({"rule": "storage_reachable", "id": key, "pass": ok, "from": "doors" if openings else "boundary" if src.any() else "none"})
    return _rate(out), details


def cell_not_blocking(scene: Scene) -> tuple[dict, list]:
    recs = {o["path"]: o for o in scene.graph.get("objects") or [] if o.get("category") == "robot_cell"}
    out, details = [], []
    openings = scene.openings()
    for c in scene.by_asset(ROBOT_CELL_ASSETS):
        poly = np.asarray(recs[c.root]["safety_polygon"], float) if c.root in recs else c.polygon()
        d, near = min(((_seg_poly_dist(o.a, o.b, poly), o.id) for o in openings), default=(math.inf, None))
        ok = d >= CELL_DOOR_MIN_M
        out.append(ok)
        details.append({"rule": "cell_not_blocking", "id": c.key, "pass": ok, "nearest_door": near,
                        "distance_m": None if math.isinf(d) else round(d, 3)})
    return _rate(out), details


def score(scene: Scene) -> dict:
    occ = scene.occupancy(HUMAN_RADIUS_M)  # (grid inflated by the human radius, collider footprints, colliders)
    rules, details = {}, []
    for name, fn in (("bench_access", lambda: bench_access(scene, occ)), ("conveyor_connected", lambda: conveyor_connected(scene)),
                     ("storage_reachable", lambda: storage_reachable(scene, occ)), ("cell_not_blocking", lambda: cell_not_blocking(scene))):
        rules[name], d = fn()
        details += d
    return {"rules": rules, "plausibility": aggregate(rules), "details": details}


def aggregate(rules: dict) -> float | None:
    """Mean of the rule pass rates; a rule with no instance in the scene (rate None) is left out, not counted as a pass."""
    rates = [r["rate"] for r in rules.values() if r["rate"] is not None]
    return round(sum(rates) / len(rates), 4) if rates else None
