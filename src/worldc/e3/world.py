"""A built scene as a 2D world: occupancy grids for the robot and for people, task points, robot spawn, body and goals.

Reads the scene's own colliders (the same nav colliders as ``check_scene``) and the ``safety`` block of ``scene_graph.json``;
no Isaac Sim.  Grid geometry is the post-build check's (``check.nav``), so 2D and the checks agree on what is free.

Robot body: the footprint is the robot table's measured bounds (``assets/library/robots``: x by y in the robot's default
pose, centred on its origin).  Which way the robot drives in that frame is the travel heading measured by the Isaac Sim drive
calibration (``drive/drive_result.json``, ``travel_heading_in_robot_frame_rad``); an E3 variant carries its base scene's value in
``e3_variant.json``.  Without either, +x is assumed and the source says so.  Braking deceleration (for the ``fields``
policy) is the table's ``brake_decel_mps2``.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..check import nav as NV
from ..check import scene as SC
from ..check.safety import HUMAN_RADIUS_M
from ..drive import _smooth

POI_SNAP_M = 0.8  # a task point moves to the nearest free cell within this distance, or is dropped
GATE_STAND_OFF_M = 0.6  # a worker at a robot-cell gate stands this far outside it
EIGHT = np.ones((3, 3), bool)


@dataclass
class World:
    name: str
    xr: tuple[float, float]
    yr: tuple[float, float]
    res: float
    polys: dict  # static nav footprints (collider index -> polygon); occluders added by a tier transform come in here too
    cells: list[np.ndarray]  # robot-cell safety footprints (people stay out)
    spawn: tuple[float, float, float]  # robot frame yaw
    robot: dict  # {asset, size_m (x, y in the robot frame), travel_heading_rad, source}
    robot_r: float
    goals: list[dict]
    raw_pois: dict[str, list[tuple[float, float]]]
    grid: NV.Grid = None  # robot: free = footprint radius clear of static colliders
    hgrid: NV.Grid = None  # people: free = HUMAN_RADIUS_M clear of static colliders and robot-cell safety footprints
    pois: dict = None
    _paths: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.grid = NV.build_grid(NV.Grid(self.xr, self.yr, self.res), self.polys, self.robot_r)
        hpolys = dict(self.polys)
        for j, P in enumerate(self.cells):
            hpolys[f"cell{j}"] = P
        self.hgrid = NV.build_grid(NV.Grid(self.xr, self.yr, self.res), hpolys, HUMAN_RADIUS_M)
        self.pois = self._snap_pois()

    @property
    def body_half(self) -> tuple[float, float]:
        return self.robot["size_m"][0] / 2, self.robot["size_m"][1] / 2

    @property
    def phi0(self) -> float:
        """Travel direction in the robot frame (rad)."""
        return float(self.robot["travel_heading_rad"])

    def with_obstacles(self, extra: dict, name: str | None = None) -> "World":
        """The same world with more static footprints (e.g. a tier's occluders)."""
        return World(name or self.name, self.xr, self.yr, self.res, {**self.polys, **extra}, self.cells, self.spawn, self.robot,
                     self.robot_r, self.goals, self.raw_pois)

    def _snap_pois(self) -> dict:
        free = self.hgrid.free
        _, (ri, ci) = ndimage.distance_transform_edt(~free, return_indices=True)
        lab, _ = ndimage.label(free, structure=EIGHT)
        out = {}
        for kind, pts in self.raw_pois.items():
            snapped = []
            for x, y in pts:
                r, c = self.hgrid.clamp(self.hgrid.cell(x, y))
                rr, cc = int(ri[r, c]), int(ci[r, c])
                px, py = self.hgrid.x0 + (cc + 0.5) * self.res, self.hgrid.y1 - (rr + 0.5) * self.res
                if math.hypot(px - x, py - y) <= POI_SNAP_M:
                    snapped.append((round(px, 3), round(py, 3)))
            if snapped:
                out[kind] = list(dict.fromkeys(snapped))
        # keep only task points connected to the (first) door: a point walled off from it is not a task anyone reaches
        ref = (out.get("door") or next(iter(out.values()), [None]))[0]
        if ref is None:
            return {}
        comp = lab[self.hgrid.clamp(self.hgrid.cell(*ref))]
        out = {k: [p for p in v if lab[self.hgrid.clamp(self.hgrid.cell(*p))] == comp] for k, v in out.items()}
        return {k: v for k, v in out.items() if v}

    def human_path(self, a, b) -> np.ndarray:
        """Smoothed shortest walking path between two task points (cached; task points are fixed per world).  Always planned
        from the smaller point to the larger and reversed for the other direction: A* breaks ties differently each way, and the
        path must not depend on which direction was asked for first (a seed's workers must be the same in every run)."""
        key = (tuple(a), tuple(b))
        if key not in self._paths:
            lo, hi = sorted(key)
            p = plan(self.hgrid, self.hgrid.free, lo, hi)
            if p is None:
                raise ValueError(f"no walking path {a} -> {b}")
            self._paths[(lo, hi)] = p
            self._paths[(hi, lo)] = p[::-1].copy()
        return self._paths[key]

    def walkway_paths(self) -> list[np.ndarray]:
        """The walking paths between every pair of task points."""
        pts = [p for v in self.pois.values() for p in v]
        return [self.human_path(a, b) for i, a in enumerate(pts) for b in pts[i + 1:]]



def astar(free: np.ndarray, start, goal):
    """8-connected A* (octile heuristic, no corner cutting) -> list of cells start..goal, or None."""
    H, W = free.shape
    if not (free[start] and free[goal]):
        return None
    gr, gc = goal
    h = lambda r, c: (lambda dr, dc: max(dr, dc) + 0.414 * min(dr, dc))(abs(r - gr), abs(c - gc))  # noqa: E731
    g = np.full((H, W), np.inf)
    prev = np.full((H, W, 2), -1, np.int32)
    g[start] = 0.0
    pq = [(h(*start), 0.0, start)]
    nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while pq:
        _, d, (r, c) = heapq.heappop(pq)
        if (r, c) == goal:
            out = [goal]
            while out[-1] != start:
                out.append(tuple(prev[out[-1]]))
            return out[::-1]
        if d > g[r, c]:
            continue
        for dr, dc, w in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and free[rr, cc]:
                if dr and dc and not (free[r, cc] and free[rr, c]):
                    continue
                nd = d + w
                if nd < g[rr, cc]:
                    g[rr, cc] = nd
                    prev[rr, cc] = (r, c)
                    heapq.heappush(pq, (nd + h(rr, cc), nd, (rr, cc)))
    return None


def plan(grid: NV.Grid, free: np.ndarray, a, b) -> np.ndarray | None:
    """A* between two points on ``free``, line-of-sight smoothed, as an (n, 2) array of xy (exact end points)."""
    s, t = grid.clamp(grid.cell(*a)), grid.clamp(grid.cell(*b))
    cells = astar(free, s, t)
    if cells is None:
        return None
    pts = [(grid.x0 + (c + 0.5) * grid.res, grid.y1 - (r + 0.5) * grid.res) for r, c in _smooth(free, cells)]
    pts[0], pts[-1] = tuple(a[:2]), tuple(b[:2])
    return np.array(pts, float)


def _gate_points(cols, cells: list[dict]) -> list[tuple[float, float]]:
    """Where a worker stands at each robot cell: outside its gate (a ``gate_*`` collider of the cell), else outside the
    middle of its longest edge."""
    out = []
    for cell in cells:
        P = np.asarray(cell["polygon"], float)
        c0 = P.mean(0)
        gates = [c for c in cols if c.category.startswith("gate") and any(c.path == m or c.path.startswith(m + "/") for m in cell["members"])]
        if gates:
            lo, hi = gates[0].obb.aabb()
            g = (np.asarray(lo[:2]) + np.asarray(hi[:2])) / 2
        else:
            e = np.roll(P, -1, 0) - P
            k = int(np.argmax(np.hypot(*e.T)))
            g = P[k] + e[k] / 2
        d = g - c0
        out.append(tuple(g + d / max(np.linalg.norm(d), 1e-9) * GATE_STAND_OFF_M))
    return out


def robot_frame(d: Path, rb: dict) -> dict:
    """Footprint size from the robot table and travel heading from the Isaac drive calibration (see module doc)."""
    from ..assets.robots import table

    info = table().get(rb.get("asset") or "")
    size = [info.size_m[0], info.size_m[1]] if info else [2 * float(rb.get("radius_m", 0.3))] * 2
    out = {"asset": rb.get("asset"), "size_m": [round(v, 4) for v in size],
           "size_source": "robot table (assets/library/robots)" if info else "footprint radius (no table entry)"}
    if info and info.brake_decel_mps2:
        out.update(brake_decel_mps2=info.brake_decel_mps2, brake_source=f"robot table: {info.brake_decel_source}")
    else:
        out.update(brake_decel_mps2=3.0, brake_source="no table entry: worldc drive's deceleration limit")
    variant = d / "e3_variant.json"
    if variant.exists():
        rf = json.loads(variant.read_text())["robot_frame"]
        return {**out, "travel_heading_rad": rf["travel_heading_rad"], "source": rf["source"]}
    dr = d / "drive" / "drive_result.json"
    cal = json.loads(dr.read_text()).get("calibration") if dr.exists() else None
    if cal and "travel_heading_in_robot_frame_rad" in cal:
        return {**out, "travel_heading_rad": float(cal["travel_heading_in_robot_frame_rad"]), "source": f"Isaac drive calibration ({dr})"}
    return {**out, "travel_heading_rad": 0.0, "source": "assumed +x (no Isaac drive calibration)"}


def load_world(scene_dir) -> World:
    from pxr import Usd

    from ..colliders import collect

    d, usd, graph = SC.locate(scene_dir)
    cols = collect(Usd.Stage.Open(str(usd)))
    cfg = SC.nav_config(graph)
    xr, yr = SC.environment_bounds(graph, cols)
    polys = NV.polygons(cols, NV.nav_colliders(cols, cfg["z_band"], set(cfg["ignore_categories"])))
    safety = graph.get("safety") or {}
    cells = safety.get("robot_cells", [])
    raw = {"door": [tuple(e["xy"]) for e in safety.get("exits", [])],
           "bench": [tuple(p) for b in safety.get("workbenches", []) for p in (b["stand_points"][:1] if b.get("has_front") else b["stand_points"])],
           "storage": [((z["rect"][0] + z["rect"][2]) / 2, (z["rect"][1] + z["rect"][3]) / 2) for z in safety.get("storage_zones", [])],
           "cell_gate": _gate_points(cols, cells)}
    rb = graph["robot"]
    goals = [{"id": g.get("id") or g.get("name"), "pose": tuple(g["base_pose"][:3])} for g in graph.get("goals", []) if g.get("reachable", True)]
    name = Path(d).parent.name if Path(d).name == "scene" else Path(d).name
    return World(name, tuple(xr), tuple(yr), float(cfg["resolution_m"]), polys, [np.asarray(c["polygon"], float) for c in cells],
                 tuple(rb["spawn"][:3]), robot_frame(Path(d), rb), SC.robot_radius(graph), goals, {k: v for k, v in raw.items() if v})
