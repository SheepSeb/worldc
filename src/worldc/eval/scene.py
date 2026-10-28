"""A built scene as the evaluators see it: scene graph, resolved spec, instance footprints, door openings and the occupancy grid.

Needs no Isaac Sim.  The occupancy grid is not saved per scene (only a PNG), so ``occupancy`` rebuilds it with ``usd-core``
from the scene's own USD colliders, exactly as ``check_scene`` does (``check/nav.py``: z band, footprints, inflation).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..assets import registry

EXIT_INSET_M = 0.6  # a door's walking-in point lies this far inside, on the opening's centre line (as in compile.py)


@dataclass
class Inst:
    key: str
    asset: str
    spec_id: str
    xy: tuple[float, float]
    yaw: float  # radians
    params: dict
    parent: str | None = None
    root: str = ""

    def asset_obj(self):
        return registry.get(self.asset) if registry.exists(self.asset) else None

    def local_polygon(self) -> np.ndarray | None:
        a = self.asset_obj()
        if a is None:
            return None
        return np.asarray(a.footprint(a.params(self.params)).polygon, float)

    def to_world(self, pts) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        p = np.asarray(pts, float).reshape(-1, 2)
        return np.stack([self.xy[0] + c * p[:, 0] - s * p[:, 1], self.xy[1] + s * p[:, 0] + c * p[:, 1]], 1)

    def polygon(self) -> np.ndarray | None:
        p = self.local_polygon()
        return None if p is None else self.to_world(p)


@dataclass
class Opening:
    id: str
    wall: str
    a: tuple[float, float]  # ends of the opening on the wall's interior line
    b: tuple[float, float]
    inside: tuple[float, float]  # walking-in point


@dataclass
class Scene:
    directory: Path
    graph: dict
    spec: dict | None
    instances: list[Inst] = field(default_factory=list)

    def by_asset(self, ids) -> list[Inst]:
        return [i for i in self.instances if i.asset in ids]

    @property
    def x_range(self):
        return tuple((self.graph.get("environment") or self.graph.get("building") or {}).get("x_range") or (0, 0))

    @property
    def y_range(self):
        return tuple((self.graph.get("environment") or self.graph.get("building") or {}).get("y_range") or (0, 0))

    def openings(self) -> list[Opening]:
        """Door openings (sill 0) from the resolved spec; empty for outdoor scenes or when there is no spec."""
        env = (self.spec or {}).get("environment") or {}
        if env.get("kind", "indoor") != "indoor":
            return []
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        out = []
        for o in (env.get("walls") or {}).get("openings") or []:
            if float(o.get("sill", 0.0)) > 0:
                continue
            c, h = float(o["center"]), float(o["width"]) / 2
            line, n = {"south": (lambda t: (t, y0), (0, 1)), "north": (lambda t: (t, y1), (0, -1)),
                       "west": (lambda t: (x0, t), (1, 0)), "east": (lambda t: (x1, t), (-1, 0))}[o["wall"]]
            m = line(c)
            out.append(Opening(o["id"], o["wall"], line(c - h), line(c + h), (m[0] + EXIT_INSET_M * n[0], m[1] + EXIT_INSET_M * n[1])))
        return out

    def zone_tags(self) -> dict[str, list[str]]:
        return {z["id"]: z.get("tags", []) for z in (self.spec or {}).get("zones", []) or []}

    def zones(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in ((self.graph.get("layout") or {}).get("zones") or {}).items()}

    def occupancy(self, radius: float):
        """(grid, collider polygons, colliders) with ``grid.occ`` (raw footprints) and ``grid.free`` (inflated by ``radius``)."""
        from pxr import Usd

        from ..check import nav as NV
        from ..check import scene as SC
        from ..colliders import collect

        d, usd, graph = SC.locate(self.directory)
        stage = Usd.Stage.Open(str(usd))
        cols = collect(stage)
        cfg = SC.nav_config(graph)
        xr, yr = SC.environment_bounds(graph, cols)
        polys = NV.polygons(cols, NV.nav_colliders(cols, cfg["z_band"], set(cfg["ignore_categories"])))
        grid = NV.Grid(xr, yr, float(cfg["resolution_m"]))
        NV.build_grid(grid, polys, radius)
        return grid, polys, cols


def load(directory) -> Scene:
    d = Path(directory)
    graph = json.loads((d / "scene_graph.json").read_text())
    spec = None
    for name in ("resolved_spec.json", "spec.json"):
        if (d / name).is_file():
            spec = json.loads((d / name).read_text())
            break
    if spec is None and (d.parent / "spec.json").is_file():  # dataset layout: runs/<id>/spec.json next to runs/<id>/scene/
        spec = json.loads((d.parent / "spec.json").read_text())
    insts = []
    raw = graph.get("instances")
    if isinstance(raw, dict):
        for k, v in raw.items():
            if not isinstance(v, dict) or "asset" not in v:
                continue
            pose = v.get("pose") or {}
            xyz = pose.get("xyz") or [0, 0, 0]
            insts.append(Inst(k, v["asset"], v.get("spec_id", k), (float(xyz[0]), float(xyz[1])), math.radians(float(pose.get("yaw_deg", 0.0))),
                              v.get("params") or {}, v.get("parent"), v.get("root", "")))
    return Scene(d, graph, spec, insts)
