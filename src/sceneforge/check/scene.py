"""Loading a built scene (stage + scene_graph.json) and the generic per-scene config with legacy fallbacks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..colliders import Collider

LEGACY_ROBOT_R = 0.38
DEFAULT_NAV = {"resolution_m": 0.05, "z_band": [0.10, 1.80], "ignore_categories": ["hazard", "floor", "roof"]}


class LoadError(Exception):
    pass


@dataclass
class SceneData:
    directory: Path
    usd_path: Path
    graph: dict
    stage: object = None
    cols: list[Collider] = field(default_factory=list)
    spec_ids: dict = field(default_factory=dict)  # prim path -> spec id (records + instances)

    def spec_id(self, prim_path: str | None) -> str | None:
        """Spec id of the record owning ``prim_path`` (nearest ancestor with a record)."""
        p = prim_path
        while p and p != "/":
            if p in self.spec_ids:
                return self.spec_ids[p]
            p = p.rsplit("/", 1)[0] if "/" in p else ""
        return None


def locate(path) -> tuple[Path, Path, dict]:
    p = Path(path)
    if p.is_dir():
        d, usd = p, None
    elif p.suffix.lower() in (".usd", ".usda", ".usdc"):
        d, usd = p.parent, p
    else:
        raise LoadError(f"{p} is neither a scene directory nor a USD file")
    gp = d / "scene_graph.json"
    if not gp.exists():
        raise LoadError(f"no scene_graph.json in {d}")
    try:
        graph = json.loads(gp.read_text())
    except Exception as e:  # noqa: BLE001
        raise LoadError(f"scene_graph.json unreadable: {e}") from e
    if usd is None:
        name = graph.get("usd")
        if name:
            usd = d / name
        else:
            cands = sorted(d.glob("*.usd*"))
            if not cands:
                raise LoadError(f"scene_graph.json has no 'usd' key and {d} holds no USD file")
            usd = cands[0]
    if not usd.exists():
        raise LoadError(f"USD file {usd} does not exist")
    return d, usd, graph


def build_spec_id_map(graph: dict) -> dict:
    m = {}
    inst = graph.get("instances") or {}
    if isinstance(inst, dict):
        for sid, paths in inst.items():
            for pp in ([paths] if isinstance(paths, str) else (paths or [])):
                if isinstance(pp, str):
                    m[pp] = sid
    for r in graph.get("objects", []) or []:
        if r.get("spec_id") and r.get("path"):
            m[r["path"]] = r["spec_id"]
    return m


def environment_bounds(graph: dict, cols: list[Collider]):
    env = graph.get("environment") or graph.get("building")
    if env and env.get("x_range") and env.get("y_range"):
        return [float(v) for v in env["x_range"]], [float(v) for v in env["y_range"]]
    import numpy as np

    lo = np.min([c.obb.aabb()[0] for c in cols], axis=0)
    hi = np.max([c.obb.aabb()[1] for c in cols], axis=0)
    return [float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]


def nav_config(graph: dict) -> dict:
    cfg = dict(DEFAULT_NAV)
    cfg.update(graph.get("nav") or {})
    return cfg


def robot_radius(graph: dict) -> float:
    return float((graph.get("robot") or {}).get("radius_m", LEGACY_ROBOT_R))


def clearance_checks(graph: dict) -> list[dict]:
    if graph.get("clearance_checks"):
        return list(graph["clearance_checks"])
    ch = graph.get("chokepoint")
    if ch and ch.get("aisle_x_range") and ch.get("y_range"):
        x0, x1 = ch["aisle_x_range"]
        y0, y1 = ch["y_range"]
        return [{"id": "chokepoint", "rect": [x0, y0, x1, y1], "axis": "x", "span": None,
                 "target": ch.get("target_clearance_m", 0.9), "tolerance": 0.03}]
    return []
