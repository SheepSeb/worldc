"""Prompt adherence: does the built scene contain what the prompt asked for?

Counts come from ``scene_graph.json``: placed instances by registry asset id (``instances``) and walking people (``hazards``;
forklift agents are ignored, forklifts are not part of the evaluation).
Scene graphs from the old warehouse generator have no ``instances``; they fall back to scene-graph record categories.
Each requested field of the prompt row is compared exactly (-1 = unspecified, skipped); ``adherence`` is the fraction of
specified fields that match.
"""
from __future__ import annotations

from collections import Counter

from .scene import Scene

# registry asset ids (assets/registry.py) per prompt field
WORKBENCH_ASSETS = {"work_table", "workbench_steel", "workbench_modular", "packing_table_heavy", "packing_table_deluxe"}  # + curated SimReady
ROBOT_CELL_ASSETS = {"robot_cell"}
CONVEYOR_ASSETS = {"conveyor_belt", "conveyor_roller"}
RACK_ASSETS = {"rack_row", "parts_shelf", "shelf_rack", "bulk_storage_rack"}  # prompts count parts shelving units
DOOR_ASSETS = {"rollup_door", "swing_door"}
HUMAN_ASSETS = {"human", "worker_static"}
# legacy scene graphs (no `instances`): scene-graph record categories, one record per instance
LEGACY_CATEGORIES = {"n_benches": {"work_table"}, "n_robot_cells": {"robot_cell"}, "n_conveyors": {"conveyor_belt", "conveyor_roller"},
                     "n_racks": {"rack_row"}, "n_doors": {"door_rollup", "door_swing"}}

# prompt column -> (scene count key, kind)
FIELDS = {"n_benches": ("n_benches", "count"), "n_robot_cells": ("n_robot_cells", "count"), "conveyor": ("n_conveyors", "bool"),
          "n_racks": ("n_racks", "count"), "n_doors": ("n_doors", "count"),
          "n_people": ("n_humans", "count")}


def counts(scene: Scene) -> dict[str, int]:
    hz = Counter((h.get("kind") or "") for h in scene.graph.get("hazards") or [])
    humans = sum(n for k, n in hz.items() if k.startswith("human"))  # human / human_capsule; forklift agents are skipped
    if scene.instances:
        a = Counter(i.asset for i in scene.instances)
        n = lambda ids: sum(a[x] for x in ids)  # noqa: E731
        return {"n_benches": n(WORKBENCH_ASSETS), "n_robot_cells": n(ROBOT_CELL_ASSETS), "n_conveyors": n(CONVEYOR_ASSETS),
                "n_racks": n(RACK_ASSETS), "n_doors": n(DOOR_ASSETS), "n_humans": humans + n(HUMAN_ASSETS)}
    cat = Counter(o.get("category") for o in scene.graph.get("objects") or [])
    out = {k: sum(cat[c] for c in cs) for k, cs in LEGACY_CATEGORIES.items()}
    return {**out, "n_humans": humans}


def _requested(v) -> int | bool | None:
    """Prompt cell -> requested value; None when unspecified (-1 / empty)."""
    if v is None or str(v).strip() in ("", "-1"):
        return None
    s = str(v).strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return int(float(s))


def score(scene: Scene, prompt_row: dict) -> dict:
    """Per field: requested, got, match (+ abs_err for counts); ``adherence`` over the specified fields (None if none)."""
    got = counts(scene)
    out: dict = {"counts": got, "fields": {}}
    for col, (key, kind) in FIELDS.items():
        req = _requested(prompt_row.get(col))
        if req is None:
            continue
        g = got[key]
        if kind == "bool":
            out["fields"][col] = {"requested": bool(req), "got": g > 0, "match": bool(req) == (g > 0)}
        else:
            out["fields"][col] = {"requested": int(req), "got": g, "match": g == int(req), "abs_err": abs(g - int(req))}
    f = out["fields"]
    out["adherence"] = round(sum(v["match"] for v in f.values()) / len(f), 4) if f else None
    return out
