"""Record the pre-refactor warehouse generator's metrics (seeds 0-4) -> tests/fixtures/warehouse_baseline.json.

Historical: it needs the pre-refactor modules (``generate.build`` of the old generator and ``static_validate``), which were
removed in phase 5.  To regenerate the fixture, check out commit f4daa5f and run:

    uv run python tests/fixtures/make_warehouse_baseline.py /tmp/somewhere
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sceneforge import generate, static_validate

SEEDS = (0, 1, 2, 3, 4)
STRUCTURAL = ("rack_row", "building", "floor", "door_rollup", "door_swing", "gate_sliding", "drawer", "conveyor_belt",
              "conveyor_roller", "work_table", "charger", "dock_leveler", "cage_fence", "hazard", "robot_spawn", "camera")


def metrics(d: Path) -> dict:
    g = json.loads((d / "scene_graph.json").read_text())
    st = static_validate.run(d)
    return {
        "counts": g["counts"],
        "n_dynamic_bodies": g["n_dynamic_bodies"],
        "lighting": g["lighting"]["setup_name"],
        "building_size_m": g["building"]["size_m"],
        "articulations": sorted((a["name"], a["kind"], a["lower"], a["upper"], a["unit"]) for a in g["articulations"]),
        "conveyors": sorted((c["name"], c["kind"], c["speed_mps"]) for c in g["conveyors"]),
        "goals": sorted((x["name"], x["type"]) for x in g["goals"]),
        "hazards": sorted((h["kind"], h["path_mode"], h["speed_mps"]) for h in g["hazards"]),
        "cameras": sorted(c["name"] for c in g["cameras"]),
        "chokepoint_clearance_m": st["navigation"]["chokepoint_min_clearance_m"],
        "static": {"spawn_interpenetrations": len(st["spawn_interpenetrations"]), "floating_bodies": len(st["floating_bodies"]),
                   "hazard_path_conflicts": len(st["hazard_path_conflicts"]), "all_reachable": st["navigation"]["all_reachable"],
                   "passed": st["passed"]},
    }


def main():
    out = Path(sys.argv[1])
    per_seed = {}
    for s in SEEDS:
        d = generate.build(s, out)
        per_seed[s] = metrics(d)
        print(s, per_seed[s]["counts"], per_seed[s]["static"], flush=True)
    cats = sorted({c for m in per_seed.values() for c in m["counts"]})
    ranges = {c: [min(m["counts"].get(c, 0) for m in per_seed.values()), max(m["counts"].get(c, 0) for m in per_seed.values())] for c in cats}
    ranges["n_dynamic_bodies"] = [min(m["n_dynamic_bodies"] for m in per_seed.values()), max(m["n_dynamic_bodies"] for m in per_seed.values())]
    res = {"source_commit": "f4daa5f", "seeds": list(SEEDS), "structural_categories": list(STRUCTURAL), "count_ranges": ranges,
           "per_seed": {str(k): v for k, v in per_seed.items()}}
    p = Path(__file__).with_name("warehouse_baseline.json")
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
