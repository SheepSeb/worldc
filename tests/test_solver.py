"""Solver: relations, constraints, determinism and actionable failure reports (no USD involved)."""
import copy
import math

import numpy as np
import pytest

from worldc.solver.geometry import overlap
from worldc.solver.solve import solve
from worldc.spec.io import parse


def room(objects, **kw):
    d = {"environment": {"footprint": {"size": [10, 8]}, "height": 4}, "objects": objects}
    d.update(kw)
    spec, issues = parse(d)
    assert spec is not None, issues
    return spec


def codes(sol):
    return [i.code for i in sol.issues if i.severity == "error"]


def no_overlaps(sol, margin=0.0):
    flat = [(p.key, P, z0, z1, p.parent) for p in sol.instances for P, z0, z1 in p.pieces]
    for i, (ka, A, a0, a1, pa) in enumerate(flat):
        for kb, B, b0, b1, pb in flat[i + 1:]:
            if ka == kb or pa == kb or pb == ka or (a1 <= b0 + 1e-3 or b1 <= a0 + 1e-3):
                continue
            assert not overlap(A, B, -1e-6), (ka, kb)


def test_inside_count_and_no_overlap():
    spec = room([{"id": "boxes", "asset": "cardboard_box", "count": 12, "params": {"size": [0.5, 0.4, 0.3]},
                  "placement": {"inside": "z"}, "yaw": "random"}], zones=[{"id": "z", "rect": [-4, -3, 0, 3]}])
    sol = solve(spec, 0)
    assert not codes(sol)
    assert len(sol.instances) == 12
    for p in sol.instances:
        P = np.concatenate([q for q, _, _ in p.pieces])
        assert P[:, 0].min() >= -4 + 0.1 - 1e-9 and P[:, 0].max() <= 0 - 0.1 + 1e-9
    no_overlaps(sol)


def test_deterministic_and_seed_sensitive():
    spec = room([{"id": "b", "asset": "cardboard_box", "count": 5, "placement": {"inside": "z"}}], zones=[{"id": "z", "rect": [-4, -3, 4, 3]}])
    a, b, c = solve(spec, 3), solve(spec, 3), solve(spec, 4)
    pa = [i.pose for i in a.instances]
    assert pa == [i.pose for i in b.instances]
    assert pa != [i.pose for i in c.instances]


def test_editing_one_object_keeps_others():
    objs = [{"id": "a", "asset": "cardboard_box", "count": 3, "placement": {"inside": "left"}},
            {"id": "b", "asset": "traffic_cone", "count": 2, "placement": {"inside": "right"}}]
    z = [{"id": "left", "rect": [-4, -3, -1, 3]}, {"id": "right", "rect": [1, -3, 4, 3]}]
    s1 = solve(room(objs, zones=z), 1)
    objs2 = copy.deepcopy(objs)
    objs2[1]["count"] = 3
    s2 = solve(room(objs2, zones=z), 1)
    assert [i.pose for i in s1.instances if i.obj_id == "a"] == [i.pose for i in s2.instances if i.obj_id == "a"]


def test_unsatisfiable_inside_reports_counts_and_hint():
    spec = room([{"id": "big", "asset": "pallet_load", "count": 30, "params": {"variant": "empty"}, "placement": {"inside": "tiny"}}],
                zones=[{"id": "tiny", "rect": [-2, -2, 2, 2]}])
    sol = solve(spec, 0)
    iss = [i for i in sol.issues if i.code == "solve.unsatisfiable.inside"]
    assert iss and iss[0].path == "/objects/0/placement"
    assert "of 30 required" in iss[0].message and "big" in iss[0].message and iss[0].hint


def test_optional_instances_dropped_silently():
    spec = room([{"id": "big", "asset": "pallet_load", "count": [1, 40], "params": {"variant": "empty"}, "placement": {"inside": "tiny"}}],
                zones=[{"id": "tiny", "rect": [-2, -2, 2, 2]}])
    sol = solve(spec, 0)
    assert not codes(sol) and 1 <= len(sol.instances) < 40


def test_on_and_fill():
    spec = room([
        {"id": "rack", "asset": "rack_row", "params": {"segments": [[-2.0, 3.4]]}, "placement": {"pose": {"xy": [-3, 0]}}},
        {"id": "pals", "asset": "pallet_load", "params": {"variant": "empty"},
         "placement": {"fill": {"anchors": "rack.slot_*", "where": {"level": [0, 1]}, "probability": 1.0}}},
        {"id": "belt", "asset": "conveyor_belt", "placement": {"pose": {"xy": [2, 0], "yaw": 0}}},
        {"id": "items", "asset": "cardboard_box", "count": 3, "params": {"size": [0.3, 0.3, 0.2]}, "placement": {"on": "belt.surface"}},
    ])
    sol = solve(spec, 0)
    assert not codes(sol), [str(i) for i in sol.issues]
    pals = [i for i in sol.instances if i.obj_id == "pals"]
    assert len(pals) == 2 * 2 * 2  # bays x slots x levels {0,1}
    assert all(abs(math.degrees(p.pose.yaw)) % 180 == pytest.approx(90, abs=1e-6) for p in pals)  # long side along the row
    items = [i for i in sol.instances if i.obj_id == "items"]
    assert all(i.parent == "belt.0" and i.pose.pos[2] == pytest.approx(0.8) for i in items)
    no_overlaps(sol)


def test_against_and_next_to():
    spec = room([{"id": "table", "asset": "work_table", "placement": {"against": "wall:north"}},
                 {"id": "bin", "asset": "trash_bin", "placement": {"next_to": {"target": "table", "side": "right", "gap": 0.3}}}])
    sol = solve(spec, 0)
    assert not codes(sol)
    t = next(i for i in sol.instances if i.obj_id == "table")
    P = np.concatenate([q for q, _, _ in t.pieces])
    assert P[:, 1].max() == pytest.approx(4 - 0.1 - 0.05, abs=1e-6)  # wall line - floor pad - default gap
    assert abs(math.degrees(t.pose.yaw)) == pytest.approx(180)


def test_layout_rack_aisles_zones():
    spec = room([{"id": "racks", "asset": "rack_row", "params": {"segments": [[-2, 3.4]]},
                  "placement": {"layout": {"type": "rack_aisles", "aisles": 2, "aisle_width": 2.5, "start_x": -4.8}}}])
    sol = solve(spec, 0)
    assert not codes(sol)
    racks = [i for i in sol.instances if i.obj_id == "racks"]
    assert len(racks) == 4 and [r.params.face for r in racks] == [1, -1, 1, -1]
    assert "racks.aisle_0" in sol.zones and sol.zones["racks.aisle_1"][2] - sol.zones["racks.aisle_1"][0] == pytest.approx(2.5)


def test_clearance_hits_target():
    spec = room([
        {"id": "wall_a", "asset": "cardboard_box", "params": {"size": [0.6, 3.0, 1.0]}, "placement": {"pose": {"xy": [-2.0, 0]}}},
        {"id": "wall_b", "asset": "cardboard_box", "params": {"size": [0.6, 3.0, 1.0]}, "placement": {"pose": {"xy": [2.0, 0]}}},
        {"id": "debris", "asset": "pallet_load", "params": {"variant": "empty"}, "placement": {"pose": {"xy": [-1.0, 0], "yaw": 10}}},
    ], zones=[{"id": "gap", "rect": [-1.7, -1.4, 1.7, 1.4]}],
        constraints=[{"type": "clearance", "id": "choke", "zone": "gap", "target": 1.5, "tolerance": 0.02, "adjust": ["debris"]}])
    sol = solve(spec, 0)
    assert not codes(sol), [str(i) for i in sol.issues]
    c = sol.clearance[0]
    assert c["ok"] and abs(c["planned_m"] - 1.5) <= 0.02


def test_clearance_impossible_reports_achievable():
    spec = room([{"id": "debris", "asset": "pallet_load", "params": {"variant": "empty"}, "placement": {"pose": {"xy": [-1.0, 0]}}}],
                zones=[{"id": "gap", "rect": [-1.7, -1.4, 1.7, 1.4]}],
                constraints=[{"type": "clearance", "id": "choke", "zone": "gap", "target": 0.5, "adjust": ["debris"]}])
    iss = [i for i in solve(spec, 0).issues if i.code == "solve.clearance"]
    assert iss and "best achievable" in iss[0].message and iss[0].data["achievable"] > 0.9


def test_unreachable_names_blockers():
    spec = room([{"id": "barrier", "asset": "cardboard_box", "params": {"size": [0.4, 7.6, 1.0]}, "placement": {"pose": {"xy": [0, 0]}}}],
                robot={"spawn": {"pose": {"xy": [-3, 0]}}, "nav_goals": [{"id": "far", "type": "x", "target": [3, 0]}]},
                constraints=[{"type": "reachable", "to": ["*"]}])
    sol = solve(spec, 0)
    iss = [i for i in sol.issues if i.code == "solve.unreachable"]
    assert iss and "barrier" in iss[0].message and iss[0].path == "/robot/nav_goals/0"


def test_unreachable_drops_optional_blockers():
    spec = room([{"id": "clutter", "asset": "cardboard_box", "count": [0, 40], "params": {"size": [0.5, 0.5, 0.5]}, "placement": {"inside": "mid"}}],
                zones=[{"id": "mid", "rect": [-0.6, -3.9, 0.6, 3.9]}],
                robot={"spawn": {"pose": {"xy": [-3, 0]}}, "nav_goals": [{"id": "far", "type": "x", "target": [3, 0]}]},
                constraints=[{"type": "reachable", "to": ["*"]}])
    sol = solve(spec, 2)
    assert not codes(sol), [str(i) for i in sol.issues]
    assert sol.goals[0]["reachable"]


def test_layout_blocked_message():
    spec = room([{"id": "a", "asset": "pallet", "placement": {"pose": {"xy": [0, 0]}}},
                 {"id": "b", "asset": "pallet", "placement": {"layout": {"type": "row", "start": [-1, 0], "spacing": 1.0}}, "count": 3}])
    sol = solve(spec, 0)
    assert "solve.unsatisfiable.layout" in codes(sol)
