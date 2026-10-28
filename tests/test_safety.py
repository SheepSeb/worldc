"""Mandatory safety checks in check_scene (human reachability, aisle width, robot-cell clearance) and declared min_clearance,
plus the robot_cell asset.  Every failure must be an error-level issue with a hint, so it feeds the repair loop."""
from __future__ import annotations

import json

import pytest

from worldc.check import safety as SF
from worldc.compile import compile_scene

DOOR = {"id": "door", "wall": "south", "center": 0.0, "width": 1.2, "height": 2.2}


def room(objects, size=(6, 6), openings=(DOOR,), constraints=()):
    return {"environment": {"footprint": {"size": list(size)}, "height": 3.0, "walls": {"openings": list(openings)}},
            "objects": objects, "constraints": list(constraints)}


def wall_block(oid, x, y, width):
    return {"id": oid, "asset": "enclosure", "params": {"size": [width, 0.4, 1.5]}, "placement": {"pose": {"xy": [x, y], "yaw": 0}}}


TABLE = {"id": "bench", "asset": "work_table", "params": {"drawers": 0}, "placement": {"pose": {"xy": [0.0, 1.3], "yaw": 0}}}  # 1.3 m free in front


def build(d, tmp_path, name="s"):
    r = compile_scene(d, 0, tmp_path / name)
    errors = [i for i in r.issues if i.severity == "error"]
    return r, errors, json.loads((tmp_path / name / "check.json").read_text())


def codes(errors):
    return sorted({i.code for i in errors})


def test_open_room_passes(tmp_path):
    r, errors, chk = build(room([TABLE]), tmp_path)
    assert r.ok, [str(i) for i in errors]
    h = chk["safety"]["human"]
    assert h["source"]["exit"] == "door" and h["targets"] == [{"kind": "workbench", "id": "bench.0", "reachable": True, "aisle_ok": True,
                                                               "narrowest_m": None}]
    assert chk["safety"]["thresholds"] == SF.THRESHOLDS


def test_no_door_is_an_error(tmp_path):
    r, errors, _ = build(room([TABLE], openings=()), tmp_path)
    assert not r.ok and codes(errors) == ["check.safety_no_exit"] and "openings" in errors[0].hint


def test_walled_off_workbench_is_unreachable(tmp_path):
    d = room([TABLE, wall_block("left", -1.5, 0.0, 2.78), wall_block("right", 1.5, 0.0, 2.78)])
    r, errors, chk = build(d, tmp_path)
    assert not r.ok and codes(errors) == ["check.human_unreachable"]
    e = errors[0]
    assert e.data["target"] == "bench.0" and "door 'door'" in e.message and "walkway" in e.hint
    assert {"left", "right"} & set(e.hint.replace("'", " ").split())  # names the partition
    assert chk["safety"]["human"]["targets"][0]["reachable"] is False


def test_narrow_gap_fails_aisle_width_but_is_reachable(tmp_path):
    d = room([TABLE, wall_block("left", -1.5, 0.0, 2.78), wall_block("right", 1.74, 0.0, 2.3)])  # 0.70 m gap at x -0.11..0.59
    r, errors, chk = build(d, tmp_path)
    assert not r.ok and codes(errors) == ["check.aisle_too_narrow"]
    t = chk["safety"]["human"]["targets"][0]
    assert t["reachable"] and not t["aisle_ok"] and 0.6 <= t["narrowest_m"] < SF.MIN_AISLE_WIDTH_M
    e = errors[0]
    assert -0.15 <= e.data["at"][0] <= 0.65 and abs(e.data["at"][1]) < 0.5  # located at the gap
    assert "Widen it by" in e.hint


def test_storage_zone_and_second_exit_are_targets(tmp_path):
    side = {"id": "side", "wall": "north", "center": 2.0, "width": 1.0, "height": 2.1}
    d = room([wall_block("left", -1.5, 0.0, 2.78), wall_block("right", 1.5, 0.0, 2.78)], openings=(DOOR, side))
    d["zones"] = [{"id": "parts", "rect": [-2.5, 1.0, -1.0, 2.5], "tags": ["storage"]}]
    r, errors, chk = build(d, tmp_path)
    kinds = sorted((t["kind"], t["reachable"]) for t in chk["safety"]["human"]["targets"])
    assert kinds == [("exit", False), ("storage_zone", False)] and codes(errors) == ["check.human_unreachable"]


def test_mandatory_checks_run_without_any_constraint(tmp_path):
    d = room([TABLE, wall_block("left", -1.5, 0.0, 2.78), wall_block("right", 1.5, 0.0, 2.78)])
    assert "constraints" not in json.dumps(d["constraints"]) and not d["constraints"]
    assert not build(d, tmp_path)[0].ok  # nothing declared, still tested


# ---------------------------------------------------------------------------------------------------- min_clearance
def test_min_clearance_violation_and_pass(tmp_path):
    objs = [wall_block("a", -0.5, 1.0, 0.6), wall_block("b", 0.4, 1.0, 0.6)]  # faces 0.3 m apart
    r, errors, chk = build(room(objs, constraints=[{"type": "min_clearance", "id": "ab", "between": ["a", "b"], "value": 0.5}]), tmp_path, "bad")
    assert not r.ok and codes(errors) == ["check.min_clearance"]
    e = errors[0]
    assert e.path == "/constraints/0" and e.data["measured_m"] == pytest.approx(0.3, abs=0.03) and "0.2" in e.hint
    assert chk["safety"]["min_clearance"][0]["ok"] is False
    r2, errors2, chk2 = build(room(objs, constraints=[{"type": "min_clearance", "between": ["a", "b"], "value": 0.25}]), tmp_path, "good")
    assert r2.ok, [str(i) for i in errors2]
    assert chk2["safety"]["min_clearance"][0]["measured_m"] == pytest.approx(0.3, abs=0.03)


def test_min_clearance_object_level_wall_and_unresolved(tmp_path):
    a = wall_block("a", 0.0, 2.6, 0.6)  # back face 0.2 m from the north wall (interior at y=3)
    a["constraints"] = [{"type": "min_clearance", "between": ["a", "wall:north"], "value": 0.5}]
    ghost = {**wall_block("ghost", 0.0, -1.0, 0.5), "count": 0}  # valid id, but nothing gets built
    d = room([a, ghost], constraints=[{"type": "min_clearance", "between": ["a", "ghost"], "value": 0.5}])
    r, errors, _ = build(d, tmp_path)
    assert codes(errors) == ["check.min_clearance", "check.min_clearance_unresolved"]
    by = {e.code: e for e in errors}
    assert by["check.min_clearance"].path == "/objects/0/constraints/0" and 0.1 < by["check.min_clearance"].data["measured_m"] < 0.25
    assert by["check.min_clearance_unresolved"].path == "/constraints/0/between"
    d["objects"] = [a]  # an unknown id is already rejected by validation, before anything is built
    r2 = compile_scene(d, 0, tmp_path / "unknown")
    assert [i.code for i in r2.issues if i.severity == "error"] == ["ref.unknown"]


# ---------------------------------------------------------------------------------------------------- robot cell
CELL = {"id": "cell", "asset": "robot_cell", "params": {"arm": "ur10", "fence_style": "cage"},  # cage: offline (panels download SimReady meshes)
        "placement": {"pose": {"xy": [0.0, 0.5], "yaw": 0}}}


def test_robot_cell_alone_passes_and_records_its_footprint(tmp_path):
    r, errors, chk = build(room([CELL], size=(10, 8)), tmp_path)
    assert r.ok, [str(i) for i in errors]
    g = json.loads((tmp_path / "s" / "scene_graph.json").read_text())
    cell = g["safety"]["robot_cells"][0]
    assert cell["spec_id"] == "cell" and cell["fenced"] and len(cell["polygon"]) == 4
    arm = next(o for o in g["objects"] if o["category"] == "robot_arm")
    assert arm["isaac_path"].endswith("ur10/ur10.usd") and arm["driven"] is False
    assert chk["safety"]["robot_cells"] == [{"id": "cell.0", "spec_id": "cell", "ok": True, "min_distance_m": None, "offenders": []}]
    assert g["robot"] is None  # the cell does not take the scene's single (mobile) robot slot


def test_robot_cell_clearance_violation(tmp_path):
    near = wall_block("crate", 2.0, 0.5, 0.3)  # 0.19 m from the fence's outer face (half-size 1.66 m)
    r, errors, chk = build(room([CELL, near], size=(10, 8)), tmp_path)
    assert not r.ok and codes(errors) == ["check.robot_cell_clearance"]
    e = errors[0]
    assert e.data["obstacle_spec_id"] == "crate" and e.data["distance_m"] == pytest.approx(0.19, abs=0.03) and "crate" in e.hint
    assert chk["safety"]["robot_cells"][0]["offenders"] == ["crate"]


def test_fixtures_inside_the_cell_are_part_of_it(tmp_path):
    fixture = {"id": "fixture", "asset": "enclosure", "params": {"size": [0.5, 0.4, 0.8]}, "placement": {"on": "cell.inside"}}
    r, errors, _ = build(room([CELL, fixture], size=(10, 8)), tmp_path)
    assert r.ok, [str(i) for i in errors]


def test_unfenced_franka_cell_is_a_keepout(tmp_path):
    cell = {"id": "cell", "asset": "robot_cell", "params": {"arm": "franka", "fence": False}, "placement": {"pose": {"xy": [0.0, 0.5], "yaw": 0}}}
    boxes = {"id": "boxes", "asset": "cardboard_box", "count": 12, "placement": {"inside": "floor"}}
    d = room([cell, boxes], size=(10, 8))
    d["zones"] = [{"id": "floor", "rect": [-4.5, -3.5, 4.5, 3.5]}]
    r, errors, chk = build(d, tmp_path)
    assert not any(e.code == "check.robot_cell_clearance" and "intrudes" in e.message for e in errors)  # solver kept the boxes out
    g = json.loads((tmp_path / "s" / "scene_graph.json").read_text())
    assert not g["safety"]["robot_cells"][0]["fenced"]
    from pxr import Usd

    st = Usd.Stage.Open(str(r.usd))
    arm = next(p for p in st.Traverse() if p.GetName() == "Arm")
    assert arm.GetAttribute("physxArticulation:articulationEnabled").Get() is False
