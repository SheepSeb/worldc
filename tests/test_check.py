"""Generic post-build checker (worldc.check.check_scene) on tiny authored scenes + a legacy warehouse."""
from __future__ import annotations

import json

import pytest
from pxr import UsdPhysics

from conftest import build_asset
from worldc.assets import Pose
from worldc.check import check_scene

ROOM = 10.0, 8.0  # building_shell interior (x, y), centred at the origin


def room(ctx):
    build_asset(ctx, "building_shell", {"size": list(ROOM), "height": 4.0, "apron_depth": 0.0})


def box(ctx, x, y, z=0.0, yaw=0.0, size=(0.4, 0.3, 0.25), sid=None):
    ctx.meta = {"spec_id": sid} if sid else {}
    return build_asset(ctx, "cardboard_box", {"size": list(size)}, Pose.xy_yaw(x, y, yaw, z))[2]


def finish(ctx, goals=((3.0, 3.0),), hazards=None, clearance=None, spawn=(-4.0, -3.0, 0.0), radius=0.3, nav=None, write=True):
    """Save the stage, write a matching minimal scene_graph.json and run check_scene."""
    ctx.kit.save()
    d = ctx.out_dir
    g = {"usd": "scene.usd",
         "environment": {"kind": "indoor", "x_range": [-ROOM[0] / 2, ROOM[0] / 2], "y_range": [-ROOM[1] / 2, ROOM[1] / 2], "height": 4.0},
         "robot": {"spawn": list(spawn), "radius_m": radius},
         "goals": [{"id": f"g{i}", "name": f"goal_{i}", "type": "pick", "base_pose": [x, y, 0.0]} for i, (x, y) in enumerate(goals)],
         "hazards": hazards if hazards is not None else ctx.scene.get("hazards", []),
         "objects": ctx.kit.records}
    if clearance:
        g["clearance_checks"] = clearance
    if nav:
        g["nav"] = nav
    (d / "scene_graph.json").write_text(json.dumps(g))
    return check_scene(d, write=write)


def codes(r):
    return [i["code"] for i in r["issues"]]


def test_clean_scene_passes(scratch_ctx):
    room(scratch_ctx)
    box(scratch_ctx, -1.0, 0.0, sid="b0")
    box(scratch_ctx, 1.0, 1.0, yaw=0.5, sid="b1")
    build_asset(scratch_ctx, "pallet_load", pose=Pose.xy_yaw(2.0, -2.0, 0.3), seed=1)
    build_asset(scratch_ctx, "rack_row", {"segments": [[-1.0, 1.0]], "bays_per_segment": 1}, Pose.xy_yaw(-3.5, 2.0, 0.0))
    r = finish(scratch_ctx)
    assert r["passed"], r["issues"]
    assert r["issues"] == []
    assert r["n_colliders"] > 5
    nav = r["navigation"]
    assert nav["spawn_free"] and nav["all_reachable"] and nav["n_goals"] == 1
    assert nav["goals"][0]["id"] == "g0" and nav["goals"][0]["path_length_m"] > 6
    assert nav["robot_radius_m"] == 0.3 and nav["resolution_m"] == 0.05
    d = scratch_ctx.out_dir
    saved = json.loads((d / "check.json").read_text())
    assert saved["passed"] and (d / "nav_occupancy.png").exists()


def test_accepts_usd_path_and_no_write(scratch_ctx):
    room(scratch_ctx)
    r = finish(scratch_ctx, write=False)
    assert not (scratch_ctx.out_dir / "check.json").exists()
    r2 = check_scene(scratch_ctx.out_dir / "scene.usd", write=False)
    assert r2["passed"] and r["passed"]


def test_interpenetration(scratch_ctx):
    room(scratch_ctx)
    box(scratch_ctx, 0.0, 0.0, sid="left")
    box(scratch_ctx, 0.2, 0.0, sid="right")
    r = finish(scratch_ctx)
    assert not r["passed"]
    pairs = r["spawn_interpenetrations"]
    assert len(pairs) == 1
    p = pairs[0]
    assert {p["a_spec_id"], p["b_spec_id"]} == {"left", "right"} and p["depth_mm"] > 100
    iss = [i for i in r["issues"] if i["code"] == "check.interpenetration"]
    assert len(iss) == 1
    assert p["a"] in iss[0]["message"] and p["b"] in iss[0]["message"] and "left" in iss[0]["hint"]


def test_jointed_pair_not_interpenetrating(scratch_ctx):
    room(scratch_ctx)
    a = box(scratch_ctx, 0.0, 0.0)
    b = box(scratch_ctx, 0.2, 0.0)
    assert "check.interpenetration" in codes(finish(scratch_ctx))
    st = scratch_ctx.kit.stage
    j = UsdPhysics.FixedJoint.Define(st, "/World/Test/J")
    j.CreateBody0Rel().SetTargets([a.root])
    j.CreateBody1Rel().SetTargets([b.root])
    r = finish(scratch_ctx)
    assert not r["spawn_interpenetrations"] and r["passed"], r["issues"]


@pytest.mark.parametrize("aid", ["work_table", "rollup_door", "swing_door"])
def test_articulated_assets_clean(scratch_ctx, aid):
    room(scratch_ctx)
    build_asset(scratch_ctx, aid, pose=Pose.xy_yaw(1.0, -1.0, 0.4))
    r = finish(scratch_ctx)
    bad = [i for i in r["issues"] if i["code"] in ("check.interpenetration", "check.floating")]
    assert not bad, bad


def test_floating(scratch_ctx):
    room(scratch_ctx)
    b = box(scratch_ctx, 1.0, 1.0, z=0.3, sid="hover")
    r = finish(scratch_ctx)
    fl = r["floating_bodies"]
    assert [f["body"] for f in fl] == [b.root] and fl[0]["spec_id"] == "hover" and abs(fl[0]["bottom_z"] - 0.3) < 0.01
    assert "check.floating" in codes(r) and not r["passed"]


def test_box_on_box_not_floating(scratch_ctx):
    room(scratch_ctx)
    box(scratch_ctx, 1.0, 1.0)
    box(scratch_ctx, 1.0, 1.0, z=0.25)
    assert finish(scratch_ctx)["passed"]


def test_goal_enclosed(scratch_ctx):
    room(scratch_ctx)
    cx, cy = 3.0, 1.5
    k = 0
    for sx in (-0.6, 0.6):  # top / bottom
        for sy in (-1.1, 1.1):
            box(scratch_ctx, cx + sx, cy + sy, size=(1.2, 0.3, 0.4), sid=f"ring_{k}")
            k += 1
    for sx in (-1.1, 1.1):  # left / right
        for sy in (-0.475, 0.475):
            box(scratch_ctx, cx + sx, cy + sy, yaw=1.5707963, size=(0.9, 0.3, 0.4), sid=f"ring_{k}")
            k += 1
    r = finish(scratch_ctx, goals=((cx, cy), (-3.0, 2.0)))
    g = r["navigation"]["goals"]
    assert [x["reachable"] for x in g] == [False, True]
    assert g[0]["goal_cell_free"]
    iss = [i for i in r["issues"] if i["code"] == "check.goal_unreachable"]
    assert len(iss) == 1 and "/World/Test/cardboard_box" in iss[0]["hint"] and "ring_" in iss[0]["hint"]
    assert not r["passed"] and not r["navigation"]["all_reachable"]


def test_goal_cell_blocked(scratch_ctx):
    room(scratch_ctx)
    box(scratch_ctx, 3.0, 3.0, sid="on_goal")
    r = finish(scratch_ctx)
    iss = [i for i in r["issues"] if i["code"] == "check.goal_unreachable"]
    assert len(iss) == 1 and "goal cell itself is blocked" in iss[0]["hint"] and "on_goal" in iss[0]["hint"]
    assert r["navigation"]["goals"][0]["goal_cell_free"] is False


def test_spawn_blocked_and_no_goals(scratch_ctx):
    room(scratch_ctx)
    box(scratch_ctx, -4.0, -3.0, sid="on_spawn")
    r = finish(scratch_ctx, goals=())
    cs = codes(r)
    assert "check.spawn_blocked" in cs and "check.no_goals" in cs
    sev = {i["code"]: i["severity"] for i in r["issues"]}
    assert sev["check.no_goals"] == "warning" and sev["check.spawn_blocked"] == "error"
    assert "on_spawn" in next(i for i in r["issues"] if i["code"] == "check.spawn_blocked")["hint"]


def test_hazard_path_through_rack(scratch_ctx):
    room(scratch_ctx)
    build_asset(scratch_ctx, "rack_row", {"segments": [[-1.0, 1.0]], "bays_per_segment": 1}, Pose.xy_yaw(0.0, 0.0, 0.0))
    build_asset(scratch_ctx, "forklift_proxy", {"waypoints": [(-3.0, 0.0), (3.0, 0.0)], "path_mode": "pingpong"},
                Pose.xy_yaw(-4.0, 0.0, 0.0))
    r = finish(scratch_ctx)
    hz = r["hazard_path_conflicts"]
    assert len(hz) == 1 and "forklift" in hz[0]["hazard"]
    assert "rack_row" in hz[0]["obstacle"]
    assert "check.hazard_path_blocked" in codes(r)


def test_hazard_path_clear(scratch_ctx):
    room(scratch_ctx)
    build_asset(scratch_ctx, "forklift_proxy", {"waypoints": [(-3.0, 3.0), (3.0, 3.0)], "path_mode": "pingpong"},
                Pose.xy_yaw(-3.0, 3.0, 0.0))
    r = finish(scratch_ctx, goals=((3.0, -3.0),))
    assert r["hazard_path_conflicts"] == [] and r["passed"], r["issues"]


def test_clearance_measured(scratch_ctx):
    room(scratch_ctx)
    box(scratch_ctx, -0.6, 0.0, size=(0.4, 1.2, 0.5))  # x in [-0.8, -0.4]
    box(scratch_ctx, 0.35, 0.0, size=(0.4, 1.2, 0.5))  # x in [0.15, 0.55] -> gap 0.55
    box(scratch_ctx, 3.0, 1.5, size=(1.2, 0.4, 0.5))  # y in [1.3, 1.7]
    box(scratch_ctx, 3.0, 2.35, size=(1.2, 0.4, 0.5))  # y in [2.15, 2.55] -> gap 0.45
    ck = [{"id": "gap_x", "rect": [-1.0, -0.5, 1.0, 0.5], "axis": "x", "span": None, "target": 0.55, "tolerance": 0.02},
          {"id": "gap_y", "rect": [2.5, 1.0, 3.5, 3.0], "axis": "y", "span": [2.7, 3.3], "target": 0.45, "tolerance": 0.02},
          {"id": "wrong", "rect": [-1.0, -0.5, 1.0, 0.5], "axis": "x", "span": [-0.4, 0.4], "target": 0.90, "tolerance": 0.03}]
    r = finish(scratch_ctx, goals=((-3.0, 3.0),), clearance=ck)
    m = {c["id"]: c for c in r["clearance"]}
    assert abs(m["gap_x"]["measured_m"] - 0.55) < 0.01 and m["gap_x"]["ok"]
    assert abs(m["gap_y"]["measured_m"] - 0.45) < 0.01 and m["gap_y"]["ok"]
    assert not m["wrong"]["ok"] and abs(m["wrong"]["measured_m"] - 0.55) < 0.01
    iss = [i for i in r["issues"] if i["code"] == "check.clearance"]
    assert len(iss) == 1 and "wrong" in iss[0]["message"] and "-0.3" in iss[0]["message"]
    assert not r["passed"]


def test_load_failed(tmp_path):
    r = check_scene(tmp_path)
    assert not r["passed"] and codes(r) == ["check.load_failed"]
    (tmp_path / "scene_graph.json").write_text("{}")
    (tmp_path / "x.usd").write_text("garbage")
    r = check_scene(tmp_path / "x.usd", write=False)
    assert codes(r) == ["check.load_failed"]


# --------------------------------------------------------------------------- full warehouse (slow)
@pytest.fixture(scope="session")
def warehouse_seed(tmp_path_factory):
    from worldc.generate import build

    return build(0, tmp_path_factory.mktemp("warehouse"))


@pytest.mark.slow
def test_warehouse_matches_baseline_validation(warehouse_seed):
    """The recipe-built warehouse passes the same pre-flight the legacy validator reported for the original (fixture)."""
    import json
    from pathlib import Path

    base = json.loads((Path(__file__).parent / "fixtures" / "warehouse_baseline.json").read_text())["per_seed"]["0"]
    r = check_scene(warehouse_seed)
    print("check_scene seconds:", r["seconds"], "colliders:", r["n_colliders"])
    assert r["passed"], r["issues"]
    assert r["seconds"] < 60
    ck = next(c for c in r["clearance"] if c["id"] == "chokepoint")
    assert abs(ck["measured_m"] - 0.90) <= 0.03
    nav = r["navigation"]
    assert nav["all_reachable"] and nav["robot_radius_m"] == 0.38 and nav["n_goals"] == len(base["goals"])
    assert len(r["spawn_interpenetrations"]) == base["static"]["spawn_interpenetrations"] == 0
    assert len(r["floating_bodies"]) == base["static"]["floating_bodies"] == 0
    assert len(r["hazard_path_conflicts"]) == base["static"]["hazard_path_conflicts"] == 0
