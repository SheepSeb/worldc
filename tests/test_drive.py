"""Route planning for the driving feature (pure Python), plus one real Isaac Sim drive marked slow."""
from __future__ import annotations

import json
import math

import pytest

from worldc.compile import compile_scene
from worldc.drive import plan_route

ROOM = {"environment": {"footprint": {"size": [14, 10]}, "walls": {"openings": [{"id": "door", "wall": "south", "center": 0.0, "width": 1.2, "height": 2.2}]}}, "zones": [{"id": "z", "rect": [-6, -4, 6, 4]}],
        "objects": [{"id": "wall", "asset": "pallet_load", "count": 4, "placement": {"pose": [{"xy": [x, 0.0], "yaw": 90} for x in (-3.0, -1.0, 1.0, 3.0)]}}],
        "robot": {"asset": "nova_carter", "spawn": {"pose": {"xy": [-5.5, -3.5], "yaw": 0}},
                  "nav_goals": [{"id": "a", "type": "inspect", "target": [5.0, 3.5]}, {"id": "b", "type": "inspect", "target": [-5.0, 3.5]},
                                {"id": "c", "type": "inspect", "target": [5.0, -3.5]}]},
        "constraints": [{"type": "reachable", "to": ["*"]}]}


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    out = tmp_path_factory.mktemp("drive") / "s"
    r = compile_scene(ROOM, 0, out)
    assert r.ok, [str(i) for i in r.issues]
    return out


def test_route_goes_through_the_goals_in_order(scene):
    r = plan_route(scene)
    assert [leg["goal"] for leg in r["legs"]] == ["a", "b", "c"] and r["robot"] == "nova_carter"
    wp = r["waypoints"]
    assert wp[0] == pytest.approx(r["spawn"][:2]) and r["length_m"] > 25
    for leg in r["legs"]:  # each leg ends exactly on its goal pose
        assert wp[leg["end_index"]] == pytest.approx(leg["pose"][:2])
    # smoothed path: far fewer waypoints than grid cells, and consecutive legs are ordered
    assert len(wp) < 40 and [l["end_index"] for l in r["legs"]] == sorted(l["end_index"] for l in r["legs"])


def test_route_options_and_errors(scene):
    only_b = plan_route(scene, goal_ids=["b"])
    assert [l["goal"] for l in only_b["legs"]] == ["b"]
    assert len(plan_route(scene, n_goals=2)["legs"]) == 2
    with pytest.raises(ValueError, match="unknown goals"):
        plan_route(scene, goal_ids=["nope"])


def test_route_keeps_clear_of_obstacles(scene):
    """Every waypoint is in free space for the robot's real footprint (same grid as the checks)."""
    from pxr import Usd

    from worldc.check import nav as NV
    from worldc.check import scene as SC
    from worldc.colliders import collect

    d, usd, graph = SC.locate(scene)
    cols = collect(Usd.Stage.Open(str(usd)))
    cfg = SC.nav_config(graph)
    xr, yr = SC.environment_bounds(graph, cols)
    grid = NV.Grid(xr, yr, float(cfg["resolution_m"]))
    NV.build_grid(grid, NV.polygons(cols, NV.nav_colliders(cols, cfg["z_band"], set(cfg["ignore_categories"]))), SC.robot_radius(graph))
    r = plan_route(scene)
    for (x0, y0), (x1, y1) in zip(r["waypoints"], r["waypoints"][1:]):
        for t in (0, 0.25, 0.5, 0.75, 1):
            assert grid.is_free(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)


def test_unsupported_robot_is_refused(tmp_path):
    from worldc.drive import run_drive

    spec = {**ROOM, "robot": {**ROOM["robot"], "asset": "franka"}}
    out = tmp_path / "f"
    assert compile_scene(spec, 0, out).ok
    with pytest.raises(ValueError, match="differential-drive"):
        run_drive(out, video=False)


@pytest.mark.slow
def test_robot_drives_to_every_goal_in_isaac_sim(scene, tmp_path):
    from worldc.drive import run_drive

    res = run_drive(scene, tmp_path / "d", n_goals=2, video=False)
    assert res["ok"], res
    assert [leg["reached"] for leg in res["legs"]] == [True, True] and res["upright"] and res["max_tilt_deg"] < 5
    assert 0.05 < res["calibration"]["forward_m_per_s_per_rad_s"] < 1.0 and abs(res["calibration"]["turn_rad_s_per_rad_s"]) > 0.05
    assert json.loads((tmp_path / "d" / "drive_result.json").read_text())["ok"] and not res["videos"]
