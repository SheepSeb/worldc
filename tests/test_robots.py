"""The scene robot is always a real Isaac Sim robot (assets/robots.py): measured table, registry, validation, build, scene graph."""
from __future__ import annotations

import json

import pytest
from pxr import Usd, UsdGeom

from worldc.assets import registry, robots
from worldc.assets.resolve import AssetResolutionError, resolve
from worldc.compile import compile_scene
from worldc.spec.validate import validate

ROOM = {"environment": {"footprint": {"size": [8, 8]}, "walls": {"openings": [{"id": "door", "wall": "south", "center": 0.0, "width": 1.2, "height": 2.2}]}}, "zones": [{"id": "z", "rect": [-3.5, -3.5, 3.5, 3.5]}],
        "objects": [{"id": "boxes", "asset": "cardboard_box", "count": 3, "placement": {"inside": "z"}}]}


def spec(robot):
    return {**ROOM, "robot": {"spawn": {"pose": {"xy": [-2.5, -2.5], "yaw": 45}}, **robot}}


def codes(issues):
    return [i.code for i in issues if i.severity == "error"]


def test_table_is_measured_and_registered():
    t = robots.table()
    assert {"nova_carter", "franka"} <= set(t) and len(t) >= 8
    for a, r in t.items():
        assert r.path.startswith("Isaac/Robots/") and r.kind in ("wheeled_amr", "mobile_manipulator", "arm", "legged", "humanoid"), a
        assert all(0.05 < v < 4.0 for v in r.size_m) and 0.05 < r.footprint_radius_m < 2.5, (a, r.size_m)
        assert r.articulation and r.n_joints > 0, a  # real robots carry their own articulation
    nova = registry.get("nova_carter")
    assert nova.source == "isaac" and {"robot", "isaac", "wheeled_amr"} <= set(nova.tags)
    assert "nova_carter" in {a["id"] for a in registry.list_assets(tags=["robot"])}
    assert nova.describe()["physics"]["body"] in ("articulated", "dynamic") and nova.describe()["physics"]["collider"] == "authored"
    assert "robot_placeholder" not in registry.ids()


def test_default_and_explicit_real_robots_validate():
    assert codes(validate(spec({}))) == []  # default robot: nova_carter
    assert codes(validate(spec({"asset": "jackal"}))) == []
    path = robots.table()["iw_hub"].path
    assert codes(validate(spec({"asset": {"catalog": "omniverse", "path": path}}))) == []
    assert resolve({"catalog": "omniverse", "path": path}).id == "iw_hub"


@pytest.mark.parametrize("robot,code", [
    ({"asset": "robot_placeholder"}, "robot.unknown"),
    ({"asset": "nova_carte"}, "robot.unknown"),
    ({"asset": "pallet"}, "robot.not_isaac"),
    ({"asset": {"generate": {"id": "gen/bot", "description": "x", "body": "dynamic", "mass_kg": 5,
                             "parts": [{"name": "b", "shape": "box", "size": [0.4, 0.4, 0.4], "pos": [0, 0, 0.2]}]}}}, "robot.not_isaac"),
    ({"asset": {"file": "bot.usd"}}, "robot.not_isaac"),
    ({"asset": {"catalog": "omniverse", "path": "Isaac/Robots/Nobody/made_up.usd"}}, "robot.unknown"),
    ({"asset": {"catalog": "omniverse", "path": "Isaac/Props/Pallet/pallet.usd"}}, "robot.unknown"),
    ({"asset": {"catalog": "omniverse", "query": "robot"}}, "robot.unknown"),
])
def test_only_real_isaac_robots_are_accepted(robot, code):
    issues = validate(spec(robot))
    assert codes(issues) == [code]
    hint = next(i.hint for i in issues if i.code == code)
    assert "nova_carter" in hint and "list_assets" in hint  # the error tells the agent what to use
    with pytest.raises(AssetResolutionError):
        resolve({"catalog": "omniverse", "path": "Isaac/Robots/Nobody/made_up.usd"})


def test_robot_is_referenced_with_its_own_physics_and_measured_footprint(tmp_path):
    r = compile_scene(spec({"asset": "nova_carter"}), 0, tmp_path / "o", run_check=False)
    assert r.ok, [str(i) for i in r.issues]
    g = json.loads((tmp_path / "o" / "scene_graph.json").read_text())
    info = robots.table()["nova_carter"]
    rob = g["robot"]
    assert rob["asset"] == "nova_carter" and rob["isaac_path"] == info.path and rob["kind"] == "wheeled_amr" and not rob["needs_controller"]
    assert rob["radius_m"] == info.footprint_radius_m  # nav radius comes from the real robot unless the spec overrides it
    stage = Usd.Stage.Open(str(r.usd))
    vis = stage.GetPrimAtPath("/World/Robot/Isaac")
    refs = [i.assetPath for i in vis.GetPrimStack()[0].referenceList.prependedItems]
    assert refs == [registry.get("nova_carter").url] and refs[0].endswith("/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd")
    assert vis.GetAttribute("physics:rigidBodyEnabled").Get() is None  # nothing switched off: the robot keeps its articulation
    assert not list(tmp_path.rglob("nova_carter*"))  # referenced by URL, never copied
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Robot/RobotCamera"))
    assert cam and any(c["name"] == "robot" for c in g["cameras"])
    assert g["objects"] and next(o for o in g["objects"] if o.get("category") == "robot_spawn" or o.get("spec_id") == "robot")["isaac_robot"] == "nova_carter"


def test_override_radius_and_blocked_spawn(tmp_path):
    s = spec({"asset": "nova_carter", "footprint_radius": 0.8})
    r = compile_scene(s, 0, tmp_path / "a", run_check=False)
    assert r.ok and json.loads((tmp_path / "a" / "scene_graph.json").read_text())["robot"]["radius_m"] == 0.8
    blocked = {**ROOM, "objects": [{"id": "b", "asset": "pallet", "placement": {"pose": {"xy": [0, 0]}}}],
               "robot": {"asset": "nova_carter", "spawn": {"pose": {"xy": [0, 0]}}}}
    res = compile_scene(blocked, 0, tmp_path / "b", run_check=False)
    assert not res.ok and any(i.code == "solve.robot_spawn_blocked" for i in res.issues)


def test_navigation_uses_the_robot_radius(tmp_path):
    """A wide robot cannot pass a narrow gap that a small one can: reachability depends on the real footprint."""
    wall = {"id": "wall", "asset": "pallet", "count": 1, "placement": {"pose": {"xy": [0, 0], "yaw": 90}}}
    base = {**ROOM, "objects": [wall]}
    goal = {"id": "g", "type": "inspect", "target": [2.5, 2.5]}
    ok = compile_scene({**base, "robot": {"asset": "turtlebot3_burger", "spawn": {"pose": {"xy": [-2.5, -2.5]}}, "nav_goals": [goal]},
                        "constraints": [{"type": "reachable", "to": ["*"]}]}, 0, tmp_path / "s", run_check=True)
    assert ok.ok and ok.check["navigation"]["all_reachable"]
