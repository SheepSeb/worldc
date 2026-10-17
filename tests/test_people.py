"""Real Isaac Sim people (assets/people.py): table, build, pose / gait animation, timeline, hazards."""
from __future__ import annotations

import json

import pytest
from pxr import Usd, UsdGeom, UsdSkel

from worldc.assets import people, registry
from worldc.compile import compile_scene
from worldc.spec.validate import validate

ROOM = {"environment": {"footprint": {"size": [12, 8]}}, "zones": [],
        "agents": [{"id": "w1", "asset": "human", "params": {"character": "male_adult_construction_03"},
                    "path": {"waypoints": [[-4, -2], [4, -2]], "mode": "pingpong", "speed": 1.0}}]}


def test_table_and_registry():
    assert len(people.CHARACTERS) >= 4
    for name, c in people.CHARACTERS.items():
        assert 1.5 < c["height_m"] < 2.0 and {"L_Upperarm", "R_Upperarm", "L_Thigh", "R_Thigh"} <= set(c["joints"]), name
        assert len(c["joints"]["L_Thigh"]["q"]) == 4 and c["skeleton"].endswith(name.replace("_new", ""))
    h = registry.get("human")
    assert h.source == "isaac" and {"human", "hazard", "isaac"} <= set(h.tags)
    assert "human_proxy" not in registry.ids() and "forklift_proxy" not in registry.ids()
    assert h.describe()["params_schema"]["properties"]["character"]["enum"] == list(people.CHARACTERS)


def test_unknown_character_and_old_proxy_are_rejected():
    bad = {**ROOM, "agents": [{**ROOM["agents"][0], "params": {"character": "blue_capsule"}}]}
    assert any(i.severity == "error" for i in validate(bad))
    old = {**ROOM, "agents": [{**ROOM["agents"][0], "asset": "human_proxy"}]}
    assert [i.code for i in validate(old) if i.severity == "error"] == ["asset.unknown"]


def test_person_is_referenced_posed_and_walking(tmp_path):
    r = compile_scene(ROOM, 0, tmp_path / "o", run_check=False)
    assert r.ok, [str(i) for i in r.issues]
    g = json.loads((tmp_path / "o" / "scene_graph.json").read_text())
    hz = next(h for h in g["hazards"] if h["kind"] == "human")
    assert hz["speed_mps"] == 1.0 and hz["path_mode"] == "pingpong"
    stage = Usd.Stage.Open(str(r.usd))
    vis = stage.GetPrimAtPath("/World/Agents/w1/Isaac")
    assert [i.assetPath for i in vis.GetPrimStack()[0].referenceList.prependedItems][0].endswith("/Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd")
    assert not list(tmp_path.rglob("male_adult_construction*"))  # referenced by URL, never copied
    # the arms-down pose + gait is authored as a skeleton animation with time samples
    info = people.CHARACTERS["male_adult_construction_03"]
    skel = stage.GetPrimAtPath(f"/World/Agents/w1/Isaac/{info['skeleton']}")
    binding = UsdSkel.BindingAPI(skel)
    anim = UsdSkel.Animation(stage.GetPrimAtPath(binding.GetAnimationSourceRel().GetTargets()[0]))
    assert anim and "L_Thigh" in " ".join(anim.GetJointsAttr().Get()) and len(anim.GetRotationsAttr().GetTimeSamples()) > 1000
    # the timeline is extended only because something animates; a scene without people stays static
    assert stage.GetEndTimeCode() == people.GAIT_S * stage.GetTimeCodesPerSecond()
    static = compile_scene({"environment": {"footprint": {"size": [6, 6]}}, "zones": []}, 0, tmp_path / "s", run_check=False)
    assert Usd.Stage.Open(str(static.usd)).GetEndTimeCode() == 0
    # the collider is an invisible capsule of the character's size (checks and kinematic drive stay exact)
    col = stage.GetPrimAtPath("/World/Agents/w1/Collider")
    assert UsdGeom.Imageable(col).GetPurposeAttr().Get() == UsdGeom.Tokens.guide


def test_gait_period_follows_walking_speed(tmp_path):
    def period(speed):
        spec = {**ROOM, "agents": [{**ROOM["agents"][0], "path": {**ROOM["agents"][0]["path"], "speed": speed}}]}
        r = compile_scene(spec, 0, tmp_path / f"p{speed}", run_check=False)
        stage = Usd.Stage.Open(str(r.usd))
        info = people.CHARACTERS["male_adult_construction_03"]
        anim = UsdSkel.Animation(stage.GetPrimAtPath(f"/World/Agents/w1/Isaac/{info['skeleton']}/Walk"))
        ts = anim.GetRotationsAttr().GetTimeSamples()
        return (ts[12] - ts[0]) / stage.GetTimeCodesPerSecond()  # 12 samples = one gait cycle
    assert period(1.0) == pytest.approx(people.STRIDE_M / 1.0, rel=0.01) and period(0.5) == pytest.approx(2 * period(1.0), rel=0.01)
