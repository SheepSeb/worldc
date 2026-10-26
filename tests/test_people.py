"""Real Isaac Sim people (assets/people.py): table, build, retargeted Isaac walk clips, timeline, hazards.  A synthetic local asset root
stands in for the Isaac Sim S3 mirror (clips with the real biped rig and a made-up swing), so nothing here touches the network."""
from __future__ import annotations

import json
import math

import pytest
from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

from worldc.assets import people, registry
from worldc.assets.skel_pose import AXES
from worldc.compile import compile_scene
from worldc.spec.validate import validate

ROOM = {"environment": {"footprint": {"size": [12, 8]}}, "zones": [],
        "agents": [{"id": "w1", "asset": "human", "params": {"character": "male_adult_construction_03"},
                    "path": {"waypoints": [[-4, -2], [4, -2]], "mode": "pingpong", "speed": 1.0}}]}




@pytest.fixture
def isaac_root(tmp_path_factory, monkeypatch):
    """Walk clips on the biped rig: the legs and arms swing, the pelvis bobs, the root travels (all of it must be handled)."""
    root = tmp_path_factory.mktemp("isaac")
    rig = people.GAITS["rig"]
    for name, g in {**people.GAITS["clips"], **people.GAITS["poses"]}.items():
        standing = name in people.GAITS["poses"]
        path = root / "Isaac" / "People" / "Animations" / f"{name}.skelanim.usd"
        path.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(str(path))
        stage.SetTimeCodesPerSecond(30)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
        anim = UsdSkel.Animation.Define(stage, "/Root")
        anim.CreateJointsAttr(Vt.TokenArray(rig["joints"]))
        anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(rig["joints"])))
        short = [j.rsplit("/", 1)[-1] for j in rig["joints"]]
        for f in range(g["start"], g["start"] + g["frames"]):
            ang = 2 * math.pi * (f - g["start"]) / g["frames"]
            quats, trans = [], []
            for i, (j, row) in enumerate(zip(short, rig["rest"])):
                q = Gf.Quatd(*row[3:])
                if standing:  # arms stay out in the rig's T-pose in LookAround, only the head turns
                    if name == "LookAround" and j == "Head":
                        h = math.radians(30 * math.sin(ang)) / 2
                        q = q * Gf.Quatd(math.cos(h), 0, math.sin(h), 0)
                elif j in ("R_UpLeg", "L_UpLeg", "R_UpArm", "L_UpArm"):
                    sgn = 1 if j[0] == "R" else -1
                    h = math.radians(25 * sgn * math.sin(ang)) / 2
                    q = q * Gf.Quatd(math.cos(h), math.sin(h), 0, 0)
                quats.append(Gf.Quatf(q))
                t = Gf.Vec3f(*row[:3])
                if i == 0:
                    t = Gf.Vec3f(0.03 * f, 0, 0.03 * f)  # root travel
                if j == "Pelvis":
                    t = Gf.Vec3f(t[0], t[1] - 0.02 * abs(math.sin(ang)), t[2])
                trans.append(t)
            anim.GetRotationsAttr().Set(Vt.QuatfArray(quats), f)
            anim.GetTranslationsAttr().Set(Vt.Vec3fArray(trans), f)
        stage.Save()
    monkeypatch.setenv("WORLDC_ISAAC_ASSET_ROOT", str(root))
    return root


def test_table_and_registry():
    assert len(people.CHARACTERS) >= 4
    for name, c in people.CHARACTERS.items():
        assert 1.5 < c["height_m"] < 2.0 and len(c["joints"]) == len(c["rest"]) == 101, name
        assert {"Hip", "L_Thigh", "R_Thigh", "L_Upperarm", "R_Upperarm", "Head"} <= {j.rsplit("/", 1)[-1] for j in c["joints"]}
        assert c["skeleton"].endswith(name.replace("_new", ""))
    assert len(people.GAITS["rig"]["joints"]) == 81 and people.GAITS["clips"]
    for name, g in people.GAITS["clips"].items():
        assert 0.7 < g["speed_mps"] < 1.5 and 30 <= g["frames"] <= 50 and g["closure_deg"] <= 2, name
    h = registry.get("human")
    assert h.source == "isaac" and {"human", "hazard", "isaac"} <= set(h.tags)
    assert "human_proxy" not in registry.ids() and "forklift_proxy" not in registry.ids()
    props = h.describe()["params_schema"]["properties"]
    assert props["character"]["enum"] == list(people.CHARACTERS) and props["gait"]["enum"] == list(people.GAITS["clips"])


def test_unknown_character_gait_and_old_proxy_are_rejected():
    bad = {**ROOM, "agents": [{**ROOM["agents"][0], "params": {"character": "blue_capsule"}}]}
    assert any(i.severity == "error" for i in validate(bad))
    bad = {**ROOM, "agents": [{**ROOM["agents"][0], "params": {"gait": "moonwalk"}}]}
    assert any(i.severity == "error" for i in validate(bad))
    old = {**ROOM, "agents": [{**ROOM["agents"][0], "asset": "human_proxy"}]}
    assert [i.code for i in validate(old) if i.severity == "error"] == ["asset.unknown"]


def _retarget(character):
    rt = people.retarget(character)
    return rt, rt.clip, rt.char


def _world_dir(rig, local_q, a, b, t=None):
    pos = rig.global_pos(local_q, t or rig.t)
    return (pos[rig.by_short[b]] - pos[rig.by_short[a]]).GetNormalized()


@pytest.mark.parametrize("character", list(people.CHARACTERS))
def test_clip_rest_pose_retargets_to_character_rest_pose(character):
    rt, clip, char = _retarget(character)
    q, hip_t = rt.frame(clip.q, clip.t[rt.pelvis])
    for i, qq in zip(rt.order, q):
        assert Gf.Dot(qq, char.q[i]) == pytest.approx(1.0, abs=1e-4) or Gf.Dot(qq, char.q[i]) == pytest.approx(-1.0, abs=1e-4), char.short[i]
    assert (hip_t - char.t[rt.hip]).GetLength() < 1e-6


def test_a_clip_bone_swing_moves_the_character_bone_the_same_way():
    rt, clip, char = _retarget("male_adult_construction_05_new")
    swing = 35.0
    local = list(clip.q)
    h = math.radians(swing) / 2
    i = clip.by_short["L_UpLeg"]
    world = clip.global_rots(clip.q)
    local[i] = Gf.Quatd(world[clip.parent[i]].GetReal(), -world[clip.parent[i]].GetImaginary()) * Gf.Quatd(math.cos(h), math.sin(h), 0, 0) * world[i]  # a swing about world x
    q, _ = rt.frame(local, clip.t[rt.pelvis])
    moved = list(char.q)
    for i, qq in zip(rt.order, q):
        moved[i] = qq
    rest_dir = _world_dir(char, char.q, "L_Thigh", "L_Calf")
    new_dir = _world_dir(char, moved, "L_Thigh", "L_Calf")
    clip_rest = _world_dir(clip, clip.q, "L_UpLeg", "L_LoLeg")
    clip_new = _world_dir(clip, local, "L_UpLeg", "L_LoLeg")
    angle = lambda a, b: math.degrees(math.acos(max(-1, min(1, Gf.Dot(a, b)))))
    assert angle(rest_dir, new_dir) == pytest.approx(swing, abs=0.5) == pytest.approx(angle(clip_rest, clip_new), abs=0.5)
    # and in the same direction: the clip's bone, turned by the quarter turn between the two models' axes, is the character's bone
    mapped = Gf.Rotation(AXES).TransformDir(clip_new)
    assert angle(mapped, new_dir) < 4.0
    # nothing else moved: the right leg and the twist bones stay where the rest pose puts them relative to the parent
    assert angle(_world_dir(char, moved, "R_Thigh", "R_Calf"), _world_dir(char, char.q, "R_Thigh", "R_Calf")) < 0.5


def test_person_is_referenced_posed_and_walking(tmp_path, isaac_root):
    r = compile_scene(ROOM, 0, tmp_path / "o", run_check=False)
    assert r.ok, [str(i) for i in r.issues]
    g = json.loads((tmp_path / "o" / "scene_graph.json").read_text())
    hz = next(h for h in g["hazards"] if h["kind"] == "human")
    assert hz["speed_mps"] == 1.0 and hz["path_mode"] == "pingpong"
    assert (hz["dwell_s"], hz["ease_s"], hz["turn_s"]) == (4.0, 0.8, 1.4)  # people stop, turn and ease by default
    stage = Usd.Stage.Open(str(r.usd))
    vis = stage.GetPrimAtPath("/World/Agents/w1/Isaac")
    assert [i.assetPath for i in vis.GetPrimStack()[0].referenceList.prependedItems][0].endswith("/Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd")
    assert not list(tmp_path.rglob("male_adult_construction*"))  # referenced by URL, never copied
    # Isaac's clips, retargeted: a skeleton animation of the moving joints (hands included) with time samples, root travel dropped
    info = people.CHARACTERS["male_adult_construction_03"]
    skel = stage.GetPrimAtPath(f"/World/Agents/w1/Isaac/{info['skeleton']}")
    binding = UsdSkel.BindingAPI(skel)
    anim = UsdSkel.Animation(stage.GetPrimAtPath(binding.GetAnimationSourceRel().GetTargets()[0]))
    joints = [str(j).rsplit("/", 1)[-1] for j in anim.GetJointsAttr().Get()]
    assert {"Hip", "L_Thigh", "L_Calf", "L_Foot", "R_Upperarm", "R_Forearm", "R_Hand", "R_Index1", "L_Thumb2", "Head"} <= set(joints) and "RL_BoneRoot" not in joints
    ts = anim.GetRotationsAttr().GetTimeSamples()
    assert len(ts) > 1000
    hip = joints.index("Hip")
    zs = [anim.GetTranslationsAttr().Get(t)[hip][2] for t in ts[:200]]
    assert 0.005 < max(zs) - min(zs) < 0.1  # the pelvis bobs as the person walks
    # the timeline is extended only because something animates; a scene without people stays static
    assert stage.GetEndTimeCode() == people.GAIT_S * stage.GetTimeCodesPerSecond()
    static = compile_scene({"environment": {"footprint": {"size": [6, 6]}}, "zones": []}, 0, tmp_path / "s", run_check=False)
    assert Usd.Stage.Open(str(static.usd)).GetEndTimeCode() == 0
    # the collider is an invisible capsule of the character's size (checks and kinematic drive stay exact)
    col = stage.GetPrimAtPath("/World/Agents/w1/Collider")
    assert UsdGeom.Imageable(col).GetPurposeAttr().Get() == UsdGeom.Tokens.guide


def _performer(isaac_root, gait="stand_walk_1", speed=1.0, **timing):
    from worldc.runtime import PathFollower
    f = PathFollower([[0, 0], [8, 0]], speed, "pingpong", 0.0, 0.0, **{"dwell_s": 4.0, "ease_s": 0.8, "turn_s": 1.4, **timing})
    return f, people.Performer("male_adult_construction_03", gait, f, 0.0)


def _pose_dirs(perf, pose, a, b):
    rt = people.retarget("male_adult_construction_03")
    lq = list(rt.char.q)
    for i, q in zip(rt.order, pose[0]):
        lq[i] = q
    return _world_dir(rt.char, lq, a, b), rt


def test_people_stop_stand_and_turn_like_people(isaac_root):
    f, perf = _performer(isaac_root)
    period = f.period_s
    assert period == pytest.approx(2 * (8 / 1.0 + 0.8 + 4.0))
    # the animation repeats exactly with the motion, so one period is authored and looped
    for t in (0.0, 3.3, 9.0, 9.9, 11.0, 13.2):
        a, b = perf.pose(t), perf.pose(t + period)
        assert all(abs(Gf.Dot(x, y)) == pytest.approx(1.0, abs=1e-4) for x, y in zip(a[0], b[0])) and (a[1] - b[1]).GetLength() < 1e-6, t
    stand = perf.pose(8.8 + 1.0)
    # walking moves the thigh joint a lot, standing does not
    rt = people.retarget("male_adult_construction_03")
    k = [j for j, i in enumerate(rt.order) if rt.char.short[i] == "L_Thigh"][0]
    turned = lambda p, q: math.degrees(2 * math.acos(min(1.0, abs(Gf.Dot(p[0][k], q[0][k])))))
    assert max(turned(perf.pose(t), stand) for t in (3.0, 3.3, 3.6, 3.9, 4.2, 4.5)) > 10
    assert turned(perf.pose(8.8 + 1.5), stand) < 1
    # the look-around layer moves the head at the far end and nothing else (the near end plays the idle clip)
    far = perf.pose(8.8 + 1.0)
    rt = people.retarget("male_adult_construction_03")
    head = [k for k, i in enumerate(rt.order) if rt.char.short[i] == "Head"][0]
    idle = perf._at(perf.idle, 1.0 * people.CLIP_FPS, loop=True)
    moved = [k for k, (x, y) in enumerate(zip(far[0], idle[0])) if abs(Gf.Dot(x, y)) < 0.9999]
    assert head in moved and {rt.char.short[rt.order[k]] for k in moved} <= set(people.LOOK_JOINTS)
    # turning round steps on the spot: the legs move while the body is held (no travel)
    mid_turn = perf.pose(8.8 + 4.0 - 0.7)
    assert turned(mid_turn, stand) > 1


def test_gait_phase_follows_distance_so_feet_do_not_slide(isaac_root):
    """The gait cycle advances one stride per `stride` metres walked, however the speed ramps; faster walking = faster stepping."""
    f, perf = _performer(isaac_root, ease_s=0.0, dwell_s=0.0, turn_s=0.0)
    g = people.GAITS["clips"]["stand_walk_1"]
    assert perf.stride == pytest.approx(8.0 / round(8.0 / (g["frames"] / people.CLIP_FPS * g["speed_mps"])))
    # walking twice as fast covers a stride in half the time: pose(t) at speed v == pose(t / 2) at 2v for the same distance
    f2, perf2 = _performer(isaac_root, speed=2.0, ease_s=0.0, dwell_s=0.0, turn_s=0.0)
    a, b = perf.pose(2.0), perf2.pose(1.0)
    assert all(abs(Gf.Dot(p, q)) == pytest.approx(1.0, abs=1e-4) for p, q in zip(a[0], b[0]))


def test_humanized_path_timing():
    from worldc.runtime import PathFollower
    f = PathFollower([[0, 0], [9, 0]], 1.0, "pingpong", 0.0, 0.0, dwell_s=4, ease_s=1, turn_s=1.5)
    assert f.humanized and f.period_s == pytest.approx(2 * (9 + 1 + 4))
    assert f.motion(0.0).speed == 0 and f.motion(0.5).speed == pytest.approx(0.5) and f.motion(5.0).speed == 1.0 and f.motion(9.9).speed < 0.2
    # arrives exactly at the end after leg time (L / v + ease), pauses, turns counter-clockwise during the last turn_s, walks back
    x, y, _, yaw = f.pose(10.0)
    assert x == pytest.approx(9.0) and yaw == pytest.approx(0.0) and f.motion(10.0).dwell_t == pytest.approx(0.0)
    assert f.pose(12.4)[3] == pytest.approx(0.0) and 0 < f.pose(13.2)[3] < math.pi and abs(f.pose(13.99)[3]) == pytest.approx(math.pi, abs=0.02)
    assert f.pose(15.0)[0] < 9.0 and abs(f.pose(15.0)[3]) == pytest.approx(math.pi)
    assert f.pose(1.234) == pytest.approx(f.pose(1.234 + f.period_s))
    # the default (forklift) timing is untouched: constant speed, instant flip
    g = PathFollower([[0, 0], [9, 0]], 1.0, "pingpong")
    assert not g.humanized and g.pose(4.5)[0] == pytest.approx(4.5) and g.pose(9.5)[0] == pytest.approx(8.5)


def test_turn_must_fit_in_the_pause():
    bad = {**ROOM, "agents": [{**ROOM["agents"][0], "params": {"dwell_s": 1.0, "turn_s": 2.0}}]}
    assert any(i.severity == "error" for i in validate(bad))


def test_missing_clip_is_a_build_error_not_a_silent_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDC_ISAAC_ASSET_ROOT", str(tmp_path / "empty"))
    r = compile_scene(ROOM, 0, tmp_path / "o", run_check=False)
    assert not r.ok and "stand_walk_1" in " ".join(str(i) for i in r.issues)


@pytest.mark.slow
@pytest.mark.parametrize("character", list(people.CHARACTERS))
def test_real_isaac_clips_look_like_a_person(character):
    """With NVIDIA's actual clips (downloaded once): arms hang down, the feet stay on the floor, the legs really stride."""
    from worldc.assets.catalog import CatalogError
    try:
        f, perf = _performer(None, gait="stand_walk_2")
    except CatalogError as e:
        pytest.skip(str(e))
    rt = people.retarget(character)
    perf = people.Performer(character, "stand_walk_2", f, 0.0)
    char, lows, thigh_swing = rt.char, [], []
    rest_foot = char.global_pos(char.q, char.t)[char.by_short["L_Foot"]][2]
    for k in range(60):
        pose = perf.pose(3.0 + k * 0.05)
        lq, lt = list(char.q), list(char.t)
        for i, q in zip(rt.order, pose[0]):
            lq[i] = q
        lt[rt.hip] = pose[1]
        P = char.global_pos(lq, lt)
        lows.append(min(P[char.by_short["L_Foot"]][2], P[char.by_short["R_Foot"]][2]))
        thigh_swing.append(_world_dir(char, lq, "L_Thigh", "L_Calf")[1])
        arm = _world_dir(char, lq, "L_Upperarm", "L_Forearm")
        assert arm[2] < -0.9
    assert abs(min(lows) - rest_foot) < 0.02 and max(thigh_swing) - min(thigh_swing) > 0.4
    stand = perf.pose(8.8 + 1.0)
    lq = list(char.q)
    for i, q in zip(rt.order, stand[0]):
        lq[i] = q
    assert _world_dir(char, lq, "L_Upperarm", "L_Forearm")[2] < -0.95


def test_the_usd_itself_moves_the_person_along_the_path(tmp_path, isaac_root):
    """Opening the scene and pressing play is enough: the body's translation and heading are time samples that equal the path follower's."""
    from worldc.runtime import hazard_followers
    r = compile_scene(ROOM, 0, tmp_path / "o", run_check=False)
    g = json.loads((tmp_path / "o" / "scene_graph.json").read_text())
    f = hazard_followers(g)["/World/Agents/w1"]
    stage = Usd.Stage.Open(str(r.usd))
    body = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Agents/w1"))
    tcps = stage.GetTimeCodesPerSecond()
    for t in (0.0, 1.0, 5.0, 9.4, 12.0, 14.0, 20.0, 31.7):
        x, y, z, yaw = f.pose(t)
        M = body.ComputeLocalToWorldTransform(t * tcps)
        assert tuple(M.ExtractTranslation())[:2] == pytest.approx((x, y), abs=0.03), t
        q = M.ExtractRotationQuat()
        assert abs(2 * (q.GetReal() * math.cos(yaw / 2) + q.GetImaginary()[2] * math.sin(yaw / 2))) / 2 == pytest.approx(1.0, abs=2e-3), t
    # without a stop the default (start) pose is what a static consumer sees
    assert tuple(body.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())[:2] == pytest.approx((-4.0, -2.0))
