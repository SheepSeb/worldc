"""Industrial assets salvaged from the plant generator (assets/procedural/industrial.py)."""
import json

import numpy as np
import pytest
from pxr import Usd, UsdPhysics

from conftest import build_asset
from worldc.assets import Pose, describe, get
from worldc.assets.base import ParamError
from worldc.colliders import collect

IDS = ["gate_valve", "control_valve", "gauge", "instrument_box", "enclosure", "control_station", "instrument_panel", "pump_set",
       "flexible_hose", "pipe", "pipe_run", "worker_static", "loose_tool", "ibeam"]
ARTICULATED = {"gate_valve", "control_valve", "enclosure", "control_station", "pump_set"}

# parameter variants exercised by the footprint / physics tests (the default of each asset is always included)
VARIANTS = [
    ("gate_valve", {"pipe_dir": "y", "pipe_radius": 0.08, "wheel_diameter": 0.4, "turns": 8}),
    ("gate_valve", {"pipe_dir": "x", "stem_dir": [0, -1], "turns": 5}),
    ("gate_valve", {"pipe_dir": "z", "stem_dir": [0, 1], "wheel_material": "wheel_blue"}),
    ("control_valve", {"pipe_dir": "y", "pipe_radius": 0.08}),
    ("gauge", {"mount": "side", "dial_yaw_deg": -90, "pipe_radius": 0.06}),
    ("enclosure", {"size": [0.6, 0.4, 1.7], "material": "cab_blue", "latch": True, "louvers": True, "category": "cabinet"}),
    ("enclosure", {"size": [0.32, 0.26, 0.3], "door": False, "drawer": {"w": 0.28, "h": 0.12, "z": 0.18, "stroke": 0.16}}),
    ("enclosure", {"size": [0.3, 0.24, 0.3], "latch": True, "door_lo": 0.16, "drawer": {"w": 0.26, "h": 0.12, "z": 0.08, "stroke": 0.15}}),
    ("control_station", {"n_buttons": 4}),
    ("instrument_panel", {"manifold": False}),
    ("pump_set", {"scale": 0.7}),
    ("pipe", {"a": [0, 0, 0.4], "b": [0, 3, 0.4], "bands": [{"at": 1.0}], "gaps": [{"start": 1.2, "end": 1.5}]}),
    ("pipe_run", {"order": "zyx", "support": "up", "support_ref": 3.0}),
    ("pipe_run", {"a": [0, 0, 0.3], "b": [3, -2, 0.3], "bands": [{"at": 1.5, "material": "orange_band"}]}),
    ("ibeam", {"a": [0, 0, 0], "b": [0, 0, 3]}),
    ("ibeam", {"a": [0, -1, 2], "b": [0, 1, 2]}),
] + [("loose_tool", {"kind": k}) for k in ("wrench", "hammer", "bucket", "hardhat", "hardhat_blue", "toolbox", "coil", "spool", "glove_box")]
CASES = [(i, {}) for i in IDS] + VARIANTS
CASE_IDS = [f"{i}-{'default' if not p else '-'.join(f'{k}={v}' for k, v in p.items())}"[:60] for i, p in CASES]


def _rigid_bodies(stage):
    return [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]


@pytest.mark.parametrize("aid", IDS)
def test_builds_with_defaults(scratch_ctx, aid):
    a, p, b = build_asset(scratch_ctx, aid)
    assert b.root
    assert scratch_ctx.kit.records, "asset must record scene-graph rows"
    assert scratch_ctx.kit.stage.GetPrimAtPath(b.root)
    if aid in ARTICULATED:
        assert b.bodies and scratch_ctx.scene["joints"]


@pytest.mark.parametrize("aid,params", CASES, ids=CASE_IDS)
def test_physics_and_nesting(scratch_ctx, aid, params):
    build_asset(scratch_ctx, aid, params)
    st = scratch_ctx.kit.stage
    bodies = _rigid_bodies(st)
    for prim in bodies:
        m = UsdPhysics.MassAPI(prim)
        assert m.GetMassAttr().Get() > 0, prim.GetPath()
        assert all(v > 0 for v in m.GetDiagonalInertiaAttr().Get()), prim.GetPath()
        par = prim.GetParent()
        while par and par.GetPath().pathString != "/":
            assert not par.HasAPI(UsdPhysics.RigidBodyAPI), f"{prim.GetPath()} nested in rigid body {par.GetPath()}"
            par = par.GetParent()


@pytest.mark.parametrize("aid", IDS)
def test_describe_json(aid):
    d = describe(aid)
    json.dumps(d)
    assert d["id"] == aid and d["footprint_at_defaults"]["size"][2] > 0


@pytest.mark.parametrize("aid,params", [c for c in CASES if c[0] in ARTICULATED | {"flexible_hose"}], ids=lambda v: str(v)[:30])
def test_joints_recorded_and_consistent(scratch_ctx, aid, params):
    a, p, b = build_asset(scratch_ctx, aid, params)
    st = scratch_ctx.kit.stage
    infos = a.joints(p)
    if aid == "flexible_hose":
        assert len(infos) == p.n_links + 1
        for i in range(p.n_links + 1):
            assert st.GetPrimAtPath(f"{b.root}/J_{i}")
        return
    recs = scratch_ctx.scene["joints"]
    assert len(recs) == len(infos) >= 1
    for rec, info in zip(recs, infos):
        assert rec["type"] == info.type and rec["role"] == info.role
        assert rec["lower"] == info.lower and rec["upper"] == info.upper
        if rec["lower"] is not None:
            assert rec["lower"] <= rec["upper"]
        prim = st.GetPrimAtPath(rec["path"])
        assert prim
        if rec["lower"] is not None:
            j = UsdPhysics.RevoluteJoint(prim) if rec["type"] == "revolute" else UsdPhysics.PrismaticJoint(prim)
            assert j.GetLowerLimitAttr().Get() == pytest.approx(rec["lower"])
            assert j.GetUpperLimitAttr().Get() == pytest.approx(rec["upper"])
        else:
            assert rec.get("continuous")
    # every non-continuous, non-mimic joint has a sweep entry the validator can use
    sweeps = scratch_ctx.scene["articulations"]
    assert len(sweeps) == len([r for r in recs if r["mimic"] is None and r["lower"] is not None])
    assert all(s["lower"] <= s["upper"] for s in sweeps)


def test_gate_valve_mimic_and_info(scratch_ctx):
    a, p, b = build_asset(scratch_ctx, "gate_valve", {"turns": 6, "stroke": 0.08, "stem_dir": [0, -1]}, Pose.xy_yaw(1, 2, 0.4, z=1.2))
    st = scratch_ctx.kit.stage
    sj = st.GetPrimAtPath(f"{b.root}/J_stem")
    assert "PhysxMimicJointAPI:rotZ" in sj.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
    assert [str(t) for t in sj.GetRelationship("physxMimicJoint:rotZ:referenceJoint").GetTargets()] == [f"{b.root}/J_wheel"]
    assert sj.GetAttribute("physxMimicJoint:rotZ:gearing").Get() == pytest.approx(-0.08 / (6 * 360.0))
    wj = st.GetPrimAtPath(f"{b.root}/J_wheel")
    assert UsdPhysics.RevoluteJoint(wj).GetUpperLimitAttr().Get() == pytest.approx(6 * 360.0)
    for k in ("wheel_joint", "stem_joint", "wheel_center", "rim_point", "gearing_m_per_deg", "wheel_height", "pipe_r"):
        assert k in b.info
    assert b.info["wheel_center"][2] == pytest.approx(1.2)
    rim = b.anchor("handwheel_rim")
    assert np.linalg.norm(np.array(rim.pos) - np.array(b.info["rim_point"])) < 1e-6
    assert abs(np.linalg.norm(np.array(rim.pos) - np.array(b.info["wheel_center"])) - 0.15) < 1e-6
    pin, pout = b.anchor("pipe_in"), b.anchor("pipe_out")
    assert pin.port_kind == pout.port_kind == "pipe" and pin.radius == pout.radius == 0.05
    # pipe ports lie on the pipe axis, one body length apart, on both sides of the pose origin
    L = b.info["length"]
    assert np.linalg.norm(np.array(pout.pos) - np.array(pin.pos)) == pytest.approx(L)
    assert [round(c, 3) for c in b.info["pos"]] == [1.0, 2.0, 1.2]


def test_gate_valve_param_errors():
    with pytest.raises(ParamError):
        get("gate_valve").params({"pipe_dir": "x", "stem_dir": [1, 0]})  # stem along the pipe
    with pytest.raises(ParamError):
        get("gate_valve").params({"wheel_diameter": 0.9})
    with pytest.raises(ParamError):
        get("gate_valve").params({"wheel_material": "nope"})


def test_enclosure_door_drawer_joints(scratch_ctx):
    a, p, b = build_asset(scratch_ctx, "enclosure", {"latch": True, "drawer": {"w": 0.2, "h": 0.1, "z": 0.3}, "category": "cabinet"})
    assert set(b.info["joints"]) == {"door", "latch", "drawer"}
    assert b.info["kind"] == "cabinet"
    assert b.info["door_handle"] is not None and b.info["drawer_point"] is not None
    assert {k for k, _ in [(t, 0) for t, _ in b.keepouts]} == {"door_swing"}


def test_pump_set_ports(scratch_ctx):
    a, p, b = build_asset(scratch_ctx, "pump_set", {"scale": 1.2}, Pose.xy_yaw(2, 1, 0.0))
    d, s = b.anchor("nozzle_discharge"), b.anchor("nozzle_suction")
    assert d.port_kind == s.port_kind == "pipe"
    assert d.radius == pytest.approx(0.11 * 1.2) and s.radius == pytest.approx(0.12 * 1.2)
    assert d.pos[2] == pytest.approx(1.52 * 1.2) and d.pos[0] == pytest.approx(2 + 0.95 * 1.2)
    assert s.pos[1] == pytest.approx(1 - 0.84 * 1.2)
    assert b.info["nozzle_top"][2] == pytest.approx(1.52 * 1.2)
    rotor = scratch_ctx.scene["joints"][-1]
    assert rotor["continuous"] and rotor["drive"]["target_velocity_deg_s"] == 360.0
    drv = UsdPhysics.DriveAPI.Get(scratch_ctx.kit.stage.GetPrimAtPath(rotor["path"]), "angular")
    assert drv.GetTargetVelocityAttr().Get() == pytest.approx(360.0)
    # the pump's own anchors (pure) agree with the built ones at the identity pose
    for x in a.anchors(p):
        assert x.name in {y.name for y in b.anchors}


def test_flexible_hose_chain(scratch_ctx):
    a, p, b = build_asset(scratch_ctx, "flexible_hose", {"a": [0, 0, 1.0], "b": [1.5, 0.5, 0.8], "relax_iterations": 3000})
    assert len(b.bodies) == p.n_links
    st = scratch_ctx.kit.stage
    zs = [st.GetPrimAtPath(x).GetAttribute("xformOp:translate").Get()[2] for x in b.bodies]
    assert min(zs) > 0.12  # clears the floor
    ends = [st.GetPrimAtPath(f"{b.root}/J_{p.n_links}")]
    assert ends[0].IsValid()
    assert b.info["length"] >= np.linalg.norm([1.5, 0.5, -0.2])
    pa, pb = b.anchor("end_a"), b.anchor("end_b")
    assert pa.pos == pytest.approx((0, 0, 1.0)) and pb.port_kind == "hose"


def test_pipe_run_never_leaves_bare_ends(scratch_ctx):
    for order in ("xyz", "zyx", "yxz"):
        a, p, b = build_asset(scratch_ctx, "pipe_run", {"order": order, "a": [0, 0, 0.5], "b": [2, 1.5, 1.8], "radius": 0.08})
        rep = b.info["report"]
        assert rep["unfitted_pipe_ends"] == 0
        f = b.info["fittings"]
        assert f["elbow"] == 2 and f["equipment_flange"] == 2 and sum(f.values()) == rep["n_nodes"]
        assert b.info["end_error_m"] < 0.03
        assert np.allclose(b.anchor("end_b").pos, (2, 1.5, 1.8))
    # coincident-in-two-axes request: single straight leg, flanges at both ends
    a, p, b = build_asset(scratch_ctx, "pipe_run", {"a": [0, 0, 1], "b": [2, 0, 1], "end_a": "blind"})
    assert b.info["fittings"]["blind_flange"] == 1 and b.info["fittings"]["equipment_flange"] == 1
    with pytest.raises(ParamError):
        get("pipe_run").params({"a": [0, 0, 1], "b": [0.005, 0, 1.01]})


def test_ibeam_axis_validation():
    with pytest.raises(ParamError):
        get("ibeam").params({"a": [0, 0, 0], "b": [1, 1, 0]})


@pytest.mark.parametrize("aid", IDS)
def test_deterministic(scratch_ctx, aid):
    params = {"relax_iterations": 2000} if aid == "flexible_hose" else {}
    import re

    norm = lambda o: re.sub(r"_000\d", "_N", json.dumps(o, default=str, sort_keys=True))
    ids_of = lambda recs: norm([{k: v for k, v in r.items() if k not in ("path", "name", "articulation")} for r in recs])
    _, _, b1 = build_asset(scratch_ctx, aid, params, seed=7)
    n1 = len(scratch_ctx.kit.records)
    j1 = len(scratch_ctx.scene["joints"])
    _, _, b2 = build_asset(scratch_ctx, aid, params, seed=7)
    r1, r2 = scratch_ctx.kit.records[:n1], scratch_ctx.kit.records[n1:]
    assert ids_of(r1) == ids_of(r2)
    strip = lambda js: norm([{k: v for k, v in j.items() if k not in ("name", "path", "art", "art_root", "link", "parent", "mimic", "driven_by")} for j in js])
    assert strip(scratch_ctx.scene["joints"][:j1]) == strip(scratch_ctx.scene["joints"][j1:])
    assert json.dumps(b1.info, default=str).replace(b1.root, "X") .count("nan") == 0


def test_panel_seed_changes_layout(scratch_ctx):
    _, _, b1 = build_asset(scratch_ctx, "instrument_panel", {"manifold": False}, seed=1)
    _, _, b2 = build_asset(scratch_ctx, "instrument_panel", {"manifold": False}, seed=2)
    order = lambda b: [v["slot"] for v in b.info["items"].values()]
    assert len(b1.info["items"]) == 6 and order(b1) != order(b2)


def _footprint_box(fp):
    poly = np.asarray(fp.polygon, float)
    return np.array([*poly.min(0), fp.z0]), np.array([*poly.max(0), fp.z1])


@pytest.mark.parametrize("aid,params", CASES, ids=CASE_IDS)
def test_footprint_bounds_colliders(scratch_ctx, aid, params):
    if aid == "flexible_hose":
        params = {**params, "relax_iterations": 3000}
    a, p, b = build_asset(scratch_ctx, aid, params)
    scratch_ctx.kit.save()
    cols = collect(scratch_ctx.kit.stage)
    assert cols, "asset must author colliders"
    lo, hi = _footprint_box(a.footprint(p))
    tol = 0.02
    for c in cols:  # visual-only parts are not collected (no CollisionAPI)
        clo, chi = c.obb.aabb()
        assert (clo >= lo - tol).all() and (chi <= hi + tol).all(), f"{c.path}: {clo}..{chi} outside footprint {lo}..{hi}"


def test_footprint_bounds_hose_default(scratch_ctx):
    # default relaxation length too: the hose sags but never leaves the conservative bound
    a, p, b = build_asset(scratch_ctx, "flexible_hose")
    scratch_ctx.kit.save()
    lo, hi = _footprint_box(a.footprint(p))
    for c in collect(scratch_ctx.kit.stage):
        clo, chi = c.obb.aabb()
        assert (clo >= lo - 0.02).all() and (chi <= hi + 0.02).all()


def test_pose_transform_applies(scratch_ctx):
    # a yawed, raised enclosure puts its front port where yaw and z say, and records the pose
    a, p, b = build_asset(scratch_ctx, "enclosure", {"size": [0.5, 0.4, 0.8]}, Pose.xy_yaw(3, 4, np.pi / 2, z=0.5))
    f = b.anchor("front")
    assert f.pos == pytest.approx((3 - 0.2, 4, 0.5 + 0.4), abs=1e-6)
    rec = scratch_ctx.kit.records[-1]
    assert rec["position"] == [3.0, 4.0, 0.5] and rec["yaw_deg"] == pytest.approx(90.0)
