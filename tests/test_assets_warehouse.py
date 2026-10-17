"""Warehouse structural assets: build at defaults, physics validity, footprint bounds colliders, describe JSON."""
import json

import numpy as np
import pytest
from pxr import UsdPhysics

from conftest import build_asset
from claude_scene_generator.assets import Pose, describe, get
from claude_scene_generator.colliders import collect

IDS = ["building_shell", "slope_zone", "rack_row", "rollup_door", "swing_door", "cage", "conveyor_belt", "conveyor_roller",
       "work_table", "charging_station", "traffic_cone", "trash_bin", "human_proxy", "forklift_proxy", "robot_placeholder",
       "pallet", "pallet_load", "cardboard_box", "tote", "pallet_jack_tipped"]


@pytest.mark.parametrize("aid", IDS)
def test_builds_and_physics_valid(scratch_ctx, aid):
    pose = Pose.xy_yaw(2.0, -1.0, 0.7) if aid not in ("building_shell",) else Pose()
    a, p, b = build_asset(scratch_ctx, aid, pose=pose, seed=1)
    json.dumps(describe(aid))
    st = scratch_ctx.kit.stage
    for prim in st.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(prim)
            assert m.GetMassAttr().Get() > 0, prim.GetPath()
            assert all(v > 0 for v in m.GetDiagonalInertiaAttr().Get()), prim.GetPath()


FOOTPRINT_EXEMPT = {"building_shell", "slope_zone", "human_proxy", "forklift_proxy", "robot_placeholder", "pallet_jack_tipped"}


@pytest.mark.parametrize("aid", [i for i in IDS if i not in FOOTPRINT_EXEMPT])
def test_footprint_bounds_colliders(scratch_ctx, aid):
    pose = Pose.xy_yaw(3.0, 1.0, 0.4)
    a, p, b = build_asset(scratch_ctx, aid, pose=pose)
    scratch_ctx.kit.save()
    fp = a.footprint(p)
    poly = pose.world_xy(np.concatenate([np.asarray(f.polygon) for f in fp.pieces()]))
    lo, hi = poly.min(0) - 0.05, poly.max(0) + 0.05
    for c in collect(scratch_ctx.kit.stage):
        clo, chi = c.obb.aabb()
        if c.shape in ("Cylinder", "Capsule", "Cone") and abs(c.obb.R[2, 2]) > 0.99:  # upright round: bound by radius
            r = float(c.obb.h[0])
            clo, chi = clo.copy(), chi.copy()
            clo[:2], chi[:2] = c.obb.c[:2] - r, c.obb.c[:2] + r
        assert np.all(clo[:2] >= lo) and np.all(chi[:2] <= hi), (aid, c.path, clo, chi, lo, hi)
        assert chi[2] <= pose.pos[2] + fp.z1 + 0.05, (aid, c.path, chi[2], fp.z1)


def test_articulations_recorded(scratch_ctx):
    for aid in ("rollup_door", "swing_door", "cage", "work_table"):
        build_asset(scratch_ctx, aid, pose=Pose.xy_yaw(0, 0, 0.3))
    arts = scratch_ctx.scene["articulations"]
    assert len(arts) == 5
    for a in arts:
        assert a["upper"] > a["lower"] and scratch_ctx.kit.stage.GetPrimAtPath(a["joint"])


def test_shell_openings_and_rack_slots(scratch_ctx):
    shell = get("building_shell")
    p = shell.params({"openings": [{"id": "dock_0", "wall": "south", "center": -15, "width": 3, "height": 3.2},
                                   {"id": "w", "wall": "west", "center": 11, "width": 1.15, "height": 2.24}]})
    ports = {a.name: a for a in shell.anchors(p)}
    assert ports["opening_dock_0"].pos[:2] == (-15, -12.5)
    assert abs(Pose(ports["opening_w"].pos, ports["opening_w"].quat).yaw + np.pi / 2) < 1e-9
    rack = get("rack_row")
    slots = [a for a in rack.anchors(rack.params()) if a.name.startswith("slot_")]
    assert len(slots) == 2 * 2 * 2 * 4
    _, _, b = build_asset(scratch_ctx, "rack_row", pose=Pose.xy_yaw(-18.95, 0))
    rec = [r for r in scratch_ctx.kit.records if r["category"] == "rack_row"][0]
    assert rec["n_uprights"] == 12 and rec["n_beams"] == 32
