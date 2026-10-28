"""Curated SimReady subset: pinned table, registered aliases, baked footprints / anchors, robot_cell panel fence.  Offline:
everything here reads the checked-in table; nothing is downloaded."""
from __future__ import annotations

import math
import re

import numpy as np
import pytest

from worldc.assets import curated, get, list_assets, registry
from worldc.assets.procedural.robot_cell import RobotCell, panel_dims, reach_of


def test_table_is_pinned_and_licensed():
    t = curated.table()
    assert len(t) == 45 <= 60
    for alias, e in t.items():
        assert re.fullmatch(r"[0-9a-f]{40}", e["rev"]), alias  # a commit, never a branch
        assert "/resolve/main/" not in curated.asset_url(e) and e["rev"] in curated.asset_url(e)
        assert e["licence"] in ("cc-by-4.0", "apache-2.0") and e["attribution"] and "simready" in e["tags"]
        assert e.get("baked"), f"{alias} not baked: run python -m worldc.assets.curated bake {alias}"


def test_aliases_are_registered_with_tags():
    scene_assets = [a for a, e in curated.table().items() if "robot_arm" not in e["tags"]]
    assert len(scene_assets) == 43 and all(registry.exists(a) for a in scene_assets)
    assert not registry.exists("fanuc_crx10ia")  # arms are robot_cell options, not scene objects
    benches = {a["id"] for a in list_assets(["workbench"])}
    assert {"workbench_modular", "workbench_steel", "packing_table_heavy", "packing_table_deluxe"} <= benches
    assert {a["id"] for a in list_assets(["klt"])} == {"klt_bin"}
    assert get("klt_bin").source == "simready"


@pytest.mark.parametrize("alias", [a for a, e in curated.table().items() if "robot_arm" not in e["tags"]])
def test_baked_footprint_matches_survey_size(alias):
    e = curated.table()[alias]
    a = get(alias)
    sx, sy, sz = a.footprint(a.params({})).size
    want = sorted(e["size_m"][:2])
    assert sorted([sx, sy]) == pytest.approx(want, abs=0.05) and sz == pytest.approx(e["size_m"][2], abs=0.05)  # units converted (WH01: cm)


SCENE_ASSETS = [a for a, e in curated.table().items() if "robot_arm" not in e["tags"]]


@pytest.mark.parametrize("alias", SCENE_ASSETS)
def test_support_anchors_sit_on_their_hulls(alias):
    """Every support anchor is at the top of the hulls it rests on (within SUPPORT_TOL) and nothing that starts below
    SUPPORT_REACH above it rises through it (uprights, rims, raised partial hulls).  Needs the asset in the local cache."""
    m = curated.cached_mesh(alias)
    if m is None:
        pytest.skip(f"{alias} not cached (python -m worldc.assets.curated warm)")
    from worldc.assets.mesh.mesh_asset import load_hulls

    hulls = [h for h, _, _ in load_hulls(m.collision_file)]
    for an in (x for x in m.sidecar.anchors if x.kind == "support"):
        P = np.array(an.polygon) + np.array(an.pos[:2])
        lo, hi = P.min(0) + 0.002, P.max(0) - 0.002
        under = [h for h in hulls if (np.minimum(h[:, :2].max(0), hi) > np.maximum(h[:, :2].min(0), lo)).all()]
        z = an.pos[2]
        assert any(abs(h[:, 2].max() - z) <= curated.SUPPORT_TOL for h in under), f"{alias}.{an.name}: no hull top at z={z}"
        blocking = [h for h in under if h[:, 2].max() > z + curated.SUPPORT_TOL and h[:, 2].min() < z + curated.SUPPORT_REACH]
        assert not blocking, f"{alias}.{an.name}: {len(blocking)} hull(s) rise through the support at z={z}"
    assert m.sidecar.anchors == [type(m.sidecar.anchors[0])(**a) for a in curated.baked_anchors(curated.table()[alias]["baked"])] or not m.sidecar.anchors


def test_bench_tops_and_shelves_are_support_anchors():
    for alias in ("workbench_steel", "packing_table_heavy"):
        a = get(alias)
        top = next(x for x in a.anchors(a.params({})) if x.name == "top")
        assert top.kind == "support" and curated.WORK_TOP_BAND[0] <= top.pos[2] <= curated.WORK_TOP_BAND[1], alias
    assert next(x for x in get("workbench_steel").anchors(get("workbench_steel").params({})) if x.name == "top").pos[2] == pytest.approx(0.85, abs=0.02)
    for alias in ("workbench_modular", "packing_table_deluxe"):  # their collision shape covers the work surface
        assert not [x for x in get(alias).anchors(get(alias).params({})) if x.kind == "support"] and "no usable work top" in get(alias).description
    for alias in ("parts_shelf", "shelf_rack", "bulk_storage_rack"):
        a = get(alias)
        zs = [x.pos[2] for x in a.anchors(a.params({})) if x.name.startswith("shelf_")]
        assert len(zs) >= 2 and zs == sorted(zs), alias


def test_fanuc_arms_are_baked_for_robot_cell():
    for alias, reach in (("fanuc_crx10ia", 1.249), ("fanuc_crx20ia_l", 1.418)):
        a = curated.arm(alias)
        assert a["reach_m"] == reach and a["url"].startswith("https://huggingface.co/datasets/FANUC-CORPORATION/")
        assert a["rigid_bodies"] and a["joints"] and "base_offset_z" in a  # what robot_cell switches off
        assert reach_of(alias) == reach


def test_robot_cell_panel_fence_geometry():
    cell = RobotCell()
    p = cell.params({"arm": "fanuc_crx10ia"})
    w, d, _ = panel_dims()
    h = cell.half(p)
    assert h >= reach_of("fanuc_crx10ia") + p.safety_margin
    n = round((2 * h - d) / (w + 0.02))
    panels = cell.panels(p)
    assert len(panels) == 4 * n - 1  # one panel left out as the gate opening
    xs = [x for x, y, _ in panels if y < -h + d]  # south side
    assert not any(abs(x - cell.gate_x(p)) < 1e-6 for x in xs)
    assert {a.name for a in cell.anchors(p)} == {"inside", "gate", "front"}
    poly = cell.safety_polygon(p)
    assert poly[2][0] == pytest.approx(h) and cell.keepouts(p) == []
    fp = cell.footprint(p)
    assert len(fp.pieces()) == len(panels) + 1 and fp.size[0] == pytest.approx(2 * h, abs=0.01)
    assert p.fence_layers == 2 and fp.z1 == pytest.approx(2 * panel_dims()[2], abs=0.01)  # two panels high, ~2.1 m
    # cage and unfenced styles are unchanged
    assert cell.half(cell.params({"arm": "ur10", "fence_style": "cage"})) == pytest.approx(1.6)
    assert [k.tag for k in cell.keepouts(cell.params({"fence": False}))] == ["robot_cell"]
    assert math.isclose(cell.pedestal(cell.params({"arm": "fanuc_crx20ia_l"}))[0], 0.75)
