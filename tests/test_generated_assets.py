"""LLM-generated assets (declarative spec -> fixed builder) and procedural TextureSpec rendering."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from sceneforge import texturegen
from sceneforge.assets import Pose
from sceneforge.assets.generated import (
    GeneratedAsset, GeneratedAssetError, GeneratedAssetSpec, load_promoted, promote, validate_generated,
)
from sceneforge.colliders import collect
from sceneforge.spec.issues import has_errors


# --------------------------------------------------------------------------- example specs
def shelf():
    posts = [{"name": f"post{i}", "shape": "box", "size": [0.04, 0.04, 1.8], "pos": [sx * 0.58, sy * 0.28, 0.9], "material": "steel_dark"}
             for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)])]
    boards = [{"name": f"board{k}", "shape": "box", "size": [1.12, 0.52, 0.03], "pos": [0, 0, z], "material": {"rgb": [0.6, 0.5, 0.4], "roughness": 0.7}}
              for k, z in enumerate([0.3, 0.9, 1.5])]
    return {"id": "gen/test_shelf", "description": "three-board static steel shelf", "tags": ["shelf", "static"], "body": "static",
            "parts": posts + boards,
            "anchors": [{"name": "top_board", "type": "support", "auto_from": "board2"},
                        {"name": "mid_board", "type": "support", "auto_from": "board1"},
                        {"name": "front_port", "type": "port", "pos": [0, 0.3, 1.0], "port_kind": "mount", "radius": 0.05, "yaw_deg": 90}]}


def cart():
    parts = [{"name": "top", "shape": "box", "size": [0.9, 0.5, 0.03], "pos": [0, 0, 0.9], "material": "steel_bright"},
             {"name": "bottom", "shape": "box", "size": [0.82, 0.42, 0.03], "pos": [0, 0, 0.2], "material": "steel"}]
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        parts.append({"name": f"post{i}", "shape": "box", "size": [0.04, 0.04, 0.785], "pos": [sx * 0.43, sy * 0.23, 0.4925], "material": "steel_dark"})
        parts.append({"name": f"caster{i}", "shape": "cylinder", "radius": 0.05, "height": 0.03, "axis": "y",
                      "pos": [sx * 0.4, sy * 0.2, 0.05], "material": "rubber"})
    return {"id": "gen/test_cart", "description": "two-shelf utility cart", "tags": ["cart", "dynamic"], "body": "dynamic", "mass_kg": 14.0,
            "parts": parts, "anchors": [{"name": "top_shelf", "type": "support", "auto_from": "top"}]}


def cabinet():
    return {
        "id": "gen/test_cabinet", "description": "cabinet with a hinged door and a drawer", "tags": ["cabinet"], "body": "articulated", "mass_kg": 40.0,
        "parts": [
            {"name": "side_l", "shape": "box", "size": [0.02, 0.4, 1.2], "pos": [-0.29, 0, 0.6], "material": "door_grey"},
            {"name": "side_r", "shape": "box", "size": [0.02, 0.4, 1.2], "pos": [0.29, 0, 0.6], "material": "door_grey"},
            {"name": "back", "shape": "box", "size": [0.56, 0.02, 1.2], "pos": [0, -0.19, 0.6], "material": "door_grey"},
            {"name": "bottom", "shape": "box", "size": [0.56, 0.38, 0.02], "pos": [0, 0.01, 0.01], "material": "door_grey"},
            {"name": "top", "shape": "box", "size": [0.56, 0.38, 0.02], "pos": [0, 0.01, 1.19], "material": "door_grey"},
        ],
        "links": [
            {"name": "door", "joint": {"type": "revolute", "axis": "Z", "limits": [0, 110], "parent": "root", "origin": [-0.28, 0.21, 0.9],
                                       "drive": {"stiffness": 0.0, "damping": 5.0}},
             "parts": [{"name": "door_panel", "shape": "box", "size": [0.56, 0.02, 0.5], "pos": [0, 0.21, 0.9], "material": "door_leaf"},
                       {"name": "handle", "shape": "cylinder", "radius": 0.01, "height": 0.12, "pos": [0.23, 0.235, 0.9], "material": "stainless"}]},
            {"name": "drawer", "joint": {"type": "prismatic", "axis": "Y", "limits": [0, 0.3], "parent": "root", "origin": [0, 0, 0.3]},
             "parts": [{"name": "drawer_box", "shape": "box", "size": [0.5, 0.34, 0.1], "pos": [0, 0, 0.3], "material": "drawer"},
                       {"name": "drawer_front", "shape": "box", "size": [0.54, 0.02, 0.16], "pos": [0, 0.19, 0.3], "material": "drawer"}]},
        ],
        "anchors": [{"name": "top_surface", "type": "support", "auto_from": "top"}],
    }


def codes(issues):
    return [i.code for i in issues]


def by_code(issues, code):
    return [i for i in issues if i.code == code]


def mutate(spec, fn):
    s = copy.deepcopy(spec)
    fn(s)
    return s


def inside_poly(poly, pt, tol=1e-6):
    """CCW convex polygon containment."""
    n = len(poly)
    for i in range(n):
        a, b = np.asarray(poly[i]), np.asarray(poly[(i + 1) % n])
        if (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]) < -tol:
            return False
    return True


def assert_footprint_bounds_colliders(ctx, asset, pose, zpad=0.0):
    ctx.kit.save()
    cols = collect(ctx.kit.stage)
    assert cols
    fp = asset.footprint()
    poly = pose.world_xy(fp.polygon)
    for c in cols:
        for corner in c.obb.corners():
            assert inside_poly(poly, corner[:2], tol=1e-4), (c.path, corner)
            assert pose.pos[2] + fp.z0 - 1e-4 <= corner[2] <= pose.pos[2] + fp.z1 + zpad + 1e-4
    return cols


# --------------------------------------------------------------------------- valid builds
def test_examples_validate_clean():
    for f in (shelf, cart, cabinet):
        iss = validate_generated(f())
        assert not has_errors(iss), [str(i) for i in iss]
        assert not iss, [str(i) for i in iss]


def test_static_shelf(scratch_ctx):
    a = GeneratedAsset(shelf())
    pose = Pose.xy_yaw(1.0, 2.0, 0.3)
    built = a.build(scratch_ctx, "/World/Test/shelf", pose, a.params(), np.random.default_rng(0))
    assert a.source == "generated" and a.id == "gen/test_shelf"
    top = built.anchor("top_board")
    assert top.kind == "support" and abs(top.pos[2] - 1.515) < 1e-6 and len(top.polygon) >= 3
    assert built.anchor("front_port").port_kind == "mount"
    cols = assert_footprint_bounds_colliders(scratch_ctx, a, pose)
    assert len(cols) == 7 and all(c.body is None for c in cols)
    assert scratch_ctx.kit.records[-1]["body_type"] == "static"
    assert a.footprint().size[2] == pytest.approx(1.8, abs=1e-6)


def test_dynamic_cart(scratch_ctx):
    from pxr import UsdPhysics

    a = GeneratedAsset(cart())
    pose = Pose.xy_yaw(0.5, -1.0, 0.7)
    built = a.build(scratch_ctx, "/World/Test/cart", pose, a.params(), np.random.default_rng(0))
    prim = scratch_ctx.kit.stage.GetPrimAtPath("/World/Test/cart")
    m = UsdPhysics.MassAPI(prim)
    assert m.GetMassAttr().Get() == pytest.approx(14.0)
    inertia = m.GetDiagonalInertiaAttr().Get()
    assert all(v > 0 for v in inertia)
    assert built.bodies == ["/World/Test/cart"]
    assert_footprint_bounds_colliders(scratch_ctx, a, pose, zpad=0.001)
    assert scratch_ctx.kit.records[-1]["mass_kg"] == pytest.approx(14.0)


def test_dynamic_density_mass():
    s = cart()
    del s["mass_kg"]
    s["density_kg_m3"] = 300.0
    a = GeneratedAsset(s)
    assert a.physics.mass_kg > 0


def test_articulated_cabinet(scratch_ctx):
    a = GeneratedAsset(cabinet())
    pose = Pose.xy_yaw(0.0, 0.0, 0.0)
    a.build(scratch_ctx, "/World/Test/cab", pose, a.params(), np.random.default_rng(0))
    joints = scratch_ctx.scene["joints"]
    assert {j["role"]: j["type"] for j in joints} == {"door": "revolute", "drawer": "prismatic"}
    door = next(j for j in joints if j["role"] == "door")
    assert (door["lower"], door["upper"], door["unit"]) == (0, 110, "deg")
    drawer = next(j for j in joints if j["role"] == "drawer")
    assert drawer["axis_local"] == "Y" and drawer["upper"] == 0.3
    assert len(scratch_ctx.scene["articulations"]) == 2
    assert [j.type for j in a.joints()] == ["revolute", "prismatic"]
    cols = assert_footprint_bounds_colliders(scratch_ctx, a, pose, zpad=0.001)
    assert any(c.group for c in cols)
    assert scratch_ctx.kit.records[-1]["articulation"] == "cab"


def test_textured_and_inline_material_build(scratch_ctx):
    s = shelf()
    s["parts"][4]["texture"] = {"base": "wood", "size_px": 128, "ops": [{"op": "noise"}, {"op": "label", "text": "ABC", "barcode": True}]}
    s["parts"][0]["texture"] = {"base": "painted", "size_px": 64}
    s["parts"][0]["shape"] = "box"
    a = GeneratedAsset(s)
    a.build(scratch_ctx, "/World/Test/shelf", Pose(), a.params(), np.random.default_rng(0))
    assert list((scratch_ctx.textures.dir).glob("gtex_*.png"))
    assert scratch_ctx.kit.stage.GetPrimAtPath("/World/Test/shelf/board0_vis") or scratch_ctx.kit.stage.GetPrimAtPath("/World/Test/shelf/post0_vis")


def test_describe_json_serialisable():
    for f in (shelf, cart, cabinet):
        d = GeneratedAsset(f()).describe()
        json.dumps(d)
        assert d["source"] == "generated" and d["footprint_at_defaults"]["size"]


def test_params_rejected():
    from sceneforge.assets.base import ParamError

    with pytest.raises(ParamError):
        GeneratedAsset(shelf()).params({"x": 1})


def test_json_schema_export():
    sch = GeneratedAssetSpec.model_json_schema()
    json.dumps(sch)
    assert "parts" in sch["properties"] and sch["additionalProperties"] is False
    assert all("description" in v for v in sch["properties"].values())
    json.dumps(texturegen.TextureSpec.model_json_schema())


def test_promote_and_load_roundtrip(tmp_path):
    out = promote(cart(), tmp_path)
    assert out == tmp_path / "generated" / "test_cart.yaml" and out.exists()
    (assets,) = load_promoted(tmp_path / "generated")
    assert assets.id == "gen/test_cart"
    assert assets.footprint().size == pytest.approx(GeneratedAsset(cart()).footprint().size)
    bad = mutate(cart(), lambda s: s.update(mass_kg=-1))
    with pytest.raises(GeneratedAssetError):
        promote(bad, tmp_path)
    with pytest.raises(GeneratedAssetError):
        GeneratedAsset(bad)


def test_registry_loader_dir_exists():
    from sceneforge.assets import generated

    assert generated.LIBRARY_GENERATED_DIR.is_dir()


# --------------------------------------------------------------------------- invalid cases
def _unstable(s):
    s["parts"] = [{"name": "foot", "shape": "box", "size": [0.1, 0.1, 0.02], "pos": [0, 0, 0.01]},
                  {"name": "heavy", "shape": "box", "size": [0.5, 0.5, 0.5], "pos": [0.5, 0, 0.27]}]
    s["anchors"] = []


def _tiny(s):
    s["parts"][0]["size"] = [0.0005, 0.5, 0.5]


CASES = [
    ("gen.schema", cart, lambda s: s.update(bogus=1), "/bogus"),
    ("gen.schema", cart, lambda s: s["parts"][0].update(shape="torus"), "/parts/0"),
    ("gen.schema", cart, lambda s: s["parts"][0].pop("size"), "/parts/0/size"),
    ("gen.id_invalid", cart, lambda s: s.update(id="cart"), "/id"),
    ("gen.id_invalid", cart, lambda s: s.update(id="gen/Bad Name"), "/id"),
    ("gen.name_duplicate", cart, lambda s: s["parts"][1].update(name="top"), "/parts/1/name"),
    ("gen.name_invalid", cart, lambda s: s["parts"][1].update(name="my part"), "/parts/1/name"),
    ("gen.mass_missing", cart, lambda s: s.pop("mass_kg"), "/mass_kg"),
    ("gen.mass_ambiguous", cart, lambda s: s.update(density_kg_m3=500), "/density_kg_m3"),
    ("gen.mass_nonpositive", cart, lambda s: s.update(mass_kg=0.0), "/mass_kg"),
    ("gen.mass_nonpositive", cart, lambda s: (s.pop("mass_kg"), s.update(density_kg_m3=-3.0)), "/density_kg_m3"),
    ("gen.mass_ignored", shelf, lambda s: s.update(mass_kg=5), "/mass_kg"),
    ("gen.mass_implausible", cart, lambda s: s.update(mass_kg=1e7), "/mass_kg"),
    ("gen.part_size_range", cart, _tiny, "/parts/0/size/0"),
    ("gen.part_size_range", cart, lambda s: s["parts"][2].update(size=[40.0, 0.04, 0.5]), "/parts/2/size/0"),
    ("gen.asset_size_range", cart, lambda s: s["parts"][0].update(pos=[60.0, 0, 0.9]), "/parts"),
    ("gen.origin_off_floor", cart, lambda s: [p.update(pos=[p["pos"][0], p["pos"][1], p["pos"][2] + 1.0]) for p in s["parts"]], "/parts"),
    ("gen.material_unknown", cart, lambda s: s["parts"][0].update(material="steal"), "/parts/0/material"),
    ("gen.physics_material_unknown", cart, lambda s: s.update(physics_material="stele"), "/physics_material"),
    ("gen.texture_unsupported_shape", cart, lambda s: s["parts"][2 + 1].update(texture={"base": "rubber"}), "/parts/3/texture"),
    ("gen.no_collider", cart, lambda s: [p.update(collide=False) for p in s["parts"]], "/parts"),
    ("gen.part_overlap", cart, lambda s: s["parts"][1].update(size=[0.9, 0.5, 0.03]), "/parts/2"),
    ("gen.unstable", cart, _unstable, "/parts"),
    ("gen.no_support_polygon", cart, lambda s: s.update(parts=[{"name": "bar", "shape": "box", "size": [0.5, 0.1, 0.1], "rpy_deg": [45, 0, 0], "pos": [0, 0, 0.07]}], anchors=[]), "/parts"),
    ("gen.links_on_non_articulated", cabinet, lambda s: s.update(body="static", mass_kg=None), "/links"),
    ("gen.articulated_without_links", cabinet, lambda s: s.update(links=[]), "/links"),
    ("gen.joint_limits", cabinet, lambda s: s["links"][0]["joint"].update(limits=[90, 10]), "/links/0/joint/limits"),
    ("gen.joint_default_range", cabinet, lambda s: s["links"][0]["joint"].update(default=200), "/links/0/joint/default"),
    ("gen.joint_parent_unknown", cabinet, lambda s: s["links"][1]["joint"].update(parent="dor"), "/links/1/joint/parent"),
    ("gen.joint_cycle", cabinet, lambda s: (s["links"][0]["joint"].update(parent="drawer"), s["links"][1]["joint"].update(parent="door")),
     "/links/0/joint/parent"),
    ("gen.anchor_part_unknown", shelf, lambda s: s["anchors"][0].update(auto_from="boardd"), "/anchors/0/auto_from"),
    ("gen.anchor_invalid", shelf, lambda s: s["anchors"][0].update(auto_from=None), "/anchors/0"),
    ("gen.anchor_invalid", shelf, lambda s: s["anchors"][2].pop("port_kind"), "/anchors/2"),
]


@pytest.mark.parametrize("code,base,fn,path", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_invalid_case(code, base, fn, path):
    iss = validate_generated(mutate(base(), fn))
    hit = by_code(iss, code)
    assert hit, [str(i) for i in iss]
    assert all(i.hint for i in iss), [str(i) for i in iss if not i.hint]
    assert any(i.path == path for i in hit), (path, [i.path for i in hit])


def test_unknown_material_suggests_close_match():
    (i,) = by_code(validate_generated(mutate(cart(), lambda s: s["parts"][0].update(material="steal"))), "gen.material_unknown")
    assert "steel" in i.hint


def test_overlap_policy():
    ov = mutate(cart(), lambda s: s["parts"][1].update(size=[0.9, 0.5, 0.03]))
    iss = validate_generated(ov)
    assert any(i.code == "gen.part_overlap" and i.severity == "warning" for i in iss)
    assert not has_errors([i for i in iss if i.code == "gen.part_overlap"])
    # an intentional overlap, flagged on either part, is silent
    ov["parts"][1]["allow_overlap"] = True
    assert not [i for i in validate_generated(ov) if i.code == "gen.part_overlap" and i.data.get("penetration_m") and "bottom" in (i.data["a"], i.data["b"])]
    assert GeneratedAsset(ov)  # warnings never block compilation


def test_unstable_hint_names_support_problem():
    (i,) = by_code(validate_generated(mutate(cart(), _unstable)), "gen.unstable")
    assert "support polygon" in i.message and "foot" in i.message and i.severity == "error"


def test_joint_cycle_reported_for_self_parent():
    iss = validate_generated(mutate(cabinet(), lambda s: s["links"][0]["joint"].update(parent="door")))
    assert "gen.joint_cycle" in codes(iss)


# --------------------------------------------------------------------------- textures
def tex_spec():
    return texturegen.TextureSpec.model_validate({
        "base": "metal_brushed", "size_px": 128, "size_m": [1.0, 0.5],
        "ops": [{"op": "noise"}, {"op": "wear", "amount": 0.4}, {"op": "dirt"}, {"op": "stains", "count": 3}, {"op": "scratches", "count": 20},
                {"op": "stripes", "region": [0, 0.8, 1, 1], "width_m": 0.05}, {"op": "grid", "tile_m": [0.25, 0.25]},
                {"op": "rect", "center": [0.2, 0.3]}, {"op": "label", "text": "BOX 7", "barcode": True, "center": [0.6, 0.4]},
                {"op": "text", "text": "A-12", "pos": [0.05, 0.05]}, {"op": "arrow"}]})


def test_texture_deterministic(tmp_path):
    spec = tex_spec()
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    texturegen.render_texture(spec, a, seed=3)
    texturegen.render_texture(spec, b, seed=3)
    texturegen.render_texture(spec, c, seed=4)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    from PIL import Image

    assert Image.open(a).size == (128, 64)


@pytest.mark.parametrize("base", ["concrete", "metal_brushed", "painted", "plastic", "wood", "cardboard", "fabric", "rubber", [0.2, 0.4, 0.6]])
def test_texture_bases_render(tmp_path, base):
    arr = texturegen.render_array(texturegen.TextureSpec(base=base, size_px=64), 0)
    assert arr.shape == (64, 64, 3) and arr.dtype == np.uint8


def test_texture_key_stable_and_sensitive():
    s = tex_spec()
    assert texturegen.texture_key(s) == texturegen.texture_key(tex_spec())
    s2 = s.model_copy(update={"size_px": 256})
    assert texturegen.texture_key(s) != texturegen.texture_key(s2)


def test_texture_spec_rejects_bad_input():
    with pytest.raises(Exception):
        texturegen.TextureSpec(size_px=32)
    with pytest.raises(Exception):
        texturegen.TextureSpec.model_validate({"base": "unobtainium"})
    with pytest.raises(Exception):
        texturegen.TextureSpec.model_validate({"ops": [{"op": "noise", "bogus": 1}]})


def test_texture_material_caches(scratch_ctx):
    spec = texturegen.TextureSpec(base="wood", size_px=64)
    m1 = texturegen.texture_material(scratch_ctx, spec)
    m2 = texturegen.texture_material(scratch_ctx, spec)
    assert m1.GetPath() == m2.GetPath()
    assert len(list(scratch_ctx.textures.dir.glob("*.png"))) == 1
