"""Omniverse catalog: offline index, BM25 search, query pinning and compilation (PLAN.md §2.6).

A synthetic local asset root stands in for the Isaac Sim S3 mirror, so nothing here touches the network."""
from __future__ import annotations

import json

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from worldc.assets import catalog_index as ci
from worldc.assets import registry
from worldc.assets.resolve import AssetResolutionError, pin_catalog, resolve, search
from worldc.compile import compile_scene
from worldc.spec.validate import validate


def _make_usd(path, size, up="Z", mpu=1.0, rigid=True, label=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, up)
    UsdGeom.SetStageMetersPerUnit(st, mpu)
    root = UsdGeom.Xform.Define(st, "/Root")
    st.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Mesh.Define(st, "/Root/Body")  # unit box, scaled to `size` (metres) in stage units
    cube.GetPointsAttr().Set([Gf.Vec3f(x - 0.5, y - 0.5, z - 0.5) for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    cube.GetFaceVertexCountsAttr().Set([4] * 6)
    cube.GetFaceVertexIndicesAttr().Set([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
    UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(*[s / mpu for s in size]))
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if label:
        a = cube.GetPrim().CreateAttribute("semantic:Semantics:params:semanticData", __import__("pxr").Sdf.ValueTypeNames.String)
        a.Set(label)
    st.Save()


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("isaac_root")
    p = r / "Isaac" / "Props"
    _make_usd(p / "Dolly" / "hand_truck.usda", (0.5, 0.6, 1.2), label="hand truck")
    _make_usd(p / "Pallet" / "pallet.usda", (1.2, 0.8, 0.15))
    _make_usd(p / "Blocks" / "basic_block.usda", (0.05, 0.05, 0.05))
    _make_usd(p / "Food" / "Soup_Can.usda", (0.066, 0.1, 0.066), up="Y", mpu=0.01)  # authored in centimetres, Y-up (height on Y)
    for i in (1001, 1002, 1003):  # numbered scene instances collapse into one family
        _make_usd(r / "Isaac" / "Environments" / "Simple_Warehouse" / "Props" / f"SM_CardBoxD_04_{i}.usda", (0.4, 0.3, 0.3))
    _make_usd(p / "Dolly" / "Materials" / "skipped.usda", (1, 1, 1))
    return r


@pytest.fixture(scope="module")
def index_file(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("idx") / "catalog.jsonl"
    ci.build_index(str(root), out, packs=("Props", "Environments"), cache=out.parent / "cache", log=lambda *_: None)
    return out


def test_index_content(index_file):
    idx = ci.CatalogIndex(index_file)
    assert idx.meta["n"] == 5 and idx.meta["deep"]
    names = {e["name"] for e in idx.entries}
    assert "skipped" not in names  # Materials/ is not an asset
    box = next(e for e in idx.entries if e["name"].startswith("SM_CardBoxD_04"))
    assert box["instances"] == 3
    can = next(e for e in idx.entries if e["name"] == "Soup_Can")
    assert can["size_m"] == pytest.approx([0.066, 0.066, 0.1], abs=1e-3)  # cm -> m, Y-up -> Z-up (height ends up last)
    truck = idx.by_path["Isaac/Props/Dolly/hand_truck.usda"]
    assert truck["physics"]["rigid_body"] and truck["physics"]["collision"] and truck["labels"] == ["hand truck"]
    assert truck["size_m"] == pytest.approx([0.5, 0.6, 1.2], abs=1e-3)


def test_search_ranks_and_synonyms(index_file):
    idx = ci.CatalogIndex(index_file)
    assert idx.search("hand truck")[0]["name"] == "hand_truck"
    assert idx.search("wooden pallet")[0]["name"] == "pallet"  # ranking still works for partial matches
    assert idx.search("cardboard box")[0]["name"].startswith("SM_CardBoxD")  # via synonyms
    assert idx.search("zzqx") == []
    assert [h["name"] for h in idx.search("block", max_size_m=0.01)] == []  # too big for the size filter
    assert idx.search("pallet")[0]["score"] == 1.0


def test_resolve_query_threshold(index_file, monkeypatch):
    monkeypatch.setenv("WORLDC_CATALOG_INDEX", str(index_file))
    assert ci.resolve_query("pallet")["path"] == "Isaac/Props/Pallet/pallet.usda"
    assert ci.resolve_query("pallet jack hydraulic") is None  # only 1 of 3 words matches
    assert ci.resolve_query("wooden pallet") is None  # 1 of 2 is not enough either: a wrong pin is worse than an error
    assert ci.resolve_query("zzqx") is None


def test_pin_and_errors(index_file, monkeypatch):
    monkeypatch.setenv("WORLDC_CATALOG_INDEX", str(index_file))
    pinned = pin_catalog({"catalog": "omniverse", "query": "hand truck", "tags": ["dynamic"]})
    assert pinned["path"] == "Isaac/Props/Dolly/hand_truck.usda" and "query" not in pinned and pinned["tags"] == ["dynamic"]
    assert pin_catalog(pinned) == pinned  # already pinned: unchanged
    with pytest.raises(AssetResolutionError) as e:
        pin_catalog({"catalog": "omniverse", "query": "zzqx"})
    assert e.value.code == "asset.catalog_no_match"
    monkeypatch.setenv("WORLDC_CATALOG_INDEX", str(index_file.with_name("missing.jsonl")))
    with pytest.raises(AssetResolutionError) as e:
        pin_catalog({"catalog": "omniverse", "query": "pallet"})
    assert e.value.code == "asset.catalog_unavailable"


def test_validate_catalog_refs(index_file, monkeypatch):
    monkeypatch.setenv("WORLDC_CATALOG_INDEX", str(index_file))
    base = {"environment": {"footprint": {"size": [6, 6]}}, "zones": [{"id": "z", "rect": [-2, -2, 2, 2]}]}

    def codes(q):
        d = {**base, "objects": [{"id": "o", "asset": {"catalog": "omniverse", "query": q}, "placement": {"inside": "z"}}]}
        return [i.code for i in validate(d) if i.severity == "error"]

    assert codes("pallet") == []
    assert codes("zzqx") == ["asset.catalog_no_match"]


def test_unified_search_prefers_registry(index_file, monkeypatch):
    monkeypatch.setenv("WORLDC_CATALOG_INDEX", str(index_file))
    hits = search("pallet")
    assert {h["source"] for h in hits} == {"registry", "omniverse"}
    assert hits[0]["source"] == "registry"  # equal relevance: existing registry asset first


def test_checked_in_index_is_usable(monkeypatch):
    monkeypatch.delenv("WORLDC_CATALOG_INDEX", raising=False)
    idx = ci.load_index()
    assert idx is not None and idx.meta["n"] > 100 and idx.meta["root"].startswith("https://")
    assert "Pallet" in idx.search("pallet")[0]["path"] or "pallet" in idx.search("pallet")[0]["path"].lower()
    assert idx.search("forklift")[0]["size_m"]
    assert idx.path.stat().st_size < 2_000_000  # checked into git: keep it reviewable


def test_compile_scene_with_catalog_query(root, index_file, tmp_path, monkeypatch):
    pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    monkeypatch.setenv("WORLDC_CATALOG_INDEX", str(index_file))
    monkeypatch.setenv("WORLDC_ISAAC_ASSET_ROOT", str(root))
    monkeypatch.setattr("worldc.assets.catalog.DEFAULT_CACHE", tmp_path / "cache")
    d = {"environment": {"footprint": {"size": [6, 6]}, "walls": {"openings": [{"id": "door", "wall": "south", "center": 0.0, "width": 1.2, "height": 2.2}]}}, "zones": [{"id": "z", "rect": [-2.5, -2.5, 2.5, 2.5]}],
         "objects": [{"id": "truck", "asset": {"catalog": "omniverse", "query": "hand truck"}, "placement": {"inside": "z"}}]}
    r = compile_scene(d, 0, tmp_path / "out")
    assert r.ok, [str(i) for i in r.issues]
    res = json.loads((tmp_path / "out" / "resolved_spec.json").read_text())
    assert res["objects"][0]["asset"]["path"] == "Isaac/Props/Dolly/hand_truck.usda" and "query" not in res["objects"][0]["asset"]
    g = json.loads((tmp_path / "out" / "scene_graph.json").read_text())
    assert next(iter(g["instances"].values()))["source"] == "mesh"
    # the scene references the original by URL/path, with no copy of the catalog mesh among the outputs
    assert not list((tmp_path / "out").rglob("hand_truck*"))
    assert registry.exists("isaac_hand_truck")
    stage = Usd.Stage.Open(str(r.usd))
    vis = next(p for p in stage.Traverse() if p.GetName() == "Visual")
    refs = [i.assetPath for i in vis.GetPrimStack()[0].referenceList.prependedItems]
    assert refs and refs[0].endswith("Isaac/Props/Dolly/hand_truck.usda")  # referenced in place, not copied
    size = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(vis).ComputeAlignedRange().GetSize()
    assert sorted(size) == pytest.approx(sorted([0.5, 0.6, 1.2]), abs=0.01)
    again = compile_scene(tmp_path / "out" / "resolved_spec.json", 0, tmp_path / "out2")
    assert again.ok


def test_limited_use_license_warning(index_file, monkeypatch):
    monkeypatch.setenv("WORLDC_CATALOG_INDEX", str(index_file))
    assert ci.license_of("Isaac/Environments/Office/Props/SM_Clock_B.usd") == "limited-use"
    assert ci.license_of("Isaac/Props/Pallet/pallet.usd") == "omniverse-content"
    d = {"environment": {"footprint": {"size": [6, 6]}}, "zones": [{"id": "z", "rect": [-2, -2, 2, 2]}],
         "objects": [{"id": "o", "asset": {"catalog": "omniverse", "query": "cardbox"}, "placement": {"inside": "z"}}]}
    issues = validate(d)
    assert [i.code for i in issues if i.severity == "warning"] == ["asset.catalog_limited_license"]
    assert not [i for i in issues if i.severity == "error"]


def test_generated_asset_with_catalog_match_warns(monkeypatch):
    monkeypatch.delenv("WORLDC_CATALOG_INDEX", raising=False)
    gen = lambda gid: {"environment": {"footprint": {"size": [6, 6]}}, "zones": [{"id": "z", "rect": [-2, -2, 2, 2]}],
                       "objects": [{"id": "o", "asset": {"generate": {"id": gid, "description": "x", "body": "dynamic", "mass_kg": 5,
                                    "parts": [{"name": "b", "shape": "box", "size": [0.4, 0.4, 0.4], "pos": [0, 0, 0.2]}]}},
                                    "placement": {"inside": "z"}}]}
    assert [i.code for i in validate(gen("gen/forklift")) if i.severity == "warning"] == ["asset.generated_but_match_exists"]
    assert not [i for i in validate(gen("gen/qzx_widget")) if i.severity == "warning"]


def test_real_index_phrase_synonyms_and_refusals(monkeypatch):
    monkeypatch.delenv("WORLDC_CATALOG_INDEX", raising=False)
    assert ci.resolve_query("hand truck")["path"] == "Isaac/Props/Dolly/dolly.usd"
    assert ci.resolve_query("tote bin")["path"].startswith("Isaac/Props/KLT_Bin/")
    assert ci.resolve_query("hand drill") is None  # shares only "hand" with the hand dryer
    assert ci.resolve_query("pallet jack") is None  # not in the catalog: a plain pallet is the wrong object


def test_wrapper_layer_assets_are_mirrored(tmp_path):
    """Many Isaac assets (e.g. KLT_Bin/small_KLT.usd) are thin wrappers that reference geometry in sibling layers."""
    pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    from worldc.assets.catalog import catalog_asset

    r = tmp_path / "root"
    _make_usd(r / "Isaac" / "Props" / "Bins" / "parts" / "bin_geo.usda", (0.3, 0.2, 0.15))
    st = Usd.Stage.CreateNew(str(r / "Isaac" / "Props" / "Bins" / "bin.usda"))
    UsdGeom.SetStageUpAxis(st, "Z")
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root_prim = UsdGeom.Xform.Define(st, "/Root").GetPrim()
    st.SetDefaultPrim(root_prim)
    root_prim.GetReferences().AddReference("./parts/bin_geo.usda")
    st.Save()
    idx = ci.build_index(str(r), tmp_path / "i.jsonl", packs=("Props",), cache=tmp_path / "c", log=lambda *_: None)
    entries = {e["name"]: e for e in ci.CatalogIndex(idx).entries}
    assert "bin_geo" not in entries  # parts/ is a component folder
    assert entries["bin"]["size_m"] == pytest.approx([0.3, 0.2, 0.15], abs=1e-3)
    a = catalog_asset("Isaac/Props/Bins/bin.usda", root=str(r), cache_dir=tmp_path / "cache")
    assert a.footprint(a.params({})).pieces()
