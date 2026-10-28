"""NVIDIA SimReady (Hugging Face): metadata index, strict query pinning, probe, BOM-checked fetch and compilation.

A synthetic local mirror laid out like ``nvidia/simready-assets`` stands in for the repo (WORLDC_SIMREADY_ROOT), so
nothing here touches the network."""
from __future__ import annotations

import hashlib
import json

import pytest
from pxr import Usd

from worldc.assets import simready as sr
from worldc.assets.resolve import AssetResolutionError, pin_catalog, search
from worldc.compile import compile_scene
from worldc.spec.validate import validate

from test_catalog import _make_usd


def _package(root, name, size, label=None, physx=True, bad_hash=False):
    d = root / name
    usd = d / "simready_usd" / f"sm_{name}_01.usda"
    _make_usd(usd, size, label=label)
    files = [usd]
    if physx:
        (d / "simready_usd" / "payloads" / "PhysX").mkdir(parents=True)
        files.append(d / "simready_usd" / "payloads" / "PhysX" / "enabled.usda")
        files[-1].write_text("#usda 1.0\n")
    thumb = d / "simready_usd" / ".thumbs" / "256x256" / f"{usd.name}.png"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"png")
    items = [{"relative_path": f.relative_to(d).as_posix(), "size": f.stat().st_size,
              "hash": {"sha256": "0" * 64 if bad_hash else hashlib.sha256(f.read_bytes()).hexdigest()}} for f in files]
    items.append({"relative_path": thumb.relative_to(d).as_posix(), "size": 3, "hash": {"sha256": hashlib.sha256(b"png").hexdigest()}})
    (d / ".metadata").mkdir()
    (d / ".metadata" / "com.nvidia.simready.packaging.bom.json").write_text(json.dumps({"format_version": "1.0", "items": items}))
    (d / ".metadata" / "LICENSE.txt").write_text("CC-BY-4.0")


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    root = tmp_path / "hf"
    _package(root, "appliance_toaster_a01", (0.3, 0.15, 0.18), label="toaster")
    _package(root, "chair_office_mesh_a01", (0.55, 0.57, 0.8), label="office chair")
    _package(root, "equipment_liftTable_scissor_a01", (1.2, 0.8, 0.9), physx=False)
    _package(root, "broken_crate_a01", (0.4, 0.4, 0.4), bad_hash=True)
    (root / "README.md").write_text("not a package")
    monkeypatch.setenv("WORLDC_SIMREADY_ROOT", str(root))
    monkeypatch.setenv("WORLDC_SIMREADY_INDEX", str(tmp_path / "cache" / "index.jsonl"))
    monkeypatch.setattr(sr, "CACHE", tmp_path / "cache")
    monkeypatch.setattr("worldc.assets.catalog.DEFAULT_CACHE", tmp_path / "mesh_cache")
    sr.build_index(log=lambda *_: None)
    return root


def test_index_is_metadata_only(mirror, tmp_path):
    idx = sr.load_index()
    assert idx.meta["n"] == 4 and idx.meta["rev"] == "local" and idx.meta["license"] == "cc-by-4.0"
    e = next(e for e in idx.entries if e["package"] == "appliance_toaster_a01")
    assert e["path"] == "appliance_toaster_a01/simready_usd/sm_appliance_toaster_a01_01.usda"
    assert e["physics"] == {"authored": True} and e["thumb"].endswith(".thumbs/256x256/sm_appliance_toaster_a01_01.usda.png")
    assert not next(e for e in idx.entries if e["package"].startswith("equipment"))["physics"]["authored"]
    assert not (tmp_path / "cache" / "pkgs").exists()  # indexing downloads nothing


def test_search_and_strict_pinning(mirror):
    assert sr.search("toaster")[0]["package"] == "appliance_toaster_a01"
    assert sr.search("toaster")[0]["source"] == "simready" and not sr.search("toaster")[0]["cached"]
    assert sr.resolve_query("office chair")["package"] == "chair_office_mesh_a01"
    assert sr.resolve_query("forklift") is None  # the synonym table ranks the lift table, but must not pin it
    assert sr.resolve_query("zzqx") is None
    pinned = pin_catalog({"catalog": "simready", "query": "toaster"})
    assert pinned == {"catalog": "simready", "path": "appliance_toaster_a01/simready_usd/sm_appliance_toaster_a01_01.usda",
                      "rev": "local"}
    with pytest.raises(AssetResolutionError) as e:
        pin_catalog({"catalog": "simready", "query": "zzqx"})
    assert e.value.code == "asset.catalog_no_match"


def test_missing_index(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDC_SIMREADY_INDEX", str(tmp_path / "none.jsonl"))
    assert sr.search("toaster") == []
    with pytest.raises(AssetResolutionError) as e:
        pin_catalog({"catalog": "simready", "query": "toaster"})
    assert e.value.code == "asset.catalog_unavailable" and "worldc simready index" in e.value.hint


def test_validate_does_not_download(mirror, tmp_path):
    base = {"environment": {"footprint": {"size": [6, 6]}}, "zones": [{"id": "z", "rect": [-2, -2, 2, 2]}]}

    def codes(q):
        d = {**base, "objects": [{"id": "o", "asset": {"catalog": "simready", "query": q}, "placement": {"inside": "z"}}]}
        return [i.code for i in validate(d) if i.severity == "error"]

    assert codes("toaster") == [] and codes("zzqx") == ["asset.catalog_no_match"]
    assert not (tmp_path / "cache" / "pkgs").exists()


def test_probe_keeps_only_metadata(mirror, tmp_path):
    d = sr.probe("chair_office_mesh_a01")
    assert d["size_m"] == pytest.approx([0.55, 0.57, 0.8], abs=1e-3) and d["labels"] == ["office chair"] and d["physics"]["authored"]
    assert not (tmp_path / "cache" / "pkgs").exists()
    hit = sr.search("office chair")[0]
    assert hit["probed"] and hit["size_m"] == pytest.approx([0.55, 0.57, 0.8], abs=1e-3)  # probes feed search results


def test_fetch_checks_bom_and_size(mirror, tmp_path):
    d = sr.fetch_package("appliance_toaster_a01")
    assert (d / "simready_usd" / "sm_appliance_toaster_a01_01.usda").is_file() and (d / ".metadata" / "LICENSE.txt").is_file()
    assert not list(d.rglob("*.png"))  # thumbnails are never downloaded
    assert sr.is_cached("appliance_toaster_a01", "local") and sr.search("toaster")[0]["cached"]
    with pytest.raises(sr.SimReadyError) as e:
        sr.fetch_package("broken_crate_a01")
    assert e.value.code == "asset.simready_corrupt" and not (tmp_path / "cache" / "pkgs" / "local" / "broken_crate_a01").exists()
    with pytest.raises(sr.SimReadyError) as e:
        sr.fetch_package("chair_office_mesh_a01", max_mb=1e-6)
    assert e.value.code == "asset.simready_too_large"
    sr.main(["clean", "appliance_toaster_a01"])
    assert not sr.is_cached("appliance_toaster_a01", "local")


def test_unified_search_includes_simready(mirror):
    assert "simready" in {h["source"] for h in search("toaster")}


def test_compile_scene_with_simready_query(mirror, tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    d = {"environment": {"footprint": {"size": [6, 6]}, "walls": {"openings": [{"id": "door", "wall": "south", "center": 0.0, "width": 1.2, "height": 2.2}]}}, "zones": [{"id": "z", "rect": [-2.5, -2.5, 2.5, 2.5]}],
         "objects": [{"id": "t", "asset": {"catalog": "simready", "query": "toaster"}, "placement": {"inside": "z"}}]}
    r = compile_scene(d, 0, tmp_path / "out")
    assert r.ok, [str(i) for i in r.issues]
    res = json.loads((tmp_path / "out" / "resolved_spec.json").read_text())
    assert res["objects"][0]["asset"]["path"].startswith("appliance_toaster_a01/") and res["objects"][0]["asset"]["rev"] == "local"
    assert [p.name for p in (tmp_path / "cache" / "pkgs" / "local").iterdir()] == ["appliance_toaster_a01"]  # only what the spec uses
    stage = Usd.Stage.Open(str(r.usd))
    vis = next(p for p in stage.Traverse() if p.GetName() == "Visual")
    ref = vis.GetPrimStack()[0].referenceList.prependedItems[0].assetPath
    assert ref == str(tmp_path / "cache" / "pkgs" / "local" / "appliance_toaster_a01" / "simready_usd" / "sm_appliance_toaster_a01_01.usda")
    assert not list((tmp_path / "out").rglob("sm_appliance_toaster*"))  # referenced in place, not copied into the scene


def test_pinned_rev_mismatch(mirror):
    with pytest.raises(sr.SimReadyError) as e:
        sr.simready_asset("appliance_toaster_a01/simready_usd/sm_appliance_toaster_a01_01.usda", rev="d94382800d8a")
    assert e.value.code == "asset.simready_rev_mismatch"
