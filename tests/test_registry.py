"""Every registered asset: describable, pure planning functions, builds deterministically, physics-valid."""
import json

import numpy as np
import pytest
from pxr import UsdPhysics

from conftest import build_asset
from worldc.assets import Pose, get, ids, list_assets
from worldc.assets.resolve import AssetResolutionError, resolve, source_of


@pytest.mark.parametrize("aid", ids())
def test_contract(aid):
    a = get(aid)
    assert a.id == aid and a.description and a.tags
    d = a.describe()
    json.dumps(d)
    p = a.params()
    fp1, fp2 = a.footprint(p), a.footprint(p)
    assert fp1.to_json() == fp2.to_json() and fp1.z1 > fp1.z0
    names = [x.name for x in a.anchors(p)]
    assert len(names) == len(set(names)), "anchor names must be unique"


@pytest.mark.parametrize("aid", ids())
def test_build_deterministic_and_valid(tmp_path, aid):
    from worldc.assets.base import BuildContext
    from worldc.materials import Materials, TextureCache
    from worldc.usdkit import Kit

    recs = []
    for k in range(2):
        kit = Kit(tmp_path / f"s{k}.usd")
        kit.ns = "scene"
        ctx = BuildContext(kit, Materials(kit), TextureCache(tmp_path / f"t{k}"), out_dir=tmp_path)
        build_asset(ctx, aid, pose=Pose.xy_yaw(1.0, 2.0, 0.5), seed=11, path="/World/X")
        recs.append(json.dumps(kit.records, sort_keys=True, default=str))
        for prim in kit.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() > 0
                par = prim.GetParent()
                while par and par.GetPath().pathString != "/":
                    assert not par.HasAPI(UsdPhysics.RigidBodyAPI), f"nested rigid body {prim.GetPath()}"
                    par = par.GetParent()
    assert recs[0] == recs[1]


def test_list_filters():
    assert {a["id"] for a in list_assets(tags=["door"])} >= {"rollup_door", "swing_door"}
    assert all("valve" in a["id"] or "valve" in a["description"].lower() or "valve" in a["tags"] for a in list_assets(query="valve"))
    assert list_assets(tags=["door", "nonexistent_tag"]) == []


def test_resolve_order(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    assert source_of("pallet") == "registry" and resolve("pallet").id == "pallet"
    m = trimesh.creation.box((0.4, 0.3, 0.2))
    f = tmp_path / "crate.obj"
    m.export(f)
    a = resolve({"file": "crate.obj", "id": "my_crate", "mass_kg": 2.0}, base_dir=tmp_path, cache_dir=tmp_path / "cache")
    assert a.source == "mesh" and a.id == "my_crate" and get("my_crate") is a
    with pytest.raises(AssetResolutionError):
        resolve({"catalog": "omniverse", "query": "zzqx nonexistent"})
    g = resolve({"generate": {"id": "gen/block", "description": "a block", "body": "static",
                              "parts": [{"name": "b", "shape": "box", "size": [1, 1, 1], "pos": [0, 0, 0.5]}]}})
    assert g.source == "generated"
    with pytest.raises(KeyError, match="did you mean"):
        resolve("palet")
