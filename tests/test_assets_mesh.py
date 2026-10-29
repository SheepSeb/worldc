"""Mesh import pipeline, MeshAsset, file_asset cache and library loader."""
from __future__ import annotations

import json

import pytest
trimesh = pytest.importorskip("trimesh")
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from worldc.assets import Pose, registry
from worldc.assets.mesh import file_asset as fa_mod
from worldc.assets.mesh import import_mesh as im
from worldc.assets.mesh import mesh_asset as ma
from conftest import build_asset

try:
    import coacd  # noqa: F401
    HAVE_COACD = True
except ImportError:
    HAVE_COACD = False


def _table(w=1.0, d=0.6, h=0.7, t=0.04, leg=0.06) -> trimesh.Trimesh:
    def box(size, c):
        b = trimesh.creation.box(extents=size)
        b.apply_translation(c)
        return b

    parts = [box((w, d, t), (0, 0, h - t / 2))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(box((leg, leg, h - t), (sx * (w / 2 - leg), sy * (d / 2 - leg), (h - t) / 2)))
    m = trimesh.util.concatenate(parts)
    m.apply_translation((3.0, -2.0, 1.0))  # off-origin on purpose
    return m


@pytest.fixture
def table_stl(tmp_path):
    p = tmp_path / "table.stl"
    _table().export(p)
    return p


@pytest.fixture
def imported(table_stl, tmp_path):
    return im.import_mesh(table_stl, tmp_path / "lib", "test_table", ["table", "wood"], 12.0, material="wood", max_hulls=8)


def test_import_sidecar(imported):
    sc = ma.read_sidecar(imported)
    assert sc.id == "test_table" and sc.mass_kg == 12.0 and sc.material == "wood"
    assert (imported.parent / sc.source).is_file() and (imported.parent / sc.collision.file).is_file()
    assert (imported.parent / sc.thumbnail).is_file()
    assert sc.bounds.min[2] == pytest.approx(0.0, abs=1e-4)
    assert sc.bounds.size == pytest.approx((1.0, 0.6, 0.7), abs=1e-3)
    assert sc.bounds.min[0] == pytest.approx(-0.5, abs=1e-3)
    assert len(sc.footprint) == 4
    assert sc.provenance.sha256 and sc.provenance.importer_version
    assert any(a.name == "top" and a.pos[2] == pytest.approx(0.7, abs=0.02) for a in sc.anchors)
    if HAVE_COACD:
        assert sc.provenance.decomposition == "coacd" and sc.collision.n_hulls >= 2
    else:
        assert sc.collision.n_hulls == 1 and sc.provenance.warnings


def test_up_axis_and_units_normalisation(tmp_path):
    """Y-up, centimetre USD 100 x 40 x 60 (x, y=height, z) -> Z-up metres, 1.0 x 0.6 x 0.4, bottom at z=0."""
    src = tmp_path / "yup.usda"
    st = Usd.Stage.CreateNew(str(src))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(st, 0.01)
    root = UsdGeom.Xform.Define(st, "/Root")
    st.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(st, "/Root/box")
    pts = [(x, y, z) for x in (0, 100) for y in (50, 90) for z in (0, 60)]  # offset in space, height 40 along Y
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr([i for q in quads for i in q])
    st.GetRootLayer().Save()
    sc = ma.read_sidecar(im.import_mesh(src, tmp_path / "o", "yup_box", [], 5.0, collision="single_hull"))
    assert sc.original.up_axis == "Y" and sc.original.meters_per_unit == 0.01
    assert sc.bounds.size == pytest.approx((1.0, 0.6, 0.4), abs=1e-4)
    assert sc.bounds.min == pytest.approx((-0.5, -0.3, 0.0), abs=1e-4)
    vis = Usd.Stage.Open(str(tmp_path / "o" / sc.source))
    assert UsdGeom.GetStageUpAxis(vis) == "Z" and UsdGeom.GetStageMetersPerUnit(vis) == 1.0
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(vis.GetDefaultPrim()).GetRange()
    assert bb.GetMin()[2] == pytest.approx(0.0, abs=1e-4) and bb.GetMax()[2] == pytest.approx(0.4, abs=1e-4)


def test_mesh_asset_builds_dynamic(imported, scratch_ctx):
    a = ma.MeshAsset.from_file(imported)
    registry.register(a)
    _, p, built = build_asset(scratch_ctx, "test_table", {"scale": 1.5}, Pose((1, 2, 0)))
    st = scratch_ctx.kit.stage
    root = st.GetPrimAtPath(built.root)
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    mass = UsdPhysics.MassAPI(root)
    assert mass.GetMassAttr().Get() == pytest.approx(12.0 * 1.5**3)
    assert all(v > 0 for v in mass.GetDiagonalInertiaAttr().Get())
    hulls = st.GetPrimAtPath(f"{built.root}/Collision").GetChildren()
    assert len(hulls) >= (2 if HAVE_COACD else 1)
    for h in hulls:
        assert h.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.MeshCollisionAPI(h).GetApproximationAttr().Get() == "convexHull"
    vis = st.GetPrimAtPath(f"{built.root}/Visual")
    assert vis.GetPrimStack()[0].referenceList.GetAddedOrExplicitItems()
    assert any(r["path"] == built.root and r["asset_id"] == "test_table" and r["body_type"] == "dynamic"
               for r in scratch_ctx.kit.records)
    assert built.anchors and built.anchors[0].pos[2] == pytest.approx(0.7 * 1.5, abs=0.05)
    assert json.dumps(a.describe())


def test_file_asset_cache(table_stl, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    ov = {"id": "my_table", "mass_kg": 7.0, "tags": ["x"]}
    a1 = fa_mod.file_asset(str(table_stl), ov, cache)
    assert a1.id == "my_table" and a1.sidecar.mass_kg == 7.0 and "x" in a1.tags
    calls = []
    monkeypatch.setattr(fa_mod, "import_mesh", lambda *a, **k: calls.append(1))
    a2 = fa_mod.file_asset(str(table_stl), ov, cache)
    assert not calls and a2.sidecar == a1.sidecar
    assert len(list(cache.iterdir())) == 1
    with pytest.raises(NotImplementedError):
        fa_mod.file_asset("https://example.com/a.glb", None, cache)


def test_library_example_registered():
    for a in ma.load_library():
        registry.register(a)
    a = registry.get("example_crate")
    assert a.source == "mesh" and a.sidecar.collision.n_hulls >= 1
    d = registry.describe("example_crate")
    json.dumps(d)
    sx, sy, sz = d["footprint_at_defaults"]["size"]  # planning footprint covers the collision hulls (a few mm past the visual)
    assert 0.6 <= sx <= 0.61 and 0.4 <= sy <= 0.41 and sz == pytest.approx(0.3, abs=1e-3)
    assert sum(f.stat().st_size for f in ma.DEFAULT_LIBRARY.rglob("*") if f.is_file() and "catalog" not in f.parts) < 200_000
