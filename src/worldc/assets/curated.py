"""Curated SimReady subset for remanufacturing / disassembly cells: short aliases over pinned Hugging Face assets.

    python -m worldc.assets.curated list
    python -m worldc.assets.curated bake [alias ...]      # download + import once, store footprint / anchors in the table
    python -m worldc.assets.curated warm [alias ...]      # download + import into the cache only (the table is not touched)

``library/catalog/simready_curated.json`` (checked in) holds, per alias, the dataset, the pinned commit and root USD, licence
and attribution, tags, body type and material, plus what ``bake`` measured: footprint, bounds, anchors, mass, hull count
(robot arms: base height and the prims whose physics is switched off).  No USD is stored in the repo.

Each alias is a registered asset, so ``list_assets(tags=[...])``, ``describe_asset`` and the solver work offline from the
table.  Building one downloads the asset at its pinned revision (once, into the catalog cache), imports it like any catalog
asset (convex-hull colliders, the asset's own physics switched off) and references the original by its pinned HF URL.
Benches flagged ``work_top`` get a ``top`` support anchor on their work surface (0.6-1.2 m), so props can go ``on: bench.top``.
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.parse
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import Asset, Built, Footprint, PhysicsInfo
from .mesh.mesh_asset import PHYSICAL, AnchorSpec, MeshParams, read_sidecar, write_sidecar

TABLE = Path(__file__).parent / "library" / "catalog" / "simready_curated.json"
HF = "https://huggingface.co"
WORK_TOP_BAND = (0.6, 1.2)  # m: where a bench's work surface may be
MIN_TOP_FRAC = 0.25


@lru_cache(maxsize=1)
def _table(mtime_ns: int) -> dict:
    return json.loads(TABLE.read_text())


def table() -> dict:
    return _table(TABLE.stat().st_mtime_ns)


def root_url(e: dict) -> str:
    return f"{HF}/datasets/{e['dataset']}/resolve/{e['rev']}"


def asset_url(e: dict) -> str:
    return f"{root_url(e)}/{urllib.parse.quote(e['path'])}"


def overrides(alias: str, e: dict) -> dict:
    return {"id": alias, "tags": e["tags"], "body": e["body"], "material": e["material"], "description": e["description"],
            **({"max_hulls": e["max_hulls"]} if e.get("max_hulls") else {})}


@lru_cache(maxsize=None)
def mesh(alias: str):
    """The imported MeshAsset (downloads and imports on first use; cached), with the baked work-top anchor applied."""
    from .catalog import catalog_asset

    e = table()[alias]
    m = catalog_asset(e["path"], overrides(alias, e), root=root_url(e), rev=e["rev"], source_tags=("catalog", "simready"))
    return _apply_baked(m, alias, e)


def _apply_baked(m, alias: str, e: dict):
    """The baked anchors (fitted to the hulls) replace the importer's in the cached sidecar."""
    b = e.get("baked") or {}
    if "anchors" not in b:
        return m
    want = [AnchorSpec(**a) for a in baked_anchors(b)]
    if m.sidecar.anchors != want:
        sc_path = m.base_dir / f"{alias}.asset.yaml"
        sc = read_sidecar(sc_path).model_copy(update={"anchors": want})
        write_sidecar(sc, sc_path)
        m.sidecar = sc
    return m


def cached_mesh(alias: str):
    """The imported MeshAsset if it is already in the cache (never downloads), else None."""
    from .catalog import cached_sidecar, ensure_overrides
    from .mesh.mesh_asset import MeshAsset

    e = table()[alias]
    _, sc = cached_sidecar(e["path"], overrides(alias, e), root=root_url(e), rev=e["rev"])
    return _apply_baked(MeshAsset.from_file(ensure_overrides(sc)), alias, e) if sc.is_file() else None


def baked_anchors(b: dict) -> list[dict]:
    """All anchors of a baked entry: the importer's (supports fitted), the bench work top and the shelf boards."""
    extra = ([b["work_top"]] if b.get("work_top") else []) + b.get("shelves", [])
    names = {a["name"] for a in extra}
    return [a for a in b.get("anchors", []) if a["name"] not in names] + extra


class CuratedAsset(Asset):
    source = "simready"
    Params = MeshParams

    def __init__(self, alias: str, e: dict):
        self.id, self.e = alias, e
        self.tags = frozenset(e["tags"])
        b = e.get("baked") or {}
        self.description = (f"{e['description']}. SimReady asset ({e['dataset']}, {e['licence'].upper()}); "
                            f"{' x '.join(f'{v:.2f}' for v in (b.get('size') or e['size_m']))} m"
                            + ("; work top is a support anchor 'top'" if b.get("work_top") else
                               "; no usable work top (its collision shape covers it): nothing can be placed on it" if e.get("work_top") else "")
                            + (f"; shelves are support anchors shelf_0..shelf_{len(b['shelves']) - 1}" if b.get("shelves") else ""))
        self.physics = PhysicsInfo(e["body"], e["material"], PHYSICAL[e["material"]][0], b.get("mass_kg"), "convex_hulls",
                                   f"{b.get('n_hulls', '?')} convex hull(s) from the SimReady mesh; its own physics is switched off")
        self.example = {"id": alias, "asset": alias, "placement": {"inside": "zone"}}

    def footprint(self, p):
        b, s = self.e.get("baked"), p.scale
        if b:
            return Footprint([(x * s, y * s) for x, y in b["footprint"]], b["bounds"][0][2] * s, b["bounds"][1][2] * s)
        x, y, z = self.e["size_m"]
        return Footprint.rect(x * s, y * s, z * s)

    def anchors(self, p):
        return [AnchorSpec(**a).to_anchor(p.scale) for a in baked_anchors(self.e.get("baked") or {})]

    def build(self, ctx, path, pose, p, rng) -> Built:
        return mesh(self.id).build(ctx, path, pose, p, rng)


def scene_aliases() -> list[str]:
    """Curated entries registered as scene assets: not robot_cell arms, not excluded (kept in the table with the reason)."""
    return [a for a, e in table().items() if "robot_arm" not in e["tags"] and not e.get("excluded")]


def load() -> list[Asset]:
    return [CuratedAsset(a, table()[a]) for a in scene_aliases()]


def arm(alias: str) -> dict:
    """Robot-arm entry (for robot_cell): url, reach, base offset, physics prims to switch off."""
    e = table()[alias]
    return {"url": asset_url(e), "reach_m": e["arm_reach_m"], **(e.get("baked") or {})}


# ------------------------------------------------------------------------------------------------------- bake
def _flat_levels(visual_usd: Path, fp, band) -> list[tuple[float, list]]:
    """Flat, upward surfaces (1 cm z bins) inside ``band`` covering >= MIN_TOP_FRAC of the footprint: [(area, triangles)]."""
    from scipy.spatial import ConvexHull

    from .mesh.import_mesh import load_parts

    parts, _, _ = load_parts(visual_usd)  # the imported visual is already normalised (m, Z-up, bottom-centre origin)
    fp_area = float(ConvexHull(np.array(fp)).volume)
    levels: dict[int, list] = {}
    for P, F, _ in parts:
        a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        n = np.cross(b - a, c - a)
        ar = np.linalg.norm(n, axis=1) / 2
        ok = (ar > 1e-10) & (n[:, 2] / np.maximum(2 * ar, 1e-12) > 0.98)
        z = (a[:, 2] + b[:, 2] + c[:, 2]) / 3
        for i in np.where(ok & (z >= band[0]) & (z <= band[1]))[0]:
            levels.setdefault(int(round(z[i] / 0.01)), []).append((ar[i], np.stack([a[i], b[i], c[i]])))
    return [(sum(t[0] for t in tris), tris) for _, tris in sorted(levels.items()) if sum(t[0] for t in tris) >= MIN_TOP_FRAC * fp_area]


def _anchor(name: str, tris) -> dict:
    from scipy.spatial import ConvexHull

    xy = np.concatenate([t[1] for t in tris])[:, :2]
    c = (xy.min(0) + xy.max(0)) / 2
    h = xy[ConvexHull(xy).vertices]
    z = float(np.mean([t[1][:, 2].mean() for t in tris]))
    return {"name": name, "kind": "support", "pos": [round(float(c[0]), 4), round(float(c[1]), 4), round(z, 4)],
            "polygon": [[round(float(x - c[0]), 4), round(float(y - c[1]), 4)] for x, y in h]}


SUPPORT_TOL = 0.003      # a hull this much above a support surface is in the way of what stands on it
SUPPORT_REACH = 0.3      # ... if it starts less than this above the surface (a board higher up is not)
MIN_SUPPORT = 0.10       # a fitted support narrower than this (m) is dropped


def fit_support(anchor: dict | None, hulls) -> dict | None:
    """Fit a support anchor to the collision hulls: its height to the top of the hulls carrying it (the ones covering >= 25 %
    of it, else every hull within 3 cm of it), and its area to the largest rectangle left once every hull that rises above it
    (uprights, rims, a raised partial hull) is cut out.  None if nothing usable is left."""
    if anchor is None or anchor.get("kind") != "support" or not anchor.get("polygon"):
        return anchor
    z0 = anchor["pos"][2]
    poly = np.array(anchor["polygon"], float) + np.array(anchor["pos"][:2], float)
    lo, hi = poly.min(0), poly.max(0)
    area = float(np.prod(hi - lo))

    def overlap(p, r):
        return np.clip(np.minimum(p[:, :2].max(0), r[2:]) - np.maximum(p[:, :2].min(0), r[:2]), 0, None)

    r = np.array([lo[0], lo[1], hi[0], hi[1]])
    near = [p for p in hulls if abs(float(p[:, 2].max()) - z0) <= 0.03 and float(np.prod(overlap(p, r))) > 0]
    carriers = [p for p in near if float(np.prod(overlap(p, r))) >= MIN_TOP_FRAC * area]
    z = max([z0] + [float(p[:, 2].max()) for p in (carriers or near)])
    obstacles = sorted((p for p in hulls if float(p[:, 2].max()) > z + SUPPORT_TOL and float(p[:, 2].min()) < z + SUPPORT_REACH),
                       key=lambda p: -float(np.prod(overlap(p, r))))
    for p in obstacles:
        if float(np.prod(overlap(p, r))) <= 0:
            continue
        (bx0, by0), (bx1, by1) = p[:, :2].min(0), p[:, :2].max(0)
        cuts = [np.array([bx1, r[1], r[2], r[3]]), np.array([r[0], r[1], bx0, r[3]]),
                np.array([r[0], by1, r[2], r[3]]), np.array([r[0], r[1], r[2], by0])]
        cuts = [c for c in cuts if c[2] - c[0] >= MIN_SUPPORT and c[3] - c[1] >= MIN_SUPPORT]
        if not cuts:
            return None
        r = max(cuts, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    c = (r[:2] + r[2:]) / 2
    return {**anchor, "pos": [round(float(c[0]), 4), round(float(c[1]), 4), round(z, 4)],
            "polygon": [[round(float(x - c[0]), 4), round(float(y - c[1]), 4)] for x, y in
                        ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3]))]}


def _work_top(visual_usd: Path, fp) -> dict | None:
    """The bench top: the largest flat surface between WORK_TOP_BAND."""
    levels = _flat_levels(visual_usd, fp, WORK_TOP_BAND)
    return _anchor("top", max(levels, key=lambda lv: lv[0])[1]) if levels else None


def _shelves(visual_usd: Path, fp, height: float) -> list[dict]:
    """Shelf boards bottom to top as support anchors shelf_0, shelf_1, ... (levels closer than 0.15 m keep the larger)."""
    kept: list[tuple[float, list]] = []
    for area, tris in _flat_levels(visual_usd, fp, (0.05, height + 0.01)):
        z = float(np.mean([t[1][:, 2].mean() for t in tris]))
        if kept and z - kept[-1][0] < 0.15:
            if area > sum(t[0] for t in kept[-1][1]):
                kept[-1] = (z, tris)
            continue
        kept.append((z, tris))
    return [_anchor(f"shelf_{i}", tris) for i, (_, tris) in enumerate(kept)]


def _bake_arm(e: dict) -> dict:
    from pxr import Usd, UsdGeom

    from .catalog_index import _mirror

    with tempfile.TemporaryDirectory(prefix="curated_arm_") as td:
        stage = Usd.Stage.Open(str(_mirror(root_url(e), e["path"], Path(td))))
        root = stage.GetDefaultPrim()
        rng = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(root).ComputeAlignedRange()
        mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
        out = {"base_offset_z": round(-float(rng.GetMin()[2]) * mpu, 4), "size": [round(float(v) * mpu, 4) for v in rng.GetSize()],
               "rigid_bodies": [], "joints": [], "articulation_roots": []}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            rel = str(prim.GetPath().MakeRelativePath(root.GetPath()))
            rel = "" if rel == "." else rel
            s = set(prim.GetAppliedSchemas())
            if "PhysicsRigidBodyAPI" in s:
                out["rigid_bodies"].append(rel)
            if "PhysicsArticulationRootAPI" in s:
                out["articulation_roots"].append(rel)
            if "Joint" in prim.GetTypeName():
                out["joints"].append(rel)
    return out


def bake_one(alias: str) -> tuple[str, dict]:
    e = table()[alias]
    if "robot_arm" in e["tags"]:
        return alias, _bake_arm(e)
    from .catalog import catalog_asset
    from .mesh.import_mesh import detect_top_anchor, load_parts
    from .mesh.mesh_asset import load_hulls

    mesh.cache_clear()
    m = catalog_asset(e["path"], overrides(alias, e), root=root_url(e), rev=e["rev"], source_tags=("catalog", "simready"))
    sc = m.sidecar
    hulls = [h for h, _, _ in load_hulls(m.collision_file)]
    parts = load_parts(m.visual_file)[0]
    importer = [a.model_dump(mode="json", exclude_defaults=True) for a in sc.anchors if a.kind != "support"]  # ports as they are
    top = detect_top_anchor(parts, sc.footprint)  # the importer's own top, measured afresh (the cached sidecar may be patched)
    if top is not None and not e.get("work_top"):
        importer.append(top.model_dump(mode="json", exclude_defaults=True))
    out = {"footprint": [list(map(float, p)) for p in sc.footprint], "bounds": [list(sc.bounds.min), list(sc.bounds.max)],
           "size": [round(v, 4) for v in sc.bounds.size], "mass_kg": sc.mass_kg, "n_hulls": sc.collision.n_hulls,
           "anchors": [a for a in (fit_support(a, hulls) for a in importer) if a is not None]}
    if e.get("work_top"):
        out["work_top"] = fit_support(_work_top(m.visual_file, sc.footprint), hulls)
    if "shelf" in e["tags"]:
        out["shelves"] = [a for a in (fit_support(a, hulls) for a in _shelves(m.visual_file, sc.footprint, sc.bounds.max[2])) if a is not None]
    return alias, out


def bake(aliases=None, workers: int = 4, log=print):
    from concurrent.futures import ProcessPoolExecutor

    t = json.loads(TABLE.read_text())
    todo = list(aliases or t)
    with ProcessPoolExecutor(workers) as ex:
        for alias, baked in ex.map(bake_one, todo):
            t[alias]["baked"] = baked
            log(f"  {alias}: {baked.get('size')} hulls={baked.get('n_hulls', '-')}" + (f" top z={baked['work_top']['pos'][2]}" if baked.get("work_top") else ""))
            TABLE.write_text(json.dumps(t, indent=1) + "\n")  # after each, so an interrupted bake keeps what it did


from . import registry  # noqa: E402

if __name__ != "__main__":
    registry.add_loader(load)

if __name__ == "__main__":
    if sys.argv[1:2] == ["bake"]:
        bake(sys.argv[2:] or None)
    elif sys.argv[1:2] == ["warm"]:  # before a frozen batch: same downloads / imports as bake, but the table (and its hash) stay as frozen
        from . import curated as _pkg

        for a in sys.argv[2:] or scene_aliases():
            _pkg.mesh(a)
            print(f"  {a}: cached")
    else:
        for a, e in table().items():
            print(f"{a:26s} {','.join(t for t in e['tags'] if t != 'simready'):40s} {e['dataset']}")
