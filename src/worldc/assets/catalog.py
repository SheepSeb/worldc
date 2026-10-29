"""Omniverse / Isaac Sim catalog assets (PLAN.md §2.6): download, import and reference by URL.  Search lives in catalog_index.

``{catalog: omniverse, path: Isaac/Props/Pallet/pallet.usd}`` resolves against the Isaac Sim asset root
(``WORLDC_ISAAC_ASSET_ROOT``, default: the 5.1 S3 root Isaac Sim itself uses).  The root USD is downloaded once into a
content-addressed cache and imported like a user file (footprint, bounds, convex-hull colliders).  The scene then
*references the original by URL*, so Isaac Sim renders NVIDIA's own materials and no mesh is copied into scene output.
The asset's own colliders / rigid bodies are switched off by overrides; our hulls are the only physics.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from .catalog_index import _mirror
from .mesh.import_mesh import ImportError_, import_mesh, load_parts, normalise
from .mesh.mesh_asset import MeshAsset, VisualXform, read_sidecar, write_sidecar

ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "worldc" / "catalog"
META_KEYS = {"id", "mass_kg", "tags", "body", "material", "description", "collision", "up_axis", "max_hulls"}


class CatalogError(RuntimeError):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.hint = code, hint


def asset_root() -> str:
    return os.environ.get("WORLDC_ISAAC_ASSET_ROOT", ASSET_ROOT).rstrip("/")


def catalog_url(path: str, root: str | None = None) -> str:
    """Full URL (or local path) of a catalog asset; absolute URLs pass through."""
    if path.startswith(("http://", "https://")):
        return path
    return f"{(root or asset_root()).rstrip('/')}/{path.lstrip('/')}"


def _fetch(url: str, dst: Path):
    if not url.startswith(("http://", "https://")):  # local asset root (tests, offline mirrors)
        if not Path(url).is_file():
            raise CatalogError("asset.catalog_not_found", f"Catalog asset not found: {url}")
        shutil.copyfile(url, dst)
        return
    try:
        with urllib.request.urlopen(url, timeout=60) as r, dst.open("wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        code = "asset.catalog_not_found" if e.code in (403, 404) else "asset.catalog_unreachable"
        raise CatalogError(code, f"Catalog asset {url} -> HTTP {e.code}",
                           "Check the path against the Isaac Sim asset browser (paths start with 'Isaac/').") from e
    except OSError as e:
        raise CatalogError("asset.catalog_unreachable", f"Cannot download {url}: {e}",
                           "Check network access or set WORLDC_ISAAC_ASSET_ROOT to a local mirror.") from e


def _physics_prims(stage) -> list[str]:
    """Prims under the default prim that carry physics APIs, as paths relative to it ('' = the default prim)."""
    root = stage.GetDefaultPrim()
    out = []
    for prim in Usd.PrimRange(root):
        if any("Physics" in s or "Physx" in s for s in prim.GetAppliedSchemas()):
            out.append(str(prim.GetPath().MakeRelativePath(root.GetPath())).replace(".", "", 1) if prim != root else "")
    return out


def _local_xforms(stage) -> dict[str, list[float]]:
    """Prims under the default prim whose xformOpOrder starts with '!resetXformStack!' (ForgeCAD writes articulated links in
    the asset's world frame), with their rest pose relative to the parent.  Referenced under a placed prim, such a part would
    otherwise ignore the placement and sit at the scene origin."""
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    out = {}
    for prim in Usd.PrimRange(root):
        if prim == root or not prim.IsA(UsdGeom.Xformable) or not UsdGeom.Xformable(prim).GetResetXformStack():
            continue
        local = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
        rel = str(prim.GetPath().MakeRelativePath(root.GetPath()))
        out[rel] = [round(float(local[i][j]), 9) for i in range(4) for j in range(4)]
    return out


def _inactive(stage) -> list[str]:
    """The asset's own physics joints and physics scenes (prim types, so _physics_prims does not see them).  Referenced into a
    scene, PhysX would still build the joints between the switched-off links (articulations, mimic joints) and every asset
    would add a physics scene next to ours (ForgeCAD ships one per asset, GPU dynamics by default): with several of them
    the first simulation step can stall.  The scene keeps exactly one physics scene, ours."""
    root = stage.GetDefaultPrim()
    return [str(p.GetPath().MakeRelativePath(root.GetPath())) for p in Usd.PrimRange(root)
            if p.IsA(UsdPhysics.Joint) or p.IsA(UsdPhysics.Scene)]


def ensure_overrides(sidecar: Path) -> Path:
    """Imports made before local_xforms / inactive were recorded: compute them from the cached source USD (no download)."""
    sc = read_sidecar(sidecar)
    if (sc.local_xforms is None or sc.inactive is None) and sc.visual_url:
        src = next((sidecar.parent / "src").rglob(Path(sc.visual_url).name), None)
        stage = Usd.Stage.Open(str(src)) if src else None
        if stage is not None and stage.GetDefaultPrim():
            write_sidecar(sc.model_copy(update={"local_xforms": _local_xforms(stage), "inactive": _inactive(stage)}), sidecar)
    return sidecar


def cached_sidecar(path: str, overrides: dict | None = None, cache_dir: Path | str | None = None, root: str | None = None,
                   rev: str | None = None) -> tuple[str, Path]:
    """(asset id, sidecar path) where catalog_asset keeps this import; the file exists only once it has been imported."""
    url = catalog_url(path, root)
    ov = {k: v for k, v in (overrides or {}).items() if v not in (None, [], "")}
    unknown = set(ov) - META_KEYS
    if unknown:
        raise ValueError(f"unknown catalog overrides {sorted(unknown)}; allowed: {sorted(META_KEYS)}")
    asset_id = ov.get("id") or "isaac_" + "".join(c if c.isalnum() else "_" for c in Path(url).stem).strip("_").lower()
    key = hashlib.sha256((url + (rev or "") + json.dumps(ov, sort_keys=True, default=str)).encode()).hexdigest()[:12]
    return asset_id, Path(cache_dir or DEFAULT_CACHE) / key / f"{asset_id}.asset.yaml"


def catalog_asset(path: str, overrides: dict | None = None, cache_dir: Path | str | None = None,
                  root: str | None = None, rev: str | None = None, source_tags=("catalog", "isaac")) -> MeshAsset:
    url = catalog_url(path, root)
    ov = {k: v for k, v in (overrides or {}).items() if v not in (None, [], "")}
    asset_id, sidecar = cached_sidecar(path, overrides, cache_dir, root, rev)
    d = sidecar.parent
    if not sidecar.is_file():
        tmp = d.with_name(d.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        base, rel = url.rsplit("/", 1) if path.startswith(("http://", "https://")) else (root or asset_root(), path.lstrip("/"))
        src = tmp / "src" / rel
        src.parent.mkdir(parents=True)
        _fetch(url, src)
        _mirror(base, rel, tmp / "src")  # many catalog assets are thin wrappers composing further layers: fetch those too
        stage = Usd.Stage.Open(str(src))
        if stage is None or not stage.GetDefaultPrim():
            raise CatalogError("asset.catalog_bad_usd", f"{url} has no default prim, so it cannot be referenced.")
        try:
            sc_path = import_mesh(src, tmp, asset_id, ov.get("tags", ()), ov.get("mass_kg"), description=ov.get("description"),
                                  max_hulls=int(ov.get("max_hulls", 16)),
                                  material=ov.get("material", "plastic"), body=ov.get("body", "dynamic"),
                                  collision=ov.get("collision", "auto"), up=ov.get("up_axis"), thumbnail=False)
        except ImportError_ as e:
            raise CatalogError("asset.catalog_bad_usd", f"{url}: {e}") from e
        # the same normalisation import_mesh applied, as a transform on the referenced original
        parts, up, mpu = load_parts(src)
        up = ov.get("up_axis") or up
        p0 = parts[0][0][0] * mpu
        if up.upper() == "Y":
            p0 = np.array([p0[0], -p0[2], p0[1]])
        shift = p0 - normalise(parts, up, mpu)[0][0][0]
        sc = read_sidecar(sc_path).model_copy(update={
            "visual_url": url,
            "visual_xform": VisualXform(translate=tuple(round(float(-v), 6) for v in shift),
                                        rotate_x_deg=90.0 if up.upper() == "Y" else 0.0, scale=mpu),
            "disable_physics": _physics_prims(stage),
            "local_xforms": _local_xforms(stage),
            "inactive": _inactive(stage),
            "tags": sorted(set(ov.get("tags", ())) | set(source_tags)),
        })
        write_sidecar(sc, sc_path)
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
    return MeshAsset.from_file(ensure_overrides(sidecar))
