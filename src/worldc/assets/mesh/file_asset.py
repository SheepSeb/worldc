"""User-provided mesh file -> MeshAsset via a content-addressed import cache (asset source priority 1, docs/PLAN.md §2.5)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .import_mesh import import_mesh
from .mesh_asset import MeshAsset, read_sidecar, write_sidecar

DEFAULT_CACHE = Path.home() / ".cache" / "worldc" / "mesh_assets"
IMPORT_KEYS = {"up", "units", "max_hulls", "collision"}
META_KEYS = {"id", "mass_kg", "tags", "body", "material", "description", "friction"}


def file_asset(path_or_url: str, overrides: dict | None = None, cache_dir: Path | str = DEFAULT_CACHE) -> MeshAsset:
    """Import ``path_or_url`` once into ``cache_dir/<sha12>/`` and return the MeshAsset.  ``overrides`` (id, mass_kg, tags,
    body, material, description, friction, and importer options up/units/max_hulls/collision) win over the sidecar and are
    part of the cache key."""
    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        raise NotImplementedError(f"downloading {s!r} is not supported yet; download the file and pass a local path")
    src = Path(s).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"asset file not found: {src}")
    ov = {k: v for k, v in (overrides or {}).items() if v is not None}
    unknown = set(ov) - IMPORT_KEYS - META_KEYS
    if unknown:
        raise ValueError(f"unknown asset overrides {sorted(unknown)}; allowed: {sorted(IMPORT_KEYS | META_KEYS)}")
    if "mass" in (overrides or {}):
        raise ValueError("use 'mass_kg' (not 'mass') in overrides")
    h = hashlib.sha256(src.read_bytes())
    h.update(json.dumps(ov, sort_keys=True, default=str).encode())
    d = Path(cache_dir) / h.hexdigest()[:12]
    asset_id = ov.get("id") or src.stem.replace("-", "_").replace(" ", "_").lower()
    sidecar = d / f"{asset_id}.asset.yaml"
    if not sidecar.is_file():
        tmp = d.with_name(d.name + ".tmp")
        import_mesh(src, tmp, asset_id, ov.get("tags", ()), ov.get("mass_kg"), description=ov.get("description"),
                    material=ov.get("material", "steel"), body=ov.get("body", "dynamic"), max_hulls=ov.get("max_hulls", 16),
                    up=ov.get("up"), units=ov.get("units"), friction=ov.get("friction"), collision=ov.get("collision", "auto"))
        if d.exists():
            import shutil

            shutil.rmtree(d)
        tmp.rename(d)
    return MeshAsset.from_file(sidecar)
