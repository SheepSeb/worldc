"""Resolve a spec ``asset`` reference to an Asset (PLAN.md §2.5).

Priority when several sources could serve an object: user-provided file > registry > Omniverse catalog > generated.
A spec states which one it uses explicitly; ``resolve`` honours that.  ``search`` (and MCP ``search_assets``) rank
registry and catalog matches together so the LLM prefers existing assets over generating new ones.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import registry
from .base import Asset

DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "claude_scene_generator" / "mesh_assets"


class AssetResolutionError(ValueError):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.hint = code, hint


def source_of(ref) -> str:
    if isinstance(ref, str):
        return "registry"
    for k in ("file", "catalog", "generate"):
        if (isinstance(ref, dict) and k in ref) or hasattr(ref, k):
            return k
    raise AssetResolutionError("asset.bad_ref", f"Unrecognised asset reference {ref!r}")


def resolve(ref, base_dir: Path | None = None, cache_dir: Path | None = None) -> Asset:
    src = source_of(ref)
    if src == "registry":
        return registry.get(ref)
    get = (lambda k: ref[k]) if isinstance(ref, dict) else (lambda k: getattr(ref, k))
    if src == "file":
        from .mesh.file_asset import file_asset

        path = Path(get("file"))
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        d = ref if isinstance(ref, dict) else ref.model_dump(exclude_none=True)
        overrides = {k: v for k, v in d.items() if k != "file" and v not in (None, [], "")}
        asset = file_asset(str(path), overrides or None, Path(cache_dir or DEFAULT_CACHE))
        if not registry.exists(asset.id):
            registry.register(asset)
        return asset
    if src == "catalog":
        raise AssetResolutionError("asset.catalog_unavailable", "Omniverse catalog assets need the catalog index (PLAN phase 5b).",
                                   "Use a registry id, a file or a generated asset for now.")
    from .generated import GeneratedAsset

    return GeneratedAsset(get("generate"))
