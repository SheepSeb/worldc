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


def pin_catalog(ref) -> dict:
    """Catalog ref (dict or model) with its ``query`` replaced by the exact ``path`` + ``rev`` it resolves to."""
    from . import catalog_index

    d = dict(ref) if isinstance(ref, dict) else ref.model_dump(exclude_none=True)
    if d.get("path") or not d.get("query"):
        return d
    hit = catalog_index.resolve_query(d["query"], d.get("tags"))
    if hit is None:
        idx = catalog_index.load_index()
        if idx is None:
            raise AssetResolutionError("asset.catalog_unavailable", "No catalog index is installed, so catalog queries cannot be resolved.",
                                       "Build one with `python -m claude_scene_generator.assets.catalog_index index`, pin `path`, or use a registry asset.")
        near = ", ".join(h["path"] for h in idx.search(d["query"], 3)) or "none"
        raise AssetResolutionError("asset.catalog_no_match", f"No catalog asset matches {d['query']!r}.",
                                   f"Closest: {near}. Try other words (search_assets), or use a registry id / generated asset.")
    d.pop("query")
    d.update(path=hit["path"], **({"rev": hit["etag"]} if hit.get("etag") else {}))
    return d


def _catalog(ref, get) -> Asset:
    from .catalog import CatalogError, catalog_asset

    d = pin_catalog(ref)
    try:
        asset = catalog_asset(d["path"], {"tags": d.get("tags") or []}, rev=d.get("rev"))
    except CatalogError as e:
        raise AssetResolutionError(e.code, str(e), e.hint) from e
    if not registry.exists(asset.id):
        registry.register(asset)
    return asset


def search(query: str, k: int = 10, tags=None) -> list[dict]:
    """One ranked list over registry assets (``source: registry``) and the Omniverse catalog (``source: omniverse``).

    Scores are relative to each source's best hit; registry matches are listed first at equal score, because an
    existing registry asset is preferred over a catalog one (PLAN.md §2.5)."""
    from . import catalog_index

    q = query.lower().split()
    reg = []
    for a in registry.list_assets(tags=tags):
        text = (a["id"] + " " + a.get("description", "") + " " + " ".join(a.get("tags", []))).lower()
        hit = sum(w in text for w in q)
        if hit:
            reg.append({**a, "source": "registry", "score": round(hit / len(q), 3)})
    reg.sort(key=lambda h: (-h["score"], h["id"]))
    cat = catalog_index.search(query, k, tags=tags)
    return sorted(reg + cat, key=lambda h: (-h["score"], h["source"] != "registry"))[:k]


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
        return _catalog(ref, get)
    from .generated import GeneratedAsset

    return GeneratedAsset(get("generate"))
