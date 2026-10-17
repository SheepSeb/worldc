"""Global asset registry.

Procedural assets register at import time with ``@register``.  Other sources (mesh sidecars in ``assets/library``,
promoted generated assets, ...) contribute through *loaders*: callables run once, lazily, on first registry access, that
return Asset instances.
"""
from __future__ import annotations

import difflib
import importlib
from typing import Callable, Iterable

from .base import Asset

_ASSETS: dict[str, Asset] = {}
_LOADERS: list[Callable[[], Iterable[Asset]]] = []
_loaded = False

# procedural modules imported on first access (each registers its assets)
PROCEDURAL_MODULES = (
    "building", "racks", "doors", "conveyors", "furniture", "logistics", "clutter", "agents", "industrial",
)
# modules that only add loaders (mesh library sidecars, promoted generated assets)
LOADER_MODULES = ("mesh.mesh_asset", "generated", "robots", "vehicles")


def register(cls_or_obj):
    """Class decorator (or call with an instance). Duplicate ids are an error."""
    obj = cls_or_obj() if isinstance(cls_or_obj, type) else cls_or_obj
    if obj.id in _ASSETS and type(_ASSETS[obj.id]) is not type(obj):
        raise ValueError(f"duplicate asset id {obj.id!r}")
    _ASSETS[obj.id] = obj
    return cls_or_obj


def add_loader(fn: Callable[[], Iterable[Asset]]):
    _LOADERS.append(fn)
    return fn


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    _loaded = True
    for m in [f"procedural.{m}" for m in PROCEDURAL_MODULES] + list(LOADER_MODULES):
        try:
            importlib.import_module(f"{__package__}.{m}")
        except ModuleNotFoundError as e:
            if e.name != f"{__package__}.{m}":
                raise
    for fn in _LOADERS:
        for a in fn():
            register(a)


def get(asset_id: str) -> Asset:
    _ensure_loaded()
    try:
        return _ASSETS[asset_id]
    except KeyError:
        close = difflib.get_close_matches(asset_id, list(_ASSETS), n=3)
        raise KeyError(f"unknown asset {asset_id!r}" + (f"; did you mean {', '.join(close)}?" if close else "")) from None


def exists(asset_id: str) -> bool:
    _ensure_loaded()
    return asset_id in _ASSETS


def ids() -> list[str]:
    _ensure_loaded()
    return sorted(_ASSETS)


def suggest(asset_id: str, n: int = 3) -> list[str]:
    _ensure_loaded()
    return difflib.get_close_matches(asset_id, list(_ASSETS), n=n)


def list_assets(tags: Iterable[str] | None = None, query: str | None = None) -> list[dict]:
    """Summaries of assets having *all* ``tags`` and whose id/description/tags contain ``query`` (case-insensitive)."""
    _ensure_loaded()
    tags = set(tags or ())
    q = (query or "").lower()
    out = []
    for aid in sorted(_ASSETS):
        a = _ASSETS[aid]
        if tags and not tags <= set(a.tags):
            continue
        if q and q not in (a.id + " " + a.description + " " + " ".join(a.tags)).lower():
            continue
        out.append(a.summary())
    return out


def describe(asset_id: str) -> dict:
    return get(asset_id).describe()
