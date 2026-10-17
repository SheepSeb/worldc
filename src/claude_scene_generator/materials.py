"""Named material presets (visual + physics) and a per-scene procedural texture cache.

Assets ask for materials by preset name (``ctx.mats.get("wood")``) or by colour (``ctx.mats.color(rgb)``); USD material
prims are created on first use under ``/World/Looks`` and ``/World/PhysicsMaterials``.
"""
from __future__ import annotations

import hashlib
import zlib
from pathlib import Path
from typing import Callable

import numpy as np

# name: (rgb, roughness, metallic[, opacity[, emissive]])
VISUAL: dict[str, tuple] = {
    # --- generic
    "steel": ((0.68, 0.70, 0.72), 0.42, 0.4),
    "steel_b": ((0.62, 0.64, 0.66), 0.45, 0.4),
    "steel_c": ((0.76, 0.77, 0.78), 0.38, 0.45),
    "steel_d": ((0.56, 0.58, 0.60), 0.5, 0.4),
    "steel_dark": ((0.34, 0.35, 0.37), 0.5, 0.45),
    "steel_bright": ((0.75, 0.76, 0.78), 0.25, 1.0),
    "stainless": ((0.72, 0.73, 0.74), 0.32, 0.9),
    "rubber": ((0.04, 0.04, 0.04), 0.9, 0.0),
    "black": ((0.03, 0.03, 0.035), 0.6, 0.0),
    "white": ((0.9, 0.9, 0.87), 0.6, 0.0),
    "safety_yellow": ((0.95, 0.75, 0.05), 0.5, 0.0),
    "yellow": ((0.98, 0.8, 0.05), 0.45, 0.0),
    "dark_void": ((0.02, 0.02, 0.02), 1.0, 0.0),
    # --- building
    "apron": ((0.30, 0.30, 0.30), 0.85, 0.0),
    "wall": ((0.78, 0.78, 0.74), 0.9, 0.0),
    "roof": ((0.62, 0.64, 0.66), 0.8, 0.0),
    "truss": ((0.25, 0.28, 0.32), 0.6, 0.6),
    # --- logistics
    "wood": ((0.58, 0.41, 0.24), 0.85, 0.0),
    "wood_light": ((0.72, 0.55, 0.34), 0.85, 0.0),
    "wood_dark": ((0.5, 0.33, 0.18), 0.8, 0.0),
    "wrap": ((0.80, 0.85, 0.90), 0.25, 0.0),
    "tote_blue": ((0.08, 0.28, 0.68), 0.5, 0.0),
    "tote_grey": ((0.45, 0.47, 0.50), 0.5, 0.0),
    "tote_red": ((0.68, 0.12, 0.10), 0.5, 0.0),
    "item_a": ((0.9, 0.8, 0.1), 0.6, 0.0),
    "item_b": ((0.85, 0.85, 0.85), 0.6, 0.0),
    "item_c": ((0.15, 0.55, 0.3), 0.6, 0.0),
    "item_d": ((0.3, 0.3, 0.8), 0.6, 0.0),
    "belt": ((0.06, 0.06, 0.065), 0.85, 0.0),
    "conv_frame": ((0.45, 0.48, 0.52), 0.4, 0.7),
    "roller": ((0.7, 0.72, 0.75), 0.3, 0.9),
    "table_top": ((0.75, 0.62, 0.42), 0.7, 0.0),
    "drawer": ((0.55, 0.58, 0.62), 0.5, 0.4),
    "door_grey": ((0.62, 0.64, 0.67), 0.45, 0.6),
    "door_slats": ((0.80, 0.82, 0.84), 0.4, 0.5),
    "door_frame": ((0.30, 0.33, 0.38), 0.5, 0.5),
    "door_leaf": ((0.18, 0.40, 0.62), 0.5, 0.3),
    "leveler": ((0.50, 0.52, 0.55), 0.5, 0.7),
    "cage_mesh": ((0.75, 0.78, 0.8), 0.5, 0.0, 0.3),
    "cage_post": ((0.90, 0.70, 0.05), 0.5, 0.0),
    "cage_gate": ((0.88, 0.50, 0.08), 0.5, 0.4, 0.6),
    "cone": ((0.95, 0.35, 0.05), 0.55, 0.0),
    "cone_base": ((0.08, 0.08, 0.08), 0.9, 0.0),
    "bin_green": ((0.10, 0.45, 0.20), 0.6, 0.0),
    "bin_dark": ((0.03, 0.04, 0.03), 0.9, 0.0),
    "jack": ((0.80, 0.15, 0.10), 0.5, 0.4),
    "forklift": ((0.95, 0.72, 0.08), 0.5, 0.0),
    "human_a": ((0.95, 0.45, 0.05), 0.7, 0.0),
    "human_b": ((0.10, 0.30, 0.65), 0.7, 0.0),
    "charger": ((0.10, 0.60, 0.30), 0.4, 0.3),
    "charger_led": ((0.1, 1.0, 0.3), 0.3, 0.0, 1.0, (0.1, 0.9, 0.3)),
    "robot_body": ((0.85, 0.87, 0.90), 0.4, 0.3),
    "robot_arm": ((0.20, 0.45, 0.80), 0.4, 0.4),
    "robot_marker": ((0.1, 0.8, 0.9), 0.5, 0.0, 0.5),
    # --- industrial
    "valve_body": ((0.62, 0.64, 0.66), 0.45, 0.45),
    "valve_red": ((0.62, 0.06, 0.05), 0.45, 0.15),
    "red_jb": ((0.66, 0.05, 0.04), 0.45, 0.15),
    "gray_box": ((0.58, 0.60, 0.62), 0.5, 0.2),
    "panel_plate": ((0.45, 0.47, 0.5), 0.55, 0.35),
    "cab_blue": ((0.06, 0.17, 0.50), 0.45, 0.2),
    "brown": ((0.26, 0.12, 0.07), 0.65, 0.1),
    "motor": ((0.55, 0.57, 0.58), 0.5, 0.3),
    "pump_gray": ((0.45, 0.47, 0.49), 0.5, 0.4),
    "insul": ((0.9, 0.9, 0.88), 0.45, 0.2),
    "orange_band": ((0.95, 0.42, 0.05), 0.5, 0.0),
    "green_band": ((0.08, 0.50, 0.20), 0.5, 0.0),
    "white_band": ((0.92, 0.92, 0.9), 0.5, 0.0),
    "blue_band": ((0.1, 0.3, 0.75), 0.5, 0.0),
    "tube": ((0.62, 0.63, 0.65), 0.35, 0.8),
    "cable_red": ((0.7, 0.05, 0.05), 0.6, 0.0),
    "cable_orange": ((0.95, 0.45, 0.05), 0.6, 0.0),
    "cable_green": ((0.06, 0.5, 0.15), 0.6, 0.0),
    "cable_blue": ((0.08, 0.2, 0.7), 0.6, 0.0),
    "btn_green": ((0.05, 0.6, 0.15), 0.4, 0.0),
    "btn_red": ((0.8, 0.04, 0.04), 0.4, 0.0),
    "btn_black": ((0.04, 0.04, 0.04), 0.4, 0.0),
    "gauge_white": ((0.92, 0.92, 0.88), 0.4, 0.0),
    "coverall_red": ((0.74, 0.06, 0.05), 0.8, 0.0),
    "boot": ((0.25, 0.14, 0.08), 0.7, 0.0),
    "glove": ((0.95, 0.5, 0.05), 0.7, 0.0),
    "skin": ((0.78, 0.58, 0.45), 0.7, 0.0),
    "hardhat": ((0.96, 0.42, 0.03), 0.4, 0.0),
    "hardhat_white": ((0.9, 0.9, 0.88), 0.4, 0.0),
    "bucket_yellow": ((0.9, 0.72, 0.06), 0.5, 0.0),
    "orange_box": ((0.9, 0.4, 0.05), 0.6, 0.0),
    "wheel_blue": ((0.08, 0.22, 0.6), 0.5, 0.1),
    "wheel_white": ((0.85, 0.85, 0.84), 0.5, 0.1),
    "wheel_red": ((0.65, 0.07, 0.06), 0.5, 0.1),
    "tank_white": ((0.78, 0.8, 0.82), 0.6, 0.1),
}

# name: (friction (static = dynamic), restitution)
PHYSICAL: dict[str, tuple] = {
    "cardboard": (0.5, 0.05),
    "steel": (0.6, 0.1),
    "concrete": (0.7, 0.05),
    "rubber": (0.9, 0.05),
    "wood": (0.5, 0.05),
    "plastic": (0.4, 0.05),
    "grating": (0.7, 0.05),
}

# materials that keep their colour under weathering (people, signals, cables)
_NO_WEATHER = ("coverall", "skin", "glove", "hardhat", "btn_", "cable", "gauge", "charger_led", "human_", "robot_")


def _usd_name(s: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in s)
    return out if out and not out[0].isdigit() else "m_" + out


class Materials:
    """Lazily-created USD materials.  ``weathering`` in [0, 1] darkens / roughens non-exempt presets (seeded)."""

    def __init__(self, kit, weathering: float = 0.0, seed: int = 0):
        self.kit = kit
        self.weathering = float(weathering)
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, 0x6D6174]))
        self._tint: dict[str, np.ndarray] = {}

    def get(self, name: str):
        if name not in VISUAL:
            raise KeyError(f"unknown material preset {name!r}")
        rgb, rough, metal, *rest = VISUAL[name]
        opacity = rest[0] if rest else 1.0
        emissive = rest[1] if len(rest) > 1 else None
        c = np.asarray(rgb, float)
        if self.weathering > 0 and not name.startswith(_NO_WEATHER):
            if name not in self._tint:
                self._tint[name] = self._rng.uniform(0.4, 1.0)
            d = self.weathering
            c = c * (1 - 0.30 * d * self._tint[name]) + np.array([0.20, 0.16, 0.11]) * 0.12 * d
            rough = min(1.0, rough + 0.25 * d)
        return self.kit.material(_usd_name(name), tuple(np.clip(c, 0, 1)), rough, metal, opacity=opacity, emissive=emissive)

    def color(self, rgb, rough: float = 0.5, metal: float = 0.0, opacity: float = 1.0, emissive=None, name: str | None = None):
        key = name or "c_" + hashlib.sha1(repr((tuple(round(float(v), 4) for v in rgb), rough, metal, opacity, emissive)).encode()).hexdigest()[:10]
        return self.kit.material(_usd_name(key), tuple(float(v) for v in rgb), rough, metal, opacity=opacity, emissive=emissive)

    def textured(self, name: str, rel_path: str, rough: float = 0.7, metal: float = 0.0):
        return self.kit.material(_usd_name(name), (0.5, 0.5, 0.5), rough, metal, tex=rel_path)

    def phys(self, name: str):
        if name not in PHYSICAL:
            raise KeyError(f"unknown physics material preset {name!r}")
        mu, rest = PHYSICAL[name]
        return self.kit.phys_material(_usd_name(name), mu, rest)

    def visual_or_phys(self, name):
        return self.get(name)


class TextureCache:
    """Procedural textures written once per scene to ``out_dir/textures/<key>.jpg``; returns the stage-relative path.

    ``maker(path, rng)`` writes the file.  The rng is derived from (scene seed, key), so a texture does not change
    when other textures are added.  ``legacy_seed`` lets recipes reproduce pre-refactor seeding exactly.
    """

    def __init__(self, out_dir: Path, seed: int = 0):
        self.dir = Path(out_dir) / "textures"
        self.seed = seed
        self._done: dict[str, str] = {}

    def rng_for(self, key: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, zlib.crc32(key.encode())]))

    def get(self, key: str, maker: Callable[[Path, np.random.Generator], object], ext: str = "jpg",
            legacy_seed: int | None = None) -> str:
        if key in self._done:
            return self._done[key]
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{key}.{ext}"
        rng = np.random.default_rng(legacy_seed) if legacy_seed is not None else self.rng_for(key)
        maker(path, rng)
        rel = f"./textures/{key}.{ext}"
        self._done[key] = rel
        return rel

    def cardboard(self, variant: int) -> str:
        from .textures import make_cardboard_texture

        return self.get(f"cardboard_{variant}", lambda p, r: make_cardboard_texture(p, r, variant), legacy_seed=self.seed * 100 + variant)
