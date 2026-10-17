"""Declarative, deterministic procedural textures (PLAN.md §2.7).

An LLM describes a texture as a :class:`TextureSpec` (base material preset or colour + a list of layered ops); a fixed
renderer (numpy + PIL, no code execution) turns it into an image.  Same spec + same seed -> identical bytes.

Conventions
-----------
* The image is mapped once over each face of the textured box (UV 0..1).  ``size_m`` is the real-world size of the whole
  image; metre-valued op parameters (stripe width, grid pitch, scratch length, ...) are converted with it.
* 2-D positions/regions are **fractions of the image** ``(u, v)`` with the origin at the top-left, ``v`` down.
* Colours are RGB floats in 0..1.
* Every op gets its own RNG derived from ``(seed, op index)``, so inserting an op does not reshuffle the others.

Public API: ``TextureSpec``, ``render_texture(spec, path, seed)``, ``texture_key(spec)``, ``texture_material(ctx, spec)``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal, Union

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import gaussian_filter

from .textures import _fbm  # reuse the fractal-noise helper (textures.py itself is not modified)

RGB = tuple[float, float, float]
UV = tuple[float, float]

# preset: (rgb, roughness, metallic, grain)
BASE_PRESETS: dict[str, tuple] = {
    "concrete": ((0.54, 0.53, 0.51), 0.9, 0.0, "speckle"),
    "metal_brushed": ((0.66, 0.68, 0.70), 0.4, 0.8, "brushed"),
    "painted": ((0.62, 0.64, 0.66), 0.5, 0.05, "smooth"),
    "plastic": ((0.55, 0.57, 0.60), 0.45, 0.0, "smooth"),
    "wood": ((0.58, 0.41, 0.24), 0.8, 0.0, "wood"),
    "cardboard": ((0.73, 0.56, 0.37), 0.9, 0.0, "fibre"),
    "fabric": ((0.40, 0.42, 0.48), 0.95, 0.0, "weave"),
    "rubber": ((0.07, 0.07, 0.07), 0.9, 0.0, "speckle"),
}
BasePreset = Literal["concrete", "metal_brushed", "painted", "plastic", "wood", "cardboard", "fabric", "rubber"]


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoiseOp(_Op):
    op: Literal["noise"] = "noise"
    scale: int = Field(6, ge=1, le=64, description="Lowest noise frequency: number of noise cells across the image.")
    octaves: int = Field(4, ge=1, le=8, description="Number of fbm octaves (more = finer detail).")
    amount: float = Field(0.3, ge=0.0, le=1.0, description="Brightness modulation strength, 0 (none) .. 1 (full).")


class WearOp(_Op):
    op: Literal["wear"] = "wear"
    amount: float = Field(0.3, ge=0.0, le=1.0, description="Fraction of the surface where paint/finish is worn through.")
    color: RGB = Field((0.78, 0.78, 0.76), description="Colour of the exposed worn material, RGB 0..1.")
    scale: int = Field(10, ge=1, le=64, description="Size of the worn patches: noise cells across the image.")


class DirtOp(_Op):
    op: Literal["dirt"] = "dirt"
    amount: float = Field(0.3, ge=0.0, le=1.0, description="Dirt strength 0..1; accumulates towards the bottom (v=1).")
    color: RGB = Field((0.12, 0.10, 0.08), description="Dirt colour, RGB 0..1.")
    scale: int = Field(4, ge=1, le=64, description="Size of the dirt blotches: noise cells across the image.")


class StainsOp(_Op):
    op: Literal["stains"] = "stains"
    count: int = Field(5, ge=1, le=200, description="Number of stains (oil spots, water marks).")
    radius_m: float = Field(0.08, gt=0.0, le=5.0, description="Typical stain radius in metres (see size_m).")
    amount: float = Field(0.5, ge=0.0, le=1.0, description="Opacity of the stains, 0..1.")
    color: RGB = Field((0.08, 0.07, 0.05), description="Stain colour, RGB 0..1.")


class ScratchesOp(_Op):
    op: Literal["scratches"] = "scratches"
    count: int = Field(40, ge=1, le=2000, description="Number of scratches.")
    length_m: float = Field(0.1, gt=0.0, le=5.0, description="Typical scratch length in metres (see size_m).")
    amount: float = Field(0.4, ge=0.0, le=1.0, description="Scratch opacity 0..1.")
    color: RGB = Field((0.85, 0.85, 0.85), description="Scratch colour, RGB 0..1 (light for bare metal under paint).")


class StripesOp(_Op):
    op: Literal["stripes"] = "stripes"
    region: tuple[float, float, float, float] = Field((0.0, 0.0, 1.0, 1.0), description="(u0, v0, u1, v1) image fractions the hatching fills.")
    angle_deg: float = Field(45.0, ge=-180.0, le=180.0, description="Direction of the stripe pattern's repeat axis in degrees (45 = classic hazard hatching).")
    width_m: float = Field(0.05, gt=0.0, le=5.0, description="Width of each stripe in metres (see size_m).")
    colors: tuple[RGB, RGB] = Field(((0.95, 0.75, 0.05), (0.05, 0.05, 0.05)), description="The two alternating stripe colours (default yellow/black).")
    opacity: float = Field(1.0, ge=0.0, le=1.0, description="Stripe opacity 0..1.")


class GridOp(_Op):
    op: Literal["grid"] = "grid"
    tile_m: tuple[float, float] = Field((0.5, 0.5), description="Pitch (x, y) of the grid lines in metres: tiles or expansion joints.")
    line_width_m: float = Field(0.006, gt=0.0, le=1.0, description="Line width in metres (min 1 px).")
    darkness: float = Field(0.4, ge=0.0, le=1.0, description="How much the lines darken the base, 0..1.")


class RectOp(_Op):
    op: Literal["rect"] = "rect"
    center: UV = Field((0.5, 0.5), description="Rectangle centre as image fractions (u, v), origin top-left.")
    size: UV = Field((0.3, 0.2), description="Rectangle size as image fractions (w, h).")
    color: RGB = Field((0.8, 0.1, 0.1), description="Fill colour, RGB 0..1.")
    outline_m: float | None = Field(None, gt=0.0, le=1.0, description="If set, draw only an outline of this width in metres instead of a fill.")
    opacity: float = Field(1.0, ge=0.0, le=1.0, description="Opacity 0..1.")


class LabelOp(_Op):
    op: Literal["label"] = "label"
    center: UV = Field((0.5, 0.5), description="Label centre as image fractions (u, v), origin top-left.")
    size: UV = Field((0.4, 0.25), description="Label size as image fractions (w, h).")
    color: RGB = Field((0.93, 0.93, 0.9), description="Label paper colour, RGB 0..1.")
    text: str | None = Field(None, max_length=64, description="Optional caption drawn on the upper part of the label (PIL default font).")
    barcode: bool = Field(False, description="Draw a barcode on the label (deterministic bars from the seed).")


class TextOp(_Op):
    op: Literal["text"] = "text"
    text: str = Field(description="Text to draw (ASCII; PIL default bitmap font).", min_length=1, max_length=64)
    pos: UV = Field((0.5, 0.5), description="Top-left of the text as image fractions (u, v).")
    color: RGB = Field((0.05, 0.05, 0.05), description="Text colour, RGB 0..1.")
    scale: int = Field(2, ge=1, le=16, description="Integer upscale factor of the default 10-px font.")


class ArrowOp(_Op):
    op: Literal["arrow"] = "arrow"
    pos: UV = Field((0.5, 0.5), description="Arrow tail as image fractions (u, v).")
    length: float = Field(0.3, gt=0.0, le=2.0, description="Arrow length as a fraction of the image width.")
    angle_deg: float = Field(90.0, ge=-360.0, le=360.0, description="Direction: 0 = right, 90 = up (towards v=0), counter-clockwise.")
    width_m: float = Field(0.02, gt=0.0, le=1.0, description="Shaft width in metres (see size_m).")
    color: RGB = Field((0.1, 0.1, 0.1), description="Arrow colour, RGB 0..1.")


TextureOp = Annotated[
    Union[NoiseOp, WearOp, DirtOp, StainsOp, ScratchesOp, StripesOp, GridOp, RectOp, LabelOp, TextOp, ArrowOp],
    Field(discriminator="op"),
]


class TextureSpec(BaseModel):
    """Declarative texture: a base (preset or colour) plus layered ops, rendered deterministically from a seed."""

    model_config = ConfigDict(extra="forbid")

    base: BasePreset | RGB = Field("painted", description="Base material: a preset name (concrete, metal_brushed, painted, plastic, wood, "
                                   "cardboard, fabric, rubber) or an explicit RGB colour (floats 0..1). Presets add their characteristic grain.")
    grain: float = Field(1.0, ge=0.0, le=2.0, description="Strength of the base material's grain (0 = flat colour).")
    tint: RGB | None = Field(None, description="Optional RGB multiplier applied to a preset base, e.g. (1.0, 0.9, 0.8) to warm it.")
    size_px: int = Field(512, ge=64, le=4096, description="Image width in pixels (64..4096). Height follows the aspect ratio of size_m.")
    size_m: tuple[float, float] = Field((1.0, 1.0), description="Real-world (width, height) in metres covered by the image; converts metre-valued op parameters to pixels.")
    ops: list[TextureOp] = Field(default_factory=list, description="Layered ops applied in order, e.g. noise, wear, dirt, stains, scratches, stripes, grid, rect, label, text, arrow.")
    roughness: float | None = Field(None, ge=0.0, le=1.0, description="Override the preset's PBR roughness.")
    metallic: float | None = Field(None, ge=0.0, le=1.0, description="Override the preset's PBR metallic.")


def texture_key(spec: TextureSpec) -> str:
    """Stable content hash of the spec (the seed is *not* part of it; the per-scene TextureCache derives the seed)."""
    blob = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "gtex_" + hashlib.sha1(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- rendering
def _dims(spec: TextureSpec) -> tuple[int, int, float, float]:
    w = int(spec.size_px)
    mw, mh = max(spec.size_m[0], 1e-3), max(spec.size_m[1], 1e-3)
    h = int(np.clip(round(w * mh / mw), 64, 4096))
    return w, h, w / mw, h / mh  # w, h, px_per_m_x, px_per_m_y


def _base_image(spec: TextureSpec, h: int, w: int, rng) -> np.ndarray:
    if isinstance(spec.base, str):
        rgb, _, _, grain = BASE_PRESETS[spec.base]
    else:
        rgb, grain = spec.base, "smooth"
    col = np.asarray(rgb, np.float32)
    if spec.tint is not None and isinstance(spec.base, str):
        col = np.clip(col * np.asarray(spec.tint, np.float32), 0, 1)
    g = float(spec.grain)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if grain == "speckle":
        v = 0.9 + 0.2 * _fbm(rng, h, w, 5, 6) + gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.8) * 0.04
    elif grain == "brushed":
        streak = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), (0.4, max(2.0, w / 40)))
        v = 1.0 + streak * 0.9
    elif grain == "wood":
        warp = _fbm(rng, h, w, 4, 3) * 6.0
        rings = 0.5 + 0.5 * np.sin((xx / w * 18.0 + warp) * math.pi)
        fine = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), (max(2.0, h / 60), 0.6)) * 0.35
        v = 0.85 + 0.2 * rings + fine * 0.3
    elif grain == "fibre":
        fib = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), (0.5, 3.0))
        v = 0.9 + 0.2 * _fbm(rng, h, w, 4, 8) + fib * 0.02
    elif grain == "weave":
        f = max(2.0, w / 96.0)
        v = 0.9 + 0.1 * np.sin(xx / f * math.pi) * np.sin(yy / f * math.pi) + 0.04 * _fbm(rng, h, w, 3, 6)
    else:  # smooth
        v = 1.0 + gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 1.0) * 0.03
    v = 1.0 + (v - 1.0) * g
    return np.clip(col[None, None, :] * v[..., None], 0, 1).astype(np.float32)


def _blend(img, color, alpha):
    a = np.clip(alpha, 0, 1)[..., None]
    return img * (1 - a) + np.asarray(color, np.float32)[None, None, :] * a


def _overlay(img: np.ndarray, draw_fn, opacity: float = 1.0) -> np.ndarray:
    """Paint with PIL on an RGBA overlay and alpha-composite it onto ``img`` (float 0..1)."""
    h, w = img.shape[:2]
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(ov))
    a = np.asarray(ov).astype(np.float32) / 255.0
    alpha = a[..., 3:4] * opacity
    return img * (1 - alpha) + a[..., :3] * alpha


def _c255(c) -> tuple:
    return tuple(int(round(float(v) * 255)) for v in c) + (255,)


def _apply_op(img: np.ndarray, op, rng, ppm_x: float, ppm_y: float) -> np.ndarray:
    h, w = img.shape[:2]
    ppm = 0.5 * (ppm_x + ppm_y)
    if isinstance(op, NoiseOp):
        n = _fbm(rng, h, w, op.octaves, op.scale)
        return np.clip(img * (1.0 + op.amount * (n[..., None] - 0.5) * 1.6), 0, 1)
    if isinstance(op, WearOp):
        n = _fbm(rng, h, w, 5, op.scale)
        mask = np.clip((n - (1.0 - op.amount * 0.6 - 0.2)) * 6.0, 0, 1) if op.amount > 0 else np.zeros((h, w), np.float32)
        return _blend(img, op.color, mask * 0.9)
    if isinstance(op, DirtOp):
        n = _fbm(rng, h, w, 5, op.scale)
        v = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        mask = np.clip((0.7 * n + 0.3 * v - 0.35) * 2.5, 0, 1) * op.amount
        return _blend(img, op.color, mask * 0.85)
    if isinstance(op, StainsOp):
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        for _ in range(op.count):
            cx, cy = rng.random() * w, rng.random() * h
            rx = op.radius_m * ppm_x * rng.uniform(0.5, 1.5)
            ry = op.radius_m * ppm_y * rng.uniform(0.5, 1.5)
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=int(rng.integers(120, 255)))
        mask = gaussian_filter(np.asarray(m).astype(np.float32) / 255.0, max(1.0, op.radius_m * ppm * 0.25))
        ragged = _fbm(rng, h, w, 4, 12)
        mask = np.clip(mask * (0.6 + 0.8 * ragged) * 1.6, 0, 1) * op.amount
        return _blend(img, op.color, mask)
    if isinstance(op, ScratchesOp):
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        for _ in range(op.count):
            x, y = rng.random() * w, rng.random() * h
            ang = rng.uniform(0, math.tau)
            ln = op.length_m * ppm * rng.uniform(0.3, 1.2)
            d.line([(x, y), (x + math.cos(ang) * ln, y + math.sin(ang) * ln)], fill=int(rng.integers(100, 255)), width=1)
        mask = gaussian_filter(np.asarray(m).astype(np.float32) / 255.0, 0.5) * 1.5
        return _blend(img, op.color, np.clip(mask, 0, 1) * op.amount)
    if isinstance(op, StripesOp):
        u0, v0, u1, v1 = op.region
        x0, x1 = sorted((int(u0 * w), int(u1 * w)))
        y0, y1 = sorted((int(v0 * h), int(v1 * h)))
        if x1 <= x0 or y1 <= y0:
            return img
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        a = math.radians(op.angle_deg)
        t = (xx / ppm_x * math.cos(a) + yy / ppm_y * math.sin(a)) / op.width_m
        band = (np.floor(t).astype(np.int64) % 2).astype(bool)
        c0, c1 = np.asarray(op.colors[0], np.float32), np.asarray(op.colors[1], np.float32)
        patch = np.where(band[..., None], c1[None, None, :], c0[None, None, :])
        out = img.copy()
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - op.opacity) + patch * op.opacity
        return out
    if isinstance(op, GridOp):
        out = img.copy()
        lw_x, lw_y = max(1, int(round(op.line_width_m * ppm_x))), max(1, int(round(op.line_width_m * ppm_y)))
        px, py = max(2, int(round(op.tile_m[0] * ppm_x))), max(2, int(round(op.tile_m[1] * ppm_y)))
        xs = (np.arange(w) % px) < lw_x
        ys = (np.arange(h) % py) < lw_y
        mask = (xs[None, :] | ys[:, None]).astype(np.float32)
        return out * (1 - op.darkness * mask[..., None])
    if isinstance(op, RectOp):
        cx, cy, sw, sh = op.center[0] * w, op.center[1] * h, op.size[0] * w, op.size[1] * h
        box = [cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2]
        lw = None if op.outline_m is None else max(1, int(round(op.outline_m * ppm)))
        return _overlay(img, lambda d: d.rectangle(box, fill=None if lw else _c255(op.color), outline=_c255(op.color) if lw else None,
                                                   width=lw or 1), op.opacity)
    if isinstance(op, LabelOp):
        cx, cy, sw, sh = op.center[0] * w, op.center[1] * h, op.size[0] * w, op.size[1] * h
        x0, y0, x1, y1 = cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2

        def draw(d):
            d.rectangle([x0, y0, x1, y1], fill=_c255(op.color), outline=(120, 120, 120, 255))
            if op.text:
                d.text((x0 + 4, y0 + 3), op.text, fill=(20, 20, 20, 255))
            if op.barcode:
                x = x0 + 5
                top, bot = y0 + (0.38 if op.text else 0.12) * sh, y1 - 0.12 * sh
                while x < x1 - 5:
                    bw = int(rng.integers(1, 4))
                    d.rectangle([x, top, x + bw, bot], fill=(15, 15, 15, 255))
                    x += bw + int(rng.integers(1, 4))

        return _overlay(img, draw)
    if isinstance(op, TextOp):
        tw, th = max(8, 8 * len(op.text) + 4), 14
        small = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(small).text((1, 1), op.text, fill=255)
        big = small.resize((tw * op.scale, th * op.scale), Image.NEAREST)
        ox, oy = int(op.pos[0] * w), int(op.pos[1] * h)
        mask = Image.new("L", (w, h), 0)
        mask.paste(big, (ox, oy))
        return _blend(img, op.color, np.asarray(mask).astype(np.float32) / 255.0)
    if isinstance(op, ArrowOp):
        x, y = op.pos[0] * w, op.pos[1] * h
        L = op.length * w
        a = math.radians(op.angle_deg)
        dx, dy = math.cos(a), -math.sin(a)
        ex, ey = x + dx * L, y + dy * L
        lw = max(1, int(round(op.width_m * ppm)))
        hl = max(lw * 3.0, L * 0.25)
        nx, ny = -dy, dx
        head = [(ex + dx * hl, ey + dy * hl), (ex + nx * hl * 0.6, ey + ny * hl * 0.6), (ex - nx * hl * 0.6, ey - ny * hl * 0.6)]

        def draw(d):
            d.line([(x, y), (ex, ey)], fill=_c255(op.color), width=lw)
            d.polygon(head, fill=_c255(op.color))

        return _overlay(img, draw)
    raise ValueError(f"unknown op {op!r}")


def render_array(spec: TextureSpec, seed: int = 0) -> np.ndarray:
    """Render to a uint8 (h, w, 3) array."""
    w, h, ppm_x, ppm_y = _dims(spec)
    img = _base_image(spec, h, w, np.random.default_rng(np.random.SeedSequence([int(seed), 0xBA5E])))
    for i, op in enumerate(spec.ops):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), i + 1]))
        img = _apply_op(img, op, rng, ppm_x, ppm_y)
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def render_texture(spec: TextureSpec, path, seed: int = 0) -> Path:
    """Render ``spec`` to ``path`` (format from the suffix: .png lossless, .jpg quality 90).  Deterministic in (spec, seed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(render_array(spec, seed))
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(path, quality=90)
    else:
        im.save(path)
    return path


def _pbr(spec: TextureSpec) -> tuple[float, float]:
    rough, metal = (BASE_PRESETS[spec.base][1:3] if isinstance(spec.base, str) else (0.6, 0.0))
    return (spec.roughness if spec.roughness is not None else rough, spec.metallic if spec.metallic is not None else metal)


def texture_material(ctx, spec: TextureSpec):
    """Textured ``UsdShade.Material`` for ``spec``: image cached per scene via ``ctx.textures`` (keyed by content hash)."""
    key = texture_key(spec)
    rel = ctx.textures.get(key, lambda p, rng: render_texture(spec, p, int(rng.integers(0, 2**31 - 1))), ext="png")
    rough, metal = _pbr(spec)
    return ctx.mats.textured(key, rel, rough=rough, metal=metal)
