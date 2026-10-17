"""Procedural textures for the plant: patchwork steel grating (visual only), concrete ground, sky dome."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


def fbm(rng, h, w, octaves=5, base=4):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * 2**o
        g = rng.random((max(2, int(n * h / w)) + 1, n + 1)).astype(np.float32)
        out += amp * zoom(g, (h / g.shape[0], w / g.shape[1]), order=3)[:h, :w]
        tot += amp
        amp *= 0.55
    return out / tot


def make_grating(path: Path, rng, W: float, D: float, dirt: float, ppm: int = 200, panel=(1.0, 0.8)):
    """Galvanised grating: dark voids, bright bearing bars (pitch 40 mm) + cross rods (100 mm), patchwork panels with
    alternating bar direction, banding bars at panel edges, seams, dirt/rust/oil. u=(x+W/2)/W, v=1-(y+D/2)/D."""
    w, h = int(W * ppm), int(D * ppm)
    xm = (np.arange(w) + 0.5) / ppm
    ym = (np.arange(h) + 0.5) / ppm
    X, Y = np.meshgrid(xm, ym)
    pw, ph = panel
    ix, iy = (X // pw).astype(int), (Y // ph).astype(int)
    orient = ((ix + iy + (rng.random() < 0.5)) % 2 == 0)
    # random flips for a patchwork look
    flip = rng.random((iy.max() + 1, ix.max() + 1)) < 0.25
    orient = orient ^ flip[iy, ix]
    lx, ly = X - ix * pw, Y - iy * ph
    pitch, bar = 0.040, 0.012
    # bearing bars along x (orient True) vary with y; along y (orient False) vary with x
    u_b = np.where(orient, ly, lx)
    u_c = np.where(orient, lx, ly)
    bars = (u_b % pitch) < bar
    rods = (u_c % 0.1) < 0.008
    edge = np.minimum.reduce([lx, pw - lx, ly, ph - ly]) < 0.022
    seam = np.minimum.reduce([lx, pw - lx, ly, ph - ly]) < 0.004
    solid = bars | rods | edge
    tone = 0.86 + 0.1 * (rng.random((iy.max() + 1, ix.max() + 1)) - 0.5)[iy, ix]
    n = fbm(rng, h, w, 5, 5)
    img = np.where(solid, tone, 0.26)[..., None] * np.array([0.93, 0.95, 1.0], np.float32)
    img = img * (0.9 + 0.2 * n[..., None])
    # dirt: dark grime, rust and oil patches
    g = np.clip((fbm(rng, h, w, 4, 3) - 0.5) * 3.0, 0, 1) * dirt
    img = img * (1 - 0.55 * g[..., None])
    rust = np.clip((fbm(rng, h, w, 5, 6) - 0.62) * 5.0, 0, 1) * dirt * solid
    img = img * (1 - rust[..., None]) + rust[..., None] * np.array([0.38, 0.18, 0.08], np.float32)
    img[seam] *= 0.35
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=90)


def make_concrete(path: Path, rng, dirt: float, size=(1024, 1024)):
    h, w = size
    n = fbm(rng, h, w, 6, 6)
    fine = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.8)
    base = np.array([0.50, 0.49, 0.46], np.float32)
    img = base * (0.78 + 0.4 * n[..., None]) + fine[..., None] * 0.02
    stains = np.clip((fbm(rng, h, w, 4, 3) - 0.6) * 5, 0, 1) * dirt
    img *= 1 - 0.5 * stains[..., None]
    for k in range(0, w, w // 8):
        img[:, max(0, k - 1) : k + 1] *= 0.6
    for k in range(0, h, h // 8):
        img[max(0, k - 1) : k + 1, :] *= 0.6
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=88)


def make_sky(path: Path, rng, kind: str, size=(2048, 1024)):
    """Equirect sky. kind: 'clear' (blue gradient + sun glow + light clouds) | 'overcast' (bright grey)."""
    h, w = size
    v = np.linspace(0, 1, h)[:, None]  # 0 = zenith, 1 = nadir
    el = (0.5 - v) * np.pi  # +pi/2 zenith
    s = np.clip(np.sin(el), 0, 1)
    if kind == "clear":
        zen = np.array([0.18, 0.40, 0.85])
        hor = np.array([0.80, 0.88, 0.97])
        sky = hor + (zen - hor) * (s ** 0.6)[..., None]
    else:
        zen = np.array([0.62, 0.65, 0.70])
        hor = np.array([0.82, 0.84, 0.86])
        sky = hor + (zen - hor) * (s ** 0.8)[..., None]
    sky = np.broadcast_to(sky, (h, w, 3)).copy()
    c = fbm(rng, h, w, 6, 3)
    cl = np.clip((c - (0.55 if kind == "clear" else 0.35)) * (3.0 if kind == "clear" else 2.0), 0, 1) * (el > -0.05)
    cl_col = np.array([1.0, 1.0, 1.0]) if kind == "clear" else np.array([0.88, 0.89, 0.9])
    sky = sky * (1 - cl[..., None] * 0.8) + cl_col * cl[..., None] * 0.8
    # ground below horizon: warm grey
    gnd = np.broadcast_to(el < 0, (h, w))
    sky[gnd] = np.array([0.42, 0.41, 0.38]) * (0.8 + 0.3 * fbm(rng, h, w, 4, 8)[gnd][:, None])
    if kind == "clear":
        phi = np.linspace(0, 2 * np.pi, w)[None, :]
        sun_az = 0.9
        d = np.arccos(np.clip(np.sin(el) * np.sin(0.8) + np.cos(el) * np.cos(0.8) * np.cos(phi - sun_az), -1, 1))
        sky += np.exp(-(d / 0.05) ** 2)[..., None] * np.array([6.0, 5.5, 4.5]) * 0.3 + np.exp(-(d / 0.4) ** 2)[..., None] * np.array([0.25, 0.22, 0.15])
    Image.fromarray((np.clip(sky, 0, 1) * 255).astype(np.uint8)).save(path, quality=90)
