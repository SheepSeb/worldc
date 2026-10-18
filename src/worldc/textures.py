"""Procedural textures (numpy + PIL): floors (worn concrete / concrete / asphalt / epoxy with painted markings, steel grating),
cardboard with labels, concrete ground and equirect skies (the last three salvaged from the former plant generator)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, zoom

FLOOR_PX_PER_M = 100  # 40 m x 25 m -> 4000 x 2500 px


def _fbm(rng, h, w, octaves=5, base=4):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * 2**o
        g = rng.random((max(2, int(n * h / w)) + 1, n + 1)).astype(np.float32)
        out += amp * zoom(g, (h / g.shape[0], w / g.shape[1]), order=1)[:h, :w]  # linear: 4x faster, same look for noise
        tot += amp
        amp *= 0.55
    return out / tot


FLOOR_STYLES = {  # base rgb, expansion-joint grid (m, 0 = none), noise contrast
    "concrete_worn": ((138, 136, 130), 5.0, 0.34),
    "concrete": ((150, 148, 142), 5.0, 0.22),
    "asphalt": ((62, 62, 64), 0.0, 0.30),
    "epoxy": ((118, 132, 126), 0.0, 0.08),
}


def make_oak_laminate(path: Path, rng, W: float, D: float, px_per_m=FLOOR_PX_PER_M):
    """Light oak laminate: 19 cm x 1.2 m planks laid along x in a staggered bond, per-plank tone and grain, dark seams."""
    w, h = int(W * px_per_m), int(D * px_per_m)
    base = np.array([236, 204, 160], np.float32)
    img = np.empty((h, w, 3), np.float32)
    grain = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), (0.7, 14.0))
    grain /= max(float(np.abs(grain).max()), 1e-6)
    pw, pl = 0.19, 1.2
    rows = int(np.ceil(D / pw)) + 1
    for r in range(rows):
        y0, y1 = int(r * pw * px_per_m), min(h, int((r + 1) * pw * px_per_m))
        if y0 >= h:
            break
        off = float(rng.uniform(0, pl))
        x = -off
        while x < W:
            x0, x1 = max(0, int(x * px_per_m)), min(w, int((x + pl) * px_per_m))
            if x1 > x0:
                tone = float(rng.uniform(0.9, 1.08))
                img[y0:y1, x0:x1] = base * tone
                img[y0:y1, x0:max(x0 + 1, x0 + 1)] *= 0.62  # butt joint
            x += pl
        img[y0:max(y0 + 1, y0 + 1)] = img[y0:y0 + 1] * 0.62  # long seam
    img *= 1.0 + 0.07 * grain[..., None]
    img += gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.6)[..., None] * 1.5
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, quality=90)
    return w, h


def make_floor_texture(path: Path, rng, W: float, D: float, markings: list[dict], wear: float, px_per_m=FLOOR_PX_PER_M,
                       style: str = "concrete_worn"):
    """Concrete + stains + tyre wear; markings are painted with wear-dependent flaking.

    World -> pixel: u=(x+W/2)/W, v=(D/2-y)/D (v=0 is the north edge). Markings are dicts:
    {"type": "rect"|"line"|"hatch"|"poly", "color": (r,g,b), ...} in world metres.
    """
    if style == "oak_laminate":
        return make_oak_laminate(path, rng, W, D, px_per_m)
    w, h = int(W * px_per_m), int(D * px_per_m)
    base_rgb, grid, contrast = FLOOR_STYLES[style]
    base = np.array(base_rgb, np.float32)
    n = _fbm(rng, h, w, 6, 6)
    fine = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.8)
    img = base[None, None, :] * (1.0 - contrast / 2 + contrast * n[..., None]) + fine[..., None] * 3.0
    if grid:  # expansion joints
        for gx in np.arange(0, W + 0.01, grid):
            x = int(gx * px_per_m)
            img[:, max(0, x - 1) : x + 1, :] *= 0.55
        for gy in np.arange(0, D + 0.01, grid):
            y = int(gy * px_per_m)
            img[max(0, y - 1) : y + 1, :, :] *= 0.55
    # wear: dark oil stains + polished tyre lanes
    stains = _fbm(rng, h, w, 4, 3)
    stains = np.clip((stains - 0.62) * 6.0, 0, 1) * wear
    img *= (1.0 - 0.45 * stains[..., None])
    img[..., 2] *= 1.0 - 0.1 * stains
    polish = gaussian_filter(_fbm(rng, h, w, 3, 2), 6)
    img *= (1.0 + wear * 0.12 * (polish[..., None] - 0.5))

    def to_px(x, y):
        return ((x + W / 2) * px_per_m, (D / 2 - y) * px_per_m)

    paint = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(paint)
    for m in markings:
        col = tuple(int(c) for c in m["color"]) + (255,)
        t = m["type"]
        if t == "rect":  # outline
            x0, y0, x1, y1 = m["bbox"]
            lw = max(1, int(m.get("width", 0.1) * px_per_m))
            p0, p1 = to_px(x0, y1), to_px(x1, y0)
            d.rectangle([p0, p1], outline=col, width=lw)
        elif t == "fill":
            x0, y0, x1, y1 = m["bbox"]
            d.rectangle([to_px(x0, y1), to_px(x1, y0)], fill=col)
        elif t == "line":
            pts = [to_px(*p) for p in m["pts"]]
            d.line(pts, fill=col, width=max(1, int(m.get("width", 0.1) * px_per_m)))
        elif t == "dashed":
            (xa, ya), (xb, yb) = m["pts"]
            L = float(np.hypot(xb - xa, yb - ya))
            dash, gap = m.get("dash", 0.6), m.get("gap", 0.4)
            s = 0.0
            while s < L:
                e = min(L, s + dash)
                pa = (xa + (xb - xa) * s / L, ya + (yb - ya) * s / L)
                pb = (xa + (xb - xa) * e / L, ya + (yb - ya) * e / L)
                d.line([to_px(*pa), to_px(*pb)], fill=col, width=max(1, int(m.get("width", 0.1) * px_per_m)))
                s += dash + gap
        elif t == "hatch":  # diagonal hazard stripes inside bbox
            x0, y0, x1, y1 = m["bbox"]
            sw = m.get("stripe", 0.25)
            k = x0 - (y1 - y0)
            sub = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sub)
            while k < x1:
                poly = [to_px(k, y0), to_px(k + sw, y0), to_px(k + sw + (y1 - y0), y1), to_px(k + (y1 - y0), y1)]
                sd.polygon(poly, fill=col)
                k += 2 * sw
            clip = Image.new("L", (w, h), 0)
            ImageDraw.Draw(clip).rectangle([to_px(x0, y1), to_px(x1, y0)], fill=255)
            paint.paste(sub, (0, 0), Image.composite(sub.split()[3], Image.new("L", (w, h), 0), clip))
        elif t == "text":
            d.text(to_px(*m["pos"]), m["text"], fill=col)
    pa = np.asarray(paint).astype(np.float32)
    alpha = pa[..., 3:4] / 255.0
    flake = _fbm(rng, h, w, 5, 24)
    flake = np.clip(1.0 - wear * np.clip((flake - 0.45) * 3.0, 0, 1) * 0.9, 0.0, 1.0)
    alpha = alpha * flake[..., None] * (0.92 - 0.25 * wear * polish[..., None])
    img = img * (1 - alpha) + pa[..., :3] * alpha
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, quality=88)
    return w, h


def make_cardboard_texture(path: Path, rng, variant: int, size=384):
    tone = np.array([186, 142, 94], float) * rng.uniform(0.82, 1.1, 3)
    n = _fbm(rng, size, size, 4, 8)
    fib = gaussian_filter(rng.standard_normal((size, size)), (0.5, 3.0))
    img = tone[None, None, :] * (0.9 + 0.2 * n[..., None]) + fib[..., None] * 4
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    s = size
    if variant % 3 != 2:  # packing tape across the middle
        tc = (196, 170, 120) if variant % 2 else (205, 190, 150)
        d.rectangle([0, int(s * 0.47), s, int(s * 0.53)], fill=tc)
    # shipping label with barcode
    lx0, ly0 = int(s * rng.uniform(0.08, 0.35)), int(s * rng.uniform(0.58, 0.72))
    lw, lh = int(s * 0.5), int(s * 0.26)
    d.rectangle([lx0, ly0, lx0 + lw, ly0 + lh], fill=(238, 238, 232))
    x = lx0 + 6
    while x < lx0 + lw - 6:
        bw = int(rng.integers(1, 4))
        d.rectangle([x, ly0 + 8, x + bw, ly0 + lh - 18], fill=(20, 20, 20))
        x += bw + int(rng.integers(1, 4))
    d.rectangle([lx0 + 6, ly0 + lh - 14, lx0 + lw // 2, ly0 + lh - 7], fill=(90, 90, 90))
    # brand mark
    brand = [(200, 40, 40), (30, 90, 170), (30, 130, 70), (230, 150, 20), (40, 40, 40)][variant % 5]
    d.rectangle([int(s * 0.1), int(s * 0.1), int(s * 0.45), int(s * 0.24)], fill=brand)
    d.polygon([(int(s * 0.75), int(s * 0.12)), (int(s * 0.85), int(s * 0.28)), (int(s * 0.65), int(s * 0.28))], outline=(60, 40, 20), width=3)
    # fragile / this-side-up arrows on some variants
    if variant % 2 == 0:
        d.line([(int(s * 0.82), int(s * 0.85)), (int(s * 0.82), int(s * 0.62))], fill=(40, 30, 20), width=4)
        d.polygon([(int(s * 0.82), int(s * 0.56)), (int(s * 0.77), int(s * 0.64)), (int(s * 0.87), int(s * 0.64))], fill=(40, 30, 20))
    im.save(path, quality=90)


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
    n = _fbm(rng, h, w, 5, 5)
    img = np.where(solid, tone, 0.26)[..., None] * np.array([0.93, 0.95, 1.0], np.float32)
    img = img * (0.9 + 0.2 * n[..., None])
    # dirt: dark grime, rust and oil patches
    g = np.clip((_fbm(rng, h, w, 4, 3) - 0.5) * 3.0, 0, 1) * dirt
    img = img * (1 - 0.55 * g[..., None])
    rust = np.clip((_fbm(rng, h, w, 5, 6) - 0.62) * 5.0, 0, 1) * dirt * solid
    img = img * (1 - rust[..., None]) + rust[..., None] * np.array([0.38, 0.18, 0.08], np.float32)
    img[seam] *= 0.35
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=90)


def make_concrete(path: Path, rng, dirt: float, size=(1024, 1024)):
    h, w = size
    n = _fbm(rng, h, w, 6, 6)
    fine = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.8)
    base = np.array([0.50, 0.49, 0.46], np.float32)
    img = base * (0.78 + 0.4 * n[..., None]) + fine[..., None] * 0.02
    stains = np.clip((_fbm(rng, h, w, 4, 3) - 0.6) * 5, 0, 1) * dirt
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
    c = _fbm(rng, h, w, 6, 3)
    cl = np.clip((c - (0.55 if kind == "clear" else 0.35)) * (3.0 if kind == "clear" else 2.0), 0, 1) * (el > -0.05)
    cl_col = np.array([1.0, 1.0, 1.0]) if kind == "clear" else np.array([0.88, 0.89, 0.9])
    sky = sky * (1 - cl[..., None] * 0.8) + cl_col * cl[..., None] * 0.8
    # ground below horizon: warm grey
    gnd = np.broadcast_to(el < 0, (h, w))
    sky[gnd] = np.array([0.42, 0.41, 0.38]) * (0.8 + 0.3 * _fbm(rng, h, w, 4, 8)[gnd][:, None])
    if kind == "clear":
        phi = np.linspace(0, 2 * np.pi, w)[None, :]
        sun_az = 0.9
        d = np.arccos(np.clip(np.sin(el) * np.sin(0.8) + np.cos(el) * np.cos(0.8) * np.cos(phi - sun_az), -1, 1))
        sky += np.exp(-(d / 0.05) ** 2)[..., None] * np.array([6.0, 5.5, 4.5]) * 0.3 + np.exp(-(d / 0.4) ** 2)[..., None] * np.array([0.25, 0.22, 0.15])
    Image.fromarray((np.clip(sky, 0, 1) * 255).astype(np.uint8)).save(path, quality=90)
