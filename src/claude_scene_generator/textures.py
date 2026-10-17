"""Procedural textures (numpy + PIL): worn concrete floor with painted markings, cardboard with labels."""
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
        out += amp * zoom(g, (h / g.shape[0], w / g.shape[1]), order=3)[:h, :w]
        tot += amp
        amp *= 0.55
    return out / tot


def make_floor_texture(path: Path, rng, W: float, D: float, markings: list[dict], wear: float, px_per_m=FLOOR_PX_PER_M):
    """Concrete + stains + tyre wear; markings are painted with wear-dependent flaking.

    World -> pixel: u=(x+W/2)/W, v=(D/2-y)/D (v=0 is the north edge). Markings are dicts:
    {"type": "rect"|"line"|"hatch"|"poly", "color": (r,g,b), ...} in world metres.
    """
    w, h = int(W * px_per_m), int(D * px_per_m)
    base = np.array([138, 136, 130], np.float32)
    n = _fbm(rng, h, w, 6, 6)
    fine = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.8)
    img = base[None, None, :] * (0.82 + 0.34 * n[..., None]) + fine[..., None] * 3.0
    # expansion joints on a 5 m grid
    for gx in np.arange(0, W + 0.01, 5.0):
        x = int(gx * px_per_m)
        img[:, max(0, x - 1) : x + 1, :] *= 0.55
    for gy in np.arange(0, D + 0.01, 5.0):
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
