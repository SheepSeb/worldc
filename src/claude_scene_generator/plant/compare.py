"""Quantitative render-vs-reference comparison (colour mix, layout grid, edge density) + side-by-side image."""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

SIZE = (600, 400)
CLASSES = {  # name: (hue range, min sat, (min val, max val))
    "red": ((0.96, 0.04), 0.5, (0.25, 1.0)), "orange": ((0.04, 0.10), 0.5, (0.4, 1.0)), "yellow": ((0.10, 0.20), 0.5, (0.4, 1.0)),
    "green": ((0.25, 0.45), 0.35, (0.15, 1.0)), "blue": ((0.55, 0.72), 0.5, (0.15, 1.0)),
}


def _hsv(img):
    a = np.asarray(img.convert("RGB").resize(SIZE, Image.LANCZOS), np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(mx)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = d > 1e-6
    h = np.where(m & (mx == r), ((g - b) / np.maximum(d, 1e-6)) % 6, h)
    h = np.where(m & (mx == g), (b - r) / np.maximum(d, 1e-6) + 2, h)
    h = np.where(m & (mx == b), (r - g) / np.maximum(d, 1e-6) + 4, h)
    return a, (h / 6.0) % 1.0, s, mx


def class_masks(h, s, v):
    out = {}
    for n, ((h0, h1), smin, (v0, v1)) in CLASSES.items():
        hm = (h >= h0) | (h < h1) if h0 > h1 else (h >= h0) & (h < h1)
        out[n] = hm & (s >= smin) & (v >= v0) & (v <= v1)
    out["gray_steel"] = (s < 0.22) & (v > 0.15) & (v <= 0.85)
    out["bright_white"] = (v > 0.85) & (s < 0.25)
    out["dark"] = v <= 0.15
    return out


def _edges(a):
    lum = a.mean(2)
    gx, gy = sobel(lum, axis=1), sobel(lum, axis=0)
    return np.hypot(gx, gy)


def describe(path):
    a, h, s, v = _hsv(Image.open(path))
    masks = class_masks(h, s, v)
    frac = {k: float(m.mean()) for k, m in masks.items()}
    e = _edges(a)
    lum = a.mean(2)
    gh, gw = 4, 6
    grid_l = np.array([[lum[i * SIZE[1] // gh:(i + 1) * SIZE[1] // gh, j * SIZE[0] // gw:(j + 1) * SIZE[0] // gw].mean() for j in range(gw)] for i in range(gh)])
    grid_e = np.array([[(e[i * SIZE[1] // gh:(i + 1) * SIZE[1] // gh, j * SIZE[0] // gw:(j + 1) * SIZE[0] // gw] > 0.25).mean() for j in range(gw)] for i in range(gh)])
    red = masks["red"]
    ys, xs = np.nonzero(red)
    cen = [float(xs.mean() / SIZE[0]), float(ys.mean() / SIZE[1])] if len(xs) > 50 else None
    return dict(class_fractions=frac, mean_luminance=float(lum.mean()), std_luminance=float(lum.std()), edge_density=float((e > 0.25).mean()), grid_luminance=grid_l, grid_edges=grid_e, red_centroid=cen)


def compare_to_reference(render, reference, out_png=None):
    A, B = describe(render), describe(reference)
    keys = list(A["class_fractions"])
    cls_l1 = float(sum(abs(A["class_fractions"][k] - B["class_fractions"][k]) for k in keys))
    grid_l = float(np.abs(A["grid_luminance"] - B["grid_luminance"]).mean())
    grid_e = float(np.abs(A["grid_edges"] - B["grid_edges"]).mean())
    edge_ratio = A["edge_density"] / max(B["edge_density"], 1e-6)
    cdist = None if not (A["red_centroid"] and B["red_centroid"]) else float(np.hypot(*(np.array(A["red_centroid"]) - np.array(B["red_centroid"]))))
    score = 0.30 * max(0, 1 - cls_l1 / 1.0) + 0.25 * max(0, 1 - grid_l * 4) + 0.30 * max(0, 1 - grid_e * 3) + 0.15 * (0.0 if cdist is None else max(0, 1 - cdist * 2))
    res = dict(similarity_score=round(score, 3), class_fraction_render={k: round(v, 3) for k, v in A["class_fractions"].items()},
               class_fraction_reference={k: round(v, 3) for k, v in B["class_fractions"].items()}, class_l1=round(cls_l1, 3),
               mean_luminance=[round(A["mean_luminance"], 3), round(B["mean_luminance"], 3)], luminance_std=[round(A["std_luminance"], 3), round(B["std_luminance"], 3)],
               edge_density=[round(A["edge_density"], 3), round(B["edge_density"], 3)], edge_density_ratio=round(edge_ratio, 2), grid_edge_mean_abs_diff=round(grid_e, 3),
               grid_luminance_mean_abs_diff=round(grid_l, 3), red_centroid=[A["red_centroid"], B["red_centroid"]],
               grid_edges_render=np.round(A["grid_edges"], 2).tolist(), grid_edges_reference=np.round(B["grid_edges"], 2).tolist(),
               grid_luminance_render=np.round(A["grid_luminance"] * 255, 0).tolist(), grid_luminance_reference=np.round(B["grid_luminance"] * 255, 0).tolist())
    if out_png is not None:
        r = Image.open(render).convert("RGB").resize(SIZE, Image.LANCZOS)
        f = Image.open(reference).convert("RGB").resize(SIZE, Image.LANCZOS)
        canvas = Image.new("RGB", (SIZE[0] * 2 + 10, SIZE[1]), (255, 255, 255))
        canvas.paste(f, (0, 0))
        canvas.paste(r, (SIZE[0] + 10, 0))
        canvas.save(out_png)
    return res
