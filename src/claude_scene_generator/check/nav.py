"""Occupancy grid, robot-radius inflation, Dijkstra reachability and blocker diagnosis, with bounds from the scene graph."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from ..colliders import Collider
from ..geom import footprint
from ..static_validate import dijkstra  # noqa: F401  (re-exported)

INFLATE_MARGIN = 0.02
SS = 5  # supersample factor; a cell is occupied iff its centre is inside a footprint


class Grid:
    def __init__(self, x_range, y_range, res):
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.res = res
        self.nx = max(3, int(round((self.x1 - self.x0) / res)))
        self.ny = max(3, int(round((self.y1 - self.y0) / res)))
        self.occ: np.ndarray | None = None
        self.free: np.ndarray | None = None

    def cell(self, x, y):
        return int((self.y1 - y) / self.res), int((x - self.x0) / self.res)

    def inside(self, x, y):
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def clamp(self, rc):
        return min(max(rc[0], 0), self.ny - 1), min(max(rc[1], 0), self.nx - 1)

    def is_free(self, x, y):
        return bool(self.inside(x, y) and self.free[self.clamp(self.cell(x, y))])


def nav_colliders(cols: list[Collider], z_band, ignore) -> list[int]:
    out = []
    for i, c in enumerate(cols):
        if c.is_mesh or c.category in ignore:
            continue
        lo, hi = c.obb.aabb()
        if hi[2] < z_band[0] or lo[2] > z_band[1]:
            continue
        out.append(i)
    return out


def polygons(cols, idx):
    polys = {}
    for i in idx:
        try:
            polys[i] = footprint(cols[i].obb)
        except Exception:  # noqa: BLE001  degenerate (flat) hull
            continue
    return polys


def build_grid(grid: Grid, polys: dict, robot_r: float) -> Grid:
    img = Image.new("L", (grid.nx * SS, grid.ny * SS), 0)
    d = ImageDraw.Draw(img)
    k = SS / grid.res
    for poly in polys.values():
        d.polygon([((x - grid.x0) * k, (grid.y1 - y) * k) for x, y in poly], fill=1)
    occ = np.asarray(img, bool)[SS // 2 :: SS, SS // 2 :: SS].copy()
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    grid.occ = occ
    grid.free = distance_transform_edt(~occ) * grid.res >= robot_r + INFLATE_MARGIN
    return grid


class Blockers:
    """Nearest-footprint queries over the nav colliders."""

    def __init__(self, cols, polys):
        self.cols, self.ids = cols, list(polys)
        self.polys = [polys[i] for i in self.ids]
        self.lo = np.array([p.min(0) for p in self.polys]).reshape(-1, 2)
        self.hi = np.array([p.max(0) for p in self.polys]).reshape(-1, 2)

    @staticmethod
    def _dist(poly, pt):
        # distance from a point to a convex CCW polygon
        n = len(poly)
        inside = True
        best = 1e18
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            e = b - a
            L2 = float(e @ e)
            t = 0.0 if L2 == 0 else min(1.0, max(0.0, float((pt - a) @ e) / L2))
            best = min(best, float(np.linalg.norm(pt - (a + t * e))))
            if e[0] * (pt[1] - a[1]) - e[1] * (pt[0] - a[0]) < 0:
                inside = False
        return 0.0 if inside else best

    def nearest(self, x, y, k=3, limit=30):
        """[(collider index, distance)] of the k nearest footprints."""
        if not self.ids:
            return []
        pt = np.array([x, y])
        gap = np.maximum(0, np.maximum(self.lo - pt, pt - self.hi))
        lb = np.hypot(gap[:, 0], gap[:, 1])
        cand = np.argsort(lb)[:limit]
        res = sorted((self._dist(self.polys[j], pt), self.ids[j]) for j in cand)
        return [(i, d) for d, i in res[:k]]


def trace_path(prev, start, cell):
    p, cur = [], cell
    while cur != start:
        p.append(cur)
        cur = tuple(prev[cur])
    return p


def render_png(path, grid: Grid, start, goals_xy, paths, spawn_ok=True):
    occ, free = grid.occ, grid.free
    rgb = np.zeros(occ.shape + (3,), np.uint8)
    rgb[...] = (235, 235, 235)
    rgb[~free] = (200, 160, 160)
    rgb[occ] = (40, 40, 40)
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    for p in paths.values():
        dr.line([(c, r) for r, c in p[::3]], fill=(30, 120, 220), width=1)
    for x, y in goals_xy:
        r, c = grid.clamp(grid.cell(x, y))
        dr.ellipse([c - 5, r - 5, c + 5, r + 5], outline=(0, 160, 60), width=2)
    r, c = start
    dr.ellipse([c - 6, r - 6, c + 6, r + 6], fill=(220, 0, 0) if spawn_ok else (255, 0, 255))
    im.save(path)
