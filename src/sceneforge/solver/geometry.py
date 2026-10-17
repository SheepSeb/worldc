"""2-D convex-polygon helpers and the 2.5-D occupancy index used by the solver.

Polygons are float arrays (n, 2), convex, CCW.  Every index entry is a convex polygon extruded over [z0, z1]
and owned by an instance key (or a keep-out tag).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def rect_poly(x0, y0, x1, y1) -> np.ndarray:
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)


def transform(poly, x, y, yaw) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    P = np.asarray(poly, float)
    return P @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def bbox(poly) -> np.ndarray:
    P = np.asarray(poly, float)
    return np.concatenate([P.min(0), P.max(0)])


def _axes(P):
    e = np.roll(P, -1, axis=0) - P
    n = np.stack([-e[:, 1], e[:, 0]], 1)
    L = np.linalg.norm(n, axis=1)
    return n[L > 1e-12] / L[L > 1e-12, None]


def overlap(A, B, margin: float = 0.0) -> bool:
    """Separating-axis test for convex polygons; True if they overlap or are closer than ~margin."""
    for ax in np.concatenate([_axes(A), _axes(B)]):
        pa, pb = A @ ax, B @ ax
        if pa.min() - pb.max() > margin or pb.min() - pa.max() > margin:
            return False
    return True


def contains(outer_rect, poly, margin: float = 0.0) -> bool:
    x0, y0, x1, y1 = outer_rect
    P = np.asarray(poly)
    return bool(np.all(P[:, 0] >= x0 + margin) and np.all(P[:, 0] <= x1 - margin) and np.all(P[:, 1] >= y0 + margin) and np.all(P[:, 1] <= y1 - margin))


def principal_angle(poly) -> float:
    """Direction (rad) of the longest edge of a convex polygon (in [0, pi))."""
    P = np.asarray(poly, float)
    e = np.roll(P, -1, axis=0) - P
    i = int(np.argmax(np.linalg.norm(e, axis=1)))
    return math.atan2(e[i, 1], e[i, 0]) % math.pi


def ray_exit(poly, origin, direction) -> float:
    """Distance along ``direction`` from ``origin`` (inside or outside) to the far boundary crossing of a convex polygon;
    0 if the ray misses."""
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    P = np.asarray(poly, float)
    best = 0.0
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        e = b - a
        den = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(den) < 1e-12:
            continue
        w = a - o
        t = (w[0] * (-e[1]) - w[1] * (-e[0])) / den
        u = (d[0] * w[1] - d[1] * w[0]) / den
        if t >= 0 and -1e-9 <= u <= 1 + 1e-9:
            best = max(best, t)
    return best


def scan_x(poly, y):
    """x-interval of a convex polygon at horizontal line y (or None)."""
    P = np.asarray(poly, float)
    xs = []
    for (ax, ay), (bx, by) in zip(P, np.roll(P, -1, axis=0)):
        if ay == by:
            if abs(ay - y) < 1e-9:
                xs += [ax, bx]
        elif min(ay, by) <= y <= max(ay, by):
            xs.append(ax + (bx - ax) * (y - ay) / (by - ay))
    return (min(xs), max(xs)) if xs else None


@dataclass
class Entry:
    poly: np.ndarray
    z0: float
    z1: float
    owner: str  # instance key, or "" for environment / keep-outs
    kind: str  # object | keepout | spawn | goal
    tag: str = ""


@dataclass
class Index:
    entries: list = field(default_factory=list)
    _bb: list = field(default_factory=list)

    def add(self, e: Entry):
        self.entries.append(e)
        self._bb.append(bbox(e.poly))

    def remove_owner(self, owner: str):
        keep = [i for i, e in enumerate(self.entries) if e.owner != owner]
        self.entries = [self.entries[i] for i in keep]
        self._bb = [self._bb[i] for i in keep]

    def near(self, bb, pad: float):
        if not self.entries:
            return []
        B = np.asarray(self._bb)
        m = (B[:, 0] <= bb[2] + pad) & (B[:, 2] >= bb[0] - pad) & (B[:, 1] <= bb[3] + pad) & (B[:, 3] >= bb[1] - pad)
        return [self.entries[i] for i in np.nonzero(m)[0]]

    def collisions(self, pieces, ignore_owners=(), margin=0.0, keepouts=True, z_eps=1e-3):
        """Entries hit by the candidate ``pieces`` [(poly, z0, z1)].  Keep-outs block only when ``keepouts``."""
        hits = []
        for poly, z0, z1 in pieces:
            for e in self.near(bbox(poly), margin + 0.01):
                if e.owner and e.owner in ignore_owners:
                    continue
                if e.kind in ("keepout", "goal", "spawn") and not keepouts:
                    continue
                if e.kind == "object" and (z1 <= e.z0 + z_eps or z0 >= e.z1 - z_eps):
                    continue
                if overlap(poly, e.poly, margin if e.kind == "object" else 0.0):
                    hits.append(e)
        return hits
