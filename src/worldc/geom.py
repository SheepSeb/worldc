"""Pure-numpy geometry helpers: quaternions (w, x, y, z), oriented boxes, rigid-body inertia."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Q_IDENT = (1.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- quaternions
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qaxis(axis, ang):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(ang / 2)
    return (math.cos(ang / 2), ax[0] * s, ax[1] * s, ax[2] * s)


def qyaw(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def qrpy(r, p, y):
    """Intrinsic Z-Y-X (yaw, pitch, roll) -> quaternion."""
    return qmul(qyaw(y), qmul(qaxis((0, 1, 0), p), qaxis((1, 0, 0), r)))


def qmat(q) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def qfrom_mat(R) -> tuple:
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def qrot(q, v):
    return qmat(q) @ np.asarray(v, float)


def compose(p0, q0, p1, q1=Q_IDENT):
    """World pose of a child whose local pose (p1, q1) is expressed in parent frame (p0, q0)."""
    return tuple(np.asarray(p0, float) + qmat(q0) @ np.asarray(p1, float)), qmul(q0, q1)


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# --------------------------------------------------------------------------- oriented boxes
@dataclass
class OBB:
    c: np.ndarray  # centre (3,)
    h: np.ndarray  # half extents (3,)
    R: np.ndarray  # columns are box axes in world (3,3)

    def aabb(self):
        ext = np.abs(self.R) @ self.h
        return self.c - ext, self.c + ext

    def corners(self):
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        return self.c + (s * self.h) @ self.R.T


def obb_from_pose(pos, quat, size) -> OBB:
    return OBB(np.asarray(pos, float), np.asarray(size, float) / 2.0, qmat(quat))


def _face_normals(P: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    try:
        return ConvexHull(P).equations[:, :3]
    except Exception:  # noqa: BLE001  flat / degenerate point set
        return np.zeros((0, 3))


def hull_penetration(P: np.ndarray, Q: np.ndarray, extra_axes=()) -> float:
    """SAT depth of two convex point sets over their face normals (+ ``extra_axes``). >0 overlap, <=0 separated.  Edge-edge
    axes are left out, so a rare edge-edge separation reads as a small overlap (never the other way round)."""
    axes = np.concatenate([_face_normals(P), _face_normals(Q)] + [np.asarray(extra_axes, float).reshape(-1, 3)])
    if not len(axes):
        return -1.0
    pa, qa = P @ axes.T, Q @ axes.T
    pen = np.minimum(pa.max(0) - qa.min(0), qa.max(0) - pa.min(0))
    return -1.0 if (pen <= 0).any() else float(pen.min())


def obb_penetration(a: OBB, b: OBB) -> float:
    """Minimum-translation SAT depth. >0 means overlap (metres); <=0 means separated."""
    d = b.c - a.c
    axes = [a.R[:, i] for i in range(3)] + [b.R[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            cr = np.cross(a.R[:, i], b.R[:, j])
            n = np.linalg.norm(cr)
            if n > 1e-6:
                axes.append(cr / n)
    best = 1e9
    for ax in axes:
        ra = np.abs(a.R.T @ ax) @ a.h
        rb = np.abs(b.R.T @ ax) @ b.h
        pen = ra + rb - abs(d @ ax)
        if pen <= 0:
            return -1.0
        best = min(best, pen)
    return best


def rect_overlap_2d(c1, h1, yaw1, c2, h2, yaw2, margin=0.0) -> bool:
    """2-D oriented-rectangle overlap with extra margin (SAT)."""
    def axes(y):
        return [(math.cos(y), math.sin(y)), (-math.sin(y), math.cos(y))]

    d = (c2[0] - c1[0], c2[1] - c1[1])
    for ax in axes(yaw1) + axes(yaw2):
        def rad(h, y):
            a0, a1 = axes(y)
            return h[0] * abs(ax[0] * a0[0] + ax[1] * a0[1]) + h[1] * abs(ax[0] * a1[0] + ax[1] * a1[1])

        if abs(d[0] * ax[0] + d[1] * ax[1]) > rad(h1, yaw1) + rad(h2, yaw2) + margin:
            return False
    return True


# --------------------------------------------------------------------------- inertia
@dataclass
class Part:
    kind: str  # 'box' (sx,sy,sz) | 'cyl' (radius, height; axis Z) | 'sph' (radius,)
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = Q_IDENT
    mass: float | None = None  # None -> share of total mass proportional to volume

    def volume(self) -> float:
        if self.kind == "box":
            return self.size[0] * self.size[1] * self.size[2]
        if self.kind == "cyl":
            return math.pi * self.size[0] ** 2 * self.size[1]
        return 4.0 / 3.0 * math.pi * self.size[0] ** 3

    def inertia_diag(self, m: float) -> np.ndarray:
        if self.kind == "box":
            sx, sy, sz = self.size
            return m / 12.0 * np.array([sy**2 + sz**2, sx**2 + sz**2, sx**2 + sy**2])
        if self.kind == "cyl":
            r, h = self.size
            return np.array([m * (3 * r * r + h * h) / 12.0, m * (3 * r * r + h * h) / 12.0, m * r * r / 2.0])
        r = self.size[0]
        return np.full(3, 0.4 * m * r * r)


def box_inertia(m, sx, sy, sz):
    return m / 12.0 * np.array([sy**2 + sz**2, sx**2 + sz**2, sx**2 + sy**2])


def composite_inertia(parts: list[Part], total_mass: float):
    """Return (com, diagonal inertia, principal-axes quaternion) of a compound rigid body.

    Parts with ``mass=None`` share ``total_mass`` (minus explicit masses) in proportion to volume.
    Inertia is the full 3x3 tensor about the centre of mass (parallel-axis theorem), then
    diagonalised so ``physics:diagonalInertia`` / ``physics:principalAxes`` are exact.
    """
    explicit = sum(p.mass for p in parts if p.mass is not None)
    vol = sum(p.volume() for p in parts if p.mass is None)
    masses = [p.mass if p.mass is not None else (total_mass - explicit) * p.volume() / vol for p in parts]
    M = sum(masses)
    com = sum(m * np.asarray(p.pos, float) for m, p in zip(masses, parts)) / M
    I = np.zeros((3, 3))
    for m, p in zip(masses, parts):
        R = qmat(p.quat)
        Ip = R @ np.diag(p.inertia_diag(m)) @ R.T
        r = np.asarray(p.pos, float) - com
        I += Ip + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return com, w, qfrom_mat(V), M


# --------------------------------------------------------------------------- scanlines
def footprint(obb: "OBB") -> np.ndarray:
    """Convex footprint polygon (xy) of an oriented box, counter-clockwise."""
    from scipy.spatial import ConvexHull

    pts = obb.corners()[:, :2]
    return pts[ConvexHull(pts).vertices]


def scan_interval(poly: np.ndarray, y: float):
    """x-interval of a convex polygon along the horizontal line at `y` (None if not crossed)."""
    xs = []
    n = len(poly)
    for i in range(n):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        if ay == by:
            if abs(ay - y) < 1e-9:
                xs += [ax, bx]
        elif min(ay, by) <= y <= max(ay, by):
            xs.append(ax + (bx - ax) * (y - ay) / (by - ay))
    return (min(xs), max(xs)) if xs else None


def largest_gap(intervals, lo_bound=None):
    """Largest free gap between consecutive merged intervals (obstacles on both sides)."""
    iv = sorted(intervals)
    best, cur = 0.0, None
    for a, b in iv:
        if cur is not None and a > cur:
            best = max(best, a - cur)
        cur = b if cur is None else max(cur, b)
    return best
