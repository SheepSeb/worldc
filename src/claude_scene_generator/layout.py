"""Static layout of the 40 m x 25 m warehouse (metres, +Z up, origin at floor centre).

        y=+12.5  ┌────────────── north lane (human 1 patrol) ───────────────┐ door N
                 │ rack seg 2 (4.3..9.7)                      PACKING zone  │
                 ├──────────── cross-aisle 2 (1.1..4.3) ─────  belts, rollers│
                 │ rack seg 1 (-4.3..1.1)                     table+drawers │
                 ├──────────── cross-aisle 1 (-7.5..-4.3) ───  cage (gate)  │
                 │ dock staging band                          CHARGING zone │
        y=-12.5  └── bay0 ──── bay1 ──── bay2 (roll-up doors + levelers) ───┘
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geom import rect_overlap_2d

X0, X1, Y0, Y1, H = -20.0, 20.0, -12.5, 12.5, 8.0
W, D = X1 - X0, Y1 - Y0
WALL_T = 0.3
AISLE_W = 3.2
RACK_D = 0.9
BACK_GAP = 0.1
LEVELS = (0.2, 1.8, 3.4, 5.0)
BAY_L = 2.7
SEGS = ((-4.3, 1.1), (4.3, 9.7))  # rack segments along y (2 bays each)
CROSS_AISLES = ((-7.5, -4.3), (1.1, 4.3))
POST = 0.08
BEAM_H, BEAM_W = 0.10, 0.05
UPRIGHT_H = 6.4
FLOOR_Z = 0.0

DOCK_BAYS_X = (-15.0, -9.0, -3.0)
DOCK_OPEN_W, DOCK_OPEN_H = 3.0, 3.2

SLOPE_ZONES = (  # (x0, y0, x1, y1): raised crown, ramps at `slope` gradient, peak <= max_h
    (2.0, -12.1, 10.5, -9.5),
    (13.4, -3.0, 19.4, -0.8),
)

PARKING = ((15.2, -10.4, math.pi / 2), (18.0, -10.4, math.pi / 2))
ROBOT_SPAWN = (12.4, -9.0, math.pi)
CAGE = (14.3, -7.6, 19.5, -3.6)
PACK_ZONE = (13.8, 1.2, 19.7, 12.4)
CHARGE_ZONE = (12.6, -12.4, 19.7, -8.3)


@dataclass
class Row:
    idx: int
    xc: float
    face: int  # +1: pallets are accessed from the +x side (aisle lies to +x)
    aisle: int


@dataclass
class Layout:
    rows: list[Row] = field(default_factory=list)
    aisles: list[tuple[float, float, float]] = field(default_factory=list)  # (x0, x1, xc)

    def __post_init__(self):
        x = -19.4
        self.rows.append(Row(0, x + RACK_D / 2, +1, 0))
        x += RACK_D
        for a in range(6):
            self.aisles.append((x, x + AISLE_W, x + AISLE_W / 2))
            x += AISLE_W
            if a < 5:
                self.rows.append(Row(len(self.rows), x + RACK_D / 2, -1, a))
                x += RACK_D + BACK_GAP
                self.rows.append(Row(len(self.rows), x + RACK_D / 2, +1, a + 1))
                x += RACK_D
            else:
                self.rows.append(Row(len(self.rows), x + RACK_D / 2, -1, a))
                x += RACK_D
        self.rack_x1 = x

    # forklift loop corners (cross-aisle centre lines x outer aisles)
    @property
    def loop(self):
        ya, yb = sum(CROSS_AISLES[0]) / 2, sum(CROSS_AISLES[1]) / 2
        xa, xb = self.aisles[0][2], self.aisles[5][2]
        return [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]

    def human_paths(self):
        a3 = self.aisles[3][2]
        return [
            [(-17.0, 11.2), (8.5, 11.2)],  # north lane patrol
            [(a3, -9.3), (a3, 10.9)],  # through aisle 3
        ]


def slope_params(rng):
    s = float(rng.uniform(0.02, 0.03))
    return s, 0.04


def floor_z(x, y, zones=SLOPE_ZONES, slope=0.025, max_h=0.04):
    """Height of the floor at (x, y): sum over crowns of min(max_h, slope * distance-to-edge)."""
    z = 0.0
    for x0, y0, x1, y1 in zones:
        if x0 <= x <= x1 and y0 <= y <= y1:
            d = min(x - x0, x1 - x, y - y0, y1 - y)
            z = max(z, min(max_h, slope * d))
    return z


class Plan:
    """2-D keep-out registry used for rejection-sampling floor clutter."""

    def __init__(self):
        self.items: list[tuple[float, float, float, float, float, str]] = []

    def add(self, cx, cy, hx, hy, yaw=0.0, tag=""):
        self.items.append((cx, cy, hx, hy, yaw, tag))

    def add_aabb(self, x0, y0, x1, y1, tag=""):
        self.add((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, 0.0, tag)

    def add_path(self, pts, half_w, tag="", closed=False):
        n = len(pts)
        for i in range(n if closed else n - 1):
            (ax, ay), (bx, by) = pts[i], pts[(i + 1) % n]
            L = math.hypot(bx - ax, by - ay)
            self.add((ax + bx) / 2, (ay + by) / 2, L / 2 + half_w, half_w, math.atan2(by - ay, bx - ax), tag)

    def free(self, cx, cy, hx, hy, yaw=0.0, margin=0.1):
        for c2x, c2y, h2x, h2y, y2, _ in self.items:
            if abs(c2x - cx) > hx + hy + h2x + h2y + margin + 0.1:
                continue
            if rect_overlap_2d((cx, cy), (hx, hy), yaw, (c2x, c2y), (h2x, h2y), y2, margin):
                return False
        return True

    def in_bounds(self, cx, cy, hx, hy, yaw, pad=0.25):
        ex = abs(math.cos(yaw)) * hx + abs(math.sin(yaw)) * hy
        ey = abs(math.sin(yaw)) * hx + abs(math.cos(yaw)) * hy
        return X0 + pad <= cx - ex and cx + ex <= X1 - pad and Y0 + pad <= cy - ey and cy + ey <= Y1 - pad
