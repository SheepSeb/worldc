"""Planning-time navigation grid: footprints in the robot z band are rasterised, inflated by the robot radius, and
searched with 8-connected Dijkstra.  A soft-cost variant finds which instances block an unreachable goal."""
from __future__ import annotations

import heapq

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

RES = 0.05
Z_BAND = (0.10, 1.80)
NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]


class Grid:
    def __init__(self, x_range, y_range, res=RES):
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.res = res
        self.nx, self.ny = int(round((self.x1 - self.x0) / res)), int(round((self.y1 - self.y0) / res))

    def w2g(self, x, y):
        return int(np.clip((self.y1 - y) / self.res, 0, self.ny - 1)), int(np.clip((x - self.x0) / self.res, 0, self.nx - 1))

    def g2w(self, r, c):
        return self.x0 + (c + 0.5) * self.res, self.y1 - (r + 0.5) * self.res

    def raster(self, polys, owners=None):
        """Occupancy (bool) and, if owners given, an int label map of the owner index per cell (-1 = free)."""
        SS = 3
        img = Image.new("I", (self.nx * SS, self.ny * SS), 0)
        d = ImageDraw.Draw(img)
        for k, P in enumerate(polys):
            pts = [((x - self.x0) / self.res * SS, (self.y1 - y) / self.res * SS) for x, y in P]
            d.polygon(pts, fill=k + 1)
        lab = np.asarray(img, np.int32)[SS // 2 :: SS, SS // 2 :: SS] - 1
        occ = lab >= 0
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
        return occ, lab

    def free(self, occ, radius):
        return distance_transform_edt(~occ) * self.res >= radius + 0.02


def dijkstra(passable_cost, start, goals):
    """passable_cost: float array, inf = blocked.  Returns dist, prev."""
    H, W = passable_cost.shape
    dist = np.full((H, W), np.inf)
    prev = np.full((H, W, 2), -1, np.int32)
    dist[start] = 0.0
    pq = [(0.0, start)]
    remaining = set(goals)
    while pq and remaining:
        dd, (r, c) = heapq.heappop(pq)
        if dd > dist[r, c]:
            continue
        remaining.discard((r, c))
        for dr, dc, w in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                cost = passable_cost[rr, cc]
                if not np.isfinite(cost):
                    continue
                if dr and dc and not (np.isfinite(passable_cost[r, cc]) and np.isfinite(passable_cost[rr, c])):
                    continue
                nd = dd + w * cost
                if nd < dist[rr, cc]:
                    dist[rr, cc] = nd
                    prev[rr, cc] = (r, c)
                    heapq.heappush(pq, (nd, (rr, cc)))
    return dist, prev


def path_to(prev, start, cell):
    out, cur = [], tuple(cell)
    while cur != tuple(start) and cur[0] >= 0:
        out.append(cur)
        cur = tuple(prev[cur])
    return out[::-1]


def nearest_free(free, cell, max_r=10):
    r0, c0 = cell
    if free[r0, c0]:
        return cell
    H, W = free.shape
    for rad in range(1, max_r + 1):
        best = None
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                r, c = r0 + dr, c0 + dc
                if 0 <= r < H and 0 <= c < W and free[r, c]:
                    d = dr * dr + dc * dc
                    if best is None or d < best[0]:
                        best = (d, (r, c))
        if best:
            return best[1]
    return None
