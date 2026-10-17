"""5 cm occupancy grid over the deck, robot-radius inflation, reachability and goal approach-pose selection."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import ConvexHull

from ..static_validate import dijkstra

RES = 0.05
X0, X1, Y0, Y1 = -3.0, 3.0, -2.0, 2.0
ROBOT_R = 0.35
Z_BAND = (0.06, 1.85)  # heights that block a ground robot (above the deck)
NX, NY = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
SKIP_CAT = ("deck_floor", "ground", "rack_structure")


def w2g(x, y):
    return int((Y1 - y) / RES), int((x - X0) / RES)


def g2w(r, c):
    return X0 + (c + 0.5) * RES, Y1 - (r + 0.5) * RES


def build_occ(cols, extra_rects=(), band=Z_BAND):
    SS = 5
    img = Image.new("L", (NX * SS, NY * SS), 0)
    d = ImageDraw.Draw(img)

    def px(x, y):
        return ((x - X0) / RES * SS, (Y1 - y) / RES * SS)

    for c in cols:
        if c.is_mesh or c.category in SKIP_CAT:
            continue
        lo, hi = c.obb.aabb()
        if hi[2] < band[0] or lo[2] > band[1]:
            continue
        pts = c.obb.corners()[:, :2]
        try:
            poly = pts[ConvexHull(pts).vertices]
        except Exception:
            continue
        d.polygon([px(x, y) for x, y in poly], fill=1)
    for (x0, y0, x1, y1) in extra_rects:
        d.rectangle([px(x0, y1), px(x1, y0)], fill=1)
    occ = np.asarray(img, bool)[SS // 2 :: SS, SS // 2 :: SS]
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return occ


def free_mask(occ, radius=ROBOT_R):
    return distance_transform_edt(~occ) * RES >= radius


def pick_spawn(free, pref=(-0.8, -1.4)):
    """Free cell nearest ``pref`` inside the largest connected free region (so the spawn is never in a walled-off pocket)."""
    lab, cnt = label(free)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes))
    rr, cc = np.nonzero(lab == main)
    xs = X0 + (cc + 0.5) * RES
    ys = Y1 - (rr + 0.5) * RES
    k = int(np.argmin(np.hypot(xs - pref[0], ys - pref[1])))
    return (int(rr[k]), int(cc[k])), lab == main


def line_clear(occ, p, q, ignore_r):
    """No occupied cell on the segment p->q except within ignore_r of q (the target)."""
    n = int(np.hypot(q[0] - p[0], q[1] - p[1]) / (RES / 2)) + 1
    for k in range(n + 1):
        t = k / n
        x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
        if math.hypot(x - q[0], y - q[1]) < ignore_r:
            break
        r, c = w2g(x, y)
        if 0 <= r < NY and 0 <= c < NX and occ[r, c]:
            return False
    return True


def approach_pose(free, comp, occ, target_xy, normal_xy=None, max_dev_deg=75.0, r_range=(0.55, 1.05), r_pref=0.75, ignore_r=0.5):
    """Best standing cell (x, y, yaw) for a manipulation target: free, in the spawn component, facing the target, line of sight."""
    best = None
    tx, ty = target_xy
    nrm = None if normal_xy is None else np.asarray(normal_xy, float) / np.linalg.norm(normal_xy)
    for r in np.arange(r_range[0], r_range[1] + 1e-6, 0.05):
        for a in np.arange(0, 360, 6.0):
            ang = math.radians(a)
            x, y = tx + r * math.cos(ang), ty + r * math.sin(ang)
            gr, gc = w2g(x, y)
            if not (0 <= gr < NY and 0 <= gc < NX) or not (free[gr, gc] and comp[gr, gc]):
                continue
            dev = 0.0
            if nrm is not None:
                dev = math.degrees(math.acos(np.clip(nrm @ np.array([math.cos(ang), math.sin(ang)]), -1, 1)))
                if dev > max_dev_deg:
                    continue
            if not line_clear(occ, (x, y), (tx, ty), ignore_r):
                continue
            score = abs(r - r_pref) + 0.004 * dev
            if best is None or score < best[0]:
                best = (score, x, y, math.atan2(ty - y, tx - x))
    if best is None:
        return None
    cx, cy = g2w(*w2g(best[1], best[2]))  # snap to the centre of the grid cell that was tested
    return (round(cx, 4), round(cy, 4), round(math.atan2(ty - cy, tx - cx), 4))


def reach_all(free, start, goals_xy):
    cells = [w2g(x, y) for x, y in goals_xy]
    dist, prev = dijkstra(free, start, [c for c in cells if free[c]])
    return [(bool(free[c]) and dist[c] < 1e17, None if dist[c] >= 1e17 else float(dist[c] * RES)) for c in cells], dist, prev


def render_png(path, occ, free, spawn_cell, goals, paths=None):
    rgb = np.zeros(occ.shape + (3,), np.uint8)
    rgb[...] = (235, 235, 235)
    rgb[~free] = (200, 160, 160)
    rgb[occ] = (40, 40, 40)
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    for p in (paths or {}).values():
        dr.line([(c, r) for r, c in p[::3]], fill=(30, 120, 220), width=1)
    for g in goals:
        r, c = w2g(*g[:2])
        dr.ellipse([c - 4, r - 4, c + 4, r + 4], outline=(0, 160, 60), width=2)
    r, c = spawn_cell
    dr.ellipse([c - 6, r - 6, c + 6, r + 6], fill=(220, 0, 0))
    im.save(path)
