"""Dynamic rigid objects: EUR pallets, cardboard boxes, totes, clutter.

Every dynamic body is an ``Xform`` rigid body whose children are analytic (box / cylinder / cone)
colliders -> convex compound shapes only, no triangle meshes on dynamic bodies.  Mass and the full
inertia tensor (diagonalised, principal axes) are authored explicitly from the compound shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geom import Part, Q_IDENT, composite_inertia, qfrom_mat, qmat, qmul, qyaw, rect_overlap_2d
from .layout import Plan, floor_z
from .usdkit import Kit

# EUR pallet (1200 x 800 x 144 mm): 3 bottom boards, 9 blocks, top deck
PAL_L, PAL_W, PAL_H = 1.2, 0.8, 0.144
BOX_MASS_RANGE = (0.2, 15.0)
JIT_XY, JIT_YAW = 0.03, math.radians(10.0)


def pallet_parts():
    parts = [Part("box", (1.2, 0.8, 0.022), (0, 0, 0.133))]
    for x in (-0.5275, 0.0, 0.5275):
        for y in (-0.3275, 0.0, 0.3275):
            parts.append(Part("box", (0.145, 0.145, 0.1), (x, y, 0.072)))
    for y in (-0.3275, 0.0, 0.3275):
        parts.append(Part("box", (1.2, 0.1, 0.022), (0, y, 0.011)))
    return parts


@dataclass
class PalletInfo:
    path: str
    pos: tuple
    yaw: float
    deck_top: float  # world z of the top deck surface
    load_top: float  # world z of the highest point of the load (== deck_top if empty)
    n_boxes: int = 0
    kind: str = "empty"


class Factory:
    def __init__(self, kit: Kit, rng, mats: dict, pm: dict, plan: Plan, box_classes, n_cardboard_variants):
        self.k, self.rng, self.m, self.pm, self.plan = kit, rng, mats, pm, plan
        self.classes = box_classes
        self.nv = n_cardboard_variants
        for s in ("Pallets", "Boxes", "Totes", "ToteItems", "Clutter"):
            kit.scope(f"/World/{s}")
        self.stats = {"misalign_xy_cm": [], "misalign_yaw_deg": []}

    # ------------------------------------------------------------------ pallets
    def pallet(self, pos, yaw, zone, name=None) -> PalletInfo:
        k, rng = self.k, self.rng
        i = k.next_id("pallet")
        path = f"/World/Pallets/Pallet_{i:04d}"
        q = qyaw(yaw)
        mass = float(rng.uniform(20.0, 25.0))
        tint = rng.uniform(0.85, 1.12)
        mat = self.m["wood"] if tint < 1.0 else self.m["wood_light"]
        parts = pallet_parts()
        body = k.rigid_body(path, pos, q, parts=parts, mass=mass)
        for j, p in enumerate(parts):
            k.cube(f"{path}/c{j:02d}", p.size, p.pos, Q_IDENT, mat, self.pm["wood"], collide=True)
        k.tag(body, "pallet", zone=zone)
        k.record(path, "pallet", pos, q, "dynamic", size=(PAL_L, PAL_W, PAL_H), mass=mass, friction=0.5,
                 zone=zone, standard="EUR1", n_collision_shapes=len(parts))
        return PalletInfo(path, tuple(pos), yaw, pos[2] + PAL_H, pos[2] + PAL_H)

    # ------------------------------------------------------------------ boxes
    def box_mass(self, vol):
        return float(np.clip(vol * self.rng.uniform(60.0, 220.0), *BOX_MASS_RANGE))

    def box(self, pos, quat, size, variant, zone, parent=None, layer=None, extra=None):
        """One cardboard carton: rigid body + box collider + textured visual mesh."""
        k = self.k
        i = k.next_id("box")
        path = f"/World/Boxes/Box_{i:05d}"
        mass = self.box_mass(size[0] * size[1] * size[2])
        part = Part("box", tuple(size))
        body = k.rigid_body(path, pos, quat, parts=[part], mass=mass)
        k.cube(f"{path}/Collision", size, (0, 0, 0), Q_IDENT, None, self.pm["cardboard"], collide=True, guide=True)
        k.mesh_box(f"{path}/Visual", size, vmat=self.m[f"cardboard_{variant % self.nv}"])
        k.tag(body, "box", zone=zone)
        k.record(path, "box", pos, quat, "dynamic", size=size, mass=mass, friction=0.5, zone=zone,
                 parent=parent, layer=layer, **(extra or {}))
        return path, mass

    def pick_dims(self, cls=None, lo=0.15):
        c = self.classes[cls if cls is not None else int(self.rng.integers(len(self.classes)))]
        s = float(self.rng.uniform(0.94, 1.06))
        return tuple(float(np.clip(d * s, lo, 0.6)) for d in c)

    def load_boxes(self, pal: PalletInfo, zone, dims, max_h, full=False, max_boxes=18, min_layers=1):
        """Stack cardboard layers on a pallet with random <=3 cm / <=10 deg misalignment per box."""
        rng = self.rng
        a, b, c = dims
        if rng.random() < 0.5:
            a, b = b, a
        gap = 0.03
        nx = max(1, int((PAL_L + 0.04) // (a + gap)))
        ny = max(1, int((PAL_W + 0.04) // (b + gap)))
        while nx * ny > max_boxes and (nx > 1 or ny > 1):
            if nx >= ny and nx > 1:
                nx -= 1
            else:
                ny -= 1
        per_layer = nx * ny
        cap = int(np.clip(int(max_h // (c + 0.002)), 1, 6))
        cap = max(1, min(cap, max(1, max_boxes // per_layer)))
        layers = cap if full else int(rng.integers(min_layers if min_layers <= cap else 1, cap + 1))
        # centre-to-centre pitch never smaller than footprint + gap
        pitch_x = max(PAL_L / nx, a + gap)
        pitch_y = max(PAL_W / ny, b + gap)
        px = (np.arange(nx) - (nx - 1) / 2) * pitch_x
        py = (np.arange(ny) - (ny - 1) / 2) * pitch_y
        variant = int(rng.integers(self.nv))
        q0 = qyaw(pal.yaw)
        R0 = qmat(q0)
        n = 0
        top = pal.deck_top
        dropped: set = set()  # cells that could not host a box: nothing may be stacked above them
        for layer in range(layers):
            placed: list[tuple] = []
            cells = [(x, y) for x in px for y in py if (x, y) not in dropped]
            if layer == layers - 1 and layers > 1 and not full:
                keep = max(1, int(round(len(cells) * rng.uniform(0.3, 1.0))))
                cells = [cells[i] for i in rng.permutation(len(cells))[:keep]]
            zc = pal.deck_top + layer * c + c / 2 + 0.0008 * (layer + 1)
            for (cx, cy) in cells:
                best = None
                for attempt in range(12):
                    s = 1.0 if attempt < 6 else (11 - attempt) / 6.0  # shrink jitter, last attempt = none
                    rad, ang = JIT_XY * math.sqrt(rng.random()) * s, rng.uniform(0, math.tau)
                    dx, dy = rad * math.cos(ang), rad * math.sin(ang)  # |offset| <= 3 cm
                    th = rng.uniform(-JIT_YAW, JIT_YAW) * s
                    x, y = cx + dx, cy + dy
                    ex = abs(math.cos(th)) * a / 2 + abs(math.sin(th)) * b / 2
                    ey = abs(math.sin(th)) * a / 2 + abs(math.cos(th)) * b / 2
                    if abs(x) + ex > PAL_L / 2 + 0.03 or abs(y) + ey > PAL_W / 2 + 0.03:
                        continue
                    if any(rect_overlap_2d((x, y), (a / 2, b / 2), th, p[:2], (a / 2, b / 2), p[2], 0.003) for p in placed):
                        continue
                    best = (x, y, th)
                    break
                if best is None:
                    dropped.add((cx, cy))  # cell cannot host a box without overlap -> leave it empty
                    continue
                placed.append(best)
                x, y, th = best
                wp = np.asarray(pal.pos) + R0 @ np.array([x, y, 0.0])
                wp[2] = zc
                self.box(tuple(wp), qmul(q0, qyaw(th)), (a, b, c), variant, zone, parent=pal.path, layer=layer,
                         extra={"misalign_xy_cm": round(100 * math.hypot(x - cx, y - cy), 2),
                                "misalign_yaw_deg": round(math.degrees(th), 2)})
                self.stats["misalign_xy_cm"].append(100 * math.hypot(x - cx, y - cy))
                self.stats["misalign_yaw_deg"].append(abs(math.degrees(th)))
                n += 1
            top = pal.deck_top + (layer + 1) * c + 0.0008 * (layer + 1)
        pal.load_top, pal.n_boxes, pal.kind = top, n, "boxes"
        return pal

    def wrapped_load(self, pos, yaw, zone, dims, max_h, layers=None):
        """Stretch-wrapped pallet = ONE rigid body: pallet shapes + consolidated load block."""
        k, rng = self.k, self.rng
        a, b, c = dims
        n_layers = layers or int(np.clip(int(max_h // c), 1, 5))
        h = n_layers * c
        L, Wd = PAL_L - 0.02, PAL_W - 0.02
        i = k.next_id("pallet")
        path = f"/World/Pallets/Pallet_{i:04d}"
        q = qyaw(yaw)
        parts = pallet_parts()
        nbox = max(1, int(round(L / (a + 0.01))) * int(round(Wd / (b + 0.01))) * n_layers)
        load_mass = sum(self.box_mass(a * b * c) for _ in range(min(nbox, 24))) * nbox / min(nbox, 24)
        pallet_mass = float(rng.uniform(20.0, 25.0))
        load_part = Part("box", (L, Wd, h), (0, 0, PAL_H + h / 2 + 0.0008), Q_IDENT, mass=load_mass)
        total = pallet_mass + load_mass
        for p in parts:
            p.mass = None
        # pallet parts share pallet_mass by volume; the load block carries load_mass
        vol = sum(p.volume() for p in parts)
        for p in parts:
            p.mass = pallet_mass * p.volume() / vol
        parts.append(load_part)
        body = k.rigid_body(path, pos, q, parts=parts, mass=total)
        for j, p in enumerate(parts[:-1]):
            k.cube(f"{path}/c{j:02d}", p.size, p.pos, Q_IDENT, self.m["wood"], self.pm["wood"], collide=True)
        lp = load_part.pos
        k.cube(f"{path}/Load", load_part.size, lp, Q_IDENT, None, self.pm["cardboard"], collide=True, guide=True)
        wrap_sz = (L + 0.016, Wd + 0.016, h + 0.012)
        k.mesh_box(f"{path}/Wrap", wrap_sz, (lp[0], lp[1], lp[2] + 0.004), vmat=self.m["wrap"], uv=False)
        k.tag(body, "pallet_wrapped", zone=zone)
        k.record(path, "pallet", pos, q, "dynamic", size=(PAL_L, PAL_W, PAL_H + h), mass=total, friction=0.5,
                 zone=zone, standard="EUR1", wrapped=True, equivalent_boxes=nbox, n_collision_shapes=len(parts))
        info = PalletInfo(path, tuple(pos), yaw, pos[2] + PAL_H, pos[2] + PAL_H + h + 0.0008, nbox, "wrapped")
        return info

    # ------------------------------------------------------------------ totes
    TOTE = (0.6, 0.4, 0.28)
    TWALL = 0.012

    def tote(self, pos, yaw, zone, n_items=None, parent=None):
        k, rng = self.k, self.rng
        L, Wd, Hh = self.TOTE
        t = self.TWALL
        i = k.next_id("tote")
        path = f"/World/Totes/Tote_{i:03d}"
        q = qyaw(yaw)
        mass = float(rng.uniform(1.4, 1.9))
        col = ("tote_blue", "tote_grey", "tote_red")[i % 3]
        parts = [
            Part("box", (L, Wd, t), (0, 0, t / 2)),
            Part("box", (L, t, Hh - t), (0, (Wd - t) / 2, t + (Hh - t) / 2)),
            Part("box", (L, t, Hh - t), (0, -(Wd - t) / 2, t + (Hh - t) / 2)),
            Part("box", (t, Wd - 2 * t, Hh - t), ((L - t) / 2, 0, t + (Hh - t) / 2)),
            Part("box", (t, Wd - 2 * t, Hh - t), (-(L - t) / 2, 0, t + (Hh - t) / 2)),
        ]
        body = k.rigid_body(path, pos, q, parts=parts, mass=mass)
        names = ("Base", "WallN", "WallS", "WallE", "WallW")
        for nm, p in zip(names, parts):
            k.cube(f"{path}/{nm}", p.size, p.pos, Q_IDENT, self.m[col], self.pm["plastic"], collide=True)
        k.tag(body, "tote", zone=zone)
        n_items = int(n_items if n_items is not None else rng.integers(3, 11))
        k.record(path, "tote", pos, q, "dynamic", size=self.TOTE, mass=mass, friction=0.4, zone=zone,
                 parent=parent, open_top=True, n_items=n_items)
        # contents: rigid cubes / cylinders on a jittered grid on the tote floor
        cols, rows_ = 4, 3
        inner_x, inner_y = L - 2 * t - 0.02, Wd - 2 * t - 0.02
        cells = [(ix, iy) for ix in range(cols) for iy in range(rows_)]
        order = rng.permutation(len(cells))[:n_items]
        R0 = qmat(q)
        for n_i, ci in enumerate(order):
            ix, iy = cells[ci]
            cw, ch = inner_x / cols, inner_y / rows_
            lx = -inner_x / 2 + (ix + 0.5) * cw
            ly = -inner_y / 2 + (iy + 0.5) * ch
            is_cyl = rng.random() < 0.35
            if is_cyl:
                r = float(rng.uniform(0.02, 0.032))
                hh = float(rng.uniform(0.05, 0.12))
                size = (r, hh)
                vol = math.pi * r * r * hh
            else:
                lim = (min(cw, ch) - 0.026) / math.sqrt(2)  # rotated footprint stays inside its own cell
                sx = float(rng.uniform(0.04, lim))
                sy = float(rng.uniform(0.04, lim))
                sz = float(rng.uniform(0.04, 0.11))
                size = (sx, sy, sz)
                vol = sx * sy * sz
            m_i = float(np.clip(vol * rng.uniform(250, 700), 0.02, 0.9))
            jx = float(rng.uniform(-0.006, 0.006))
            jy = float(rng.uniform(-0.006, 0.006))
            hz = size[1] if is_cyl else size[2]
            lz = t + hz / 2 + 0.0008
            wp = np.asarray(pos) + R0 @ np.array([lx + jx, ly + jy, lz])
            self.item(wp, qmul(q, qyaw(float(rng.uniform(0, math.tau)))), is_cyl, size, m_i, path, i, n_i, zone)
        return path

    def item(self, pos, quat, is_cyl, size, mass, tote_path, tote_i, n, zone):
        k, rng = self.k, self.rng
        path = f"/World/ToteItems/Item_{tote_i:03d}_{n:02d}"
        part = Part("cyl", size) if is_cyl else Part("box", size)
        body = k.rigid_body(path, pos, quat, parts=[part], mass=mass, iters=(8, 1))
        mat = self.m[("item_a", "item_b", "item_c", "item_d")[int(rng.integers(4))]]
        if is_cyl:
            k.cylinder(f"{path}/Shape", size[0], size[1], (0, 0, 0), Q_IDENT, mat, self.pm["plastic"], collide=True)
        else:
            k.cube(f"{path}/Shape", size, (0, 0, 0), Q_IDENT, mat, self.pm["plastic"], collide=True)
        k.tag(body, "tote_item", zone=zone)
        k.record(path, "tote_item", pos, quat, "dynamic", size=size, mass=mass, friction=0.4, zone=zone,
                 parent=tote_path, shape="cylinder" if is_cyl else "box")

    def pallet_of_totes(self, pos, yaw, zone, n_items_each=None):
        """Pallet carrying a single layer of 4 totes (2 x 2)."""
        pal = self.pallet(pos, yaw, zone)
        q = qyaw(yaw)
        R0 = qmat(q)
        L, Wd, Hh = self.TOTE
        # totes 0.6 x 0.4: 2 along pallet length, 2 along width
        for ix in (-1, 1):
            for iy in (-1, 1):
                lp = np.array([ix * (L / 2 + 0.004), iy * (Wd / 2 + 0.004), 0.0])
                wp = np.asarray(pos) + R0 @ lp
                wp[2] = pal.deck_top + 0.0008
                self.tote(tuple(wp), yaw, zone, n_items_each, parent=pal.path)
        pal.load_top = pal.deck_top + Hh
        pal.kind = "totes"
        pal.n_boxes = 4
        return pal

    # ------------------------------------------------------------------ clutter
    def loose_box(self, x, y, yaw, zone, dims=None, z0=None, stack_on=None):
        dims = dims or tuple(float(self.rng.uniform(0.15, 0.6)) for _ in range(3))
        z = (z0 if z0 is not None else floor_z(x, y)) + dims[2] / 2 + 0.001
        return self.box((x, y, z), qyaw(yaw), dims, int(self.rng.integers(self.nv)), zone, parent=None,
                        extra={"loose": True}), dims

    def cone(self, x, y, zone):
        k, rng = self.k, self.rng
        i = k.next_id("cone")
        path = f"/World/Clutter/Cone_{i:02d}"
        h, r, base = 0.7, 0.14, 0.38
        mass = 1.4
        parts = [Part("box", (base, base, 0.03), (0, 0, 0.015)), Part("cyl", (r * 0.55, h), (0, 0, 0.03 + h / 2))]
        q = qyaw(float(rng.uniform(0, math.tau)))
        pos = (x, y, floor_z(x, y) + 0.0008)
        body = k.rigid_body(path, pos, q, parts=parts, mass=mass)
        k.cube(f"{path}/Base", (base, base, 0.03), (0, 0, 0.015), Q_IDENT, self.m["cone_base"], self.pm["rubber"], collide=True)
        cone = k.cone(f"{path}/Cone", r, h, (0, 0, 0.03 + h / 2), Q_IDENT, self.m["cone"])
        from pxr import UsdPhysics

        UsdPhysics.CollisionAPI.Apply(cone)
        k.bind(cone, self.pm["rubber"], "physics")
        k.tag(body, "cone", zone=zone)
        k.record(path, "clutter", pos, q, "dynamic", size=(base, base, h + 0.03), mass=mass, friction=0.9, zone=zone,
                 subtype="traffic_cone")
        self.plan.add(x, y, base / 2, base / 2, 0.0, "cone")

    def trash_bin(self, x, y, zone):
        k = self.k
        path = "/World/Clutter/TrashBin"
        r, h, mass = 0.28, 0.8, 6.5
        parts = [Part("cyl", (r, h), (0, 0, h / 2))]
        pos = (x, y, floor_z(x, y) + 0.0008)
        body = k.rigid_body(path, pos, Q_IDENT, parts=parts, mass=mass)
        k.cylinder(f"{path}/Shape", r, h, (0, 0, h / 2), Q_IDENT, self.m["bin_green"], self.pm["plastic"], collide=True)
        k.cylinder(f"{path}/LidDisc", r * 0.92, 0.01, (0, 0, h + 0.002), Q_IDENT, self.m["bin_dark"])
        k.tag(body, "trash_bin", zone=zone)
        k.record(path, "clutter", pos, Q_IDENT, "dynamic", size=(2 * r, 2 * r, h), mass=mass, friction=0.4, zone=zone,
                 subtype="trash_bin")
        self.plan.add(x, y, r + 0.05, r + 0.05, 0.0, "bin")

    def pallet_jack_tipped(self, x, y, yaw, zone):
        """Manual pallet jack lying on its side: orientation chosen by support-polygon analysis."""
        k = self.k
        path = "/World/Clutter/PalletJack_tipped"
        mass = 72.0
        fork = (1.15, 0.16, 0.06)
        parts = [
            Part("box", fork, (0.575, 0.26, 0.03)),
            Part("box", fork, (0.575, -0.26, 0.03)),
            Part("box", (0.22, 0.68, 0.08), (0.05, 0, 0.1)),  # pump housing / yoke
            Part("box", (0.14, 0.10, 0.9), (0.0, 0, 0.55)),  # handle column
            Part("box", (0.08, 0.38, 0.05), (-0.1, 0, 1.05)),  # handle bar
        ]
        com, diag, axes, _ = composite_inertia(parts, mass)
        R_rest, zmin, margin = stable_orientation(parts, com)
        q_rest = qfrom_mat(R_rest)
        q = qmul(qyaw(yaw), q_rest)
        pos = (x, y, floor_z(x, y) - zmin + 0.0008)
        body = k.rigid_body(path, pos, q, parts=parts, mass=mass)
        for j, p in enumerate(parts):
            k.cube(f"{path}/p{j}", p.size, p.pos, Q_IDENT, self.m["jack"], self.pm["steel"], collide=True)
        k.tag(body, "pallet_jack", zone=zone)
        k.record(path, "clutter", pos, q, "dynamic", size=(1.2, 1.1, 0.9), mass=mass, friction=0.6, zone=zone,
                 subtype="pallet_jack_tipped", rest_stability_margin_m=round(float(margin), 3))
        self.plan.add(x, y, 0.8, 0.8, yaw, "jack")


def _axis_rotations():
    """The 24 proper rotations of the cube group (axis permutations with signs)."""
    import itertools

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def stable_orientation(parts, com):
    """Pick the axis-aligned orientation whose support polygon contains the COM with max margin.

    Returns (R, zmin_of_rotated_body, margin_m).  The body is rotated by R and dropped so its lowest
    face touches the floor, i.e. the pose is a true static equilibrium (no toppling during settle).
    """
    from scipy.spatial import ConvexHull

    best = None
    for R in _axis_rotations():
        pts = []
        for p in parts:
            c = R @ np.asarray(p.pos)
            h = np.abs(R) @ (np.asarray(p.size) / 2)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(c + np.array([sx, sy, sz]) * h)
        pts = np.array(pts)
        zmin = pts[:, 2].min()
        contact = pts[pts[:, 2] < zmin + 0.004][:, :2]
        if len(contact) < 3:
            continue
        try:
            hull = ConvexHull(contact)
        except Exception:
            continue
        cxy = (R @ com)[:2]
        eq = hull.equations  # n.x + d <= 0 inside
        margin = float(-(eq[:, :2] @ cxy + eq[:, 2]).max())
        score = margin - 0.1 * ((R @ com)[2] - zmin)
        if margin > 0.02 and (best is None or score > best[0]):
            best = (score, R, zmin, margin)
    if best is None:
        raise RuntimeError("no stable resting orientation found")
    return best[1], best[2], best[3]
