"""Static / articulated structures: building shell, racks, dock doors, swing doors, cage, conveyors, table."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

from . import layout as LY
from .geom import Part, Q_IDENT, compose, qaxis, qmul, qyaw
from .usdkit import Kit, T


@dataclass
class Ctx:
    kit: Kit
    rng: np.random.Generator
    plan: LY.Plan
    m: dict
    pm: dict
    L: LY.Layout
    slope: float
    max_h: float


class Frame:
    """Local assembly frame (origin on the wall's interior line, local +y pointing into the building)."""

    def __init__(self, xy, yaw, z=0.0):
        self.p, self.q = (xy[0], xy[1], z), qyaw(yaw)
        self.yaw = yaw

    def world(self, local_p, local_q=Q_IDENT):
        return compose(self.p, self.q, local_p, local_q)


class Structures:
    def __init__(self, c: Ctx):
        self.c = c
        self.k = c.kit
        self.articulations: list[dict] = []
        self.conveyors: list[dict] = []
        self.rack_info: list[dict] = []

    # ------------------------------------------------------------------ shell
    def shell(self):
        c, k = self.c, self.k
        k.scope("/World/Building")
        T_ = LY.WALL_T
        # collider slab (top at z=0) covering building + dock apron; render geometry is separate
        slab = k.cube("/World/Building/FloorSlab", (LY.W + 2 * T_, LY.D + 2 * T_ + 6.5, 0.5),
                      (0, (-19.0 + 12.8) / 2, -0.25), Q_IDENT, None, c.pm["concrete"], collide=True, guide=True)
        k.tag(slab, "floor")
        k.record("/World/Building/FloorSlab", "floor", (0, -3.1, -0.25), Q_IDENT, "static",
                 size=(LY.W + 2 * T_, LY.D + 2 * T_ + 6.5, 0.5), friction=0.7)
        # floor render quad (textured) + apron
        u0, v0 = 0.0, 0.0
        pts = [(LY.X0, LY.Y0, 0), (LY.X1, LY.Y0, 0), (LY.X1, LY.Y1, 0), (LY.X0, LY.Y1, 0)]
        k.mesh("/World/Building/FloorVisual", pts, [4], [0, 1, 2, 3], uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
               vmat=c.m["floor"])
        ap = [(-20.3, -19.0, 0), (20.3, -19.0, 0), (20.3, LY.Y0 - T_, 0), (-20.3, LY.Y0 - T_, 0)]
        k.mesh("/World/Building/ApronVisual", ap, [4], [0, 1, 2, 3], uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
               vmat=c.m["apron"])
        # slope zones: crown meshes (triangle-mesh colliders, static only)
        for i, (x0, y0, x1, y1) in enumerate(LY.SLOPE_ZONES):
            self._crown(i, x0, y0, x1, y1)
        # walls
        bays = [(x, LY.DOCK_OPEN_W, LY.DOCK_OPEN_H) for x in LY.DOCK_BAYS_X]
        self.wall("South", (LY.X0, LY.Y0), (LY.X1, LY.Y0), (0, 1), [(x - LY.X0, w, h) for x, w, h in bays])
        self.wall("North", (LY.X1, LY.Y1), (LY.X0, LY.Y1), (0, -1), [(LY.X1 - 16.4, 1.15, 2.24)])
        self.wall("East", (LY.X1, LY.Y0), (LY.X1, LY.Y1), (-1, 0), [])
        self.wall("West", (LY.X0, LY.Y1), (LY.X0, LY.Y0), (1, 0), [(LY.Y1 - 11.0, 1.15, 2.24)])
        # roof / ceiling slab (hidden for the overhead camera render) + visual-only trusses
        k.scope("/World/Building/Roof")
        roof = k.cube("/World/Building/Roof/Slab", (LY.W + 2 * T_, LY.D + 2 * T_, 0.3), (0, 0, LY.H + 0.15), Q_IDENT,
                      c.m["roof"], c.pm["steel"], collide=True)
        k.tag(roof, "roof")
        k.record("/World/Building/Roof/Slab", "building", (0, 0, LY.H + 0.15), Q_IDENT, "static",
                 size=(LY.W + 2 * T_, LY.D + 2 * T_, 0.3), friction=0.6, subtype="roof")
        for i, x in enumerate(np.arange(-17.5, 18, 5.0)):
            k.cube(f"/World/Building/Roof/Truss_{i}", (0.25, LY.D, 0.45), (x, 0, LY.H - 0.225), Q_IDENT, c.m["truss"])

    def _crown(self, i, x0, y0, x1, y1):
        c, k = self.c, self.k
        s, mh = c.slope, c.max_h
        r = mh / s
        def lines(a, b):
            r2 = min(r, (b - a) / 2)
            return sorted({a, a + r2, b - r2, b})
        xs, ys = lines(x0, x1), lines(y0, y1)
        H = lambda x, y: min(mh, s * min(x - x0, x1 - x, y - y0, y1 - y))
        pts, uvs = [], []
        for y in ys:
            for x in xs:
                pts.append((x, y, H(x, y) + 0.0005))
                uvs.append(((x - LY.X0) / LY.W, (y - LY.Y0) / LY.D))
        nx = len(xs)
        counts, idx = [], []
        for j in range(len(ys) - 1):
            for i2 in range(nx - 1):
                a, b, c2, d = j * nx + i2, j * nx + i2 + 1, (j + 1) * nx + i2 + 1, (j + 1) * nx + i2
                counts += [3, 3]
                idx += [a, b, c2, a, c2, d]
        path = f"/World/Building/Crown_{i}"
        prim = k.mesh(path, pts, counts, idx, uvs=uvs, vmat=c.m["floor"])
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")  # static triangle mesh
        k.bind(prim, c.pm["concrete"], "physics")
        peak = max(p[2] for p in pts)
        k.record(path, "floor", (x0, y0, 0), Q_IDENT, "static", size=(x1 - x0, y1 - y0, peak), friction=0.7,
                 subtype="slope_zone", gradient=round(s, 4), peak_height_m=round(peak, 4), bbox=[x0, y0, x1, y1])
        c.plan.add_aabb(x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + 0.3, "slope")

    def wall(self, name, p0, p1, n_in, openings):
        """Wall along p0->p1 (the interior line); thickness extends outward. openings=(s, width, height)."""
        c, k = self.c, self.k
        T_ = LY.WALL_T
        t = np.array([p1[0] - p0[0], p1[1] - p0[1]], float)
        L = float(np.linalg.norm(t))
        t /= L
        yaw = math.atan2(t[1], t[0])
        n_in = np.array(n_in, float)
        base = np.array(p0, float) - n_in * (T_ / 2)
        cuts = sorted(openings)
        cur = -T_
        pieces = []  # (s0, s1, z0, z1)
        for s, w, h in cuts:
            pieces.append((cur, s - w / 2, 0.0, LY.H))
            pieces.append((s - w / 2, s + w / 2, h, LY.H))
            cur = s + w / 2
        pieces.append((cur, L + T_, 0.0, LY.H))
        k.scope("/World/Building/Walls")
        for j, (s0, s1, z0, z1) in enumerate(pieces):
            if s1 - s0 < 1e-3:
                continue
            ctr = base + t * ((s0 + s1) / 2)
            pos = (ctr[0], ctr[1], (z0 + z1) / 2)
            size = (s1 - s0, T_, z1 - z0)
            path = f"/World/Building/Walls/{name}_{j}"
            prim = k.cube(path, size, pos, qyaw(yaw), c.m["wall"], c.pm["concrete"], collide=True)
            k.tag(prim, "wall")
            k.record(path, "building", pos, qyaw(yaw), "static", size=size, friction=0.7, subtype="wall")

    # ------------------------------------------------------------------ racks
    def racks(self):
        c, k = self.c, self.k
        k.scope("/World/Racks")
        slots = []
        pal_c = c.m
        for row in c.L.rows:
            rp = f"/World/Racks/Row_{row.idx:02d}"
            k.xform(rp)
            tone = float(c.rng.uniform(0.9, 1.08))
            parts = []
            up_m = c.m[f"rack_up_{row.idx % 3}"]
            bm_m = c.m[f"rack_beam_{row.idx % 3}"]
            for s, (ya, yb) in enumerate(LY.SEGS):
                frames = [ya + i * LY.BAY_L for i in range(3)]
                for fi, yf in enumerate(frames):
                    for side, sx in (("F", row.face), ("B", -row.face)):
                        pos = (row.xc + sx * 0.40, yf, LY.UPRIGHT_H / 2)
                        size = (LY.POST, LY.POST, LY.UPRIGHT_H)
                        name = f"Upright_s{s}_f{fi}_{side}"
                        prim = k.cube(f"{rp}/{name}", size, pos, Q_IDENT, up_m, c.pm["steel"], collide=True)
                        parts.append({"type": "upright", "name": name, "position": [round(v, 3) for v in pos], "size": list(size)})
                for b in range(2):
                    for li, z in enumerate(LY.LEVELS):
                        for side, sx in (("F", row.face), ("B", -row.face)):
                            pos = (row.xc + sx * 0.335, frames[b] + LY.BAY_L / 2, z - LY.BEAM_H / 2)
                            size = (LY.BEAM_W, LY.BAY_L - LY.POST, LY.BEAM_H)
                            name = f"Beam_s{s}_b{b}_L{li}_{side}"
                            k.cube(f"{rp}/{name}", size, pos, Q_IDENT, bm_m, c.pm["steel"], collide=True)
                            parts.append({"type": "beam", "name": name, "position": [round(v, 3) for v in pos], "size": list(size), "level": li, "top_z": z})
                        for p in range(2):
                            yc = frames[b] + 0.713 + p * 1.273
                            slots.append({"row": row.idx, "seg": s, "bay": b, "slot": p, "level": li, "x": row.xc, "y": yc,
                                          "z": z, "face": row.face, "aisle": row.aisle})
                # column protectors at both segment ends (yellow, static)
                for end_y, d in ((ya - 0.2, -1), (yb + 0.2, 1)):
                    pos = (row.xc, end_y, 0.25)
                    size = (0.95, 0.12, 0.5)
                    name = f"Guard_s{s}_{'S' if d < 0 else 'N'}"
                    k.cube(f"{rp}/{name}", size, pos, Q_IDENT, c.m["safety_yellow"], c.pm["steel"], collide=True)
                    parts.append({"type": "guard", "name": name, "position": [round(v, 3) for v in pos], "size": list(size)})
            k.tag(k.stage.GetPrimAtPath(rp), "rack_row", aisle=row.aisle)
            k.record(rp, "rack_row", (row.xc, 0, 0), Q_IDENT, "static", friction=0.6, zone="racking",
                     face=row.face, faces_aisle=row.aisle, levels=list(LY.LEVELS), n_uprights=sum(1 for p in parts if p["type"] == "upright"),
                     n_beams=sum(1 for p in parts if p["type"] == "beam"), parts=parts)
            for ya, yb in LY.SEGS:
                c.plan.add_aabb(row.xc - 0.5, ya - 0.35, row.xc + 0.5, yb + 0.35, "rack")
        return slots

    # ------------------------------------------------------------------ articulations
    def _art(self, **kw):
        self.articulations.append(kw)

    def rollup_door(self, i, x):
        c, k = self.c, self.k
        base = f"/World/Doors/RollUp_{i}"
        fr = Frame((x, LY.Y0), 0.0)
        k.xform(base, fr.p, fr.q)
        W_, H_ = LY.DOCK_OPEN_W, LY.DOCK_OPEN_H
        steel, pm = c.m["door_grey"], c.pm["steel"]
        for side in (-1, 1):  # guide rails
            k.cube(f"{base}/Guide_{'L' if side < 0 else 'R'}", (0.08, 0.12, 6.6), (side * (W_ / 2 + 0.2), 0.06, 3.3), Q_IDENT, steel, pm, collide=True)
        k.cube(f"{base}/Hood", (W_ + 0.7, 0.4, 0.5), (0, 0.33, 3.55), Q_IDENT, steel, pm, collide=True)
        k.cube(f"{base}/Sill", (W_ + 0.4, 0.15, 0.04), (0, 0.075, 0.02), Q_IDENT, c.m["safety_yellow"], pm, collide=True)
        # panel
        pw, ph, pt = W_ + 0.2, LY.DOCK_OPEN_H + 0.05, 0.06
        center = (0.0, 0.04, ph / 2 + 0.045)
        mass = 85.0
        parts = [Part("box", (pw, pt, ph), (0, 0, 0)), Part("box", (pw, 0.1, 0.1), (0, 0.02, -ph / 2 + 0.05), mass=8.0)]
        wp, wq = fr.world(center)
        body = f"{base}/Panel"
        b = k.rigid_body(body, center, Q_IDENT, parts=parts, mass=mass, disable_gravity=True, iters=(16, 4))
        k.cube(f"{body}/Slats", (pw, pt, ph), (0, 0, 0), Q_IDENT, c.m["door_slats"], c.pm["steel"], collide=True)
        k.cube(f"{body}/BottomBar", (pw, 0.1, 0.1), (0, 0.02, -ph / 2 + 0.05), Q_IDENT, c.m["safety_yellow"], c.pm["steel"], collide=True)
        bp = f"{base}/ArtBase"
        k.fixed_base(bp, fr.p, fr.q)
        limit = LY.DOCK_OPEN_H + 0.05
        jp = f"{base}/Joint"
        k.joint(jp, "prismatic", bp, body, center, (0, 0, 0), axis="Z", lower=0.0, upper=limit,
                stiffness=4e4, damping=1.2e4, max_force=5e4)
        k.tag(b, "door_rollup", bay=i)
        k.record(body, "door_rollup", wp, wq, "dynamic", size=(pw, pt, ph), mass=mass, friction=0.6, zone="dock",
                 articulation={"type": "prismatic", "root": body, "joint": jp, "axis_local": "Z", "unit": "m",
                               "limits": [0.0, limit], "closed_position": 0.0, "open_position": limit,
                               "drive": {"stiffness": 4e4, "damping": 1.2e4, "max_force": 5e4}, "fixed_base": True, "articulation_root": bp})
        self._art(name=f"RollUp_{i}", root=bp, link=body, joint=jp, kind="prismatic", lower=0.0, upper=limit, unit="m",
                  parent=base, moving_collider_paths=[f"{body}/Slats", f"{body}/BottomBar"])
        # dock leveler plate (static, 1.5 deg slope, inner edge flush with floor) + bumpers
        ang = math.radians(1.5)
        L_, Wd, th = 2.2, 2.0, 0.04
        lp = (x, -11.2, 0.0)
        z_edge = math.sin(ang) * Wd / 2
        pos = (x, -11.2, z_edge - th / 2 * math.cos(ang))
        q = qaxis((1, 0, 0), -ang)
        k.cube(f"/World/Dock/Leveler_{i}", (L_, Wd, th), pos, q, c.m["leveler"], c.pm["steel"], collide=True)
        k.record(f"/World/Dock/Leveler_{i}", "dock_leveler", pos, q, "static", size=(L_, Wd, th), friction=0.6, zone="dock",
                 tilt_deg=1.5)
        for sx in (-1, 1):
            k.cube(f"/World/Dock/Bumper_{i}_{'L' if sx < 0 else 'R'}", (0.25, 0.18, 0.4), (x + sx * 1.35, LY.Y0 - LY.WALL_T - 0.09, 1.2),
                   Q_IDENT, c.m["rubber"], c.pm["rubber"], collide=True)
        c.plan.add_aabb(x - 1.8, LY.Y0, x + 1.8, LY.Y0 + 2.9, "dock_lane")

    def swing_door(self, i, origin, yaw):
        c, k = self.c, self.k
        base = f"/World/Doors/Swing_{i}"
        fr = Frame(origin, yaw)
        k.xform(base, fr.p, fr.q)
        pm, jam = c.pm["steel"], c.m["door_frame"]
        T_ = LY.WALL_T
        for sx in (-1, 1):
            k.cube(f"{base}/Jamb_{'L' if sx < 0 else 'R'}", (0.1, T_, 2.24), (sx * 0.525, -T_ / 2, 1.12), Q_IDENT, jam, pm, collide=True)
        k.cube(f"{base}/Header", (1.15, T_, 0.1), (0, -T_ / 2, 2.19), Q_IDENT, jam, pm, collide=True)
        lw, lh, lt = 0.9, 2.1, 0.045
        hx = -0.47
        center = (hx + lw / 2, -lt / 2, lh / 2 + 0.005)
        mass = 32.0
        parts = [Part("box", (lw, lt, lh)), Part("box", (0.12, 0.03, 0.03), (0.33, lt / 2 + 0.015, -0.05), mass=0.4)]
        body = f"{base}/Leaf"
        b = k.rigid_body(body, center, Q_IDENT, parts=parts, mass=mass, iters=(16, 4))
        k.cube(f"{body}/Slab", (lw, lt, lh), (0, 0, 0), Q_IDENT, c.m["door_leaf"], c.pm["steel"], collide=True)
        k.cube(f"{body}/Handle", (0.12, 0.03, 0.03), (0.33, lt / 2 + 0.015, -0.05), Q_IDENT, c.m["steel_bright"], c.pm["steel"], collide=True)
        bp = f"{base}/ArtBase"
        k.fixed_base(bp, fr.p, fr.q)
        hinge0 = (hx, 0.0, center[2])
        hinge1 = (-lw / 2, lt / 2, 0.0)
        jp = f"{base}/Joint"
        up = 100.0
        k.joint(jp, "revolute", bp, body, hinge0, hinge1, axis="Z", lower=0.0, upper=up, stiffness=60.0, damping=14.0, max_force=1e5)
        wp, wq = fr.world(center)
        k.tag(b, "door_swing", index=i)
        k.record(body, "door_swing", wp, wq, "dynamic", size=(lw, lt, lh), mass=mass, friction=0.6,
                 articulation={"type": "revolute", "root": body, "joint": jp, "axis_local": "Z", "unit": "deg", "limits": [0.0, up],
                               "closed_position": 0.0, "open_position": up,
                               "drive": {"stiffness": 60.0, "damping": 14.0, "max_force": 1e5}, "fixed_base": True, "articulation_root": bp})
        self._art(name=f"Swing_{i}", root=bp, link=body, joint=jp, kind="revolute", lower=0.0, upper=up, unit="deg", parent=base,
                  moving_collider_paths=[f"{body}/Slab", f"{body}/Handle"])
        # swing-arc keep-out (radius = leaf width) on the interior side
        (cx, cy, _), _ = fr.world((hx + 0.45, 0.55, 0))
        c.plan.add(cx, cy, 1.05, 1.05, 0.0, "door_arc")

    def cage(self, gate_x=16.9):
        c, k = self.c, self.k
        x0, y0, x1, y1 = LY.CAGE
        base = "/World/Cage"
        k.xform(base)
        pm, mat, post_m = c.pm["steel"], c.m["cage_mesh"], c.m["cage_post"]
        gw = 1.5
        sides = [
            ("S", (x0, y0), (x1, y0), True),
            ("N", (x0, y1), (x1, y1), False),
            ("W", (x0, y0), (x0, y1), False),
            ("E", (x1, y0), (x1, y1), False),
        ]
        h, z0 = 2.0, 0.1
        for name, a, b, has_gate in sides:
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            t = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
            yaw = math.atan2(t[1], t[0])
            segs = [(0.0, L)]
            if has_gate:
                gs = gate_x - gw / 2 - a[0]
                segs = [(0.0, gs), (gs + gw, L)]
            for j, (s0, s1) in enumerate(segs):
                ctr = (a[0] + t[0] * (s0 + s1) / 2, a[1] + t[1] * (s0 + s1) / 2)
                k.cube(f"{base}/Panel_{name}{j}", (s1 - s0, 0.02, h), (ctr[0], ctr[1], z0 + h / 2), qyaw(yaw), mat, pm, collide=True)
                n_posts = max(1, int(math.ceil((s1 - s0) / 1.4)))
                for pi in range(n_posts + 1):
                    s = s0 + (s1 - s0) * pi / n_posts
                    k.cube(f"{base}/Post_{name}{j}_{pi}", (0.06, 0.06, 2.2), (a[0] + t[0] * s, a[1] + t[1] * s, 1.1), Q_IDENT, post_m, pm, collide=True)
            k.record(f"{base}/Panel_{name}", "cage_fence", ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 1.1), qyaw(yaw), "static", size=(L, 0.02, 2.2), friction=0.6, zone="cage")
        c.plan.add_aabb(x0 - 0.2, y0 - 0.2, x1 + 0.2, y1 + 0.2, "cage_fence")
        # sliding gate: leaf slides +x along the outside face of the south fence
        leaf = (gw + 0.1, 0.03, h)
        outside_y = y0 - 0.055
        center = (gate_x, outside_y, z0 + h / 2)
        mass = 28.0
        body = f"{base}/Gate"
        b = k.rigid_body(body, center, Q_IDENT, parts=[Part("box", leaf)], mass=mass, iters=(16, 4), disable_gravity=True)
        k.cube(f"{body}/Leaf", leaf, (0, 0, 0), Q_IDENT, c.m["cage_gate"], c.pm["steel"], collide=True)
        bp = f"{base}/GateBase"
        k.fixed_base(bp, (0, 0, 0), Q_IDENT)
        jp = f"{base}/GateJoint"
        up = 1.7
        k.joint(jp, "prismatic", bp, body, center, (0, 0, 0), axis="X", lower=0.0, upper=up, stiffness=2e4, damping=4e3, max_force=1e5)
        k.tag(b, "gate_sliding")
        k.record(body, "gate_sliding", center, Q_IDENT, "dynamic", size=leaf, mass=mass, friction=0.6, zone="cage",
                 articulation={"type": "prismatic", "root": body, "joint": jp, "axis_local": "X", "unit": "m", "limits": [0.0, up],
                               "closed_position": 0.0, "open_position": up,
                               "drive": {"stiffness": 2e4, "damping": 4e3, "max_force": 1e5}, "fixed_base": True, "articulation_root": bp})
        self._art(name="CageGate", root=bp, link=body, joint=jp, kind="prismatic", lower=0.0, upper=up, unit="m", parent=base,
                  moving_collider_paths=[f"{body}/Leaf"])
        c.plan.add_aabb(gate_x - 0.9, y0 - 1.0, gate_x + 2.6, y0 - 0.05, "gate_slide")

    # ------------------------------------------------------------------ conveyors
    def belt(self, name, cx, y0, y1, speed=0.5, width=0.6, top_z=0.8):
        c, k = self.c, self.k
        L = y1 - y0
        base = f"/World/Conveyors/{name}"
        k.xform(base)
        pm_steel = c.pm["steel"]
        cy = (y0 + y1) / 2
        bp = f"{base}/Belt"
        belt_t = 0.04
        body = k.rigid_body(bp, (cx, cy, top_z - belt_t / 2), Q_IDENT, parts=[Part("box", (width, L, belt_t))], mass=40.0, kinematic=True)
        k.cube(f"{bp}/Surface", (width, L, belt_t), (0, 0, 0), Q_IDENT, c.m["belt"], c.pm["rubber"], collide=True)
        k.physx(body, "PhysxSurfaceVelocityAPI", **{
            "physxSurfaceVelocity:surfaceVelocityEnabled": (T.Bool, True),
            "physxSurfaceVelocity:surfaceVelocity": (T.Vector3f, Gf.Vec3f(0.0, float(speed), 0.0)),
            "physxSurfaceVelocity:surfaceVelocityLocalSpace": (T.Bool, True),
        })
        for sx in (-1, 1):
            k.cube(f"{base}/Rail_{'L' if sx < 0 else 'R'}", (0.03, L, 0.12), (cx + sx * (width / 2 + 0.015), cy, top_z + 0.02), Q_IDENT, c.m["conv_frame"], pm_steel, collide=True)
        k.cube(f"{base}/EndStop", (width + 0.06, 0.03, 0.16), (cx, y1 - 0.015, top_z + 0.08), Q_IDENT, c.m["safety_yellow"], pm_steel, collide=True)
        k.cube(f"{base}/Frame", (width + 0.04, L, 0.06), (cx, cy, top_z - belt_t - 0.03), Q_IDENT, c.m["conv_frame"], pm_steel, collide=True)
        for i, (lx, ly) in enumerate([(-1, 0.05), (1, 0.05), (-1, 0.95), (1, 0.95)]):
            k.cube(f"{base}/Leg_{i}", (0.05, 0.05, top_z - belt_t - 0.06), (cx + lx * (width / 2 - 0.02), y0 + ly * L if ly > 0.5 else y0 + 0.2, (top_z - belt_t - 0.06) / 2), Q_IDENT, c.m["conv_frame"], pm_steel, collide=True)
        k.tag(body, "conveyor_belt", speed=float(speed))
        info = {"name": name, "kind": "belt", "belt_path": bp, "speed_mps": speed, "direction_world": [0, 1, 0],
                "bbox": [cx - width / 2, y0, cx + width / 2, y1], "top_z": top_z}
        self.conveyors.append(info)
        k.record(bp, "conveyor_belt", (cx, cy, top_z - belt_t / 2), Q_IDENT, "kinematic", size=(width, L, belt_t), mass=40.0, friction=0.9,
                 zone="packing", surface_velocity_mps=[0.0, speed, 0.0])
        c.plan.add_aabb(cx - width / 2 - 0.1, y0 - 0.1, cx + width / 2 + 0.1, y1 + 0.1, "belt")
        return info

    def roller_conveyor(self, name, cx, y0, y1, speed=0.5, width=0.6, top_z=0.8):
        c, k = self.c, self.k
        L = y1 - y0
        base = f"/World/Conveyors/{name}"
        k.xform(base)
        cy = (y0 + y1) / 2
        rp = f"{base}/Rollers"
        n = int(L / 0.1)
        r = 0.03
        parts, offs = [], []
        for i in range(n):
            ly = -L / 2 + 0.05 + i * (L - 0.1) / (n - 1)
            parts.append(Part("cyl", (r, width - 0.02), (0, ly, 0), qaxis((0, 1, 0), math.pi / 2)))
            offs.append(ly)
        body = k.rigid_body(rp, (cx, cy, top_z - r), Q_IDENT, parts=parts, mass=60.0, kinematic=True)
        for i, ly in enumerate(offs):
            k.cylinder(f"{rp}/Roller_{i:02d}", r, width - 0.02, (0, ly, 0), qaxis((0, 1, 0), math.pi / 2), c.m["roller"], c.pm["steel"], collide=True)
        k.physx(body, "PhysxSurfaceVelocityAPI", **{
            "physxSurfaceVelocity:surfaceVelocityEnabled": (T.Bool, True),
            "physxSurfaceVelocity:surfaceVelocity": (T.Vector3f, Gf.Vec3f(0.0, float(speed), 0.0)),
            "physxSurfaceVelocity:surfaceVelocityLocalSpace": (T.Bool, True),
        })
        for sx in (-1, 1):
            k.cube(f"{base}/Rail_{'L' if sx < 0 else 'R'}", (0.04, L, 0.1), (cx + sx * (width / 2 + 0.02), cy, top_z - 0.02), Q_IDENT, c.m["conv_frame"], c.pm["steel"], collide=True)
        k.cube(f"{base}/EndStop", (width + 0.08, 0.03, 0.16), (cx, y1 - 0.015, top_z + 0.08), Q_IDENT, c.m["safety_yellow"], c.pm["steel"], collide=True)
        for i, (lx, ly) in enumerate([(-1, 0.2), (1, 0.2), (-1, L - 0.2), (1, L - 0.2)]):
            k.cube(f"{base}/Leg_{i}", (0.05, 0.05, top_z - 0.08), (cx + lx * (width / 2), y0 + ly, (top_z - 0.08) / 2), Q_IDENT, c.m["conv_frame"], c.pm["steel"], collide=True)
        k.tag(body, "conveyor_roller", speed=float(speed))
        self.conveyors.append({"name": name, "kind": "roller", "belt_path": rp, "speed_mps": speed, "direction_world": [0, 1, 0],
                               "bbox": [cx - width / 2, y0, cx + width / 2, y1], "top_z": top_z})
        k.record(rp, "conveyor_roller", (cx, cy, top_z - r), Q_IDENT, "kinematic", size=(width, L, 2 * r), mass=60.0, friction=0.6,
                 zone="packing", surface_velocity_mps=[0.0, speed, 0.0], n_rollers=n)
        c.plan.add_aabb(cx - width / 2 - 0.1, y0 - 0.1, cx + width / 2 + 0.1, y1 + 0.1, "roller")

    # ------------------------------------------------------------------ work table
    def work_table(self, cx, cy):
        c, k = self.c, self.k
        base = "/World/WorkTable"
        k.xform(base, (cx, cy, 0))
        pm = c.pm["steel"]
        top_z0, top_z1 = 0.81, 0.85
        k.cube(f"{base}/Top", (1.6, 0.8, 0.04), (0, 0, 0.83), Q_IDENT, c.m["table_top"], c.pm["wood"], collide=True)
        for i, (lx, ly) in enumerate([(-0.76, -0.36), (0.76, -0.36), (-0.76, 0.36), (0.76, 0.36)]):
            k.cube(f"{base}/Leg_{i}", (0.06, 0.06, 0.81), (lx, ly, 0.405), Q_IDENT, c.m["conv_frame"], pm, collide=True)
        for nm, x in (("PanelL", -0.54), ("PanelM", 0.0), ("PanelR", 0.54)):
            k.cube(f"{base}/{nm}", (0.02, 0.76, 0.26), (x, 0, 0.68), Q_IDENT, c.m["conv_frame"], pm, collide=True)
        k.cube(f"{base}/Back", (1.1, 0.02, 0.26), (0, -0.38, 0.68), Q_IDENT, c.m["conv_frame"], pm, collide=True)
        k.cube(f"{base}/Shelf", (1.1, 0.76, 0.02), (0, 0, 0.54), Q_IDENT, c.m["conv_frame"], pm, collide=True)
        k.record(base, "work_table", (cx, cy, 0), Q_IDENT, "static", size=(1.6, 0.8, 0.85), friction=0.5, zone="packing",
                 top_z=top_z1)
        c.plan.add_aabb(cx - 0.9, cy - 0.5, cx + 0.9, cy + 1.1, "table")
        # drawers (prismatic, pull towards +y)
        for di, dx in enumerate((-0.27, 0.27)):
            dp = f"{base}/Drawer_{di}"
            dw, dl, dh = 0.5, 0.62, 0.12
            center = (dx, -0.05, 0.58 + dh / 2 + 0.02)  # local to table
            mass = 3.0
            t = 0.015
            parts = [
                Part("box", (dw, dl, t), (0, 0, -dh / 2 + t / 2)),
                Part("box", (t, dl, dh), (-dw / 2 + t / 2, 0, 0)),
                Part("box", (t, dl, dh), (dw / 2 - t / 2, 0, 0)),
                Part("box", (dw - 2 * t, t, dh), (0, -dl / 2 + t / 2, 0)),
                Part("box", (dw, 0.02, dh + 0.04), (0, dl / 2 + 0.01, 0.02)),
            ]
            b = k.rigid_body(dp, center, Q_IDENT, parts=parts, mass=mass, iters=(16, 4), disable_gravity=False)
            for nm, p in zip(("Bottom", "WallL", "WallR", "Back", "Front"), parts):
                k.cube(f"{dp}/{nm}", p.size, p.pos, Q_IDENT, c.m["drawer"], c.pm["wood"], collide=True)
            k.cube(f"{dp}/Handle", (0.18, 0.03, 0.025), (0, dl / 2 + 0.035, 0.03), Q_IDENT, c.m["steel_bright"], pm, collide=True)
            bp = f"{base}/DrawerBase_{di}"
            k.fixed_base(bp, (cx, cy, 0.0), Q_IDENT)
            jp = f"{dp}/Joint"
            up = 0.35
            k.joint(jp, "prismatic", bp, dp, center, (0, 0, 0), axis="Y", lower=0.0, upper=up, stiffness=3000.0, damping=600.0, max_force=1e4)
            wp = (cx + center[0], cy + center[1], center[2])
            k.tag(b, "drawer", index=di)
            k.record(dp, "drawer", wp, Q_IDENT, "dynamic", size=(dw, dl + 0.04, dh), mass=mass, friction=0.5, zone="packing",
                     articulation={"type": "prismatic", "root": dp, "joint": jp, "axis_local": "Y", "unit": "m", "limits": [0.0, up],
                                   "closed_position": 0.0, "open_position": up,
                                   "drive": {"stiffness": 3000.0, "damping": 600.0, "max_force": 1e4}, "fixed_base": True, "articulation_root": bp})
            self._art(name=f"Drawer_{di}", root=bp, link=dp, joint=jp, kind="prismatic", lower=0.0, upper=up, unit="m", parent=base,
                      moving_collider_paths=[f"{dp}/{n}" for n in ("Bottom", "WallL", "WallR", "Back", "Front", "Handle")])
        return top_z1
