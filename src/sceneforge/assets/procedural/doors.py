"""Articulated doors and enclosures: roll-up dock door (prismatic) with leveler + bumpers, hinged personnel door
(revolute), wire-mesh cage with a sliding gate (prismatic).  Ported from ``Structures.rollup_door`` / ``swing_door`` /
``cage`` with identical joints and drives.

Door frames: origin on the wall's interior line at the opening centre, local +y pointing into the building (the
``building_shell`` opening ports use the same convention, so ``attach_to: shell.opening_<id>`` places them).
"""
from __future__ import annotations

import math

from pydantic import Field

from ...geom import Part, Q_IDENT, qaxis, qyaw
from ..base import Anchor, Asset, Built, Footprint, JointInfo, Keepout, Params, PhysicsInfo
from ..registry import register


def _rect(cx, cy, sx, sy):
    return [(cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy - sy / 2), (cx + sx / 2, cy + sy / 2), (cx - sx / 2, cy + sy / 2)]


class RollupParams(Params):
    width: float = Field(3.0, ge=1.0, le=8.0, description="Opening width (m)")
    height: float = Field(3.2, ge=1.5, le=8.0, description="Opening height (m)")
    wall_thickness: float = Field(0.3, ge=0.05, le=1.0)
    leveler: bool = Field(True, description="Static dock-leveler plate (1.5 deg) inside + rubber bumpers outside")


@register
class RollupDoor(Asset):
    id = "rollup_door"
    description = ("Roll-up (sectional) dock door: guide rails, hood, sill and a slatted panel on a driven prismatic joint "
                   "(0 = closed, height = open). Optional dock leveler and bumpers. Attach to a wall opening port.")
    tags = frozenset({"door", "dock", "articulated", "prismatic", "warehouse"})
    Params = RollupParams
    physics = PhysicsInfo("articulated", "steel", 0.6, 85.0, "primitives", "fixed-base articulation, panel 85 kg, gravity disabled")

    def footprint(self, p):
        door = Footprint.rect(p.width + 0.7, 0.55 + p.wall_thickness + 0.2, 6.6, 0.0, (0.55 - p.wall_thickness - 0.2) / 2)
        if not p.leveler:
            return door
        return Footprint.union([door, Footprint.rect(2.25, 2.05, 0.06, 0.0, 1.3)])

    def joints(self, p):
        lim = p.height + 0.05
        return [JointInfo("Joint", "prismatic", "Z", 0.0, lim, "m", 0.0, {"stiffness": 4e4, "damping": 1.2e4, "max_force": 5e4}, "Panel", "door")]

    def keepouts(self, p):
        return [Keepout(_rect(0.0, 1.45, p.width + 0.6, 2.9), "dock_lane")]

    def anchors(self, p):
        return [Anchor("inside", "port", (0.0, 1.5, 0.0), qyaw(-math.pi / 2), port_kind="approach", meta={"note": "robot approach point in front of the door"})]

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        k.xform(path, pose.pos, pose.quat)
        W_, H_ = p.width, p.height
        steel, pm = m.get("door_grey"), m.phys("steel")
        for side in (-1, 1):
            k.cube(f"{path}/Guide_{'L' if side < 0 else 'R'}", (0.08, 0.12, 6.6), (side * (W_ / 2 + 0.2), 0.06, 3.3), Q_IDENT, steel, pm, collide=True)
        k.cube(f"{path}/Hood", (W_ + 0.7, 0.4, 0.5), (0, 0.33, H_ + 0.35), Q_IDENT, steel, pm, collide=True)
        k.cube(f"{path}/Sill", (W_ + 0.4, 0.15, 0.04), (0, 0.075, 0.02), Q_IDENT, m.get("safety_yellow"), pm, collide=True)
        pw, ph, pt = W_ + 0.2, H_ + 0.05, 0.06
        center = (0.0, 0.04, ph / 2 + 0.045)
        mass = 85.0
        parts = [Part("box", (pw, pt, ph), (0, 0, 0)), Part("box", (pw, 0.1, 0.1), (0, 0.02, -ph / 2 + 0.05), mass=8.0)]
        body = f"{path}/Panel"
        b = k.rigid_body(body, center, Q_IDENT, parts=parts, mass=mass, disable_gravity=True, iters=(16, 4))
        k.cube(f"{body}/Slats", (pw, pt, ph), (0, 0, 0), Q_IDENT, m.get("door_slats"), pm, collide=True)
        k.cube(f"{body}/BottomBar", (pw, 0.1, 0.1), (0, 0.02, -ph / 2 + 0.05), Q_IDENT, m.get("safety_yellow"), pm, collide=True)
        bp = f"{path}/ArtBase"
        k.fixed_base(bp, pose.pos, pose.quat)
        limit = H_ + 0.05
        jp = f"{path}/Joint"
        k.joint(jp, "prismatic", bp, body, center, (0, 0, 0), axis="Z", lower=0.0, upper=limit, stiffness=4e4, damping=1.2e4, max_force=5e4)
        wp, wq = pose.world(center)
        k.tag(b, "door_rollup")
        ctx.record(body, "door_rollup", wp, wq, "dynamic", size=(pw, pt, ph), mass=mass, friction=0.6,
                   articulation={"type": "prismatic", "root": body, "joint": jp, "axis_local": "Z", "unit": "m",
                                 "limits": [0.0, limit], "closed_position": 0.0, "open_position": limit,
                                 "drive": {"stiffness": 4e4, "damping": 1.2e4, "max_force": 5e4}, "fixed_base": True, "articulation_root": bp})
        ctx.scene["articulations"].append(dict(name=path.rsplit("/", 1)[-1], root=bp, link=body, joint=jp, kind="prismatic", lower=0.0,
                                               upper=limit, unit="m", parent=path, moving_collider_paths=[f"{body}/Slats", f"{body}/BottomBar"]))
        if p.leveler:
            ang = math.radians(1.5)
            L_, Wd, th = 2.2, 2.0, 0.04
            z_edge = math.sin(ang) * Wd / 2
            lpos = (0.0, 1.3, z_edge - th / 2 * math.cos(ang))
            q = qaxis((1, 0, 0), -ang)
            k.cube(f"{path}/Leveler", (L_, Wd, th), lpos, q, m.get("leveler"), pm, collide=True)
            lw, lq = pose.world(lpos, q)
            ctx.record(f"{path}/Leveler", "dock_leveler", lw, lq, "static", size=(L_, Wd, th), friction=0.6, tilt_deg=1.5)
            for sx in (-1, 1):
                k.cube(f"{path}/Bumper_{'L' if sx < 0 else 'R'}", (0.25, 0.18, 0.4), (sx * (W_ / 2 - 0.15), -p.wall_thickness - 0.09, 1.2),
                       Q_IDENT, m.get("rubber"), m.phys("rubber"), collide=True)
        return Built(path, [a.to_world(pose) for a in self.anchors(p)], [("dock_lane", pose.world_xy(self.keepouts(p)[0].polygon).tolist())],
                     info={"joint": jp, "panel": body, "articulation_root": bp}, bodies=[body, bp])


class SwingParams(Params):
    leaf_width: float = Field(0.9, ge=0.6, le=1.4)
    leaf_height: float = Field(2.1, ge=1.8, le=3.0)
    open_angle: float = Field(100.0, ge=30, le=180, description="Upper joint limit (deg)")
    wall_thickness: float = Field(0.3, ge=0.05, le=1.0)


@register
class SwingDoor(Asset):
    id = "swing_door"
    description = ("Hinged personnel door: jambs + header and a 32 kg leaf on a revolute joint with a weak position drive "
                   "(0 = closed). Swings into the room; swing arc is a keep-out. Attach to a wall opening port.")
    tags = frozenset({"door", "articulated", "revolute", "indoor"})
    Params = SwingParams
    physics = PhysicsInfo("articulated", "steel", 0.6, 32.0, "primitives")

    def footprint(self, p):
        return Footprint.rect(p.leaf_width + 0.25, p.wall_thickness + 0.1, p.leaf_height + 0.14, 0.0, -p.wall_thickness / 2)

    def joints(self, p):
        return [JointInfo("Joint", "revolute", "Z", 0.0, p.open_angle, "deg", 0.0, {"stiffness": 60.0, "damping": 14.0, "max_force": 1e5}, "Leaf", "door")]

    def keepouts(self, p):
        r = p.leaf_width + 0.15
        return [Keepout(_rect(-0.02, 0.55, 2 * r, 2 * r), "door_arc")]

    def anchors(self, p):
        return [Anchor("handle", "port", (0.33 - 0.47 + p.leaf_width / 2, 0.06, 1.0), port_kind="handle"),
                Anchor("inside", "port", (0.0, 1.2, 0.0), qyaw(-math.pi / 2), port_kind="approach")]

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        k.xform(path, pose.pos, pose.quat)
        pm, jam = m.phys("steel"), m.get("door_frame")
        T_ = p.wall_thickness
        ow = p.leaf_width + 0.25
        for sx in (-1, 1):
            k.cube(f"{path}/Jamb_{'L' if sx < 0 else 'R'}", (0.1, T_, p.leaf_height + 0.14), (sx * (ow / 2 - 0.05), -T_ / 2, (p.leaf_height + 0.14) / 2), Q_IDENT, jam, pm, collide=True)
        k.cube(f"{path}/Header", (ow, T_, 0.1), (0, -T_ / 2, p.leaf_height + 0.09), Q_IDENT, jam, pm, collide=True)
        lw, lh, lt = p.leaf_width, p.leaf_height, 0.045
        hx = -(lw / 2 + 0.02)
        center = (hx + lw / 2, -lt / 2, lh / 2 + 0.005)
        mass = 32.0
        parts = [Part("box", (lw, lt, lh)), Part("box", (0.12, 0.03, 0.03), (lw / 2 - 0.12, lt / 2 + 0.015, -0.05), mass=0.4)]
        body = f"{path}/Leaf"
        b = k.rigid_body(body, center, Q_IDENT, parts=parts, mass=mass, iters=(16, 4))
        k.cube(f"{body}/Slab", (lw, lt, lh), (0, 0, 0), Q_IDENT, m.get("door_leaf"), pm, collide=True)
        k.cube(f"{body}/Handle", (0.12, 0.03, 0.03), (lw / 2 - 0.12, lt / 2 + 0.015, -0.05), Q_IDENT, m.get("steel_bright"), pm, collide=True)
        bp = f"{path}/ArtBase"
        k.fixed_base(bp, pose.pos, pose.quat)
        jp = f"{path}/Joint"
        k.joint(jp, "revolute", bp, body, (hx, 0.0, center[2]), (-lw / 2, lt / 2, 0.0), axis="Z", lower=0.0, upper=p.open_angle,
                stiffness=60.0, damping=14.0, max_force=1e5)
        wp, wq = pose.world(center)
        k.tag(b, "door_swing")
        ctx.record(body, "door_swing", wp, wq, "dynamic", size=(lw, lt, lh), mass=mass, friction=0.6,
                   articulation={"type": "revolute", "root": body, "joint": jp, "axis_local": "Z", "unit": "deg", "limits": [0.0, p.open_angle],
                                 "closed_position": 0.0, "open_position": p.open_angle,
                                 "drive": {"stiffness": 60.0, "damping": 14.0, "max_force": 1e5}, "fixed_base": True, "articulation_root": bp})
        ctx.scene["articulations"].append(dict(name=path.rsplit("/", 1)[-1], root=bp, link=body, joint=jp, kind="revolute", lower=0.0,
                                               upper=p.open_angle, unit="deg", parent=path, moving_collider_paths=[f"{body}/Slab", f"{body}/Handle"]))
        return Built(path, [a.to_world(pose) for a in self.anchors(p)], [("door_arc", pose.world_xy(self.keepouts(p)[0].polygon).tolist())],
                     info={"joint": jp, "leaf": body, "articulation_root": bp}, bodies=[body, bp])


class CageParams(Params):
    size: tuple[float, float] = Field((5.2, 4.0), description="Fenced area x, y (m)")
    fence_height: float = Field(2.0, ge=1.0, le=4.0)
    gate_width: float = Field(1.5, ge=0.8, le=4.0)
    gate_offset: float = Field(0.0, description="Gate centre along the -y (front) side, relative to the cage centre (m)")
    gate_travel: float = Field(1.7, ge=0.5, le=4.0, description="Sliding gate travel (+x)")


@register
class Cage(Asset):
    id = "cage"
    description = ("Wire-mesh security cage: translucent mesh panels + posts on all four sides and a sliding gate on the "
                   "front (-y) side on a prismatic joint (slides +x along the outside face). Interior is a support area "
                   "'inside'; 'gate' port is just outside the gate.")
    tags = frozenset({"cage", "fence", "enclosure", "articulated", "prismatic", "warehouse"})
    Params = CageParams
    physics = PhysicsInfo("articulated", "steel", 0.6, 28.0, "primitives", "static fence + fixed-base gate articulation")

    def footprint(self, p):
        """Hollow: four fence strips (posts included), so contents can be placed inside."""
        sx, sy, h, t = p.size[0], p.size[1], p.fence_height + 0.2, 0.1
        return Footprint.union([Footprint.rect(sx + t, t, h, 0, -sy / 2), Footprint.rect(sx + t, t, h, 0, sy / 2),
                                Footprint.rect(t, sy + t, h, -sx / 2, 0), Footprint.rect(t, sy + t, h, sx / 2, 0),
                                Footprint.rect(p.gate_width + 0.1, 0.08, h, p.gate_offset, -sy / 2 - 0.055)])

    def anchors(self, p):
        sx, sy = p.size
        return [Anchor("inside", "support", (0, 0, 0), polygon=_rect(0, 0, sx - 0.4, sy - 0.4)),
                Anchor("gate", "port", (p.gate_offset, -sy / 2 - 1.1, 0.0), qyaw(math.pi / 2), port_kind="approach")]

    def keepouts(self, p):
        sy = p.size[1]
        g = p.gate_offset
        return [Keepout([(g - 0.9, -sy / 2 - 1.0), (g + 2.6, -sy / 2 - 1.0), (g + 2.6, -sy / 2 - 0.05), (g - 0.9, -sy / 2 - 0.05)], "gate_slide")]

    def joints(self, p):
        return [JointInfo("GateJoint", "prismatic", "X", 0.0, p.gate_travel, "m", 0.0, {"stiffness": 2e4, "damping": 4e3, "max_force": 1e5}, "Gate", "gate")]

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        k.xform(path, pose.pos, pose.quat)
        sx_, sy_ = p.size
        x0, y0, x1, y1 = -sx_ / 2, -sy_ / 2, sx_ / 2, sy_ / 2
        pm, mat, post_m = m.phys("steel"), m.get("cage_mesh"), m.get("cage_post")
        gw, h, z0 = p.gate_width, p.fence_height, 0.1
        gate_x = p.gate_offset
        sides = [("S", (x0, y0), (x1, y0), True), ("N", (x0, y1), (x1, y1), False), ("W", (x0, y0), (x0, y1), False), ("E", (x1, y0), (x1, y1), False)]
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
                k.cube(f"{path}/Panel_{name}{j}", (s1 - s0, 0.02, h), (ctr[0], ctr[1], z0 + h / 2), qyaw(yaw), mat, pm, collide=True)
                n_posts = max(1, int(math.ceil((s1 - s0) / 1.4)))
                for pi in range(n_posts + 1):
                    s = s0 + (s1 - s0) * pi / n_posts
                    k.cube(f"{path}/Post_{name}{j}_{pi}", (0.06, 0.06, h + 0.2), (a[0] + t[0] * s, a[1] + t[1] * s, (h + 0.2) / 2), Q_IDENT, post_m, pm, collide=True)
            wp, wq = pose.world(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 1.1), qyaw(yaw))
            ctx.record(f"{path}/Panel_{name}", "cage_fence", wp, wq, "static", size=(L, 0.02, h + 0.2), friction=0.6)
        leaf = (gw + 0.1, 0.03, h)
        center = (gate_x, y0 - 0.055, z0 + h / 2)
        mass = 28.0
        body = f"{path}/Gate"
        b = k.rigid_body(body, center, Q_IDENT, parts=[Part("box", leaf)], mass=mass, iters=(16, 4), disable_gravity=True)
        k.cube(f"{body}/Leaf", leaf, (0, 0, 0), Q_IDENT, m.get("cage_gate"), pm, collide=True)
        bp = f"{path}/GateBase"
        k.fixed_base(bp, pose.pos, pose.quat)
        jp = f"{path}/GateJoint"
        k.joint(jp, "prismatic", bp, body, center, (0, 0, 0), axis="X", lower=0.0, upper=p.gate_travel, stiffness=2e4, damping=4e3, max_force=1e5)
        k.tag(b, "gate_sliding")
        wp, wq = pose.world(center)
        ctx.record(body, "gate_sliding", wp, wq, "dynamic", size=leaf, mass=mass, friction=0.6,
                   articulation={"type": "prismatic", "root": body, "joint": jp, "axis_local": "X", "unit": "m", "limits": [0.0, p.gate_travel],
                                 "closed_position": 0.0, "open_position": p.gate_travel,
                                 "drive": {"stiffness": 2e4, "damping": 4e3, "max_force": 1e5}, "fixed_base": True, "articulation_root": bp})
        ctx.scene["articulations"].append(dict(name=path.rsplit("/", 1)[-1] + "Gate", root=bp, link=body, joint=jp, kind="prismatic", lower=0.0,
                                               upper=p.gate_travel, unit="m", parent=path, moving_collider_paths=[f"{body}/Leaf"]))
        return Built(path, [a.to_world(pose) for a in self.anchors(p)], [("gate_slide", pose.world_xy(self.keepouts(p)[0].polygon).tolist())],
                     info={"joint": jp, "gate": body, "articulation_root": bp}, bodies=[body, bp])
