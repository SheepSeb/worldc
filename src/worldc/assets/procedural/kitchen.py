"""Kitchen assets: L-shaped base-cabinet counter (doors + drawers on driven joints), fridge, island, window, tableware.

Cabinet frames: x = width, y = front (+y), z up; the back is at -depth/2.  Moving parts (doors: revolute, drawers:
prismatic) are rigid bodies on fixed-base articulations, authored like ``work_table`` drawers.  Static carcasses, tops
and handles are plain colliders.
"""
from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from ...geom import Part, Q_IDENT, qaxis, qfrom_mat, qmat, qmul, qyaw
from ..base import Anchor, Asset, Built, Footprint, JointInfo, Keepout, Params, PhysicsInfo
from ..registry import register

EPS = 0.0008
CARCASS_T = 0.018  # cabinet side / back / bottom panel thickness
PLINTH = 0.10  # recessed kick plate height
FRONT_T = 0.02  # door / drawer front thickness
QFLIP = qaxis((1, 0, 0), math.pi)


def _rect(cx, cy, sx, sy):
    return [(cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy - sy / 2), (cx + sx / 2, cy + sy / 2), (cx - sx / 2, cy + sy / 2)]


def _cube(ctx, path, size, pos, mat, pm="wood", q=Q_IDENT):
    ctx.kit.cube(path, size, pos, q, ctx.mats.get(mat), ctx.mats.phys(pm), collide=True)


def _rot(yaw, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


def _articulate(ctx, path, pose, name, center, cyaw, boxes, mass, jtype, axis, lower, upper, init, hinge=None, stiffness=3000.0,
                damping=600.0, category="drawer"):
    """One moving part: a rigid body of ``boxes`` [(size, pos, material)] (cabinet frame) on a joint to a world-fixed base.

    ``center`` is the body origin (asset frame) with the joint at 0; ``init`` shifts it to the starting joint position.
    A revolute part pivots about ``hinge`` (cabinet frame, relative to the body origin); a prismatic one slides along its +y."""
    k = ctx.kit
    qb = qyaw(cyaw)
    start = tuple(np.asarray(center, float) + np.asarray(_rot(cyaw, (0.0, init if jtype == "prismatic" else 0.0, 0.0))))
    parts = [Part("box", s, p) for s, p, _ in boxes]
    dp = f"{path}/{name}"
    b = k.rigid_body(dp, start, qb, parts=parts, mass=mass, iters=(16, 4), disable_gravity=False)
    names = []
    for i, (s, p, mat) in enumerate(boxes):
        nm = f"P{i}"
        names.append(nm)
        _cube(ctx, f"{dp}/{nm}", s, p, mat, "wood")
    bp = f"{path}/{name}_Base"
    k.fixed_base(bp, pose.pos, pose.quat)
    jp = f"{dp}/Joint"
    if jtype == "prismatic":
        pos0, pos1 = center, (0.0, 0.0, 0.0)
    else:
        pos0 = tuple(np.asarray(center, float) + np.asarray(_rot(cyaw, hinge)))
        pos1 = tuple(hinge)
    flip = jtype == "revolute" and hinge[0] > 0  # right-hinged: mirror the axis so a positive angle still opens the door
    rot0, rot1 = (qmul(qb, QFLIP), QFLIP) if flip else (qb, Q_IDENT)
    k.joint(jp, jtype, bp, dp, pos0, pos1, axis=axis, lower=lower, upper=upper, target=init, stiffness=stiffness, damping=damping,
            max_force=1e4 if jtype == "prismatic" else 1e5, rot0=rot0, rot1=rot1, init=init)
    wp, wq = pose.world(start, qb)
    k.tag(b, category, index=int(name.rsplit("_", 1)[-1]))
    unit = "m" if jtype == "prismatic" else "deg"
    drive = {"stiffness": stiffness, "damping": damping, "max_force": 1e4 if jtype == "prismatic" else 1e5}
    ext = boxes[0][0]
    ctx.record(dp, category, wp, wq, "dynamic", size=ext, mass=mass, friction=0.5,
               articulation={"type": jtype, "root": dp, "joint": jp, "axis_local": axis, "unit": unit, "limits": [lower, upper],
                             "closed_position": lower, "open_position": upper, "initial_position": init, "drive": drive,
                             "fixed_base": True, "articulation_root": bp})
    ctx.scene["articulations"].append(dict(name=f"{path.rsplit('/', 1)[-1]}_{name}", root=bp, link=dp, joint=jp, kind=jtype, lower=lower, upper=upper,
                                           unit=unit, initial=init, parent=path, moving_collider_paths=[f"{dp}/{n}" for n in names]))
    return [dp, bp]


def _bar_handle(vertical: bool, length: float, pos, standoff=0.03):
    """Brushed-steel bar handle: a bar on two short posts. ``pos`` is the bar centre; the posts reach back to the front face."""
    x, y, z = pos
    t = 0.012
    post = (0.01, standoff, 0.01)
    if vertical:
        bar = ((t, t, length), pos, "brushed_steel")
        posts = [((post[0], post[1], post[2]), (x, y - standoff / 2, z + dz), "brushed_steel") for dz in (-length * 0.35, length * 0.35)]
    else:
        bar = ((length, t, t), pos, "brushed_steel")
        posts = [((post[0], post[1], post[2]), (x + dx, y - standoff / 2, z), "brushed_steel") for dx in (-length * 0.35, length * 0.35)]
    return [bar] + posts


# =========================================================================== L-shaped counter
class CounterParams(Params):
    cabinets_along_wall: int = Field(4, ge=1, le=8, description="Base cabinets in the main run (local x), not counting the corner unit")
    cabinets_return: int = Field(2, ge=0, le=6, description="Base cabinets in the return leg (the L's short arm, local +y beyond the corner)")
    cabinet_width: float = Field(0.6, ge=0.4, le=0.9)
    corner_gap: float = Field(0.2, ge=0.0, le=0.5, description="Extra filler between the corner and the first return cabinet, so doors on both legs clear each other")
    depth: float = Field(0.6, ge=0.4, le=0.9)
    height: float = Field(0.9, ge=0.7, le=1.1, description="Counter height including the top (m)")
    top_thickness: float = Field(0.04, ge=0.02, le=0.08)
    drawers: int = Field(4, ge=0, le=14, description="Drawers above the doors of the first N cabinets (main run first)")
    drawer_travel: float = Field(0.4, ge=0.2, le=0.55)
    open_drawer: int | None = Field(None, ge=0, le=13, description="Index of one drawer to start open (None = all closed)")
    open_amount: float = Field(0.15, ge=0.0, le=0.5, description="How far that drawer starts open (m)")
    door_open_angle: float = Field(100.0, ge=30, le=170)
    front_material: str = Field("sage_matte", description="Visual material preset of doors, drawer fronts and carcass")
    top_material: str = Field("quartz_white", description="Visual material preset of the countertop")


def _layout(p):
    """[(kind, index, cx, cy, yaw)] of every unit, the corner filler included, in the counter frame."""
    w, d = p.cabinet_width, p.depth
    X = p.cabinets_along_wall * w + (d if p.cabinets_return else 0.0)
    units = [("cabinet", i, -X / 2 + w * (i + 0.5), d / 2, 0.0) for i in range(p.cabinets_along_wall)]
    if p.cabinets_return:
        units.append(("corner", 0, X / 2 - d / 2, (d + p.corner_gap) / 2, 0.0))
        units += [("cabinet", p.cabinets_along_wall + j, X / 2 - d / 2, d + p.corner_gap + w * (j + 0.5), math.pi / 2) for j in range(p.cabinets_return)]
    return X, units


@register
class KitchenCounterL(Asset):
    id = "kitchen_counter_l"
    description = ("L-shaped kitchen base-cabinet counter (default 0.9 m high, 0.6 m deep): a main run plus a return leg joined by a "
                   "blind corner unit, stone top, a hinged door on every cabinet (revolute) and drawers above the first N doors "
                   "(prismatic), all with bar handles; one drawer can start partly open. The back of the main run is at local y=0, "
                   "the return leg is at the local +x end. Supports 'top_main' and 'top_return'.")
    tags = frozenset({"counter", "cabinet", "kitchen", "furniture", "articulated", "revolute", "prismatic", "support", "drawer", "door"})
    Params = CounterParams
    flush_walls = True
    example = {"id": "counter", "asset": "kitchen_counter_l", "placement": {"pose": {"xy": [-0.5, 1.75], "yaw": 180}}}
    physics = PhysicsInfo("articulated", "wood", 0.5, None, "primitives", "static carcass + fixed-base door (4 kg) and drawer (3 kg) articulations")

    def footprint(self, p):
        X, _ = _layout(p)
        d, H = p.depth, p.height
        parts = [Footprint.rect(X, d + 0.02, H, 0.0, (d + 0.02) / 2)]
        if p.cabinets_return:
            L = p.cabinets_return * p.cabinet_width + p.corner_gap
            parts.append(Footprint.rect(d + 0.02, L - 0.02, H, X / 2 - (d + 0.02) / 2, d + 0.02 + (L - 0.02) / 2))
        return Footprint.union(parts) if len(parts) > 1 else parts[0]

    def anchors(self, p):
        X, _ = _layout(p)
        d, H = p.depth, p.height
        out = [Anchor("top_main", "support", (0.0, (d + 0.02) / 2, H), polygon=_rect(0, 0, X - 0.1, d - 0.08))]
        if p.cabinets_return:
            L = p.cabinets_return * p.cabinet_width + p.corner_gap
            out.append(Anchor("top_return", "support", (X / 2 - (d + 0.02) / 2, d + 0.02 + (L - 0.02) / 2, H), polygon=_rect(0, 0, d - 0.08, L - 0.1)))
        out.append(Anchor("front", "port", (0.0, d + 0.7, 0), qyaw(-math.pi / 2), port_kind="approach"))
        return out

    def joints(self, p):
        _, units = _layout(p)
        n = sum(1 for u in units if u[0] == "cabinet")
        js = [JointInfo(f"Drawer_{i}", "prismatic", "Y", 0.0, p.drawer_travel, "m", p.open_amount if p.open_drawer == i else 0.0,
                        {"stiffness": 3000.0, "damping": 600.0, "max_force": 1e4}, f"Drawer_{i}", "drawer") for i in range(min(p.drawers, n))]
        js += [JointInfo(f"Door_{i}", "revolute", "Z", 0.0, p.door_open_angle, "deg", 0.0,
                         {"stiffness": 40.0, "damping": 8.0, "max_force": 1e5}, f"Door_{i}", "door") for i in range(n)]
        return js

    def build(self, ctx, path, pose, p, rng):
        k = ctx.kit
        k.xform(path, pose.pos, pose.quat)
        X, units = _layout(p)
        w, d, H, tt = p.cabinet_width, p.depth, p.height, p.top_thickness
        Hc = H - tt  # carcass top
        fm, tm = p.front_material, p.top_material
        bodies = []
        n_cab = 0
        for kind, idx, cx, cy, yaw in units:
            cp = f"{path}/Cab_{idx}" if kind == "cabinet" else f"{path}/Corner"
            k.xform(cp, (cx, cy, 0.0), qyaw(yaw))
            _cube(ctx, f"{cp}/Plinth", (w - 0.02, d - 0.08, PLINTH), (0, -0.02, PLINTH / 2), "steel_dark")
            if kind == "corner":  # blind corner: a solid block, reachable from neither side
                _cube(ctx, f"{cp}/Block", (d, d + p.corner_gap, Hc - PLINTH), (0, 0, PLINTH + (Hc - PLINTH) / 2), fm)
                wp, wq = pose.world((cx, cy, Hc / 2), qyaw(yaw))
                ctx.record(cp, "cabinet", wp, wq, "static", size=(d, d + p.corner_gap, Hc), friction=0.5, subtype="corner_unit")
                continue
            has_drawer = n_cab < p.drawers
            _cube(ctx, f"{cp}/SideL", (CARCASS_T, d - FRONT_T, Hc - PLINTH), (-w / 2 + CARCASS_T / 2, -FRONT_T / 2, PLINTH + (Hc - PLINTH) / 2), fm)
            _cube(ctx, f"{cp}/SideR", (CARCASS_T, d - FRONT_T, Hc - PLINTH), (w / 2 - CARCASS_T / 2, -FRONT_T / 2, PLINTH + (Hc - PLINTH) / 2), fm)
            _cube(ctx, f"{cp}/Back", (w - 2 * CARCASS_T, CARCASS_T, Hc - PLINTH), (0, -d / 2 + CARCASS_T / 2, PLINTH + (Hc - PLINTH) / 2), fm)
            _cube(ctx, f"{cp}/Bottom", (w - 2 * CARCASS_T, d - FRONT_T - CARCASS_T, CARCASS_T), (0, -FRONT_T / 2 + CARCASS_T / 2, PLINTH + CARCASS_T / 2), fm)
            dr_h = 0.17 if has_drawer else 0.0
            door_top = Hc - 0.005 - (dr_h + 0.005 if has_drawer else 0.0)
            door_bot = PLINTH - 0.005
            dh = door_top - door_bot
            if has_drawer:
                _cube(ctx, f"{cp}/Divider", (w - 2 * CARCASS_T, d - FRONT_T - CARCASS_T, CARCASS_T), (0, -FRONT_T / 2 + CARCASS_T / 2, door_top - 0.005), fm)
            wp, wq = pose.world((cx, cy, Hc / 2), qyaw(yaw))
            ctx.record(cp, "cabinet", wp, wq, "static", size=(w, d, Hc), friction=0.5, subtype="base_cabinet")
            # door: hinged on its local -x edge, swings out through +y
            dw = w - 0.004
            door_boxes = [((dw, FRONT_T - 0.001, dh), (0, 0, 0), fm)] + _bar_handle(True, 0.14, (dw / 2 - 0.05, FRONT_T / 2 + 0.03, dh / 2 - 0.11))
            body_c = (cx, cy, 0.0)
            door_center_cab = (0.0, d / 2 - FRONT_T / 2, door_bot + dh / 2)
            dc = tuple(np.asarray(body_c) + np.asarray(_rot(yaw, door_center_cab)))
            bodies += _articulate(ctx, path, pose, f"Door_{idx}", dc, yaw, door_boxes, 4.0, "revolute", "Z", 0.0, p.door_open_angle, 0.0,
                                  hinge=(-dw / 2, FRONT_T / 2, 0.0), stiffness=40.0, damping=8.0, category="cabinet_door")
            if has_drawer:
                dwid, dl, th = w - 2 * CARCASS_T - 0.01, d - FRONT_T - 0.05, 0.012
                zc = Hc - 0.005 - dr_h / 2  # drawer body origin: front-panel centre height
                fy = d / 2 - FRONT_T / 2  # front-panel centre y (cabinet frame)
                ty = fy - FRONT_T / 2 - dl / 2  # tray centre y
                zb = -dr_h / 2 + 0.01  # tray bottom, relative to the body origin (z)
                dboxes = [
                    ((dw, FRONT_T - 0.001, dr_h), (0, 0, 0), fm),
                    ((dwid, dl, th), (0, ty - fy, zb + th / 2), "wood_light"),
                    ((th, dl, dr_h - 0.04), (-dwid / 2 + th / 2, ty - fy, zb + (dr_h - 0.04) / 2), "wood_light"),
                    ((th, dl, dr_h - 0.04), (dwid / 2 - th / 2, ty - fy, zb + (dr_h - 0.04) / 2), "wood_light"),
                    ((dwid - 2 * th, th, dr_h - 0.04), (0, ty - fy - dl / 2 + th / 2, zb + (dr_h - 0.04) / 2), "wood_light"),
                ] + _bar_handle(False, 0.16, (0.0, FRONT_T / 2 + 0.03, 0.0))
                dcen = tuple(np.asarray(body_c) + np.asarray(_rot(yaw, (0.0, fy, zc))))
                init = p.open_amount if p.open_drawer == n_cab else 0.0
                bodies += _articulate(ctx, path, pose, f"Drawer_{n_cab}", dcen, yaw, dboxes, 3.0, "prismatic", "Y", 0.0, p.drawer_travel, init,
                                      stiffness=3000.0, damping=600.0, category="drawer")
            n_cab += 1
        # stone top: main run + return leg (front overhang 2 cm)
        top_a = (X, d + 0.02, tt)
        pos_a = (0.0, (d + 0.02) / 2, H - tt / 2)
        _cube(ctx, f"{path}/TopMain", top_a, pos_a, tm, "ceramic")
        wp, wq = pose.world(pos_a)
        ctx.record(f"{path}/TopMain", "countertop", wp, wq, "static", size=top_a, friction=0.5)
        if p.cabinets_return:
            L = p.cabinets_return * p.cabinet_width + p.corner_gap
            top_b = (d + 0.02, L - 0.02, tt)
            pos_b = (X / 2 - (d + 0.02) / 2, d + 0.02 + (L - 0.02) / 2, H - tt / 2)
            _cube(ctx, f"{path}/TopReturn", top_b, pos_b, tm, "ceramic")
            wp, wq = pose.world(pos_b)
            ctx.record(f"{path}/TopReturn", "countertop", wp, wq, "static", size=top_b, friction=0.5)
        return Built(path, [a.to_world(pose) for a in self.anchors(p)], info={"top_z": pose.pos[2] + H}, bodies=bodies)


# =========================================================================== fridge
class FridgeParams(Params):
    width: float = Field(0.6, ge=0.5, le=0.95)
    depth: float = Field(0.65, ge=0.5, le=0.9)
    height: float = Field(1.8, ge=1.2, le=2.2)
    hinge: str = Field("left", pattern="^(left|right)$", description="Side of the door hinge, seen from the front")
    open_angle: float = Field(110.0, ge=30, le=170)
    door_angle: float = Field(0.0, ge=0.0, le=170, description="Starting door angle (0 = closed)")


@register
class Fridge(Asset):
    id = "fridge"
    description = ("Stainless-steel single-door fridge (default 0.6 x 0.65 x 1.8 m): static body plus a hinged door on a revolute joint "
                   "(0 = closed) with a vertical bar handle. Front is local +y; the door swing is a keep-out.")
    tags = frozenset({"fridge", "appliance", "kitchen", "articulated", "revolute", "door"})
    Params = FridgeParams
    flush_walls = True
    example = {"id": "fridge", "asset": "fridge", "placement": {"pose": {"xy": [1.7, 1.425], "yaw": 180}}}
    physics = PhysicsInfo("articulated", "steel", 0.5, 12.0, "primitives", "static body + fixed-base door articulation (12 kg)")

    def footprint(self, p):
        return Footprint.rect(p.width, p.depth, p.height, 0.0, p.depth / 2)

    def keepouts(self, p):
        return [Keepout(_rect(0.0, p.depth + 0.35, p.width, 0.7), "door_arc")]

    def anchors(self, p):
        return [Anchor("front", "port", (0.0, p.depth + 0.7, 0.0), qyaw(-math.pi / 2), port_kind="approach"),
                Anchor("handle", "port", ((1 if p.hinge == "left" else -1) * (p.width / 2 - 0.05), p.depth + 0.03, 1.2), port_kind="handle")]

    def joints(self, p):
        return [JointInfo("Door/Joint", "revolute", "Z", 0.0, p.open_angle, "deg", p.door_angle,
                          {"stiffness": 150.0, "damping": 30.0, "max_force": 1e5}, "Door", "door")]

    def build(self, ctx, path, pose, p, rng):
        k = ctx.kit
        k.xform(path, pose.pos, pose.quat)
        w, d, h = p.width, p.depth, p.height
        t = 0.03  # door thickness; the body spans y in [0, d - t]
        body_c = (0.0, (d - t) / 2, 0.03 + (h - 0.03) / 2)
        _cube(ctx, f"{path}/Body", (w, d - t, h - 0.03), body_c, "stainless_fridge", "steel")
        _cube(ctx, f"{path}/Plinth", (w - 0.04, d - 0.1, 0.03), (0, (d - 0.1) / 2, 0.015), "steel_dark", "steel")
        wp, wq = pose.world((0, (d - t) / 2, h / 2))
        ctx.record(f"{path}/Body", "fridge", wp, wq, "static", size=(w, d, h), friction=0.5)
        dw, dh = w - 0.004, h - 0.06
        hinge_x = -dw / 2 if p.hinge == "left" else dw / 2
        free = 1.0 if p.hinge == "left" else -1.0  # the handle sits on the free edge
        boxes = [((dw, t - 0.002, dh), (0, 0, 0), "stainless_fridge")] + _bar_handle(True, 0.55, (free * (dw / 2 - 0.05), t / 2 + 0.03, 0.25))
        center = (0.0, d - t / 2, 0.03 + dh / 2)
        bodies = _articulate(ctx, path, pose, "Door_0", center, 0.0, boxes, 12.0, "revolute", "Z", 0.0, p.open_angle, p.door_angle,
                             hinge=(hinge_x, t / 2, 0.0), stiffness=150.0, damping=30.0, category="fridge_door")
        return Built(path, [a.to_world(pose) for a in self.anchors(p)], [("door_arc", pose.world_xy(self.keepouts(p)[0].polygon).tolist())],
                     info={"door": f"{path}/Door_0"}, bodies=bodies)


# =========================================================================== island
class IslandParams(Params):
    size: tuple[float, float] = Field((1.2, 0.8), description="Footprint x (length), y (depth) in metres")
    height: float = Field(0.9, ge=0.7, le=1.1)
    top_thickness: float = Field(0.05, ge=0.03, le=0.1)
    top_overhang: float = Field(0.03, ge=0.0, le=0.3)
    front_material: str = "sage_matte"
    top_material: str = "butcher_block"


@register
class KitchenIsland(Asset):
    id = "kitchen_island"
    description = ("Freestanding kitchen island (default 1.2 x 0.8 m, 0.9 m high): cabinet base with two door fronts and bar handles "
                   "on the +y side, butcher-block top that overhangs the base. Static. Supports 'top'.")
    tags = frozenset({"island", "counter", "kitchen", "furniture", "static", "support"})
    Params = IslandParams
    example = {"id": "island", "asset": "kitchen_island", "placement": {"pose": {"xy": [0.4, -0.25], "yaw": 0}}}
    physics = PhysicsInfo("static", "wood", 0.5, None, "primitives")

    def footprint(self, p):
        return Footprint.rect(p.size[0], p.size[1], p.height)

    def anchors(self, p):
        return [Anchor("top", "support", (0, 0, p.height), polygon=_rect(0, 0, p.size[0] - 0.1, p.size[1] - 0.1)),
                Anchor("front", "port", (0, p.size[1] / 2 + 0.7, 0), qyaw(-math.pi / 2), port_kind="approach")]

    def build(self, ctx, path, pose, p, rng):
        k = ctx.kit
        k.xform(path, pose.pos, pose.quat)
        sx, sy = p.size
        H, tt, ov = p.height, p.top_thickness, p.top_overhang
        bx, by = sx - 2 * ov, sy - 2 * ov
        hb = H - tt
        _cube(ctx, f"{path}/Plinth", (bx - 0.06, by - 0.06, PLINTH), (0, 0, PLINTH / 2), "steel_dark")
        _cube(ctx, f"{path}/Carcass", (bx, by, hb - PLINTH), (0, 0, PLINTH + (hb - PLINTH) / 2), p.front_material)
        fy = by / 2 + FRONT_T / 2
        fw = bx / 2 - 0.004
        for i, sgn in enumerate((-1, 1)):
            _cube(ctx, f"{path}/DoorFront_{i}", (fw - 0.002, FRONT_T, hb - PLINTH - 0.02), (sgn * (fw / 2 + 0.002), fy, PLINTH + (hb - PLINTH) / 2), p.front_material)
            hx = -sgn * 0.05  # handles sit at the centre seam
            for j, (s, pos, m) in enumerate(_bar_handle(True, 0.14, (hx, fy + FRONT_T / 2 + 0.03, hb - 0.16))):
                _cube(ctx, f"{path}/Handle_{i}_{j}", s, pos, m, "steel")
        _cube(ctx, f"{path}/Top", (sx, sy, tt), (0, 0, H - tt / 2), p.top_material)
        wp, wq = pose.world((0, 0, H / 2))
        ctx.record(f"{path}/Island", "island", wp, wq, "static", size=(sx, sy, H), friction=0.5)
        return Built(path, [a.to_world(pose) for a in self.anchors(p)], info={"top_z": pose.pos[2] + H})


# =========================================================================== window
class WindowParams(Params):
    width: float = Field(1.2, ge=0.3, le=4.0)
    height: float = Field(2.0, ge=0.5, le=3.5, description="Top edge of the opening above the floor (the opening's `height`)")
    sill: float = Field(0.9, ge=0.0, le=2.0, description="Bottom edge of the opening above the floor")
    wall_thickness: float = Field(0.3, ge=0.05, le=1.0)
    mullions: int = Field(1, ge=0, le=4, description="Vertical glazing bars")
    frame_material: str = "window_frame"


@register
class Window(Asset):
    id = "window"
    description = ("Window for a wall opening: frame, glazing bars, translucent glass pane and an interior sill board. Set `sill` on the "
                   "opening so the wall below it stays solid. Attach through the opening's `door` field; with sun/overcast lighting "
                   "a soft skylight panel is added in its plane.")
    tags = frozenset({"window", "building", "glass", "static", "indoor", "daylight"})
    Params = WindowParams
    physics = PhysicsInfo("static", "steel", 0.5, None, "primitives")

    def footprint(self, p):
        return Footprint.rect(p.width, p.wall_thickness + 0.08, p.height - p.sill, 0.0, -p.wall_thickness / 2 + 0.04, z0=p.sill)

    def build(self, ctx, path, pose, p, rng):
        k = ctx.kit
        k.xform(path, pose.pos, pose.quat)
        W, T_, z0, z1 = p.width, p.wall_thickness, p.sill, p.height
        h = z1 - z0
        fb, fd = 0.05, 0.06  # frame bar width / depth
        fm = p.frame_material
        zc = (z0 + z1) / 2
        _cube(ctx, f"{path}/FrameBottom", (W, fd, fb), (0, -T_ / 2, z0 + fb / 2), fm, "steel")
        _cube(ctx, f"{path}/FrameTop", (W, fd, fb), (0, -T_ / 2, z1 - fb / 2), fm, "steel")
        for i, sx in enumerate((-1, 1)):
            _cube(ctx, f"{path}/FrameSide_{i}", (fb, fd, h - 2 * fb), (sx * (W / 2 - fb / 2), -T_ / 2, zc), fm, "steel")
        for i in range(p.mullions):
            x = -W / 2 + fb + (W - 2 * fb) * (i + 1) / (p.mullions + 1)
            _cube(ctx, f"{path}/Mullion_{i}", (0.03, fd * 0.8, h - 2 * fb), (x, -T_ / 2, zc), fm, "steel")
        k.cube(f"{path}/Glass", (W - 2 * fb, 0.008, h - 2 * fb), (0, -T_ / 2, zc), Q_IDENT, ctx.mats.get("glass"), ctx.mats.phys("glass"), collide=True)
        _cube(ctx, f"{path}/Sill", (W + 0.1, 0.08, 0.03), (0, 0.03, z0 + 0.015), fm, "steel")
        wp, wq = pose.world((0, -T_ / 2, zc))
        ctx.record(f"{path}/Glass", "window", wp, wq, "static", size=(W, fd, h), friction=0.3)
        return Built(path, [])


# =========================================================================== tableware
class PlateParams(Params):
    diameter: float = Field(0.26, ge=0.12, le=0.4)
    material: str = "ceramic_white"
    mass_kg: float = Field(0.45, ge=0.05, le=2.0)


@register
class Plate(Asset):
    id = "plate"
    description = "Round dinner plate (default 26 cm, white ceramic) with a shallow well; dynamic, single cylinder collider."
    tags = frozenset({"plate", "tableware", "kitchen", "ceramic", "dynamic", "clutter"})
    Params = PlateParams
    physics = PhysicsInfo("dynamic", "ceramic", 0.45, 0.45, "primitives")

    def footprint(self, p):
        return Footprint.circle(p.diameter / 2, 0.02)

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        r, h = p.diameter / 2, 0.02
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + EPS)
        body = k.rigid_body(path, pos, pose.quat, parts=[Part("cyl", (r, h), (0, 0, h / 2))], mass=p.mass_kg)
        k.cylinder(f"{path}/Rim", r, h, (0, 0, h / 2), Q_IDENT, m.get(p.material), m.phys("ceramic"), collide=True)
        k.cylinder(f"{path}/Well", r * 0.68, 0.002, (0, 0, h + 0.0005), Q_IDENT, m.get("white"))
        k.tag(body, "plate", zone=ctx.zone or "")
        ctx.record(path, "clutter", pos, pose.quat, "dynamic", size=(2 * r, 2 * r, h), mass=p.mass_kg, friction=0.45, subtype="plate")
        return Built(path, bodies=[path])


class MugParams(Params):
    radius: float = Field(0.04, ge=0.025, le=0.07)
    height: float = Field(0.095, ge=0.06, le=0.16)
    material: str = "ceramic_red"
    mass_kg: float = Field(0.3, ge=0.05, le=1.0)


@register
class Mug(Asset):
    id = "mug"
    description = "Ceramic coffee mug (default red, 8 x 9.5 cm) with a side handle; dynamic, cylinder + handle colliders."
    tags = frozenset({"mug", "cup", "tableware", "kitchen", "ceramic", "dynamic", "clutter"})
    Params = MugParams
    physics = PhysicsInfo("dynamic", "ceramic", 0.45, 0.3, "primitives")

    def footprint(self, p):
        return Footprint.circle(p.radius + 0.035, p.height)

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        r, h = p.radius, p.height
        hx = r + 0.016
        # handle: a vertical bar 1.6 cm outside the wall, joined to it by two short stubs
        hs = [((0.012, 0.012, h * 0.55), (hx, 0, h * 0.5)),
              ((0.032, 0.012, 0.012), (r + 0.004, 0, h * 0.5 + h * 0.275 - 0.006)),
              ((0.032, 0.012, 0.012), (r + 0.004, 0, h * 0.5 - h * 0.275 + 0.006))]
        parts = [Part("cyl", (r, h), (0, 0, h / 2))] + [Part("box", s, c) for s, c in hs]
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + EPS)
        body = k.rigid_body(path, pos, pose.quat, parts=parts, mass=p.mass_kg)
        k.cylinder(f"{path}/Body", r, h, (0, 0, h / 2), Q_IDENT, m.get(p.material), m.phys("ceramic"), collide=True)
        k.cylinder(f"{path}/Coffee", r - 0.005, 0.002, (0, 0, h + 0.0005), Q_IDENT, m.get("brown"))
        for i, (s, c) in enumerate(hs):
            _cube(ctx, f"{path}/Handle_{i}", s, c, p.material, "ceramic")
        k.tag(body, "mug", zone=ctx.zone or "")
        ctx.record(path, "clutter", pos, pose.quat, "dynamic", size=(2 * r + 0.05, 2 * r, h), mass=p.mass_kg, friction=0.45, subtype="mug")
        return Built(path, bodies=[path])


class BoardParams(Params):
    size: tuple[float, float, float] = Field((0.4, 0.28, 0.022), description="Length, width, thickness (m)")
    material: str = "wood_light"
    mass_kg: float = Field(0.9, ge=0.1, le=5.0)


@register
class CuttingBoard(Asset):
    id = "cutting_board"
    description = "Rectangular wooden cutting board (default 40 x 28 x 2.2 cm); dynamic, single box collider."
    tags = frozenset({"cutting_board", "board", "kitchen", "wood", "dynamic", "clutter"})
    Params = BoardParams
    physics = PhysicsInfo("dynamic", "wood", 0.5, 0.9, "primitives")

    def footprint(self, p):
        return Footprint.rect(p.size[0], p.size[1], p.size[2])

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        sx, sy, sz = p.size
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + EPS)
        body = k.rigid_body(path, pos, pose.quat, parts=[Part("box", p.size, (0, 0, sz / 2))], mass=p.mass_kg)
        k.cube(f"{path}/Board", p.size, (0, 0, sz / 2), Q_IDENT, m.get(p.material), m.phys("wood"), collide=True)
        k.cube(f"{path}/HandleHole", (0.05, 0.012, 0.001), (sx / 2 - 0.05, 0, sz + 0.0005), Q_IDENT, m.get("wood_dark"))
        k.tag(body, "cutting_board", zone=ctx.zone or "")
        ctx.record(path, "clutter", pos, pose.quat, "dynamic", size=p.size, mass=p.mass_kg, friction=0.5, subtype="cutting_board")
        return Built(path, bodies=[path])


class BowlParams(Params):
    apples: int = Field(3, ge=0, le=3, description="Green apples resting in the bowl (separate dynamic bodies)")
    apple_radius: float = Field(0.037, ge=0.03, le=0.045)


@register
class FruitBowl(Asset):
    id = "fruit_bowl"
    description = ("Glass fruit bowl (26 cm across, 9 cm deep, compound wall colliders) holding up to three green apples that rest "
                   "on its floor as their own dynamic bodies.")
    tags = frozenset({"bowl", "fruit", "apple", "glass", "tableware", "kitchen", "dynamic", "clutter"})
    Params = BowlParams
    physics = PhysicsInfo("dynamic", "glass", 0.35, 0.45, "primitives", "bowl 0.45 kg + apples 0.18 kg each")

    RB, RT, H, FLOOR = 0.095, 0.13, 0.09, 0.008

    def footprint(self, p):
        return Footprint.circle(self.RT + 0.004, self.H)

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        n = 16
        L = math.hypot(self.RT - self.RB, self.H - self.FLOOR)
        alpha = math.atan2(self.RT - self.RB, self.H - self.FLOOR)  # wall tilt from vertical
        rmid, zmid = (self.RB + self.RT) / 2, (self.FLOOR + self.H) / 2
        seg = 2 * math.pi * rmid / n * 1.12
        parts = [Part("cyl", (self.RB + 0.004, self.FLOOR), (0, 0, self.FLOOR / 2))]
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + EPS)
        walls = []
        for i in range(n):
            th = 2 * math.pi * i / n
            t = np.array([-math.sin(th), math.cos(th), 0.0])
            rad = np.array([math.cos(th), math.sin(th), 0.0])
            s = rad * math.sin(alpha) + np.array([0, 0, math.cos(alpha)])
            nrm = rad * math.cos(alpha) - np.array([0, 0, math.sin(alpha)])
            R = np.stack([t, -nrm, s], axis=1)
            walls.append(((seg, 0.006, L), (rmid * math.cos(th), rmid * math.sin(th), zmid), qfrom_mat(R)))
        parts += [Part("box", sz, c, q) for sz, c, q in walls]
        body = k.rigid_body(path, pos, pose.quat, parts=parts, mass=0.45)
        k.cylinder(f"{path}/Floor", self.RB + 0.004, self.FLOOR, (0, 0, self.FLOOR / 2), Q_IDENT, m.get("glass"), m.phys("glass"), collide=True)
        for i, (sz, c, q) in enumerate(walls):
            k.cube(f"{path}/Wall_{i}", sz, c, q, m.get("glass"), m.phys("glass"), collide=True)
        k.tag(body, "fruit_bowl", zone=ctx.zone or "")
        ctx.record(path, "clutter", pos, pose.quat, "dynamic", size=(2 * self.RT, 2 * self.RT, self.H), mass=0.45, friction=0.35, subtype="fruit_bowl")
        bodies = [path]
        r = p.apple_radius
        for i in range(p.apples):
            ang = math.pi / 2 + 2 * math.pi * i / 3
            lp = (0.052 * math.cos(ang), 0.052 * math.sin(ang), self.FLOOR + r + 0.0005)
            wp, _ = pose.world((lp[0], lp[1], lp[2] + EPS))
            ap = f"{path}_Apple_{i}"
            ab = k.rigid_body(ap, wp, Q_IDENT, parts=[Part("sph", (r,))], mass=0.18)
            k.sphere(f"{ap}/Fruit", r, (0, 0, 0), m.get("apple_green"), m.phys("fruit"), collide=True)
            k.cube(f"{ap}/Stem", (0.004, 0.004, 0.014), (0, 0, r + 0.003), Q_IDENT, m.get("brown"))
            k.tag(ab, "apple", zone=ctx.zone or "")
            ctx.record(ap, "clutter", wp, Q_IDENT, "dynamic", size=(2 * r, 2 * r, 2 * r), mass=0.18, friction=0.6, subtype="apple")
            bodies.append(ap)
        return Built(path, bodies=bodies)
