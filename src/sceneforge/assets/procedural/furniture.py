"""Furniture / fixtures: packing work table with two prismatic drawers, robot charging station (wall charger + parking
pad).  Ported from ``Structures.work_table`` and the charger part of ``generate.build_robot``."""
from __future__ import annotations

import math

from pydantic import Field

from ...geom import Part, Q_IDENT, qyaw
from ..base import Anchor, Asset, Built, Footprint, JointInfo, Keepout, Params, PhysicsInfo
from ..registry import register


class TableParams(Params):
    drawers: int = Field(2, ge=0, le=2, description="Number of pull-out drawers (prismatic, sliding along local +y)")
    drawer_travel: float = Field(0.35, ge=0.1, le=0.5)


@register
class WorkTable(Asset):
    id = "work_table"
    description = ("Packing work table 1.6 x 0.8 m, top at 0.85 m, under-shelf and back panel, with up to two drawers on "
                   "driven prismatic joints sliding along local +y. Top is a support surface ('top').")
    tags = frozenset({"table", "furniture", "packing", "articulated", "prismatic", "support"})
    Params = TableParams
    physics = PhysicsInfo("articulated", "wood", 0.5, 3.0, "primitives", "static table + fixed-base drawer articulations (3 kg each)")

    def footprint(self, p):
        return Footprint.rect(1.6, 0.8, 0.85)

    def anchors(self, p):
        return [Anchor("top", "support", (0, 0, 0.85), polygon=[(-0.75, -0.35), (0.75, -0.35), (0.75, 0.35), (-0.75, 0.35)]),
                Anchor("front", "port", (0, 1.7, 0), qyaw(-math.pi / 2), port_kind="approach")]

    def joints(self, p):
        return [JointInfo(f"Drawer_{i}/Joint", "prismatic", "Y", 0.0, p.drawer_travel, "m", 0.0,
                          {"stiffness": 3000.0, "damping": 600.0, "max_force": 1e4}, f"Drawer_{i}", "drawer") for i in range(p.drawers)]

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        k.xform(path, pose.pos, pose.quat)
        st = m.phys("steel")
        k.cube(f"{path}/Top", (1.6, 0.8, 0.04), (0, 0, 0.83), Q_IDENT, m.get("table_top"), m.phys("wood"), collide=True)
        for i, (lx, ly) in enumerate([(-0.76, -0.36), (0.76, -0.36), (-0.76, 0.36), (0.76, 0.36)]):
            k.cube(f"{path}/Leg_{i}", (0.06, 0.06, 0.81), (lx, ly, 0.405), Q_IDENT, m.get("conv_frame"), st, collide=True)
        for nm, x in (("PanelL", -0.54), ("PanelM", 0.0), ("PanelR", 0.54)):
            k.cube(f"{path}/{nm}", (0.02, 0.76, 0.26), (x, 0, 0.68), Q_IDENT, m.get("conv_frame"), st, collide=True)
        k.cube(f"{path}/Back", (1.1, 0.02, 0.26), (0, -0.38, 0.68), Q_IDENT, m.get("conv_frame"), st, collide=True)
        k.cube(f"{path}/Shelf", (1.1, 0.76, 0.02), (0, 0, 0.54), Q_IDENT, m.get("conv_frame"), st, collide=True)
        ctx.record(path, "work_table", pose.pos, pose.quat, "static", size=(1.6, 0.8, 0.85), friction=0.5, top_z=pose.pos[2] + 0.85)
        bodies = []
        for di, dx in enumerate((-0.27, 0.27)[: p.drawers]):
            dp = f"{path}/Drawer_{di}"
            dw, dl, dh = 0.5, 0.62, 0.12
            center = (dx, -0.05, 0.58 + dh / 2 + 0.02)
            mass, t = 3.0, 0.015
            parts = [
                Part("box", (dw, dl, t), (0, 0, -dh / 2 + t / 2)),
                Part("box", (t, dl, dh), (-dw / 2 + t / 2, 0, 0)),
                Part("box", (t, dl, dh), (dw / 2 - t / 2, 0, 0)),
                Part("box", (dw - 2 * t, t, dh), (0, -dl / 2 + t / 2, 0)),
                Part("box", (dw, 0.02, dh + 0.04), (0, dl / 2 + 0.01, 0.02)),
            ]
            b = k.rigid_body(dp, center, Q_IDENT, parts=parts, mass=mass, iters=(16, 4), disable_gravity=False)
            for nm, pp in zip(("Bottom", "WallL", "WallR", "Back", "Front"), parts):
                k.cube(f"{dp}/{nm}", pp.size, pp.pos, Q_IDENT, m.get("drawer"), m.phys("wood"), collide=True)
            k.cube(f"{dp}/Handle", (0.18, 0.03, 0.025), (0, dl / 2 + 0.035, 0.03), Q_IDENT, m.get("steel_bright"), st, collide=True)
            bp = f"{path}/DrawerBase_{di}"
            k.fixed_base(bp, pose.pos, pose.quat)
            jp = f"{dp}/Joint"
            k.joint(jp, "prismatic", bp, dp, center, (0, 0, 0), axis="Y", lower=0.0, upper=p.drawer_travel, stiffness=3000.0, damping=600.0, max_force=1e4)
            wp, wq = pose.world(center)
            k.tag(b, "drawer", index=di)
            ctx.record(dp, "drawer", wp, wq, "dynamic", size=(dw, dl + 0.04, dh), mass=mass, friction=0.5,
                       articulation={"type": "prismatic", "root": dp, "joint": jp, "axis_local": "Y", "unit": "m", "limits": [0.0, p.drawer_travel],
                                     "closed_position": 0.0, "open_position": p.drawer_travel,
                                     "drive": {"stiffness": 3000.0, "damping": 600.0, "max_force": 1e4}, "fixed_base": True, "articulation_root": bp})
            ctx.scene["articulations"].append(dict(name=f"{path.rsplit('/', 1)[-1]}_Drawer_{di}", root=bp, link=dp, joint=jp, kind="prismatic", lower=0.0,
                                                   upper=p.drawer_travel, unit="m", parent=path,
                                                   moving_collider_paths=[f"{dp}/{n}" for n in ("Bottom", "WallL", "WallR", "Back", "Front", "Handle")]))
            bodies += [dp, bp]
        return Built(path, [a.to_world(pose) for a in self.anchors(p)], info={"top_z": pose.pos[2] + 0.85}, bodies=bodies)


class ChargerParams(Params):
    pad_offset: float = Field(1.8, ge=0.8, le=5.0, description="Distance from the charger (at the wall) to the parking pad centre (local +y)")
    pad: bool = True


@register
class ChargingStation(Asset):
    id = "charging_station"
    description = ("Mobile-robot charging station: wall-mounted charger cabinet with a green LED (static collider) and a "
                   "translucent parking pad marker in front of it. Port 'parking' = the robot's docking pose (facing the charger).")
    tags = frozenset({"charger", "robot", "parking", "static", "warehouse"})
    Params = ChargerParams
    physics = PhysicsInfo("static", "steel", 0.6, None, "primitives")

    def footprint(self, p):
        return Footprint.rect(0.5, 0.3, 1.0)

    def keepouts(self, p):
        y = p.pad_offset
        return [Keepout([(-0.9, y - 1.1), (0.9, y - 1.1), (0.9, y + 1.1), (-0.9, y + 1.1)], "parking")]

    def anchors(self, p):
        return [Anchor("parking", "port", (0, p.pad_offset, 0), qyaw(-math.pi / 2), port_kind="parking")]

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        k.xform(path, pose.pos, pose.quat)
        k.cube(f"{path}/Body", (0.5, 0.3, 1.0), (0, 0, 0.5), Q_IDENT, m.get("charger"), m.phys("steel"), collide=True)
        k.cube(f"{path}/LED", (0.06, 0.02, 0.06), (0, 0.16, 0.75), Q_IDENT, m.get("charger_led"))
        if p.pad:
            k.cube(f"{path}/Pad", (1.2, 1.6, 0.01), (0, p.pad_offset, 0.005), Q_IDENT, m.get("robot_marker"))
        wp, wq = pose.world((0, 0, 0.5))
        ctx.record(path, "charger", wp, wq, "static", size=(0.5, 0.3, 1.0), friction=0.6)
        return Built(path, [a.to_world(pose) for a in self.anchors(p)], info={})
