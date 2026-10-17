"""Small floor clutter: traffic cone, wheelie/trash bin.  Ported from ``Factory.cone`` / ``trash_bin``."""
from __future__ import annotations

from pxr import UsdPhysics
from pydantic import Field

from ...geom import Part, Q_IDENT
from ..base import Asset, Built, Footprint, Params, PhysicsInfo
from ..registry import register

EPS = 0.0008


@register
class TrafficCone(Asset):
    id = "traffic_cone"
    description = "Orange traffic cone (0.73 m) on a square rubber base; dynamic, base box + cylinder collider."
    tags = frozenset({"cone", "clutter", "safety", "dynamic", "obstacle"})
    physics = PhysicsInfo("dynamic", "rubber", 0.9, 1.4, "primitives")

    def footprint(self, p):
        return Footprint.rect(0.38, 0.38, 0.73)

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        h, r, base, mass = 0.7, 0.14, 0.38, 1.4
        parts = [Part("box", (base, base, 0.03), (0, 0, 0.015)), Part("cyl", (r * 0.55, h), (0, 0, 0.03 + h / 2))]
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + EPS)
        body = k.rigid_body(path, pos, pose.quat, parts=parts, mass=mass)
        k.cube(f"{path}/Base", (base, base, 0.03), (0, 0, 0.015), Q_IDENT, m.get("cone_base"), m.phys("rubber"), collide=True)
        cone = k.cone(f"{path}/Cone", r, h, (0, 0, 0.03 + h / 2), Q_IDENT, m.get("cone"))
        UsdPhysics.CollisionAPI.Apply(cone)
        k.bind(cone, m.phys("rubber"), "physics")
        k.tag(body, "cone", zone=ctx.zone or "")
        ctx.record(path, "clutter", pos, pose.quat, "dynamic", size=(base, base, h + 0.03), mass=mass, friction=0.9, subtype="traffic_cone")
        return Built(path, bodies=[path])


class BinParams(Params):
    radius: float = Field(0.28, ge=0.1, le=0.6)
    height: float = Field(0.8, ge=0.3, le=1.5)
    mass_kg: float = Field(6.5, ge=0.5, le=50)


@register
class TrashBin(Asset):
    id = "trash_bin"
    description = "Cylindrical green trash bin with dark lid disc; dynamic, single cylinder collider."
    tags = frozenset({"bin", "trash", "clutter", "dynamic", "plastic", "container"})
    Params = BinParams
    physics = PhysicsInfo("dynamic", "plastic", 0.4, 6.5, "primitives")

    def footprint(self, p):
        return Footprint.circle(p.radius, p.height)

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        r, h = p.radius, p.height
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + EPS)
        body = k.rigid_body(path, pos, pose.quat, parts=[Part("cyl", (r, h), (0, 0, h / 2))], mass=p.mass_kg)
        k.cylinder(f"{path}/Shape", r, h, (0, 0, h / 2), Q_IDENT, m.get("bin_green"), m.phys("plastic"), collide=True)
        k.cylinder(f"{path}/LidDisc", r * 0.92, 0.01, (0, 0, h + 0.002), Q_IDENT, m.get("bin_dark"))
        k.tag(body, "trash_bin", zone=ctx.zone or "")
        ctx.record(path, "clutter", pos, pose.quat, "dynamic", size=(2 * r, 2 * r, h), mass=p.mass_kg, friction=0.4, subtype="trash_bin")
        return Built(path, bodies=[path])
