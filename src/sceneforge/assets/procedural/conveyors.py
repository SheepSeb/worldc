"""Conveyors: kinematic belt / roller beds driven by PhysX surface velocity.  Ported from ``Structures.belt`` /
``roller_conveyor``.  Local frame: transport direction is local +y, centred on the bed, z = 0 floor."""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf
from pydantic import Field

from ...geom import Part, Q_IDENT, qaxis, qmat, qyaw
from ...usdkit import T
from ..base import Anchor, Asset, Built, Footprint, Params, PhysicsInfo
from ..registry import register


class ConveyorParams(Params):
    length: float = Field(4.0, ge=0.5, le=30, description="Bed length along the transport direction (m)")
    width: float = Field(0.6, ge=0.2, le=2.0)
    top_z: float = Field(0.8, ge=0.2, le=1.5, description="Transport surface height (m)")
    speed: float = Field(0.5, ge=-3.0, le=3.0, description="Surface speed along local +y (m/s); negative reverses")
    end_stop: bool = Field(True, description="Yellow stop bar at the +y end")


def _surface_velocity(k, body, speed):
    k.physx(body, "PhysxSurfaceVelocityAPI", **{
        "physxSurfaceVelocity:surfaceVelocityEnabled": (T.Bool, True),
        "physxSurfaceVelocity:surfaceVelocity": (T.Vector3f, Gf.Vec3f(0.0, float(speed), 0.0)),
        "physxSurfaceVelocity:surfaceVelocityLocalSpace": (T.Bool, True),
    })


def _surface_anchor(p):
    hx, hy = p.width / 2 - 0.02, p.length / 2 - 0.1
    return Anchor("surface", "support", (0, 0, p.top_z), polygon=[(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)],
                  meta={"moving": True, "speed_mps": p.speed})


class _Conveyor(Asset):
    Params = ConveyorParams
    kind = "belt"

    def footprint(self, p):
        return Footprint.rect(p.width + 0.1, p.length, p.top_z + 0.16)

    def anchors(self, p):
        return [_surface_anchor(p), Anchor("infeed", "port", (0, -p.length / 2 - 1.1, 0), qyaw(math.pi / 2), port_kind="approach"),
                Anchor("outfeed", "port", (0, p.length / 2 + 1.0, 0), qyaw(-math.pi / 2), port_kind="approach")]

    def _register(self, ctx, path, pose, p, body_path, built_anchors, extra):
        R = qmat(pose.quat)
        d = R @ np.array([0.0, 1.0, 0.0])
        lo = pose.world((-p.width / 2, -p.length / 2, 0))[0]
        hi = pose.world((p.width / 2, p.length / 2, 0))[0]
        info = {"name": path.rsplit("/", 1)[-1], "kind": self.kind, "belt_path": body_path, "speed_mps": p.speed,
                "direction_world": [round(float(c), 4) for c in d], "bbox": [min(lo[0], hi[0]), min(lo[1], hi[1]), max(lo[0], hi[0]), max(lo[1], hi[1])],
                "top_z": p.top_z + pose.pos[2]}
        ctx.scene["conveyors"].append(info)
        return Built(path, built_anchors, info=info | extra, bodies=[body_path])


@register
class ConveyorBelt(_Conveyor):
    id = "conveyor_belt"
    description = "Flat belt conveyor (kinematic belt with PhysX surface velocity), side rails, frame, legs, end stop."
    tags = frozenset({"conveyor", "belt", "kinematic", "packing", "support", "warehouse"})
    physics = PhysicsInfo("kinematic", "rubber", 0.9, 40.0, "primitives", "surface velocity drives payloads")
    kind = "belt"

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        k.xform(path, pose.pos, pose.quat)
        L, width, top_z = p.length, p.width, p.top_z
        bp = f"{path}/Belt"
        belt_t = 0.04
        body = k.rigid_body(bp, (0, 0, top_z - belt_t / 2), Q_IDENT, parts=[Part("box", (width, L, belt_t))], mass=40.0, kinematic=True)
        k.cube(f"{bp}/Surface", (width, L, belt_t), (0, 0, 0), Q_IDENT, m.get("belt"), m.phys("rubber"), collide=True)
        _surface_velocity(k, body, p.speed)
        st = m.phys("steel")
        for sx in (-1, 1):
            k.cube(f"{path}/Rail_{'L' if sx < 0 else 'R'}", (0.03, L, 0.12), (sx * (width / 2 + 0.015), 0, top_z + 0.02), Q_IDENT, m.get("conv_frame"), st, collide=True)
        if p.end_stop:
            k.cube(f"{path}/EndStop", (width + 0.06, 0.03, 0.16), (0, L / 2 - 0.015, top_z + 0.08), Q_IDENT, m.get("safety_yellow"), st, collide=True)
        k.cube(f"{path}/Frame", (width + 0.04, L, 0.06), (0, 0, top_z - belt_t - 0.03), Q_IDENT, m.get("conv_frame"), st, collide=True)
        leg_h = top_z - belt_t - 0.06
        for i, (lx, ly) in enumerate([(-1, -L / 2 + 0.2), (1, -L / 2 + 0.2), (-1, L * 0.45), (1, L * 0.45)]):
            k.cube(f"{path}/Leg_{i}", (0.05, 0.05, leg_h), (lx * (width / 2 - 0.02), ly, leg_h / 2), Q_IDENT, m.get("conv_frame"), st, collide=True)
        k.tag(body, "conveyor_belt", speed=float(p.speed))
        wp, wq = pose.world((0, 0, top_z - belt_t / 2))
        ctx.record(bp, "conveyor_belt", wp, wq, "kinematic", size=(width, L, belt_t), mass=40.0, friction=0.9,
                   surface_velocity_mps=[0.0, p.speed, 0.0])
        return self._register(ctx, path, pose, p, bp, [a.to_world(pose) for a in self.anchors(p)], {})


@register
class ConveyorRoller(_Conveyor):
    id = "conveyor_roller"
    description = "Roller conveyor: one kinematic body of 6 cm rollers every ~10 cm with PhysX surface velocity, rails, legs, end stop."
    tags = frozenset({"conveyor", "roller", "kinematic", "packing", "support", "warehouse"})
    physics = PhysicsInfo("kinematic", "steel", 0.6, 60.0, "primitives")
    kind = "roller"

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        k.xform(path, pose.pos, pose.quat)
        L, width, top_z = p.length, p.width, p.top_z
        rp = f"{path}/Rollers"
        n = int(L / 0.1)
        r = 0.03
        parts, offs = [], []
        for i in range(n):
            ly = -L / 2 + 0.05 + i * (L - 0.1) / (n - 1)
            parts.append(Part("cyl", (r, width - 0.02), (0, ly, 0), qaxis((0, 1, 0), math.pi / 2)))
            offs.append(ly)
        body = k.rigid_body(rp, (0, 0, top_z - r), Q_IDENT, parts=parts, mass=60.0, kinematic=True)
        for i, ly in enumerate(offs):
            k.cylinder(f"{rp}/Roller_{i:02d}", r, width - 0.02, (0, ly, 0), qaxis((0, 1, 0), math.pi / 2), m.get("roller"), m.phys("steel"), collide=True)
        _surface_velocity(k, body, p.speed)
        st = m.phys("steel")
        for sx in (-1, 1):
            k.cube(f"{path}/Rail_{'L' if sx < 0 else 'R'}", (0.04, L, 0.1), (sx * (width / 2 + 0.02), 0, top_z - 0.02), Q_IDENT, m.get("conv_frame"), st, collide=True)
        if p.end_stop:
            k.cube(f"{path}/EndStop", (width + 0.08, 0.03, 0.16), (0, L / 2 - 0.015, top_z + 0.08), Q_IDENT, m.get("safety_yellow"), st, collide=True)
        for i, (lx, ly) in enumerate([(-1, -L / 2 + 0.2), (1, -L / 2 + 0.2), (-1, L / 2 - 0.2), (1, L / 2 - 0.2)]):
            k.cube(f"{path}/Leg_{i}", (0.05, 0.05, top_z - 0.08), (lx * (width / 2), ly, (top_z - 0.08) / 2), Q_IDENT, m.get("conv_frame"), st, collide=True)
        k.tag(body, "conveyor_roller", speed=float(p.speed))
        wp, wq = pose.world((0, 0, top_z - r))
        ctx.record(rp, "conveyor_roller", wp, wq, "kinematic", size=(width, L, 2 * r), mass=60.0, friction=0.6,
                   surface_velocity_mps=[0.0, p.speed, 0.0], n_rollers=n)
        return self._register(ctx, path, pose, p, rp, [a.to_world(pose) for a in self.anchors(p)], {"n_rollers": n})
