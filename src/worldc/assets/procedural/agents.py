"""Moving-agent proxies (kinematic, driven at runtime by ``runtime.PathFollower`` from the waypoints authored here) and
the visual-only robot placeholder.  Ported from ``generate.build_hazards`` / ``build_robot``.

Waypoints are world-frame [x, y]; the instance pose only supplies the start (first waypoint) and z.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pxr import Gf, UsdGeom, Vt
from pydantic import Field

from ...geom import Part, Q_IDENT, qfrom_mat
from ...usdkit import T
from ..base import Asset, Built, Footprint, Params, PhysicsInfo
from ..registry import register


class PathParams(Params):
    waypoints: list[tuple[float, float]] = Field(default_factory=lambda: [(0.0, 0.0), (5.0, 0.0)], min_length=2,
                                                 description="World-frame [x, y] path vertices")
    speed_mps: float = Field(0.8, gt=0, le=5)
    path_mode: Literal["loop", "pingpong"] = "pingpong"
    path_phase_m: float = Field(0.0, ge=0, description="Start offset along the path (m)")


def _author_path_agent(ctx, path, p, size, z, mass, kind, shape_fn):
    k = ctx.kit
    x0, y0 = p.waypoints[0]
    pos = (float(x0), float(y0), z)
    body = k.rigid_body(path, pos, Q_IDENT, parts=[Part("box", size)], mass=mass, kinematic=True)
    shape_fn()
    ns = k.ns
    pts3 = [(float(x), float(y), float(z)) for x, y in p.waypoints]
    k.attr(body, f"{ns}:waypoints", T.Double3Array, Vt.Vec3dArray([Gf.Vec3d(*q) for q in pts3]))
    k.attr(body, f"{ns}:speed_mps", T.Double, float(p.speed_mps))
    k.attr(body, f"{ns}:path_mode", T.String, p.path_mode)
    k.attr(body, f"{ns}:path_phase_m", T.Double, float(p.path_phase_m))
    k.tag(body, "hazard", kind=kind)
    ctx.record(path, "hazard", pos, Q_IDENT, "kinematic", size=size, mass=mass, friction=0.5, subtype=kind,
               waypoints=[[round(x, 3), round(y, 3)] for x, y in p.waypoints], z=z, speed_mps=p.speed_mps, path_mode=p.path_mode,
               path_phase_m=p.path_phase_m, driven_by="runtime.PathFollower (set kinematic pose each physics step)")
    hz = {"path": path, "kind": kind, "waypoints": [[round(x, 3), round(y, 3)] for x, y in p.waypoints], "z": z,
          "speed_mps": p.speed_mps, "path_mode": p.path_mode, "path_phase_m": p.path_phase_m}
    ctx.scene.setdefault("hazards", []).append(hz)
    return Built(path, info=hz, bodies=[path])


class HumanParams(PathParams):
    height: float = Field(1.7, ge=1.2, le=2.1)
    radius: float = Field(0.22, ge=0.15, le=0.35)
    mass_kg: float = Field(80.0, ge=30, le=150)
    outfit: Literal["vest", "blue"] = "vest"


@register
class HumanProxy(Asset):
    id = "human_proxy"
    description = ("Walking person proxy: kinematic capsule following a waypoint path (loop or ping-pong) at runtime. "
                   "Its corridor should be kept clear (keep_clear).")
    tags = frozenset({"human", "agent", "dynamic_obstacle", "kinematic", "hazard"})
    Params = HumanParams
    physics = PhysicsInfo("kinematic", "plastic", 0.5, 80.0, "primitives", "pose set every physics step by runtime.PathFollower")

    def footprint(self, p):
        return Footprint.circle(p.radius, p.height)

    def build(self, ctx, path, pose, p, rng):
        size = (2 * p.radius, 2 * p.radius, p.height)
        z = pose.pos[2] + p.height / 2 + 0.001
        mat = ctx.mats.get("human_a" if p.outfit == "vest" else "human_b")
        r, h = p.radius, p.height - 2 * p.radius
        return _author_path_agent(ctx, path, p, size, z, p.mass_kg, "human_capsule",
                                  lambda: ctx.kit.capsule(f"{path}/Shape", r, h, (0, 0, 0), Q_IDENT, mat, ctx.mats.phys("plastic"), collide=True))


class ForkliftParams(PathParams):
    size: tuple[float, float, float] = Field((2.6, 1.2, 2.1), description="Body box length, width, height (m)")
    mass_kg: float = Field(3500.0, ge=500, le=10000)
    path_mode: Literal["loop", "pingpong"] = "loop"


@register
class ForkliftProxy(Asset):
    id = "forklift_proxy"
    description = "Driving forklift proxy: kinematic box + forks following a waypoint path at runtime (loop by default)."
    tags = frozenset({"forklift", "vehicle", "agent", "dynamic_obstacle", "kinematic", "hazard", "warehouse"})
    Params = ForkliftParams
    physics = PhysicsInfo("kinematic", "steel", 0.5, 3500.0, "primitives")

    def footprint(self, p):
        return Footprint.rect(p.size[0] + 0.9, p.size[1], p.size[2])

    def build(self, ctx, path, pose, p, rng):
        size = tuple(p.size)
        z = pose.pos[2] + size[2] / 2 + 0.001
        k, m = ctx.kit, ctx.mats

        def shape():
            k.cube(f"{path}/Shape", size, (0, 0, 0), Q_IDENT, m.get("forklift"), m.phys("steel"), collide=True)
            k.cube(f"{path}/Forks", (1.0, 1.0, 0.08), (size[0] / 2 + 0.45, 0, -size[2] / 2 + 0.12), Q_IDENT, m.get("steel_bright"))

        return _author_path_agent(ctx, path, p, size, z, p.mass_kg, "forklift_box", shape)


def look_at_quat(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    if abs(f[2]) > 0.99:
        up = (0, 1, 0)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return qfrom_mat(np.stack([r, u, -f], axis=1))


class RobotParams(Params):
    footprint_radius: float = Field(0.38, ge=0.1, le=1.5)
    camera_offset: tuple[float, float, float] = Field((0.2, 0.0, 1.5), description="Head camera position in the robot frame")
    camera_focal_mm: float = Field(15.0, ge=5, le=100)


@register
class RobotPlaceholder(Asset):
    id = "robot_placeholder"
    description = ("Visual-only mobile-manipulator placeholder (no physics) marking the robot spawn pose, with a head "
                   "camera. Replace with the real robot USD / ArticulationCfg at runtime.")
    tags = frozenset({"robot", "spawn", "visual"})
    Params = RobotParams
    physics = PhysicsInfo("visual", "steel", None, None, "none")

    def footprint(self, p):
        return Footprint.circle(p.footprint_radius, 1.3)

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        k.xform(path, pose.pos, pose.quat)
        k.tag(k.stage.GetPrimAtPath(path), "robot_spawn", placeholder=True, footprint_radius_m=p.footprint_radius)
        k.cylinder(f"{path}/Base", p.footprint_radius, 0.35, (0, 0, 0.2), Q_IDENT, m.get("robot_body"))
        k.cube(f"{path}/Torso", (0.22, 0.3, 1.35), (-0.08, 0, 0.875), Q_IDENT, m.get("robot_body"))
        k.cube(f"{path}/ArmLink", (0.5, 0.08, 0.08), (0.17, 0.0, 1.15), Q_IDENT, m.get("robot_arm"))
        k.cube(f"{path}/Gripper", (0.1, 0.18, 0.06), (0.45, 0.0, 1.15), Q_IDENT, m.get("robot_arm"))
        eye = tuple(p.camera_offset)
        cam = UsdGeom.Camera.Define(k.stage, f"{path}/RobotCamera")
        cam.CreateFocalLengthAttr(float(p.camera_focal_mm))
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(11.79)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
        k.set_pose(cam.GetPrim(), eye, look_at_quat(eye, (eye[0] + 2.8, eye[1], eye[2] - 0.3)))
        ctx.record(path, "robot_spawn", pose.pos, pose.quat, "none", size=(2 * p.footprint_radius,) * 2 + (1.3,), placeholder=True,
                   note="visual-only placeholder (no physics); replace with the real robot USD / ArticulationCfg",
                   footprint_radius_m=p.footprint_radius)
        return Built(path, info={"camera": f"{path}/RobotCamera", "prim": path})
