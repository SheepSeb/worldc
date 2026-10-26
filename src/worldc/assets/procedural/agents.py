"""Moving-agent proxies (kinematic, driven at runtime by ``runtime.PathFollower`` from the waypoints authored here) and
the visual-only robot placeholder.  Ported from ``generate.build_hazards`` / ``build_robot``.

Waypoints are world-frame [x, y]; the instance pose only supplies the start (first waypoint) and z.
"""
from __future__ import annotations

from typing import Literal

import math

import numpy as np
from pxr import Gf, UsdGeom, Vt
from pydantic import Field, model_validator

from ...geom import Part, Q_IDENT, qfrom_mat
from ...runtime import PathFollower
from ...usdkit import T
from ..base import Asset, Built, Footprint, Params, PhysicsInfo
from ..registry import register


class PathParams(Params):
    waypoints: list[tuple[float, float]] = Field(default_factory=lambda: [(0.0, 0.0), (5.0, 0.0)], min_length=2,
                                                 description="World-frame [x, y] path vertices")
    speed_mps: float = Field(0.8, gt=0, le=5)
    path_mode: Literal["loop", "pingpong"] = "pingpong"
    path_phase_m: float = Field(0.0, ge=0, description="Start offset along the path (m)")
    dwell_s: float = Field(0.0, ge=0, le=60, description="Ping-pong only: seconds the agent stands still at each end of the path")
    ease_s: float = Field(0.0, ge=0, le=5, description="Ping-pong only: seconds to accelerate / brake at the ends of a leg")
    turn_s: float = Field(0.0, ge=0, le=10, description="Ping-pong only: seconds the turn-around takes (the last part of the pause, so <= dwell_s)")

    @model_validator(mode="after")
    def _turn_fits_pause(self):
        if self.turn_s > self.dwell_s:
            raise ValueError("turn_s must not exceed dwell_s (the turn happens at the end of the pause)")
        return self


MOTION_S = 600.0  # seconds of path motion authored as time samples (the timeline is extended to this)
MOTION_HZ = 20  # time samples per second (positions and headings are interpolated between them)


def follower_of(p) -> PathFollower:
    return PathFollower(p.waypoints, p.speed_mps, p.path_mode, 0.0, p.path_phase_m, dwell_s=p.dwell_s, ease_s=p.ease_s, turn_s=p.turn_s)


def _author_motion(ctx, body, p, z):
    """Animate the body along its path in the USD itself (translation and heading as time samples, one period repeated), so opening the
    scene in Isaac Sim and pressing play moves the agent; ``runtime.PathFollower`` computes the same poses for the drive flows."""
    f = follower_of(p)
    xf = UsdGeom.Xformable(body)
    ops = xf.GetOrderedXformOps()
    tr = next(o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
    ori = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeOrient), None) or xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    ori.Set(Gf.Quatd(1, 0, 0, 0) if ori.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf(1, 0, 0, 0))  # default value: the start pose
    tcps = ctx.kit.stage.GetTimeCodesPerSecond() or 24.0
    n = max(8, round(f.period_s * MOTION_HZ))
    step = f.period_s / n * tcps
    dbl = ori.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
    cycle = []
    for i in range(n):
        x, y, _, yaw = f.pose(i * f.period_s / n)
        q = Gf.Quatd(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        cycle.append((Gf.Vec3d(x, y, z), q if dbl else Gf.Quatf(q)))
    for k in range(int(MOTION_S * tcps / step) + 1):
        pos, q = cycle[k % n]
        tr.Set(pos, k * step)
        ori.Set(q, k * step)
    ctx.scene["animation_end_time_code"] = max(float(ctx.scene.get("animation_end_time_code", 0.0)), MOTION_S * tcps)


def _author_path_agent(ctx, path, p, size, z, mass, kind, shape_fn):
    k = ctx.kit
    x0, y0 = p.waypoints[0]
    pos = (float(x0), float(y0), z)
    body = k.rigid_body(path, pos, Q_IDENT, parts=[Part("box", size)], mass=mass, kinematic=True)
    shape_fn()
    _author_motion(ctx, body, p, z)
    ns = k.ns
    pts3 = [(float(x), float(y), float(z)) for x, y in p.waypoints]
    k.attr(body, f"{ns}:waypoints", T.Double3Array, Vt.Vec3dArray([Gf.Vec3d(*q) for q in pts3]))
    k.attr(body, f"{ns}:speed_mps", T.Double, float(p.speed_mps))
    k.attr(body, f"{ns}:path_mode", T.String, p.path_mode)
    k.attr(body, f"{ns}:path_phase_m", T.Double, float(p.path_phase_m))
    k.tag(body, "hazard", kind=kind)
    ctx.record(path, "hazard", pos, Q_IDENT, "kinematic", size=size, mass=mass, friction=0.5, subtype=kind,
               waypoints=[[round(x, 3), round(y, 3)] for x, y in p.waypoints], z=z, speed_mps=p.speed_mps, path_mode=p.path_mode,
               path_phase_m=p.path_phase_m, dwell_s=p.dwell_s, ease_s=p.ease_s, turn_s=p.turn_s, driven_by="runtime.PathFollower (set kinematic pose each physics step)")
    hz = {"path": path, "kind": kind, "waypoints": [[round(x, 3), round(y, 3)] for x, y in p.waypoints], "z": z,
          "speed_mps": p.speed_mps, "path_mode": p.path_mode, "path_phase_m": p.path_phase_m,
          "dwell_s": p.dwell_s, "ease_s": p.ease_s, "turn_s": p.turn_s}
    ctx.scene.setdefault("hazards", []).append(hz)
    return Built(path, info=hz, bodies=[path])


def look_at_quat(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    if abs(f[2]) > 0.99:
        up = (0, 1, 0)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return qfrom_mat(np.stack([r, u, -f], axis=1))
