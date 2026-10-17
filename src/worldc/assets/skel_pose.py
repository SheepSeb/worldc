"""Pose a referenced Isaac Sim character by authoring a UsdSkel animation over its skeleton.  Dependency-free (pxr only) so it also runs
under Isaac's python.  A pose is a dict {joint short name: (axis, degrees)}: the joint's rest rotation, then a rotation about its own
local axis, replacing the joint's local transform.  Other joints keep their rest pose."""
from __future__ import annotations

import math

from pxr import Gf, Sdf, UsdSkel, Vt


def _delta(axis: str, deg: float) -> Gf.Quatf:
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return Gf.Quatf(math.cos(h), *(math.sin(h) * c for c in v))


def author_pose(stage, skeleton_path: str, joints: dict, pose: dict, order: str = "rest_then_delta", name: str = "Pose"):
    """``joints``: {short: {"path": joint token, "t": [x, y, z], "q": [w, x, y, z]}} (rest local transforms, see people.CHARACTERS)."""
    skel = stage.OverridePrim(skeleton_path)
    anim_path = f"{skeleton_path}/{name}"
    anim = UsdSkel.Animation.Define(stage, anim_path)
    names = [n for n in pose if n in joints]
    anim.CreateJointsAttr(Vt.TokenArray([joints[n]["path"] for n in names]))
    anim.CreateTranslationsAttr(Vt.Vec3fArray([Gf.Vec3f(*joints[n]["t"]) for n in names]))
    quats = []
    for n in names:
        rest = Gf.Quatf(*joints[n]["q"])
        d = _delta(*pose[n])
        quats.append((rest * d) if order == "rest_then_delta" else (d * rest))
    anim.CreateRotationsAttr(Vt.QuatfArray(quats))
    anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(names)))
    UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([Sdf.Path(anim_path)])
    return anim


def author_walk(stage, skeleton_path: str, joints: dict, base: dict, swing: dict, period_s: float, total_s: float, tcps: float,
                phase: float = 0.0, name: str = "Walk", steps_per_cycle: int = 12):
    """A looping gait as time samples: ``base`` is a fixed pose (arms down), ``swing`` = {joint: (axis, peak degrees)} oscillating
    sinusoidally with period ``period_s`` for ``total_s`` seconds (time codes at ``tcps`` per second)."""
    skel = stage.OverridePrim(skeleton_path)
    anim_path = f"{skeleton_path}/{name}"
    anim = UsdSkel.Animation.Define(stage, anim_path)
    names = sorted(n for n in set(base) | set(swing) if n in joints)
    anim.CreateJointsAttr(Vt.TokenArray([joints[n]["path"] for n in names]))
    anim.CreateTranslationsAttr(Vt.Vec3fArray([Gf.Vec3f(*joints[n]["t"]) for n in names]))
    anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(names)))
    rot = anim.CreateRotationsAttr()
    dt_code = period_s * tcps / steps_per_cycle
    for k in range(int(total_s / period_s * steps_per_cycle) + 1):
        ang = 2 * math.pi * k / steps_per_cycle + phase
        quats = []
        for n in names:
            q = Gf.Quatf(*joints[n]["q"])
            if n in base:
                q = q * _delta(*base[n])
            if n in swing:
                ax, deg = swing[n]
                q = q * _delta(ax, deg * math.sin(ang))
            quats.append(q)
        rot.Set(Vt.QuatfArray(quats), k * dt_code)
    UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([Sdf.Path(anim_path)])
    return anim
