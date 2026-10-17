"""Real Isaac Sim people that walk along waypoint paths as moving hazards.

``human``: one of NVIDIA's Isaac Sim characters (``Isaac/People/Characters/...``) referenced by URL as the visual of a kinematic body that
``runtime.PathFollower`` drives along the waypoints (loop or ping-pong).  The body's collider is a capsule of the character's size, so the
hazard checks and the drive logic stay exact.  The characters are authored in a T-pose and face -y, so they are turned +90 degrees about Z
to face the walking direction (+x).  Feet are at the asset origin (measured in Isaac Sim).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..geom import Q_IDENT
from .base import Asset, Footprint, PhysicsInfo
from .catalog import catalog_url
from .procedural.agents import PathParams, _author_path_agent
from .registry import register

BASE = "Isaac/People/Characters"
# folder name -> {height_m, skeleton, joints}: measured in Isaac Sim (feet at the origin; rest transforms of the arm and leg joints)
TABLE = Path(__file__).parent / "library" / "people" / "isaac_5.1_characters.json"
CHARACTERS = json.loads(TABLE.read_text())
RADIUS = 0.26
STRIDE_M = 1.1  # one full gait cycle (two steps) covers this distance
ARMS_DOWN = {"L_Upperarm": ("z", -75.0), "R_Upperarm": ("z", 75.0)}  # the characters are authored in a T-pose
GAIT_S = 600.0  # seconds of gait animation authored (the timeline is extended to this)


class HumanParams(PathParams):
    character: Literal[tuple(CHARACTERS)] = Field("male_adult_construction_05_new", description="Which Isaac Sim character (hi-vis workers, an officer)")  # type: ignore[valid-type]
    mass_kg: float = Field(80.0, ge=30, le=150)
    speed_mps: float = Field(0.9, gt=0, le=3)


@register
class IsaacHuman(Asset):
    id = "human"
    source = "isaac"
    description = ("Real Isaac Sim person (hi-vis construction worker or officer; T-pose model) walking along a waypoint path (loop or ping-pong) as a "
                   "moving hazard: kinematic capsule driven at runtime. Keep its corridor clear (keep_clear).")
    tags = frozenset({"human", "agent", "dynamic_obstacle", "kinematic", "hazard", "isaac"})
    Params = HumanParams
    physics = PhysicsInfo("kinematic", "plastic", 0.5, 80.0, "primitives", "visual referenced from Isaac/People/Characters; pose set every physics step by runtime.PathFollower")
    example = {"id": "worker", "asset": "human", "params": {"character": "male_adult_construction_05_new", "waypoints": [[-8, 6], [8, 6]], "path_mode": "pingpong"}}

    def footprint(self, p):
        return Footprint.circle(RADIUS, CHARACTERS[p.character]["height_m"])

    def build(self, ctx, path, pose, p, rng):
        from pxr import Gf, UsdGeom

        info = CHARACTERS[p.character]
        h = info["height_m"]
        z = pose.pos[2] + h / 2 + 0.001
        k = ctx.kit

        def shape():
            cap = k.capsule(f"{path}/Collider", RADIUS, h - 2 * RADIUS, (0, 0, 0), Q_IDENT, ctx.mats.get("human_a"), ctx.mats.phys("plastic"), collide=True)
            UsdGeom.Imageable(cap).CreatePurposeAttr(UsdGeom.Tokens.guide)  # not drawn over the character
            vis = k.xform(f"{path}/Isaac")
            vis.GetReferences().AddReference(catalog_url(f"{BASE}/{p.character}/{p.character}.usd"))
            xf = UsdGeom.Xformable(vis)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -h / 2))  # feet on the floor
            xf.AddRotateZOp().Set(90.0)  # the character faces -y; now it faces +x, the walking direction

        def pose():
            """Arms down and a walking gait (thigh swing) as time samples; the timeline is extended so Isaac Sim plays it."""
            from .skel_pose import author_walk

            tcps = k.stage.GetTimeCodesPerSecond() or 24.0
            author_walk(k.stage, f"{path}/Isaac/{info['skeleton']}", info["joints"], ARMS_DOWN, {"L_Thigh": ("x", 28.0), "R_Thigh": ("x", -28.0)},
                        period_s=STRIDE_M / max(p.speed_mps, 0.2), total_s=GAIT_S, tcps=tcps, phase=float(rng.uniform(0, 2 * math.pi)))
            ctx.scene["animation_end_time_code"] = max(float(ctx.scene.get("animation_end_time_code", 0.0)), GAIT_S * tcps)

        built = _author_path_agent(ctx, path, p, (2 * RADIUS, 2 * RADIUS, h), z, p.mass_kg, "human", shape)
        pose()
        return built
