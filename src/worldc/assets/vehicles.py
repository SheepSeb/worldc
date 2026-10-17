"""Real Isaac Sim vehicles that drive along waypoint paths as moving hazards.

``forklift``: NVIDIA's Isaac Sim forklift (``Isaac/Props/Forklift/forklift.usd``), referenced by URL as the visual of a kinematic
body that ``runtime.PathFollower`` drives along the waypoints.  The body's collider is a box of the measured size, so the hazard checks
and the drive logic stay exact; the Isaac asset's own collision / rigid-body prims are switched off so nothing nests or double-collides.
The asset is measured in Isaac Sim: bounds x +-0.607, y +-1.747, z 0..2.153 m, forks toward -y, so it is turned +90 degrees about Z to
make the forks lead (+x is the driving direction).
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from ..geom import Q_IDENT
from .base import Asset, Footprint, PhysicsInfo
from .catalog import catalog_url
from .procedural.agents import PathParams, _author_path_agent
from .registry import register

FORKLIFT_PATH = "Isaac/Props/Forklift/forklift.usd"
FORKLIFT_SIZE = (3.494, 1.214, 2.153)  # length (driving direction), width, height in metres (measured in Isaac Sim)
FORKLIFT_PHYSICS_PRIMS = ("S_ForkliftBody", "S_ForkliftFork/S_ForkliftFork", "S_ForkliftFork/collision_box")


class ForkliftParams(PathParams):
    mass_kg: float = Field(3500.0, ge=500, le=10000)
    path_mode: Literal["loop", "pingpong"] = "loop"


@register
class IsaacForklift(Asset):
    id = "forklift"
    source = "isaac"
    description = ("Real Isaac Sim forklift driving along a waypoint path (loop by default) as a moving hazard: kinematic body with a "
                   "box collider, driven at runtime. 3.5 x 1.2 x 2.2 m. Keep its corridor clear (keep_clear).")
    tags = frozenset({"forklift", "vehicle", "agent", "dynamic_obstacle", "kinematic", "hazard", "warehouse", "isaac"})
    Params = ForkliftParams
    physics = PhysicsInfo("kinematic", "steel", 0.5, 3500.0, "primitives", f"visual referenced from {FORKLIFT_PATH}; pose set every physics step by runtime.PathFollower")
    example = {"id": "forklift", "asset": "forklift", "params": {"waypoints": [[-9, -5], [8, -5], [8, 3], [-9, 3]], "speed_mps": 0.8}}

    def footprint(self, p):
        return Footprint.rect(FORKLIFT_SIZE[0] + 0.2, FORKLIFT_SIZE[1] + 0.1, FORKLIFT_SIZE[2])

    def build(self, ctx, path, pose, p, rng):
        from pxr import Gf, Sdf, UsdGeom

        size = FORKLIFT_SIZE
        z = pose.pos[2] + size[2] / 2 + 0.001
        k = ctx.kit

        def shape():
            box = k.cube(f"{path}/Collider", size, (0, 0, 0), Q_IDENT, ctx.mats.get("steel_bright"), ctx.mats.phys("steel"), collide=True)
            UsdGeom.Imageable(box).CreatePurposeAttr(UsdGeom.Tokens.guide)  # the collider is not drawn over the forklift model
            vis = k.xform(f"{path}/Isaac")
            vis.GetReferences().AddReference(catalog_url(FORKLIFT_PATH))
            xf = UsdGeom.Xformable(vis)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -size[2] / 2 + 0.002))  # bottom of the asset on the floor
            xf.AddRotateZOp().Set(90.0)  # forks (-y in the asset) now lead along +x
            for rel in FORKLIFT_PHYSICS_PRIMS:  # the body's box is the only collider
                over = k.stage.OverridePrim(f"{path}/Isaac/{rel}")
                over.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                over.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)

        return _author_path_agent(ctx, path, p, size, z, p.mass_kg, "forklift", shape)
