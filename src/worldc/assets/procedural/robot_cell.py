"""Static robot work cell: an arm on a pedestal plus a fenced (or keep-out) safety footprint, e.g. a disassembly cell.

The arm is a non-driven visual referenced by URL: the Isaac Sim UR10 (its own ``Physics = None`` variant) or Franka Panda,
or a SimReady FANUC CRX-10iA / CRX-20iA/L (curated table, pinned HF revision).  Arms without a physics-free variant get
their rigid bodies, joints and articulation root switched off by overrides.  The scene's mobile ``robot`` stays the only
driven robot.  UR10 stands on NVIDIA's ``ur10_mount`` (``Isaac/Props/Mounts/ur10_mount.usd``, 0.515 m, top at its origin) by
default; every other arm on a procedural steel column.

Fence (``fence_style``): ``panels`` = SimReady ``safety_fence_panel`` sections (1.07 m) stacked ``fence_layers`` high (default
2, ~2.1 m) in a pinwheel square (each side runs from one corner to the next minus the panel depth, so corners close without
overlaps), one panel column left out on the front (-y) as the gate opening; ``cage`` = the procedural wire-mesh cage with a sliding gate.  The safeguarded square (half-size >= reach
+ ``safety_margin``) is recorded as ``safety_polygon`` on the ``robot_cell`` scene-graph record, which ``check_scene``'s
mandatory robot-cell clearance reads; unfenced cells make it a keep-out the solver keeps free.  Children placed on the
cell's ``inside`` anchor (fixtures, workpieces) count as part of the cell.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from ...geom import Q_IDENT, qyaw
from ..base import Anchor, Asset, Built, Footprint, Keepout, Params, PhysicsInfo, Pose
from ..registry import get, register

UR10_MOUNT = "Isaac/Props/Mounts/ur10_mount.usd"
UR10_MOUNT_SIZE = (0.27, 0.27, 0.515)
ISAAC_ARMS = {  # alias -> (Isaac path, nominal reach m); measured bounds come from the robot table
    "ur10": ("Isaac/Robots/UniversalRobots/ur10/ur10.usd", 1.30),
    "franka": ("Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", 0.855),
}
SIMREADY_ARMS = ("fanuc_crx10ia", "fanuc_crx20ia_l")  # curated table entries (reach + physics prims baked there)
COLUMN_HEIGHT = 0.75
# Franka has no physics variant: these prims (relative to its default prim) carry rigid bodies / joints and are switched off
FRANKA_LINKS = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7",
                "panda_hand", "panda_leftfinger", "panda_rightfinger"]
FRANKA_JOINTS = ["rootJoint", "panda_link0/panda_joint1", "panda_link1/panda_joint2", "panda_link2/panda_joint3", "panda_link3/panda_joint4",
                 "panda_link4/panda_joint5", "panda_link5/panda_joint6", "panda_link6/panda_joint7", "panda_link7/panda_hand_joint",
                 "panda_hand/panda_finger_joint1", "panda_hand/panda_finger_joint2"]
PANEL = "safety_fence_panel"
PANEL_GAP = 0.02  # between neighbouring panels along a side


def reach_of(arm: str) -> float:
    if arm in ISAAC_ARMS:
        return ISAAC_ARMS[arm][1]
    from ..curated import arm as curated_arm

    return curated_arm(arm)["reach_m"]


def panel_dims() -> tuple[float, float, float]:
    """(width along the fence, depth incl. feet, height) of the SimReady fence panel, from the curated table."""
    from ..curated import table

    e = table()[PANEL]
    return tuple((e.get("baked") or {}).get("size") or e["size_m"])


class RobotCellParams(Params):
    arm: Literal["ur10", "franka", "fanuc_crx10ia", "fanuc_crx20ia_l"] = Field(
        "ur10", description="Arm on the pedestal (non-driven visual): Isaac UR10 / Franka, or SimReady FANUC CRX-10iA / CRX-20iA/L")
    arm_yaw: float = Field(0.0, ge=-180, le=180, description="Arm base rotation about z, degrees (its zero pose points along local +x)")
    pedestal_height: float | None = Field(None, ge=0.3, le=1.5, description="Pedestal height (m). Default: UR10 on the Isaac ur10_mount "
                                          "(0.515 m), other arms on a 0.75 m steel column. Setting it uses a steel column for any arm")
    fence: bool = Field(True, description="Fenced cell with a gate opening on the front (-y); False = unfenced keep-out zone")
    fence_style: Literal["panels", "cage"] = Field("panels", description="panels: SimReady metal safety fence panels stacked "
                                                   "fence_layers high; cage: procedural wire-mesh cage with a sliding gate")
    fence_layers: int = Field(2, ge=1, le=3, description="panels style: panels stacked this many high (1.07 m each; 2 = ~2.1 m)")
    safety_margin: float = Field(0.3, ge=0.0, le=2.0, description="Added to the arm reach for the safeguarded square (m)")
    fence_height: float = Field(2.0, ge=1.4, le=3.0, description="cage style only")
    gate_width: float = Field(1.0, ge=0.8, le=2.0, description="cage style only (panels: the gate is one panel wide)")


@register
class RobotCell(Asset):
    id = "robot_cell"
    description = ("Static robot work cell (e.g. disassembly): an arm (Isaac UR10 / Franka or SimReady FANUC CRX) on a pedestal as a "
                   "non-driven visual, inside a fence of SimReady safety panels (default; gate opening on the front, -y), a wire-mesh "
                   "cage, or an unfenced keep-out square of half-size >= reach + safety_margin. The safeguarded footprint is checked "
                   "for 0.5 m clearance to everything else. Fenced cells expose an 'inside' support (fixtures / workpieces) and a "
                   "'gate' port; all have a 'front' port outside. Not the scene robot: the mobile `robot` stays the only driven one.")
    tags = frozenset({"robot_cell", "robot", "arm", "cell", "fence", "safety", "industrial", "disassembly", "isaac", "simready", "static"})
    Params = RobotCellParams
    physics = PhysicsInfo("static", "steel", 0.6, None, "primitives", "static pedestal + fence; arm physics off")
    example = {"id": "cell", "asset": "robot_cell", "params": {"arm": "ur10"}, "placement": {"inside": "assembly"}}

    # ---- geometry --------------------------------------------------------------------------------------------
    @staticmethod
    def half(p) -> float:
        """Half-size of the safeguarded square (panels: rounded up to whole panels)."""
        h = reach_of(p.arm) + p.safety_margin
        if p.fence and p.fence_style == "panels":
            w, d, _ = panel_dims()
            n = max(2, math.ceil((2 * h - d) / (w + PANEL_GAP)))
            return (n * (w + PANEL_GAP) + d) / 2
        return h

    @staticmethod
    def pedestal(p) -> tuple[float, bool]:
        """(height, uses the Isaac ur10_mount)."""
        if p.pedestal_height is None:
            return (UR10_MOUNT_SIZE[2], True) if p.arm == "ur10" else (COLUMN_HEIGHT, False)
        return p.pedestal_height, False

    def panels(self, p) -> list[tuple[float, float, float]]:
        """Local (x, y, yaw) of each fence panel, pinwheel order S, E, N, W; the gate gap is left out on the south side."""
        w, d, _ = panel_dims()
        S = 2 * self.half(p)
        n = round((S - d) / (w + PANEL_GAP))
        step = w + PANEL_GAP
        out = []
        for side, yaw in enumerate((0.0, math.pi / 2, math.pi, -math.pi / 2)):
            for i in range(n):
                t = -S / 2 + step * (i + 0.5)  # along the side, starting at its first corner
                if side == 0 and i == self.gate_index(n):
                    continue
                x, y = {0: (t, -S / 2 + d / 2), 1: (S / 2 - d / 2, t), 2: (-t, S / 2 - d / 2), 3: (-S / 2 + d / 2, -t)}[side]
                out.append((x, y, yaw))
        return out

    @staticmethod
    def gate_index(n: int) -> int:
        return n - 2 if n >= 3 else n - 1

    def gate_x(self, p) -> float:
        if p.fence_style == "panels":
            w, d, _ = panel_dims()
            S = 2 * self.half(p)
            n = round((S - d) / (w + PANEL_GAP))
            return -S / 2 + (w + PANEL_GAP) * (self.gate_index(n) + 0.5)
        return self.half(p) - p.gate_width / 2 - 0.25

    def cage_params(self, p):
        h = self.half(p)
        return get("cage").params({"size": (2 * h, 2 * h), "fence_height": p.fence_height, "gate_width": p.gate_width,
                                   "gate_offset": h - p.gate_width / 2 - 0.25, "gate_travel": min(p.gate_width + 0.2, 4.0)})

    def safety_polygon(self, p) -> list[tuple[float, float]]:
        h = self.half(p) + (0.06 if p.fence and p.fence_style == "cage" else 0.0)  # cage: the fence's outer face
        return [(-h, -h), (h, -h), (h, h), (-h, h)]

    def footprint(self, p):
        ped = Footprint.rect(0.4, 0.4, self.pedestal(p)[0] + 1.2)
        if not p.fence:
            return ped
        if p.fence_style == "cage":
            return Footprint.union(get("cage").footprint(self.cage_params(p)).pieces() + [ped])
        w, d, ht = panel_dims()
        ht *= p.fence_layers
        pieces = []
        for x, y, yaw in self.panels(p):
            along_x = abs(math.sin(yaw)) < 0.5
            pieces.append(Footprint.rect(w if along_x else d, d if along_x else w, ht, x, y))
        return Footprint.union(pieces + [ped])

    def keepouts(self, p):
        if not p.fence:
            return [Keepout(self.safety_polygon(p), "robot_cell", (0.0, 2.5))]
        return get("cage").keepouts(self.cage_params(p)) if p.fence_style == "cage" else []

    def anchors(self, p):
        h = self.half(p)
        front = Anchor("front", "port", (0.0, -h - 1.0, 0.0), qyaw(math.pi / 2), port_kind="approach")
        if not p.fence:
            return [front]
        if p.fence_style == "cage":
            return [a for a in get("cage").anchors(self.cage_params(p)) if a.name in ("inside", "gate")] + [front]
        d = panel_dims()[1]
        r = h - d - 0.2
        return [Anchor("inside", "support", (0, 0, 0), polygon=[(-r, -r), (r, -r), (r, r), (-r, r)]),
                Anchor("gate", "port", (self.gate_x(p), -h - 0.6, 0.0), qyaw(math.pi / 2), port_kind="approach"), front]

    # ---- build -----------------------------------------------------------------------------------------------
    def _arm(self, ctx, path, pose, p, z_top):
        from pxr import Sdf

        from ..catalog import catalog_url
        from ..robots import table

        k = ctx.kit
        if p.arm in ISAAC_ARMS:
            url, reach = catalog_url(ISAAC_ARMS[p.arm][0]), ISAAC_ARMS[p.arm][1]
            info = table().get(p.arm)
            off = info.base_offset_z if info else 0.0
        else:
            from ..curated import arm as curated_arm

            a = curated_arm(p.arm)
            url, reach, off = a["url"], a["reach_m"], a.get("base_offset_z", 0.0)
        wp, wq = pose.world((0.0, 0.0, z_top + off), qyaw(math.radians(p.arm_yaw)))
        prim = k.xform(f"{path}/Arm", wp, wq)
        prim.GetReferences().AddReference(url)
        if p.arm == "ur10":
            prim.GetVariantSets().AddVariantSet("Physics").SetVariantSelection("None")  # UR10 ships a physics-free variant
        else:
            if p.arm == "franka":
                bodies, joints, roots = FRANKA_LINKS, FRANKA_JOINTS, [""]
            else:
                bodies, joints, roots = a.get("rigid_bodies", []), a.get("joints", []), a.get("articulation_roots", [])
            for rel in roots:
                k.stage.OverridePrim(f"{path}/Arm/{rel}".rstrip("/")).CreateAttribute("physxArticulation:articulationEnabled", Sdf.ValueTypeNames.Bool).Set(False)
            for rel in bodies:
                k.stage.OverridePrim(f"{path}/Arm/{rel}").CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
            for rel in joints:
                k.stage.OverridePrim(f"{path}/Arm/{rel}").CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        ctx.record(f"{path}/Arm", "robot_arm", wp, wq, "none", isaac_path=url, arm=p.arm, driven=False, reach_m=reach,
                   note=f"non-driven visual referenced from {url}")
        return reach

    def build(self, ctx, path, pose, p, rng):
        from pxr import Gf, UsdGeom

        from ..catalog import catalog_url

        k, m = ctx.kit, ctx.mats
        k.xform(path)  # identity: children carry world poses, so the cage / panels can be built as is
        ph, isaac_mount = self.pedestal(p)
        ped = k.xform(f"{path}/Pedestal", pose.pos, pose.quat)
        if isaac_mount:
            col = k.cube(f"{path}/Pedestal/Collider", UR10_MOUNT_SIZE, (0, 0, UR10_MOUNT_SIZE[2] / 2), Q_IDENT, m.get("steel_bright"), m.phys("steel"), collide=True)
            UsdGeom.Imageable(col).CreatePurposeAttr(UsdGeom.Tokens.guide)  # drawn by the Isaac mount itself
            vis = k.xform(f"{path}/Pedestal/Isaac")
            vis.GetReferences().AddReference(catalog_url(UR10_MOUNT))
            xf = UsdGeom.Xformable(vis)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, UR10_MOUNT_SIZE[2]))  # the mount's top is at its origin
        else:
            k.cube(f"{path}/Pedestal/Base", (0.5, 0.5, 0.02), (0, 0, 0.01), Q_IDENT, m.get("conv_frame"), m.phys("steel"), collide=True)
            k.cube(f"{path}/Pedestal/Column", (0.3, 0.3, ph - 0.04), (0, 0, 0.02 + (ph - 0.04) / 2), Q_IDENT, m.get("conv_frame"), m.phys("steel"), collide=True)
            k.cube(f"{path}/Pedestal/Plate", (0.34, 0.34, 0.02), (0, 0, ph - 0.01), Q_IDENT, m.get("steel_bright"), m.phys("steel"), collide=True)
        k.tag(ped, "robot_pedestal")
        reach = self._arm(ctx, path, pose, p, ph)

        keepouts, bodies = [], []
        if p.fence and p.fence_style == "cage":
            cb = get("cage").build(ctx, f"{path}/Fence", pose, self.cage_params(p), rng)
            keepouts, bodies = cb.keepouts, cb.bodies
        elif p.fence:
            panel = get(PANEL)
            k.xform(f"{path}/Fence")
            ph_ = panel_dims()[2]
            for i, (x, y, yaw) in enumerate(self.panels(p)):
                for layer in range(p.fence_layers):  # stacked: each layer's feet stand on the top rail below
                    wp, wq = pose.world((x, y, layer * ph_), qyaw(yaw))
                    panel.build(ctx, f"{path}/Fence/Panel_{i:02d}_{layer}", Pose(wp, wq), panel.params({}), rng)
        else:
            keepouts = [("robot_cell", pose.world_xy(self.safety_polygon(p)).tolist())]
            sq = self.safety_polygon(p)
            for i, (x, y) in enumerate(sq):  # floor marking of the keep-out square (visual only)
                x2, y2 = sq[(i + 1) % 4]
                mp, mq = pose.world(((x + x2) / 2, (y + y2) / 2, 0.003), qyaw(math.atan2(y2 - y, x2 - x)))
                k.cube(f"{path}/Marking_{i}", (math.hypot(x2 - x, y2 - y), 0.1, 0.004), mp, mq, m.get("safety_yellow"))
        poly = pose.world_xy(self.safety_polygon(p)).tolist()
        h = self.half(p)
        gate = pose.world((self.gate_x(p), -h - 0.6, 0.0))[0] if p.fence else None
        height = (p.fence_height if p.fence_style == "cage" else panel_dims()[2] * p.fence_layers) if p.fence else ph
        ctx.record(path, "robot_cell", pose.pos, pose.quat, "static", size=(2 * h, 2 * h, height), arm=p.arm, reach_m=reach,
                   fenced=p.fence, fence_style=p.fence_style if p.fence else None,
                   safety_polygon=[[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                   **({"gate_xy": [round(float(gate[0]), 4), round(float(gate[1]), 4)]} if gate is not None else {}))
        return Built(path, [a.to_world(pose) for a in self.anchors(p)], keepouts,
                     info={"arm": f"{path}/Arm", "safety_polygon": poly, "reach_m": reach, "fenced": p.fence}, bodies=bodies)
