"""Real Isaac Sim robots as scene assets.  A scene robot is always an NVIDIA / partner robot USD from the Isaac Sim asset root,
referenced by URL with its own articulation and physics left on; there is no generated robot.

    robot: {asset: nova_carter, spawn: {pose: {xy: [2, -3], yaw: 90}}}                     # alias (see list_assets tags=[robot])
    robot: {asset: {catalog: omniverse, path: Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd}, spawn: ...}

The table of supported robots (``CURATED`` below + measured bounds in ``library/robots/isaac_5.1_robots.json``) is built once with
``python -m worldc.assets.robots build``, which downloads each robot's layers and measures its bounds / articulation.
Only robots in that table are accepted, so footprint, height and physics are known before anything is built.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from .base import Asset, Built, Footprint, Params, PhysicsInfo, Pose
from .catalog import catalog_url

TABLE = Path(__file__).parent / "library" / "robots" / "isaac_5.1_robots.json"

# alias -> (path under the Isaac asset root, kind, description)
CURATED: dict[str, tuple[str, str, str]] = {
    "nova_carter": ("Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd", "wheeled_amr",
                    "NVIDIA Nova Carter autonomous mobile robot: differential drive, lidar and stereo cameras. Default warehouse robot."),
    "carter_v1": ("Isaac/Robots/NVIDIA/Carter/carter_v1.usd", "wheeled_amr", "NVIDIA Carter v1 differential-drive mobile robot."),
    "iw_hub": ("Isaac/Robots/Idealworks/iwhub/iw_hub.usd", "wheeled_amr", "idealworks iw.hub warehouse AMR / pallet mover."),
    "jackal": ("Isaac/Robots/Clearpath/Jackal/jackal.usd", "wheeled_amr", "Clearpath Jackal small skid-steer outdoor/indoor robot."),
    "dingo": ("Isaac/Robots/Clearpath/Dingo/dingo.usd", "wheeled_amr", "Clearpath Dingo compact indoor differential-drive robot."),
    "turtlebot3_burger": ("Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd", "wheeled_amr", "TurtleBot3 Burger small research robot."),
    "kaya": ("Isaac/Robots/NVIDIA/Kaya/kaya.usd", "wheeled_amr", "NVIDIA Kaya holonomic three-wheel robot."),
    "o3dyn": ("Isaac/Robots/Fraunhofer/O3dyn/o3dyn.usd", "wheeled_amr", "Fraunhofer O3dyn omnidirectional logistics platform."),
    "ridgeback_franka": ("Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd", "mobile_manipulator",
                         "Clearpath Ridgeback omnidirectional base with a Franka Panda arm."),
    "franka": ("Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", "arm", "Franka Panda 7-DoF arm with gripper, fixed base."),
    "ur10": ("Isaac/Robots/UniversalRobots/ur10/ur10.usd", "arm", "Universal Robots UR10 6-DoF industrial arm (1.3 m reach, 10 kg payload), fixed base."),
    "so101": ("Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd", "arm",
              "RobotStudio SO-101 low-cost 5-DoF desktop arm with gripper, fixed base (about 35 cm reach)."),
    "spot": ("Isaac/Robots/BostonDynamics/spot/spot.usd", "legged", "Boston Dynamics Spot quadruped (needs a locomotion controller to walk)."),
    "anymal_d": ("Isaac/Robots/ANYbotics/anymal_d/anymal_d.usd", "legged", "ANYbotics ANYmal D quadruped (needs a locomotion controller)."),
    "unitree_go1": ("Isaac/Robots/Unitree/Go1/go1.usd", "legged", "Unitree Go1 quadruped (needs a locomotion controller)."),
    "unitree_h1": ("Isaac/Robots/Unitree/H1/h1.usd", "humanoid", "Unitree H1 humanoid (needs a balance controller)."),
}
NEEDS_CONTROLLER = {"legged", "humanoid"}

# Differential-drive wheel joints (velocity-driven in the robot's USD) of the wheeled robots we can drive with ``worldc drive``.
# Wheel radius and track are not hard-coded: the driver calibrates them by moving the robot (render/isaac_drive.py).
DIFF_DRIVE = {
    "nova_carter": {"left": ["joint_wheel_left"], "right": ["joint_wheel_right"]},
    "carter_v1": {"left": ["left_wheel"], "right": ["right_wheel"]},
    "dingo": {"left": ["left_wheel_joint"], "right": ["right_wheel_joint"]},
    "iw_hub": {"left": ["left_wheel_joint"], "right": ["right_wheel_joint"]},
    "jackal": {"left": ["front_left_wheel_joint", "rear_left_wheel_joint"], "right": ["front_right_wheel_joint", "rear_right_wheel_joint"]},
}


@dataclass
class RobotInfo:
    alias: str
    path: str
    kind: str
    description: str
    size_m: tuple[float, float, float]          # bounds in metres, Z-up, in the robot's default pose
    base_offset_z: float                        # lift so the lowest point sits on the floor
    footprint_radius_m: float                   # circumscribed circle of the xy bounds, centred on the robot origin
    articulation: bool
    n_joints: int
    brake_decel_mps2: float | None = None       # braking deceleration (E3 field policy); see brake_decel_source
    brake_decel_source: str | None = None

    @property
    def needs_controller(self) -> bool:
        return self.kind in NEEDS_CONTROLLER


_CACHE: dict[str, RobotInfo] | None = None


def table() -> dict[str, RobotInfo]:
    """Measured robots (alias -> info).  Empty if the table has not been built."""
    global _CACHE
    if _CACHE is None:
        raw = json.loads(TABLE.read_text()) if TABLE.is_file() else {}
        _CACHE = {a: RobotInfo(alias=a, **{k: (tuple(v) if k == "size_m" else v) for k, v in d.items()}) for a, d in raw.items()}
    return _CACHE


def aliases() -> list[str]:
    return sorted(table())


def by_path(path: str) -> RobotInfo | None:
    return next((r for r in table().values() if r.path == path.lstrip("/")), None)


class RobotParams(Params):
    camera_height: float | None = Field(None, ge=0.1, le=3.0, description="Onboard camera height in metres (default: 80% of the robot's height, max 1.4)")
    camera_focal_mm: float = Field(15.0, ge=5, le=100)


class IsaacRobot(Asset):
    source = "isaac"
    Params = RobotParams

    def __init__(self, info: RobotInfo):
        self.info = info
        self.id = info.alias
        self.description = f"{info.description} Real Isaac Sim robot ({info.kind}), {info.size_m[0]:.2f} x {info.size_m[1]:.2f} x {info.size_m[2]:.2f} m, footprint radius {info.footprint_radius_m:.2f} m."
        self.tags = frozenset({"robot", "isaac", info.kind})
        self.physics = PhysicsInfo("articulated" if info.articulation else "dynamic", "steel", None, None, "authored",
                                   f"referenced from {info.path}; its own articulation ({info.n_joints} joints) and physics are kept"
                                   + ("; needs a controller to move" if info.needs_controller else ""))
        self.example = {"robot": {"asset": info.alias, "spawn": {"pose": {"xy": [0.0, 0.0], "yaw": 0}}}}

    @property
    def url(self) -> str:
        return catalog_url(self.info.path)

    def footprint(self, p) -> Footprint:
        return Footprint.circle(self.info.footprint_radius_m, self.info.size_m[2])

    def build(self, ctx, path: str, pose: Pose, p, rng) -> Built:
        from pxr import Gf, UsdGeom

        from ..geom import Q_IDENT
        from .procedural.agents import look_at_quat

        k, i = ctx.kit, self.info
        k.xform(path, pose.pos, pose.quat)
        k.tag(k.stage.GetPrimAtPath(path), "robot_spawn", isaac_robot=i.alias, isaac_path=i.path, footprint_radius_m=i.footprint_radius_m,
              kind=i.kind)
        vis = k.xform(f"{path}/Isaac")
        vis.GetReferences().AddReference(self.url)  # the robot's own articulation / joints / drives / sensors come with it
        xf = UsdGeom.Xformable(vis)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, i.base_offset_z + 0.002))
        h = p.camera_height or min(1.4, 0.8 * i.size_m[2])
        eye = (min(0.2, i.size_m[0] / 4), 0.0, h)
        cam = UsdGeom.Camera.Define(k.stage, f"{path}/RobotCamera")
        cam.CreateFocalLengthAttr(float(p.camera_focal_mm))
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(11.79)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
        k.set_pose(cam.GetPrim(), eye, look_at_quat(eye, (eye[0] + 2.8, eye[1], eye[2] - 0.3)))
        ctx.record(path, "robot_spawn", pose.pos, pose.quat, "none", size=(2 * i.footprint_radius_m,) * 2 + (i.size_m[2],), isaac_robot=i.alias,
                   isaac_path=i.path, kind=i.kind, needs_controller=i.needs_controller, articulation=i.articulation,
                   note=f"real Isaac Sim robot referenced from {i.path}")
        return Built(path, info={"camera": f"{path}/RobotCamera", "prim": path, "isaac_robot": i.alias, "isaac_path": i.path, "kind": i.kind,
                                 "needs_controller": i.needs_controller})


def load() -> list[IsaacRobot]:
    """Registry loader: one asset per measured robot."""
    return [IsaacRobot(r) for r in table().values()]


from . import registry  # noqa: E402  (after IsaacRobot / load exist; the registry imports this module lazily)

if __name__ != "__main__":  # `python -m worldc.assets.robots` would otherwise register the robots twice (as __main__ and as the package module)
    registry.add_loader(load)


# ------------------------------------------------------------------------------------------------------ building the table
def build_table(out: Path = TABLE, only: list[str] | None = None, log=print) -> Path:
    """Measure the curated robots inside Isaac Sim (every payload loaded) and write the table."""
    import os
    import subprocess
    import tempfile

    from ..isaac_env import require_isaac_sim

    sim = require_isaac_sim()
    items = {a: v for a, v in CURATED.items() if not only or a in only}
    with tempfile.TemporaryDirectory() as td:
        res_file = Path(td) / "measured.json"
        cmd = [str(sim.python), "-m", "worldc.assets.robots_measure", str(res_file)] + [f"{a}={catalog_url(v[0])}" for a, v in items.items()]
        log(f"measuring {len(items)} robots in Isaac Sim ...")
        p = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2])), capture_output=True, text=True)
        for ln in p.stdout.splitlines():
            if ln.startswith(("MEASURED", "FAILED")):
                log("  " + ln)
        if not res_file.is_file():
            raise RuntimeError("Isaac Sim measurement failed:\n" + (p.stderr or p.stdout)[-1500:])
        measured = json.loads(res_file.read_text())
    table = json.loads(out.read_text()) if out.is_file() else {}
    for alias, (path, kind, desc) in items.items():
        m = measured.get(alias, {})
        if not m.get("ok"):
            log(f"  {alias}: not usable ({m.get('error') or m}); left out")
            table.pop(alias, None)
            continue
        table[alias] = {"path": path, "kind": kind, "description": desc, **{k: m[k] for k in ("size_m", "base_offset_z", "footprint_radius_m", "articulation", "n_joints")},
                          **{k: v for k, v in table.get(alias, {}).items() if k.startswith("brake_")}}  # not measured: kept
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(sorted(table.items())), indent=1))
    return out


def settle_offsets(out: Path = TABLE, only: list[str] | None = None, sim_seconds: float = 3.0, log=print) -> Path:
    """Calibrate each robot's spawn height by physics: drop it on a flat floor in Isaac Sim, measure how far its articulation root moves
    while it comes to rest, and fold that into ``base_offset_z``.  The visual bounding box alone can sit lower than the wheels actually
    rest (e.g. suspension), which makes a statically rendered robot float.  Robots that need a controller (legged / humanoid) sag by
    design and keep the bounding-box offset."""
    import os
    import subprocess
    import tempfile

    from ..compile import compile_scene
    from ..isaac_env import require_isaac_sim

    global _CACHE
    sim = require_isaac_sim()
    table_ = json.loads(out.read_text())
    for alias, d in table_.items():
        if (only and alias not in only) or d["kind"] in NEEDS_CONTROLLER:
            continue
        _CACHE = None
        scene = {"environment": {"footprint": {"size": [8, 8]}}, "zones": [], "robot": {"asset": alias, "spawn": {"pose": {"xy": [0.0, 0.0], "yaw": 0}}}}
        with tempfile.TemporaryDirectory() as td:
            res = compile_scene(scene, 0, Path(td) / "s", run_check=False)
            if not res.ok:
                log(f"  {alias}: scene build failed: {[str(i) for i in res.issues][:1]}")
                continue
            subprocess.run([str(sim.python), "-m", "worldc.isaac_validate", str(Path(td) / "s"), "--sim-seconds", str(sim_seconds), "--skip-render", "--skip-art"],
                           env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2])), capture_output=True, text=True)
            f = Path(td) / "s" / "isaac_validation.json"
            rob = json.loads(f.read_text()).get("stages", {}).get("robot", {}) if f.is_file() else {}
        if "dz_m" not in rob:
            log(f"  {alias}: no settle result ({rob.get('error')})")
            continue
        new = round(d["base_offset_z"] + rob["dz_m"] + 0.002, 4)
        log(f"  {alias}: settled dz {rob['dz_m']:+.4f} m  base_offset_z {d['base_offset_z']} -> {new}")
        table_[alias]["base_offset_z"] = max(new, -0.05)
        out.write_text(json.dumps(table_, indent=1))
    _CACHE = None
    return out


if __name__ == "__main__":
    from worldc.assets import robots as _pkg  # the package copy owns the registry; this __main__ copy only dispatches

    if sys.argv[1:2] == ["settle"]:
        _pkg.settle_offsets(only=sys.argv[2:] or None)
    elif sys.argv[1:2] == ["build"]:
        _pkg.build_table(only=sys.argv[2:] or None)
    else:
        for a, r in table().items():
            print(f"{a:20s} {r.kind:18s} {r.size_m}  r={r.footprint_radius_m}")
