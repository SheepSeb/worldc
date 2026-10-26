"""Real Isaac Sim people that walk along waypoint paths as moving hazards.

``human``: one of NVIDIA's Isaac Sim characters (``Isaac/People/Characters/...``) referenced by URL as the visual of a kinematic body that
``runtime.PathFollower`` drives along the waypoints (loop or ping-pong).  The body's collider is a capsule of the character's size, so the
hazard checks and the drive logic stay exact.  The characters are authored in a T-pose and face -y, so they are turned +90 degrees about Z
to face the walking direction (+x).  Feet are at the asset origin (measured in Isaac Sim).

The walk is one of Isaac Sim's own motion-capture clips (``Isaac/People/Animations/stand_walk_*.skelanim.usd``, the ones its People
extension plays), retargeted onto the character (``skel_pose``).  One clean gait cycle of the clip is looped, and played at the rate that
makes its stride match ``speed_mps`` so the feet do not slide.  ``library/people/*.json`` holds the measurements; ``people_bake`` makes them.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..geom import Q_IDENT
from .base import Asset, Footprint, PhysicsInfo
from .catalog import DEFAULT_CACHE, CatalogError, _fetch, catalog_url
from .procedural.agents import MOTION_S, PathParams, _author_path_agent, follower_of
from .registry import register
from .skel_pose import blend

BASE = "Isaac/People/Characters"
LIB = Path(__file__).parent / "library" / "people"
# folder name -> {height_m, skeleton, joints, rest}: measured in Isaac Sim (feet at the origin; every joint's rest transform)
CHARACTERS = json.loads((LIB / "isaac_5.1_characters.json").read_text())
# the clips' rig ("joints", "rest") and, per walk clip, its clean gait cycle {start, frames, speed_mps} (people_bake)
GAITS = json.loads((LIB / "isaac_5.1_gaits.json").read_text())
CLIP_FPS = 30.0
RADIUS = 0.26
LOOK_JOINTS = ("NeckTwist01", "NeckTwist02", "Head")  # the character joints the look-around clip drives
SAMPLE_HZ = 20  # animation samples per second (rotations are interpolated between them)
GAIT_S = MOTION_S  # seconds of gait animation authored (the timeline is extended to this)


def fetch_asset(path: str) -> Path:
    """An Isaac Sim asset: downloaded once into the catalog cache, or read in place when WORLDC_ISAAC_ASSET_ROOT is a local mirror."""
    url = catalog_url(path)
    if not url.startswith(("http://", "https://")):
        if not Path(url).is_file():
            raise CatalogError("asset.catalog_not_found", f"Catalog asset not found: {url}")
        return Path(url)
    dst = DEFAULT_CACHE / "people" / (hashlib.sha1(url.encode()).hexdigest()[:12] + "_" + path.rsplit("/", 1)[-1])
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        _fetch(url, tmp)
        tmp.replace(dst)
    return dst


@functools.lru_cache(maxsize=None)
def _clip_cycle(clip: str, url: str):
    from pxr import Usd

    from .skel_pose import read_clip

    stage = Usd.Stage.Open(str(fetch_asset(f"Isaac/People/Animations/{clip}.skelanim.usd")))
    joints, rots, trans, times, _ = read_clip(stage)
    if joints != GAITS["rig"]["joints"]:
        raise ValueError(f"{clip}: joints differ from the biped rig in library/people/isaac_5.1_gaits.json")
    g = GAITS["clips"].get(clip) or GAITS["poses"][clip]
    pelvis = joints.index("Root/Pelvis")
    keep = [i for i, t in enumerate(times) if g["start"] <= t < g["start"] + g["frames"]]
    if len(keep) != g["frames"]:
        raise ValueError(f"{clip}: expected frames {g['start']}..{g['start'] + g['frames'] - 1}, found {len(keep)}")
    frames = [(rots[i], trans[i][pelvis]) for i in keep]
    return frames + [frames[0]]


def clip_cycle(clip: str):
    """One looping cycle of an Isaac Sim clip: [(local rotations, pelvis translation)], the first frame repeated at the end."""
    return _clip_cycle(clip, catalog_url(f"Isaac/People/Animations/{clip}.skelanim.usd"))


@functools.lru_cache(maxsize=None)
def retarget(character: str):
    from .skel_pose import Retarget, Rig

    c = CHARACTERS[character]
    return Retarget(Rig(GAITS["rig"]["joints"], GAITS["rig"]["rest"]), Rig(c["joints"], c["rest"]))


def _smooth(f: float) -> float:
    f = min(max(f, 0.0), 1.0)
    return f * f * (3 - 2 * f)


class Performer:
    """What one person looks like at any moment of the stop-and-go path (``runtime.PathFollower.motion``): the walk clip with its gait
    phase following the distance walked (no sliding, whatever the speed or easing), standing still (Isaac's idle clip at one end of the
    path, its look-around clip at the other), and stepping round through the turn, blended by how fast they are moving."""

    STEP_HZ = 0.75  # gait cycles per second while turning on the spot

    def __init__(self, character: str, gait: str, follower, phase: float):
        rt = retarget(character)
        self.f, self.phase = follower, phase / (2 * math.pi)
        g = GAITS["clips"][gait]
        stride = g["frames"] / CLIP_FPS * g["speed_mps"]  # metres per gait cycle in the clip
        self.stride = follower.total / max(1, round(follower.total / stride))  # a whole number of cycles per leg: the phase closes
        self.walk = [rt.frame(*f) for f in clip_cycle(gait)]
        self.idle = [rt.frame(*f) for f in clip_cycle("stand_idle_loop")]
        self.look = [rt.frame(*f) for f in clip_cycle("LookAround")]
        # the look-around clip only moves the neck and head (its arms stay in the rig's T-pose): it is a layer on top of the idle
        self.look_joints = [k for k, i in enumerate(rt.order) if rt.char.short[i] in LOOK_JOINTS]

    @staticmethod
    def _at(cycle, x: float, loop: bool):
        n = len(cycle) - 1
        x = x % n if loop else min(max(x, 0.0), n - 1e-9)
        i = int(x)
        return blend(cycle[i], cycle[i + 1], x - i)

    def _standing(self, end: int, t: float):
        """The pose of standing at path end ``end`` (0 start, 1 far end) ``t`` seconds into the pause."""
        idle = self._at(self.idle, t * CLIP_FPS, loop=True)
        if end == 0:
            return idle
        look = self._at(self.look, t * CLIP_FPS, loop=False)
        return [look[0][k] if k in self.look_joints else q for k, q in enumerate(idle[0])], idle[1]

    def pose(self, t: float):
        f, m = self.f, self.f.motion(t)
        if m.dwell_t is not None:  # paused at an end (turning round during the last turn_s)
            end = 0 if m.back else 1
            stand = self._standing(end, m.dwell_t)
            if f.turn_s > 0 and 0 < m.turn < 1:  # step round: the walk cycle at a stroll, fading in and out
                step = self._at(self.walk, ((m.turn * f.turn_s * self.STEP_HZ + self.phase) % 1.0) * (len(self.walk) - 1), loop=True)
                return blend(stand, step, 0.7 * math.sin(math.pi * m.turn))
            return stand
        departing = m.odo < f.total / 2
        end = 1 if departing == m.back else 0  # the path end this leg is leaving / approaching
        stand = self._standing(end, f.dwell_s if departing else 0.0)
        step = self._at(self.walk, ((m.odo / self.stride + self.phase) % 1.0) * (len(self.walk) - 1), loop=True)
        return blend(stand, step, _smooth(m.speed / f.speed / 0.6))


class HumanParams(PathParams):
    dwell_s: float = Field(4.0, ge=0, le=60, description="Ping-pong only: seconds the person stands still at each end of the path (looking around)")
    ease_s: float = Field(0.8, ge=0, le=5, description="Ping-pong only: seconds to start / stop walking")
    turn_s: float = Field(1.4, ge=0, le=10, description="Ping-pong only: seconds the turn-around takes, at the end of the pause")
    character: Literal[tuple(CHARACTERS)] = Field("male_adult_construction_05_new", description="Which Isaac Sim character (hi-vis workers, an officer)")  # type: ignore[valid-type]
    mass_kg: float = Field(80.0, ge=30, le=150)
    speed_mps: float = Field(0.9, gt=0, le=3)
    gait: Literal[tuple(GAITS["clips"])] = Field("stand_walk_1", description="Which Isaac Sim walk clip is played (_mirror: starts on the other foot)")  # type: ignore[valid-type]


@register
class IsaacHuman(Asset):
    id = "human"
    source = "isaac"
    description = ("Real Isaac Sim person (hi-vis construction worker or officer) walking along a waypoint path (loop or ping-pong) as a moving "
                   "hazard, animated with Isaac Sim's own walk clips: kinematic capsule driven at runtime. Keep its corridor clear (keep_clear).")
    tags = frozenset({"human", "agent", "dynamic_obstacle", "kinematic", "hazard", "isaac"})
    Params = HumanParams
    physics = PhysicsInfo("kinematic", "plastic", 0.5, 80.0, "primitives", "visual referenced from Isaac/People/Characters; pose set every physics step by runtime.PathFollower")
    example = {"id": "worker", "asset": "human", "params": {"character": "male_adult_construction_05_new", "gait": "stand_walk_2", "waypoints": [[-8, 6], [8, 6]], "path_mode": "pingpong"}}

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
            """Isaac Sim's walk / idle / look-around clips, retargeted and played along the path's timing, as time samples (one period of
            the motion, repeated); the timeline is extended so Isaac Sim plays it."""
            from .skel_pose import author_samples

            tcps = k.stage.GetTimeCodesPerSecond() or 24.0
            follower = follower_of(p)
            perf = Performer(p.character, p.gait, follower, float(rng.uniform(0, 2 * math.pi)))
            n = max(8, round(follower.period_s * SAMPLE_HZ))
            poses = [perf.pose(i * follower.period_s / n) for i in range(n)]
            author_samples(k.stage, f"{path}/Isaac/{info['skeleton']}", retarget(p.character), poses, follower.period_s / n * tcps, GAIT_S * tcps)
            ctx.scene["animation_end_time_code"] = max(float(ctx.scene.get("animation_end_time_code", 0.0)), GAIT_S * tcps)

        built = _author_path_agent(ctx, path, p, (2 * RADIUS, 2 * RADIUS, h), z, p.mass_kg, "human", shape)
        pose()
        return built
