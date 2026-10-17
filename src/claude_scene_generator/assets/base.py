"""Asset contract shared by procedural, mesh, catalog and generated assets.

An asset is a parameterised thing that can (1) describe itself to an LLM, (2) tell the solver its footprint, anchors and
keep-outs *without* touching USD, and (3) author itself into a stage via ``build``.

Frames: an instance is placed with a ``Pose`` (world position of the footprint's bottom-centre origin + wxyz quaternion).
``footprint``/``anchors``/``keepouts`` are in that local frame; ``Built.anchors`` are in world frame (post-build, so they
can reflect randomness such as the real height of a pallet load).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from ..geom import Q_IDENT, compose, qmat, qyaw, yaw_of

BodyKind = Literal["static", "dynamic", "kinematic", "articulated", "mixed", "visual"]


# --------------------------------------------------------------------------- poses / shapes
@dataclass(frozen=True)
class Pose:
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = Q_IDENT  # (w, x, y, z)

    @staticmethod
    def xy_yaw(x: float, y: float, yaw: float = 0.0, z: float = 0.0) -> "Pose":
        return Pose((float(x), float(y), float(z)), qyaw(yaw))

    @property
    def yaw(self) -> float:
        return yaw_of(self.quat)

    def world(self, local_p=(0, 0, 0), local_q=Q_IDENT):
        """World (pos, quat) of a point/frame given in this pose's local frame."""
        return compose(self.pos, self.quat, local_p, local_q)

    def world_xy(self, pts) -> np.ndarray:
        R = qmat(self.quat)[:2, :2]
        return np.asarray(pts, float) @ R.T + np.asarray(self.pos[:2], float)


@dataclass
class Footprint:
    """Convex xy polygon (local, CCW) extruded from z0 to z1 (local)."""

    polygon: list
    z0: float = 0.0
    z1: float = 1.0
    parts: list | None = None  # optional finer decomposition (e.g. two rack segments with a cross-aisle between)

    def pieces(self) -> list["Footprint"]:
        return self.parts or [self]

    @staticmethod
    def union(parts: list["Footprint"]) -> "Footprint":
        """Bounding footprint (convex hull of all pieces) that remembers the pieces for precise overlap tests."""
        from scipy.spatial import ConvexHull

        pts = np.concatenate([np.asarray(p.polygon, float) for p in parts])
        hull = pts[ConvexHull(pts).vertices]
        return Footprint([tuple(map(float, v)) for v in hull], min(p.z0 for p in parts), max(p.z1 for p in parts), list(parts))

    @staticmethod
    def rect(sx: float, sy: float, h: float, cx: float = 0.0, cy: float = 0.0, z0: float = 0.0) -> "Footprint":
        hx, hy = sx / 2, sy / 2
        return Footprint([(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)], z0, z0 + h)

    @staticmethod
    def circle(r: float, h: float, n: int = 12, z0: float = 0.0) -> "Footprint":
        R = r / math.cos(math.pi / n)  # circumscribed polygon: never smaller than the circle
        return Footprint([(R * math.cos(2 * math.pi * i / n), R * math.sin(2 * math.pi * i / n)) for i in range(n)], z0, z0 + h)

    @property
    def size(self) -> tuple:
        p = np.asarray(self.polygon, float)
        lo, hi = p.min(0), p.max(0)
        return (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(self.z1 - self.z0))

    def to_json(self) -> dict:
        d = {"polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in self.polygon], "z": [round(self.z0, 4), round(self.z1, 4)],
             "size": [round(v, 4) for v in self.size]}
        if self.parts:
            d["n_parts"] = len(self.parts)
        return d


@dataclass
class Anchor:
    """Named attachment point.  ``support``: a horizontal surface (``polygon`` in the anchor frame) things can rest on.
    ``port``: a frame other assets snap to (pipe nozzle, door opening, hose end).  ``meta`` carries addressing data
    (e.g. rack slot level/face/aisle) used by ``fill`` filters and nav goals."""

    name: str
    kind: Literal["support", "port"]
    pos: tuple
    quat: tuple = Q_IDENT
    polygon: list | None = None  # support only, xy relative to pos
    port_kind: str | None = None  # port only: "pipe", "hinge", "hose", "mount", ...
    radius: float | None = None
    meta: dict = field(default_factory=dict)

    def to_world(self, pose: Pose) -> "Anchor":
        p, q = pose.world(self.pos, self.quat)
        return Anchor(self.name, self.kind, tuple(float(c) for c in p), q, self.polygon, self.port_kind, self.radius, dict(self.meta))

    def to_json(self) -> dict:
        d = {"name": self.name, "kind": self.kind, "pos": [round(float(c), 4) for c in self.pos]}
        if self.quat != Q_IDENT:
            d["yaw_deg"] = round(math.degrees(yaw_of(self.quat)), 2)
        if self.polygon is not None:
            d["polygon"] = [[round(float(x), 4), round(float(y), 4)] for x, y in self.polygon]
        for k in ("port_kind", "radius"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.meta:
            d["meta"] = self.meta
        return d


@dataclass
class Keepout:
    """Area that must stay free of other objects (door swing arc, gate slide, dock lane), local frame."""

    polygon: list
    tag: str
    z: tuple = (0.0, 2.0)


@dataclass
class JointInfo:
    name: str
    type: Literal["revolute", "prismatic", "spherical", "fixed"]
    axis: str = "Z"
    lower: float | None = None
    upper: float | None = None
    unit: str = "deg"
    default: float = 0.0
    drive: dict | None = None
    link: str = ""
    role: str = ""


@dataclass
class PhysicsInfo:
    body: BodyKind
    material: str = "steel"  # physics material preset (materials.PHYSICAL)
    friction: float | None = None
    mass_kg: float | tuple | None = None  # exact, (min, max) range, or None if parameter-dependent
    collider: str = "primitives"  # primitives | convex_hulls | triangle_mesh_static | none
    note: str = ""

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


# --------------------------------------------------------------------------- build context / result
@dataclass
class Built:
    root: str
    anchors: list = field(default_factory=list)  # world-frame Anchors (actual, post-build)
    keepouts: list = field(default_factory=list)  # world-frame xy polygons [(tag, polygon)]
    info: dict = field(default_factory=dict)  # asset-specific outputs (load_top, conveyor info, valve info, ...)
    bodies: list = field(default_factory=list)  # rigid-body prim paths authored by this instance

    def anchor(self, name: str) -> Anchor:
        for a in self.anchors:
            if a.name == name:
                return a
        raise KeyError(f"no anchor {name!r} on {self.root}")


@dataclass
class Environment:
    """What an asset may need to know about the scene it is placed into."""

    x_range: tuple = (-20.0, 20.0)
    y_range: tuple = (-12.5, 12.5)
    height: float | None = 8.0
    wall_thickness: float = 0.3
    floor_z: Callable[[float, float], float] = lambda x, y: 0.0
    kind: str = "indoor"

    @property
    def size(self) -> tuple:
        return (self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0])


class BuildContext:
    """Per-scene authoring state handed to ``Asset.build``.

    kit       usdkit.Kit for the stage being written
    mats      materials.Materials (named visual/physics presets, created on first use)
    textures  materials.TextureCache (procedural textures written to ``out_dir/textures``)
    env       Environment
    scene     shared registries assets append to: articulations, conveyors, stats
    """

    def __init__(self, kit, mats, textures, env: Environment | None = None, seed: int = 0, out_dir: Path | None = None):
        self.kit, self.mats, self.textures = kit, mats, textures
        self.env = env or Environment()
        self.seed = seed
        self.out_dir = Path(out_dir) if out_dir else Path(kit.path).parent
        self.scene: dict[str, Any] = {"articulations": [], "conveyors": [], "joints": [], "stats": {}}
        # per-instance metadata set by the compiler before each build (spec object id, zone label, extra record fields)
        self.meta: dict[str, Any] = {}
        self._scopes: set[str] = set()

    @property
    def zone(self) -> str | None:
        return self.meta.get("zone")

    def record(self, path, category, pos, quat, body_type, **kw):
        """``kit.record`` + the current instance's zone / spec id / extra fields."""
        kw.setdefault("zone", self.zone)
        extra = dict(self.meta.get("record_extra", {}))
        extra.update({k: v for k, v in kw.items()})
        if self.meta.get("spec_id"):
            extra.setdefault("spec_id", self.meta["spec_id"])
        return self.kit.record(path, category, pos, quat, body_type, **extra)

    def stat(self, key: str, value):
        self.scene["stats"].setdefault(key, []).append(value)

    def scope(self, path: str) -> str:
        if path not in self._scopes:
            if not self.kit.stage.GetPrimAtPath(path):
                self.kit.scope(path)
            self._scopes.add(path)
        return path

    def alloc(self, parent: str, prefix: str, width: int = 4) -> str:
        """Unique child path ``parent/prefix_NNNN`` (parent scope created on demand)."""
        self.scope(parent)
        i = self.kit.next_id(f"{parent}/{prefix}")
        return f"{parent}/{prefix}_{i:0{width}d}"


# --------------------------------------------------------------------------- params
class Params(BaseModel):
    """Base for asset parameter models: strict, documented, JSON-schema exportable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


DIST_KEYS = ("uniform", "randint", "choice", "normal")


def sample_value(v, rng: np.random.Generator):
    """Resolve a distribution literal into a value. ``{"uniform": [a, b]}`` | ``{"randint": [a, b]}`` (inclusive) |
    ``{"choice": [..], "p": [..]?}`` | ``{"normal": [mu, sigma], "clip": [lo, hi]?}``; lists/dicts recurse."""
    if isinstance(v, dict) and len(set(v) & set(DIST_KEYS)) == 1:
        if "uniform" in v:
            a, b = v["uniform"]
            return float(rng.uniform(a, b))
        if "randint" in v:
            a, b = v["randint"]
            return int(rng.integers(a, b + 1))
        if "choice" in v:
            opts = v["choice"]
            return opts[int(rng.choice(len(opts), p=v.get("p")))]
        mu, sd = v["normal"]
        x = float(rng.normal(mu, sd))
        if "clip" in v:
            x = float(np.clip(x, *v["clip"]))
        return x
    if isinstance(v, dict):
        return {k: sample_value(x, rng) for k, x in v.items()}
    if isinstance(v, list):
        return [sample_value(x, rng) for x in v]
    return v


class ParamError(ValueError):
    def __init__(self, asset_id: str, errors: list[dict]):
        self.asset_id, self.errors = asset_id, errors
        super().__init__(f"{asset_id}: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errors))


# --------------------------------------------------------------------------- asset
class Asset(ABC):
    id: ClassVar[str]
    description: ClassVar[str]
    tags: ClassVar[frozenset] = frozenset()
    Params: ClassVar[type[BaseModel]] = Params
    physics: ClassVar[PhysicsInfo] = PhysicsInfo("static")
    source: ClassVar[str] = "procedural"
    example: ClassVar[dict | None] = None  # one example spec object using this asset

    # -- params
    def params(self, raw: dict | BaseModel | None = None, rng: np.random.Generator | None = None) -> BaseModel:
        if isinstance(raw, BaseModel):
            return raw
        raw = dict(raw or {})
        if rng is not None:
            raw = sample_value(raw, rng)
        try:
            return self.Params(**raw)
        except ValidationError as e:
            raise ParamError(self.id, [{"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in e.errors()]) from None

    # -- planning (pure, no USD)
    @abstractmethod
    def footprint(self, p) -> Footprint: ...

    def anchors(self, p) -> list[Anchor]:
        return []

    def keepouts(self, p) -> list[Keepout]:
        return []

    def joints(self, p) -> list[JointInfo]:
        return []

    # -- authoring
    @abstractmethod
    def build(self, ctx: BuildContext, path: str, pose: Pose, p, rng: np.random.Generator) -> Built: ...

    # -- LLM-facing description
    def describe(self, max_anchors: int = 24) -> dict:
        p = self.params()
        anchors = self.anchors(p)
        d = {
            "id": self.id,
            "source": self.source,
            "description": self.description,
            "tags": sorted(self.tags),
            "params_schema": self.Params.model_json_schema(),
            "defaults": p.model_dump(mode="json"),
            "footprint_at_defaults": self.footprint(p).to_json(),
            "physics": self.physics.to_json(),
            "joints": [j.__dict__ for j in self.joints(p)],
            "n_anchors": len(anchors),
            "anchors": [a.to_json() for a in anchors[:max_anchors]],
        }
        if len(anchors) > max_anchors:
            d["anchors_truncated"] = True
        if self.example:
            d["example"] = self.example
        return d

    def summary(self) -> dict:
        return {"id": self.id, "source": self.source, "description": self.description.split("\n")[0], "tags": sorted(self.tags)}
