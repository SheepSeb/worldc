"""Mesh assets: a visual USD referenced into the scene plus precomputed convex-hull colliders, described by a sidecar YAML.

Sidecar (``<name>.asset.yaml``) is written by ``import_mesh`` and may be hand-edited (anchors, mass, material).  Meshes are
already normalised: metres, Z-up, origin at the bottom-centre of the footprint.  Dynamic bodies only ever get
``convexHull`` colliders (never triangle meshes).
"""
from __future__ import annotations

import functools
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from ...geom import Part, Q_IDENT, qyaw
from ...materials import PHYSICAL
from .. import registry
from ..base import Anchor, Asset, BuildContext, Built, Footprint, Params, PhysicsInfo, Pose

DEFAULT_LIBRARY = Path(__file__).resolve().parent.parent / "library"
SIDECAR_SUFFIX = ".asset.yaml"


# --------------------------------------------------------------------------- sidecar model
class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bounds(_M):
    min: tuple[float, float, float]
    max: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(float(b - a) for a, b in zip(self.min, self.max))


class Original(_M):
    up_axis: Literal["Y", "Z"] = "Z"
    meters_per_unit: float = 1.0


class CollisionSpec(_M):
    type: Literal["convex_hulls", "convex_hull", "box"] = "convex_hulls"
    file: str
    n_hulls: int = Field(ge=1)


class AnchorSpec(_M):
    name: str
    kind: Literal["support", "port"]
    pos: tuple[float, float, float]
    yaw_deg: float = 0.0
    polygon: list[tuple[float, float]] | None = None
    port_kind: str | None = None
    radius: float | None = None
    meta: dict = Field(default_factory=dict)

    def to_anchor(self, scale: float = 1.0) -> Anchor:
        poly = None if self.polygon is None else [(x * scale, y * scale) for x, y in self.polygon]
        return Anchor(self.name, self.kind, tuple(c * scale for c in self.pos), qyaw(math.radians(self.yaw_deg)), poly,
                      self.port_kind, None if self.radius is None else self.radius * scale, dict(self.meta))


class Provenance(_M):
    original_file: str
    sha256: str
    importer_version: str
    decomposition: Literal["coacd", "single_hull", "box"]
    warnings: list[str] = Field(default_factory=list)


class VisualXform(_M):
    """Maps the referenced original asset onto the normalised frame: p_norm = translate + rotX(rotate_x_deg) * scale * p."""
    translate: tuple[float, float, float]
    rotate_x_deg: float = 0.0
    scale: float = 1.0


class MeshSidecar(_M):
    id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    source: str  # visual USD, relative to the sidecar
    up_axis: Literal["Z"] = "Z"
    meters_per_unit: Literal[1.0] = 1.0
    original: Original = Field(default_factory=Original)
    bounds: Bounds
    footprint: list[tuple[float, float]]  # convex hull of the xy projection, CCW, local frame
    mass_kg: float = Field(gt=0)
    material: str = "steel"
    friction: float | None = None
    body: Literal["static", "dynamic", "kinematic"] = "dynamic"
    collision: CollisionSpec
    anchors: list[AnchorSpec] = Field(default_factory=list)
    articulation: str | None = None
    thumbnail: str | None = None
    provenance: Provenance
    # catalog assets: the scene references the original by URL instead of the normalised local copy
    visual_url: str | None = None
    visual_xform: VisualXform | None = None
    disable_physics: list[str] = Field(default_factory=list)  # prims (relative to the default prim) whose own physics is switched off

    @field_validator("material")
    @classmethod
    def _mat(cls, v):
        if v not in PHYSICAL:
            raise ValueError(f"unknown physics material {v!r}; choose from {sorted(PHYSICAL)}")
        return v

    @field_validator("footprint")
    @classmethod
    def _fp(cls, v):
        if len(v) < 3:
            raise ValueError("footprint needs >= 3 points")
        return v

    def dump(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)


def write_sidecar(sc: MeshSidecar, path: Path) -> Path:
    Path(path).write_text(yaml.safe_dump(sc.dump(), sort_keys=False, default_flow_style=None))
    return Path(path)


def read_sidecar(path: Path) -> MeshSidecar:
    return MeshSidecar.model_validate(yaml.safe_load(Path(path).read_text()))


# --------------------------------------------------------------------------- hull loading
@functools.lru_cache(maxsize=64)
def _load_hulls(file: str, mtime: float) -> tuple:
    """Hull meshes (points (n,3), face_counts, face_indices) from a collision USD, in file order."""
    from pxr import Usd

    stage = Usd.Stage.Open(file)
    out = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts = np.array(m.GetPointsAttr().Get(), float).reshape(-1, 3)
            out.append((pts, list(m.GetFaceVertexCountsAttr().Get()), list(m.GetFaceVertexIndicesAttr().Get())))
    return tuple(out)


def load_hulls(file: Path) -> tuple:
    return _load_hulls(str(file), os.path.getmtime(file))


def _rect(sx, sy):
    return [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]


# --------------------------------------------------------------------------- asset
class MeshParams(Params):
    scale: float = Field(1.0, ge=0.1, le=10.0, description="Uniform scale applied to visual, colliders, anchors")
    mass_kg: float | None = Field(None, gt=0, description="Override the sidecar mass (kg, after scaling)")
    body: Literal["static", "dynamic", "kinematic"] | None = Field(None, description="Override the sidecar body type")


class MeshAsset(Asset):
    source = "mesh"

    def __init__(self, sidecar: MeshSidecar, base_dir: Path):
        self.sidecar = sidecar
        self.base_dir = Path(base_dir)
        self.id = sidecar.id
        self.description = sidecar.description or f"Imported mesh asset {sidecar.id}."
        self.tags = frozenset(sidecar.tags) | {"mesh"}
        mu = sidecar.friction if sidecar.friction is not None else PHYSICAL[sidecar.material][0]
        self.physics = PhysicsInfo(sidecar.body, sidecar.material, mu, sidecar.mass_kg, sidecar.collision.type,
                                   f"{sidecar.collision.n_hulls} convex hull(s); decomposition={sidecar.provenance.decomposition}")
        self.Params = MeshParams

    @classmethod
    def from_file(cls, sidecar_path: Path) -> "MeshAsset":
        sidecar_path = Path(sidecar_path)
        return cls(read_sidecar(sidecar_path), sidecar_path.parent)

    @property
    def visual_file(self) -> Path:
        return self.base_dir / self.sidecar.source

    @property
    def collision_file(self) -> Path:
        return self.base_dir / self.sidecar.collision.file

    # -- planning
    def footprint(self, p) -> Footprint:
        s, b = p.scale, self.sidecar.bounds
        return Footprint([(x * s, y * s) for x, y in self.sidecar.footprint], b.min[2] * s, b.max[2] * s)

    def anchors(self, p) -> list[Anchor]:
        return [a.to_anchor(p.scale) for a in self.sidecar.anchors]

    def _mass(self, p) -> float:
        return float(p.mass_kg if p.mass_kg is not None else self.sidecar.mass_kg * p.scale**3)

    # -- authoring
    def build(self, ctx: BuildContext, path: str, pose: Pose, p, rng) -> Built:
        sc, k, s = self.sidecar, ctx.kit, p.scale
        body = p.body or sc.body
        mass = self._mass(p)
        hulls = load_hulls(self.collision_file)
        # convex decomposition can bulge a few mm below the visual's bottom: lift dynamic bodies so their hulls start on the support
        sink = max(0.0, -min(float(pts[:, 2].min()) for pts, _, _ in hulls) * s) if hulls and body == "dynamic" else 0.0
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + (0.001 + sink if body == "dynamic" else 0.0))
        parts = []
        for pts, _, _ in hulls:
            lo, hi = pts.min(0) * s, pts.max(0) * s
            parts.append(Part("box", tuple(np.maximum(hi - lo, 1e-3)), tuple((lo + hi) / 2)))
        if body == "static":
            root = k.xform(path, pos, pose.quat)
        else:
            root = k.rigid_body(path, pos, pose.quat, parts=parts, mass=mass, kinematic=(body == "kinematic"))
        vis = k.xform(f"{path}/Visual")
        if sc.visual_url:
            # catalog asset: reference the original (its materials render in Isaac Sim), mapped onto the normalised frame
            vx = sc.visual_xform or VisualXform(translate=(0.0, 0.0, 0.0))
            vis.GetReferences().AddReference(sc.visual_url)
            xf = UsdGeom.Xformable(vis)
            xf.ClearXformOpOrder()  # k.xform already added an identity translate
            xf.AddTranslateOp().Set(Gf.Vec3d(*(s * t for t in vx.translate)))
            if vx.rotate_x_deg:
                xf.AddRotateXOp().Set(vx.rotate_x_deg)
            if s * vx.scale != 1.0:
                xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*(3 * [s * vx.scale])))
            for rel in sc.disable_physics:  # our hulls are the only colliders; never nest the asset's own bodies
                over = k.stage.OverridePrim(f"{path}/Visual/{rel}" if rel else f"{path}/Visual")
                over.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                over.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        else:
            # visual: USD reference to the normalised visual file
            ref = self.visual_file
            try:
                asset_path = "./" + os.path.relpath(ref, Path(k.path).resolve().parent)
            except ValueError:
                asset_path = str(ref)
            vis.GetReferences().AddReference(asset_path)
            if s != 1.0:
                UsdGeom.Xformable(vis).AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(s, s, s))
        # collision: precomputed convex hulls
        pmat = self._phys_material(ctx)
        k.scope(f"{path}/Collision")
        for i, (pts, counts, idx) in enumerate(hulls):
            prim = k.mesh(f"{path}/Collision/hull_{i:02d}", pts * s, counts, idx)
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            k.bind(prim, pmat, "physics")
        cat = sorted(sc.tags)[0] if sc.tags else "mesh_asset"
        k.tag(root, cat, zone=ctx.zone or "")
        size = tuple(v * s for v in sc.bounds.size)
        ctx.record(path, cat, pos, pose.quat, body, size=size, mass=mass if body != "static" else None,
                   friction=self.physics.friction, asset_id=sc.id, n_collision_shapes=len(hulls))
        anchors = [a.to_world(Pose(pos, pose.quat)) for a in self.anchors(p)]
        return Built(path, anchors, info={"n_hulls": len(hulls), "scale": s, "mass_kg": mass},
                     bodies=[path] if body != "static" else [])

    def _phys_material(self, ctx):
        sc = self.sidecar
        mu0, rest = PHYSICAL[sc.material]
        if sc.friction is None or abs(sc.friction - mu0) < 1e-9:
            return ctx.mats.phys(sc.material)
        return ctx.kit.phys_material(f"{sc.material}_mu{int(round(sc.friction * 100)):03d}", sc.friction, rest)


# --------------------------------------------------------------------------- library
def load_library(directory: Path | str = DEFAULT_LIBRARY) -> list[MeshAsset]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return [MeshAsset.from_file(f) for f in sorted(d.glob(f"**/*{SIDECAR_SUFFIX}"))]


def _library_loader():
    return load_library(DEFAULT_LIBRARY)


registry.add_loader(_library_loader)
if registry._loaded:  # registry already finished loading before this module was imported: register now
    for _a in _library_loader():
        registry.register(_a)
