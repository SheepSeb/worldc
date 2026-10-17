"""LLM-generated assets: a declarative spec (``GeneratedAssetSpec``) compiled by a fixed builder (PLAN.md §2.7).

The LLM never writes code that we execute.  It writes a ``GeneratedAssetSpec`` (primitives, links/joints, anchors,
materials, optional ``TextureSpec``); :func:`validate_generated` reports structured, actionable ``Issue``s with JSON
pointers into the spec; :class:`GeneratedAsset` compiles a valid spec through ``procedural/_assembly.py`` (``emit``,
``body``, ``Art``) so it gets the same colliders, inertia and articulation handling as hand-written assets.
:func:`promote` stores a validated spec as ``<library>/generated/<slug>.yaml``; the registry loader registers all
promoted specs found in ``assets/library/generated/`` (this module must be imported for the loader to be registered).

Conventions: asset frame is Z-up, metres, kg; the asset's lowest point should be at z=0 (pose z = support height);
joint angles are degrees, prismatic travel metres; part ``pos`` is the *centre* of the primitive; cylinder/capsule/cone
axis is ``z`` unless ``axis`` says otherwise, then ``rpy_deg``/``yaw_deg`` rotates the part.  Mass of a body is shared
over its *colliding* parts by volume; with ``density_kg_m3`` the mass is density x total colliding volume.  Textures are
applied to ``box`` parts only (UVs exist only on the box mesh).  Dynamic assets are placed with a 0.6 mm settle gap.
Not modelled: keep-out zones (door swing) -- add them in the layout spec.

Overlap policy.  Colliding parts inside ONE body (the root, or one link) that interpenetrate by more than 5 mm
(shrunk-OBB SAT, ``geom.obb_penetration``) are a *warning* (``gen.part_overlap``), not an error: compound bodies
routinely overlap (legs sunk into a top, a handle through a door) and PhysX handles overlapping shapes of one body.
The warning stays because overlap double-counts volume in the mass share / inertia and usually signals a placement
mistake.  Setting ``allow_overlap: true`` on either part declares it intentional and silences it.  Overlap between
parts of different bodies is never reported (articulation self-collision is disabled).

Issue codes (severity: E error / W warning)
-------------------------------------------
=================================  ===  ==============================================================================
code                               sev  meaning
=================================  ===  ==============================================================================
gen.schema                         E    pydantic schema violation (missing/unknown field, wrong type, bad enum, ...)
gen.id_invalid                     E    id must be ``gen/<slug>`` with slug ``[a-z0-9][a-z0-9_]*`` (<= 63 chars)
gen.name_invalid                   E    part/link/anchor name is not ``[A-Za-z_][A-Za-z0-9_]*`` (<= 48) or link is "root"
gen.name_duplicate                 E    duplicate part, link or anchor name
gen.mass_missing                   E    dynamic/articulated body without mass_kg or density_kg_m3
gen.mass_ambiguous                 E    both mass_kg and density_kg_m3 given
gen.mass_nonpositive               E    mass_kg or density_kg_m3 <= 0
gen.mass_ignored                   W    mass given for a static body (ignored)
gen.mass_implausible               W    effective density of the colliding volume outside 20..22000 kg/m3
gen.part_size_range                E    a primitive dimension is outside [0.002, 30] m
gen.asset_size_range               E    overall extent (largest AABB side) outside [0.01, 50] m
gen.origin_off_floor               W    lowest point of the asset is not at z=0 (|z0| > 0.05 m)
gen.material_unknown               E    material preset not in materials.VISUAL (hint: close matches / inline form)
gen.physics_material_unknown       E    physics_material not in materials.PHYSICAL
gen.texture_unsupported_shape      W    texture on a non-box part is ignored (only boxes carry UVs)
gen.no_collider                    E/W  body has no colliding part (error for dynamic/kinematic/articulated root+links,
                                        warning for static)
gen.part_overlap                   W    colliding parts of one body interpenetrate (see overlap policy)
gen.unstable                       E    dynamic asset's centre of mass is outside the support polygon on the floor
gen.marginal_stability             W    COM inside but < 2 cm from the support polygon edge
gen.no_support_polygon             E    dynamic asset touches the floor at fewer than 3 non-collinear points
gen.links_on_non_articulated       E    ``links`` given but body is not "articulated"
gen.articulated_without_links      E    body "articulated" but ``links`` is empty
gen.joint_limits                   E    joint limits lo >= hi
gen.joint_default_range            E    joint ``default`` outside [lo, hi]
gen.joint_parent_unknown           E    joint parent is neither "root" nor another link
gen.joint_cycle                    E    link parent chain loops (including parent == itself)
gen.anchor_part_unknown            E    ``auto_from`` names a part that does not exist
gen.anchor_invalid                 E    support needs ``polygon``+``pos`` xor ``auto_from``; port needs ``pos``+``port_kind``
=================================  ===  ==============================================================================
"""
from __future__ import annotations

import difflib
import math
import re
import warnings
from pathlib import Path
from typing import Annotated, Literal, Union

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..geom import OBB, Part, Q_IDENT, composite_inertia, obb_penetration, qmat, qmul, qrpy, qyaw
from ..materials import PHYSICAL, VISUAL
from ..spec.issues import Issue, error, has_errors, pointer, warning
from ..texturegen import TextureSpec, texture_material
from .base import Anchor, Asset, BuildContext, Built, Footprint, JointInfo, Params, PhysicsInfo, Pose
from .procedural._assembly import Art, Q_X, Q_Y, Sh, body as _body, emit, shape_part, static as _static
from .registry import add_loader

LIBRARY_GENERATED_DIR = Path(__file__).parent / "library" / "generated"

PART_MIN, PART_MAX = 0.002, 30.0  # m, per primitive dimension
ASSET_MIN, ASSET_MAX = 0.01, 50.0  # m, overall extent
OVERLAP_TOL = 0.005  # m
SETTLE_GAP = 0.0006  # m, dynamic assets are dropped from this height
SHAPES = ("box", "cylinder", "capsule", "sphere", "cone")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,47}$")
_ID_RE = re.compile(r"^gen/[a-z0-9][a-z0-9_]{0,62}$")

Vec3 = tuple[float, float, float]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]


# --------------------------------------------------------------------------- spec models
class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InlineMaterial(_M):
    rgb: tuple[Unit, Unit, Unit] = Field(description="Base colour, RGB floats 0..1.")
    roughness: Unit = Field(0.5, description="PBR roughness 0 (mirror) .. 1 (matte).")
    metallic: Unit = Field(0.0, description="PBR metallic 0 (dielectric) .. 1 (metal).")


class _PartBase(_M):
    name: str = Field(description="Unique part name (letters, digits, underscore; <= 48 chars); becomes the USD prim name.")
    pos: Vec3 = Field((0.0, 0.0, 0.0), description="Centre of the primitive in the asset frame, metres (Z up). Link parts use the asset frame too.")
    rpy_deg: Vec3 | None = Field(None, description="Part rotation (roll, pitch, yaw) in degrees, applied after `axis`. Mutually exclusive with yaw_deg.")
    yaw_deg: float | None = Field(None, description="Shortcut for rotation about Z only, degrees. Mutually exclusive with rpy_deg.")
    material: str | InlineMaterial = Field("steel", description="A preset name from materials.VISUAL (e.g. steel, wood, rubber, tote_blue) "
                                           "or inline {rgb: [r,g,b], roughness, metallic}.")
    collide: bool = Field(True, description="Whether the part has a collider. False = visual only (no mass share).")
    allow_overlap: bool = Field(False, description="Declare that this part intentionally interpenetrates other colliding parts of the same body "
                                "(silences the gen.part_overlap warning).")
    texture: TextureSpec | None = Field(None, description="Optional procedural texture (box parts only); replaces `material` for appearance.")


class BoxPart(_PartBase):
    shape: Literal["box"] = "box"
    size: Vec3 = Field(description="Full extents (x, y, z) in metres, each 0.002..30.")


class CylinderPart(_PartBase):
    shape: Literal["cylinder"] = "cylinder"
    radius: float = Field(description="Radius in metres, 0.002..30.")
    height: float = Field(description="Length along the axis in metres, 0.002..30.")
    axis: Literal["x", "y", "z"] = Field("z", description="Axis of the cylinder before rpy_deg/yaw_deg is applied.")


class CapsulePart(_PartBase):
    shape: Literal["capsule"] = "capsule"
    radius: float = Field(description="Radius in metres, 0.002..30.")
    length: float = Field(description="Length of the cylindrical section (without the end caps) in metres, 0.002..30.")
    axis: Literal["x", "y", "z"] = Field("z", description="Axis of the capsule before rpy_deg/yaw_deg is applied.")


class SpherePart(_PartBase):
    shape: Literal["sphere"] = "sphere"
    radius: float = Field(description="Radius in metres, 0.002..30.")


class ConePart(_PartBase):
    shape: Literal["cone"] = "cone"
    radius: float = Field(description="Base radius in metres, 0.002..30.")
    height: float = Field(description="Height along the axis in metres, 0.002..30.")
    axis: Literal["x", "y", "z"] = Field("z", description="Axis of the cone before rpy_deg/yaw_deg is applied.")


Primitive = Annotated[Union[BoxPart, CylinderPart, CapsulePart, SpherePart, ConePart], Field(discriminator="shape")]


class Drive(_M):
    stiffness: float = Field(0.0, ge=0.0, description="Position-drive stiffness (N/m or Nm/deg). 0 = free joint (use damping for friction).")
    damping: float = Field(1.0, ge=0.0, description="Drive damping.")
    max_force: float = Field(1e4, gt=0.0, description="Maximum drive force/torque.")


class Joint(_M):
    type: Literal["revolute", "prismatic"] = Field(description="revolute (limits in degrees) or prismatic (limits in metres).")
    axis: Literal["X", "Y", "Z"] = Field(description="Joint axis in the asset frame.")
    limits: tuple[float, float] = Field(description="[lower, upper]; must satisfy lower < upper (degrees for revolute, metres for prismatic).")
    parent: str = Field("root", description="Parent link name, or 'root' for the fixed body made of the top-level `parts`.")
    origin: Vec3 = Field(description="Pivot / link frame origin in the asset frame, metres. The link's parts keep asset-frame positions.")
    drive: Drive | None = Field(None, description="Optional position drive. Omit for a passive joint with light damping.")
    default: float = Field(0.0, description="Initial joint position (also the drive target); must lie within limits.")


class Link(_M):
    name: str = Field(description="Unique link name (not 'root').")
    parts: list[Primitive] = Field(description="Primitives of this link; positions are in the asset frame. At least one must collide.")
    joint: Joint = Field(description="Joint connecting this link to its parent.")


class Anchor_(_M):
    name: str = Field(description="Unique anchor name.")
    type: Literal["support", "port"] = Field(description="support = horizontal surface things can rest on; port = snap frame (pipe nozzle, hinge, mount).")
    pos: Vec3 | None = Field(None, description="Anchor origin in the asset frame (required unless support uses auto_from).")
    polygon: list[tuple[float, float]] | None = Field(None, description="Support only: >= 3 xy points relative to pos. Exclusive with auto_from.")
    auto_from: str | None = Field(None, description="Support only: name of a part whose top face becomes the support polygon.")
    yaw_deg: float = Field(0.0, description="Anchor orientation about Z, degrees.")
    port_kind: str | None = Field(None, description="Port only: kind of port, e.g. 'pipe', 'hinge', 'hose', 'mount'.")
    radius: float | None = Field(None, description="Port only: radius of the connection in metres.")


class GeneratedAssetSpec(_M):
    """A new asset described declaratively (primitives + links + anchors)."""

    id: str = Field(description="Namespaced id 'gen/<slug>' (slug: lowercase letters, digits, underscore), e.g. gen/rolling_cart.")
    description: str = Field(description="One-line description shown to the planner.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags used for asset search, e.g. [cart, metal, dynamic].")
    body: Literal["static", "dynamic", "kinematic", "articulated"] = Field(description="Physics body kind.")
    mass_kg: float | None = Field(None, description="Total mass in kg (> 0). Exactly one of mass_kg / density_kg_m3 for dynamic and articulated bodies.")
    density_kg_m3: float | None = Field(None, description="Density in kg/m3 (> 0); mass = density x volume of colliding parts. Alternative to mass_kg.")
    physics_material: str = Field("steel", description="Physics material preset (materials.PHYSICAL): cardboard, steel, concrete, rubber, wood, plastic, grating.")
    parts: list[Primitive] = Field(min_length=1, description="Primitives of the root body (the whole asset unless `links` are given).")
    links: list[Link] = Field(default_factory=list, description="Articulated only: moving links, each joined to a parent by a revolute/prismatic joint.")
    anchors: list[Anchor_] = Field(default_factory=list, description="Named support surfaces and ports exposed to the layout solver.")


class GeneratedAssetError(ValueError):
    """Raised when a spec with errors is compiled or promoted; ``issues`` holds all issues."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("\n".join(str(i) for i in issues if i.severity == "error"))


# --------------------------------------------------------------------------- geometry helpers
def _axis_q(axis: str):
    return {"z": Q_IDENT, "x": Q_X, "y": Q_Y}[axis]


def _quat(p) -> tuple:
    if p.rpy_deg is not None:
        r = qrpy(*[math.radians(a) for a in p.rpy_deg])
    elif p.yaw_deg is not None:
        r = qyaw(math.radians(p.yaw_deg))
    else:
        r = Q_IDENT
    return qmul(r, _axis_q(getattr(p, "axis", "z")))


def _dims(p) -> list[tuple[str, float]]:
    """(field-pointer-name, value) of every size dimension."""
    if p.shape == "box":
        return [(f"size/{i}", v) for i, v in enumerate(p.size)]
    if p.shape in ("cylinder", "cone"):
        return [("radius", p.radius), ("height", p.height)]
    if p.shape == "capsule":
        return [("radius", p.radius), ("length", p.length)]
    return [("radius", p.radius)]


def _half(p, shrink: bool = False) -> np.ndarray:
    """Half extents of the part's local bounding box (shrink: round shapes towards their inscribed box)."""
    k = 0.85 if shrink else 1.0
    if p.shape == "box":
        return np.asarray(p.size, float) / 2
    if p.shape in ("cylinder", "cone"):
        return np.array([p.radius * k, p.radius * k, p.height / 2])
    if p.shape == "capsule":
        return np.array([p.radius * k, p.radius * k, p.length / 2 + p.radius])
    return np.full(3, p.radius * k)


def _obb(p, shrink: bool = False) -> OBB:
    return OBB(np.asarray(p.pos, float), _half(p, shrink), qmat(_quat(p)))


def _volume(p) -> float:
    if p.shape == "box":
        return float(np.prod(p.size))
    if p.shape == "cylinder":
        return math.pi * p.radius**2 * p.height
    if p.shape == "capsule":
        return math.pi * p.radius**2 * p.length + 4 / 3 * math.pi * p.radius**3
    if p.shape == "sphere":
        return 4 / 3 * math.pi * p.radius**3
    return math.pi * p.radius**2 * p.height / 3


def _sh(p, offset=(0.0, 0.0, 0.0), textured_box: bool = False) -> Sh:
    pos = tuple(float(a) - float(b) for a, b in zip(p.pos, offset))
    q = _quat(p)
    col = bool(p.collide)
    if p.shape == "box":
        return Sh("cube", tuple(p.size), pos, q, "steel", col, p.name)
    if p.shape == "cylinder":
        return Sh("cyl", (p.radius, p.height), pos, q, "steel", col, p.name)
    if p.shape == "capsule":
        return Sh("cap", (p.radius, p.length), pos, q, "steel", col, p.name)
    if p.shape == "sphere":
        return Sh("sph", (p.radius,), pos, Q_IDENT, "steel", col, p.name)
    return Sh("cone", (p.radius, p.height), pos, q, "steel", col, p.name)


def _preset(p) -> str:
    return p.material if isinstance(p.material, str) and p.material in VISUAL else "steel"


def _iter_parts(spec: GeneratedAssetSpec):
    """Yield (pointer-parts tuple, body name, part)."""
    for i, p in enumerate(spec.parts):
        yield ("parts", i), "root", p
    for j, ln in enumerate(spec.links):
        for i, p in enumerate(ln.parts):
            yield ("links", j, "parts", i), ln.name, p


def _bodies(spec: GeneratedAssetSpec) -> dict[str, list]:
    out = {"root": list(spec.parts)}
    for ln in spec.links:
        out[ln.name] = list(ln.parts)
    return out


def _corners(p) -> np.ndarray:
    return _obb(p).corners()


def _total_mass(spec: GeneratedAssetSpec) -> float | None:
    if spec.mass_kg is not None:
        return float(spec.mass_kg)
    if spec.density_kg_m3 is not None:
        return float(spec.density_kg_m3) * sum(_volume(p) for _, _, p in _iter_parts(spec) if p.collide)
    return None


def _extent(spec: GeneratedAssetSpec):
    pts = np.vstack([_corners(p) for _, _, p in _iter_parts(spec)])
    return pts.min(0), pts.max(0), pts


def _footprint(spec: GeneratedAssetSpec) -> Footprint:
    from scipy.spatial import ConvexHull

    lo, hi, pts = _extent(spec)
    try:
        hull = ConvexHull(pts[:, :2])
        poly = [tuple(float(c) for c in pts[i, :2]) for i in hull.vertices]  # CCW in 2-D
    except Exception:
        poly = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    return Footprint(poly, float(lo[2]), float(hi[2]))


def _top_face(p) -> tuple[tuple, list]:
    """(anchor pos, polygon relative to pos) of the top face of a part."""
    o = _obb(p)
    c = o.c
    R = o.R
    if p.shape in ("cylinder", "cone", "capsule", "sphere") and abs(R[2, 2]) > 0.9999 or p.shape == "sphere":
        r = p.radius
        z = c[2] + (r if p.shape == "sphere" else (p.length / 2 + r if p.shape == "capsule" else p.height / 2))
        n = 12
        return (float(c[0]), float(c[1]), float(z)), [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    pts = o.corners()
    zmax = pts[:, 2].max()
    top = pts[pts[:, 2] > zmax - 1e-6][:, :2]
    if len(np.unique(np.round(top, 6), axis=0)) >= 3:
        from scipy.spatial import ConvexHull

        h = ConvexHull(top)
        ctr = top[h.vertices].mean(0)
        return (float(ctr[0]), float(ctr[1]), float(zmax)), [tuple(float(v) for v in (top[i] - ctr)) for i in h.vertices]
    lo, hi = pts.min(0), pts.max(0)
    ctr = (lo[:2] + hi[:2]) / 2
    hx, hy = (hi[:2] - lo[:2]) / 2
    return (float(ctr[0]), float(ctr[1]), float(zmax)), [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]


def _anchors(spec: GeneratedAssetSpec) -> list[Anchor]:
    parts = {p.name: p for _, _, p in _iter_parts(spec)}
    out = []
    for a in spec.anchors:
        q = qyaw(math.radians(a.yaw_deg))
        if a.type == "support":
            if a.auto_from is not None:
                pos, poly = _top_face(parts[a.auto_from])
            else:
                pos, poly = tuple(a.pos), [tuple(pt) for pt in a.polygon]
            out.append(Anchor(a.name, "support", tuple(float(v) for v in pos), q, polygon=poly, meta={"generated": True}))
        else:
            out.append(Anchor(a.name, "port", tuple(float(v) for v in a.pos), q, port_kind=a.port_kind, radius=a.radius, meta={"generated": True}))
    return out


# --------------------------------------------------------------------------- validation
_FIELD_NAMES: set[str] = set()


def _collect_fields():
    if _FIELD_NAMES:
        return
    for m in (InlineMaterial, BoxPart, CylinderPart, CapsulePart, SpherePart, ConePart, Drive, Joint, Link, Anchor_, GeneratedAssetSpec, TextureSpec):
        _FIELD_NAMES.update(m.model_fields)


def _schema_issues(e: ValidationError) -> list[Issue]:
    _collect_fields()
    out = []
    for err in e.errors(include_url=False):
        loc = []
        for i, part in enumerate(err["loc"]):
            if isinstance(part, str) and part in SHAPES and loc and isinstance(loc[-1], int):
                continue  # discriminator tag injected by pydantic
            loc.append(part)
        t, msg = err["type"], err["msg"]
        last = loc[-1] if loc else ""
        if t == "missing":
            hint = f"Add the required field {last!r}."
        elif t == "extra_forbidden":
            close = difflib.get_close_matches(str(last), sorted(_FIELD_NAMES), n=3)
            hint = f"Remove unknown field {last!r}" + (f" (did you mean {', '.join(close)}?)" if close else "") + "; see GeneratedAssetSpec.model_json_schema()."
        elif t in ("literal_error", "enum"):
            hint = f"Use one of: {err.get('ctx', {}).get('expected', 'see schema')}."
        elif t in ("union_tag_invalid", "union_tag_not_found"):
            hint = f"Set `shape` to one of {', '.join(SHAPES)} (links/anchors/ops have their own discriminator)."
        else:
            hint = "Fix the value to match GeneratedAssetSpec.model_json_schema()."
        out.append(error("gen.schema", msg, pointer(*loc), hint, type=t))
    return out


def _closest(name: str, options, n=3) -> str:
    c = difflib.get_close_matches(name, list(options), n=n, cutoff=0.5)
    return f" Closest: {', '.join(c)}." if c else ""


def validate_generated(spec_or_dict) -> list[Issue]:
    """Validate a ``GeneratedAssetSpec`` (model or plain dict) -> list of Issues (errors and warnings)."""
    if isinstance(spec_or_dict, GeneratedAssetSpec):
        try:
            spec = GeneratedAssetSpec.model_validate(spec_or_dict.model_dump())
        except ValidationError as e:  # re-validation catches models built with model_construct
            return _schema_issues(e)
    else:
        try:
            spec = GeneratedAssetSpec.model_validate(spec_or_dict)
        except ValidationError as e:
            return _schema_issues(e)
    iss: list[Issue] = []
    body = spec.body

    # ---- id / names
    if not _ID_RE.match(spec.id):
        iss.append(error("gen.id_invalid", f"id {spec.id!r} is not a valid generated-asset id.", "/id",
                         "Use 'gen/<slug>' with a lowercase slug of letters, digits and underscores, e.g. 'gen/rolling_cart'."))
    seen: dict[str, tuple] = {}
    for ptr, _, p in _iter_parts(spec):
        if not _NAME_RE.match(p.name):
            iss.append(error("gen.name_invalid", f"part name {p.name!r} is not a valid prim name.", pointer(*ptr, "name"),
                             "Use letters, digits and underscores only (start with a letter), at most 48 characters."))
        if p.name in seen:
            iss.append(error("gen.name_duplicate", f"part name {p.name!r} is used twice (also at {pointer(*seen[p.name])}).", pointer(*ptr, "name"),
                             "Give every part (including parts inside links) a unique name."))
        seen.setdefault(p.name, ptr)
    lnames: dict[str, int] = {}
    for j, ln in enumerate(spec.links):
        if not _NAME_RE.match(ln.name) or ln.name == "root":
            iss.append(error("gen.name_invalid", f"link name {ln.name!r} is invalid or reserved.", pointer("links", j, "name"),
                             "Use letters, digits, underscores; 'root' is reserved for the top-level parts."))
        if ln.name in lnames:
            iss.append(error("gen.name_duplicate", f"link name {ln.name!r} is used twice.", pointer("links", j, "name"), "Rename one of the links."))
        lnames.setdefault(ln.name, j)
    anames: set[str] = set()
    for k, a in enumerate(spec.anchors):
        if a.name in anames:
            iss.append(error("gen.name_duplicate", f"anchor name {a.name!r} is used twice.", pointer("anchors", k, "name"), "Rename one of the anchors."))
        anames.add(a.name)

    # ---- mass
    has_m, has_d = spec.mass_kg is not None, spec.density_kg_m3 is not None
    if body in ("dynamic", "articulated"):
        if not has_m and not has_d:
            iss.append(error("gen.mass_missing", f"{body} asset needs a mass.", "/mass_kg",
                             "Set mass_kg (e.g. 14.0) or density_kg_m3 (e.g. 7800 for solid steel, ~500 for a hollow steel structure)."))
        if has_m and has_d:
            iss.append(error("gen.mass_ambiguous", "Both mass_kg and density_kg_m3 are set.", "/density_kg_m3", "Keep exactly one of them."))
    elif has_m or has_d:
        iss.append(warning("gen.mass_ignored", f"{body} assets have no dynamic mass; the value is ignored.", "/mass_kg" if has_m else "/density_kg_m3",
                           "Remove it, or change body to dynamic."))
    if has_m and spec.mass_kg <= 0:
        iss.append(error("gen.mass_nonpositive", f"mass_kg={spec.mass_kg} must be > 0.", "/mass_kg", "Use a positive mass in kg, e.g. 5.0."))
    if has_d and spec.density_kg_m3 <= 0:
        iss.append(error("gen.mass_nonpositive", f"density_kg_m3={spec.density_kg_m3} must be > 0.", "/density_kg_m3", "Use a positive density, e.g. 7800 for steel."))

    # ---- sizes
    size_ok = True
    for ptr, _, p in _iter_parts(spec):
        for fld, v in _dims(p):
            if not (PART_MIN <= v <= PART_MAX):
                size_ok = False
                iss.append(error("gen.part_size_range", f"{p.name}.{fld.replace('/', '[')}{']' if '/' in fld else ''} = {v} m is outside [{PART_MIN}, {PART_MAX}] m.",
                                 pointer(*ptr, *fld.split("/")),
                                 "Use metres: thinnest sheet 0.002 m, largest primitive 30 m. Combine several primitives for bigger things."))
    if size_ok:
        lo, hi, _ = _extent(spec)
        big = float((hi - lo).max())
        if not (ASSET_MIN <= big <= ASSET_MAX):
            iss.append(error("gen.asset_size_range", f"Overall largest dimension {big:.3f} m is outside [{ASSET_MIN}, {ASSET_MAX}] m.", "/parts",
                             "Check units (metres, not mm/cm) and part positions."))
        if abs(float(lo[2])) > 0.05:
            iss.append(warning("gen.origin_off_floor", f"Lowest point is at z={lo[2]:.3f} m; the asset frame should have its bottom at z=0.", "/parts",
                               f"Shift all part z positions by {-lo[2]:+.3f} m so pose z equals the support height."))

    # ---- materials / textures
    for ptr, _, p in _iter_parts(spec):
        if isinstance(p.material, str) and p.material not in VISUAL:
            iss.append(error("gen.material_unknown", f"Unknown material preset {p.material!r}.", pointer(*ptr, "material"),
                             "Pick a preset from materials.VISUAL." + _closest(p.material, VISUAL)
                             + " Or give an inline material {rgb: [r,g,b], roughness: 0.5, metallic: 0.0}."))
        if p.texture is not None and p.shape != "box":
            iss.append(warning("gen.texture_unsupported_shape", f"Texture on a {p.shape} is ignored (only box parts carry UVs).", pointer(*ptr, "texture"),
                               "Make this part a box, or drop `texture` and use a material."))
    if spec.physics_material not in PHYSICAL:
        iss.append(error("gen.physics_material_unknown", f"Unknown physics material {spec.physics_material!r}.", "/physics_material",
                         "Use one of: " + ", ".join(PHYSICAL) + "." + _closest(spec.physics_material, PHYSICAL, 2)))

    # ---- links / joints
    if spec.links and body != "articulated":
        iss.append(error("gen.links_on_non_articulated", f"`links` given but body is {body!r}.", "/links", "Set body: articulated, or remove the links."))
    if body == "articulated" and not spec.links:
        iss.append(error("gen.articulated_without_links", "body is articulated but there are no links.", "/links",
                         "Add at least one link with a revolute/prismatic joint, or use body: static/dynamic."))
    parent_of = {}
    for j, ln in enumerate(spec.links):
        jt = ln.joint
        jp = ("links", j, "joint")
        lo_, hi_ = jt.limits
        if lo_ >= hi_:
            iss.append(error("gen.joint_limits", f"Joint limits [{lo_}, {hi_}] for link {ln.name!r} must satisfy lower < upper.", pointer(*jp, "limits"),
                             "Swap or widen the limits, e.g. [0, 110] degrees for a door, [0, 0.35] m for a drawer."))
        elif not (lo_ <= jt.default <= hi_):
            iss.append(error("gen.joint_default_range", f"Joint default {jt.default} is outside limits [{lo_}, {hi_}].", pointer(*jp, "default"),
                             "Set default within the limits (usually the closed position)."))
        if jt.parent != "root" and jt.parent not in lnames:
            iss.append(error("gen.joint_parent_unknown", f"Link {ln.name!r} has unknown parent {jt.parent!r}.", pointer(*jp, "parent"),
                             "Parent must be 'root' or another link's name." + _closest(jt.parent, ["root", *lnames])))
        else:
            parent_of[ln.name] = jt.parent
    for j, ln in enumerate(spec.links):
        cur, seen_ = ln.name, set()
        while cur in parent_of and cur not in seen_:
            seen_.add(cur)
            cur = parent_of[cur]
            if cur == ln.name:
                iss.append(error("gen.joint_cycle", f"Link {ln.name!r} is its own ancestor.", pointer("links", j, "joint", "parent"),
                                 "Make the parent chain end at 'root'."))
                break

    # ---- colliders / overlap / stability
    for bname, parts in _bodies(spec).items():
        ptr_b = ("parts",) if bname == "root" else ("links", next(j for j, l in enumerate(spec.links) if l.name == bname), "parts")
        cols = [(i, p) for i, p in enumerate(parts) if p.collide]
        if not cols:
            sev = warning if body == "static" else error
            iss.append(sev("gen.no_collider", f"Body {bname!r} has no colliding part.", pointer(*ptr_b),
                           "Keep collide: true on at least one structural part" + ("" if body == "static" else " (needed for mass and inertia).")))
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                (ia, pa), (ib, pb) = cols[a], cols[b]
                if pa.allow_overlap or pb.allow_overlap:
                    continue
                pen = obb_penetration(_obb(pa, True), _obb(pb, True))
                if pen > OVERLAP_TOL:
                    iss.append(warning("gen.part_overlap", f"Colliding parts {pa.name!r} and {pb.name!r} of body {bname!r} overlap by {pen * 1000:.0f} mm.",
                                       pointer(*ptr_b, ib),
                                       f"Move them apart, make one collide: false, or set allow_overlap: true on {pb.name!r} if the overlap is intentional "
                                       "(overlap double-counts volume in the mass share).", a=pa.name, b=pb.name, penetration_m=round(pen, 4)))
    if body == "dynamic" and size_ok:
        iss.extend(_stability_issues(spec))
    if size_ok and _total_mass(spec):
        m, vol = _total_mass(spec), sum(_volume(p) for _, _, p in _iter_parts(spec) if p.collide)
        if m > 0 and vol > 0 and body in ("dynamic", "articulated") and spec.mass_kg is not None:
            dens = m / vol
            if not (20.0 <= dens <= 22000.0):
                iss.append(warning("gen.mass_implausible", f"Mass {m:.2f} kg over {vol:.4f} m3 of colliding parts is {dens:.0f} kg/m3 (plausible 20..22000).",
                                   "/mass_kg", "Check the mass (kg) and the part sizes (m); for mostly-hollow assets this can be legitimately low."))

    # ---- anchors
    parts_by_name = {p.name: p for _, _, p in _iter_parts(spec)}
    for k, a in enumerate(spec.anchors):
        ap = ("anchors", k)
        if a.type == "support":
            if (a.auto_from is None) == (a.polygon is None):
                iss.append(error("gen.anchor_invalid", "A support anchor needs exactly one of `polygon` or `auto_from`.", pointer(*ap),
                                 "Use auto_from: <part name> for the top face of a part, or polygon: [[x,y],...] with pos."))
            elif a.polygon is not None and (a.pos is None or len(a.polygon) < 3):
                iss.append(error("gen.anchor_invalid", "An explicit support polygon needs `pos` and >= 3 points.", pointer(*ap, "polygon"),
                                 "Add pos: [x, y, z] and at least three [x, y] points relative to it."))
            if a.auto_from is not None and a.auto_from not in parts_by_name:
                iss.append(error("gen.anchor_part_unknown", f"auto_from references unknown part {a.auto_from!r}.", pointer(*ap, "auto_from"),
                                 "Use the `name` of an existing part." + _closest(a.auto_from, parts_by_name)))
        else:
            if a.pos is None or not a.port_kind:
                iss.append(error("gen.anchor_invalid", "A port anchor needs `pos` and `port_kind`.", pointer(*ap),
                                 "Add pos: [x, y, z] and port_kind such as 'pipe', 'hinge', 'hose' or 'mount'."))
    return iss


def _stability_issues(spec: GeneratedAssetSpec) -> list[Issue]:
    """Static stability of a dynamic asset resting on the floor in its as-authored orientation."""
    from scipy.spatial import ConvexHull

    cols = [p for _, _, p in _iter_parts(spec) if p.collide]
    if not cols:
        return []
    inertia_parts = [pp for pp in (shape_part(_sh(p)) for p in cols) if pp is not None]
    com, _, _, _ = composite_inertia(inertia_parts, 1.0)
    pts, owner = [], []
    for p in cols:
        c = _corners(p)
        pts.append(c)
        owner += [p.name] * len(c)
    pts = np.vstack(pts)
    zmin = pts[:, 2].min()
    sel = pts[:, 2] < zmin + 0.004
    contact = pts[sel][:, :2]
    names = sorted({n for n, s in zip(owner, sel) if s})
    ptr = pointer("parts")
    try:
        hull = ConvexHull(contact)
    except Exception:
        return [error("gen.no_support_polygon", f"The asset touches the floor at fewer than 3 non-collinear points (parts at the bottom: {', '.join(names)}).", ptr,
                      "Give it a base: add a plate or at least three feet/wheels whose lowest points are at the same height and not in a line.")]
    margin = float(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]).max())
    if margin > 0.02:
        return []
    bb = (contact.min(0), contact.max(0))
    info = f"COM xy=({com[0]:.3f}, {com[1]:.3f}) vs support polygon x[{bb[0][0]:.3f}, {bb[1][0]:.3f}] y[{bb[0][1]:.3f}, {bb[1][1]:.3f}] spanned by {', '.join(names)}"
    if margin > 0.005:
        return [warning("gen.marginal_stability", f"Resting margin is only {margin * 100:.1f} cm; it may tip. {info}.", ptr,
                        "Widen the footprint of the lowest parts or lower the centre of mass.", margin_m=round(margin, 4))]
    hint = "Widen the support polygon (larger base plate / wider feet under the COM) or move heavy parts towards its centre"
    try:
        from .procedural.logistics import stable_orientation

        proxies = []
        for p in cols:
            o = _obb(p)
            proxies.append(Part("box", tuple(2 * (np.abs(o.R) @ o.h)), tuple(o.c)))
        R, _, m2 = stable_orientation(proxies, com)
        d = -R[2, :]
        hint += f"; alternatively author it so the authored direction ({d[0]:+.0f}, {d[1]:+.0f}, {d[2]:+.0f}) points down (stable margin {m2 * 100:.0f} cm)"
    except Exception:
        pass
    return [error("gen.unstable", f"Not statically stable on the floor: COM lies {-margin * 100:.1f} cm outside the support polygon. {info}.", ptr,
                  hint + ".", margin_m=round(margin, 4))]


# --------------------------------------------------------------------------- asset
class _NoParams(Params):
    """Generated assets are fully described by their spec and take no instance params."""


class GeneratedAsset(Asset):
    source = "generated"
    Params = _NoParams

    def __init__(self, spec: GeneratedAssetSpec | dict):
        if isinstance(spec, dict):
            issues = validate_generated(spec)
            if has_errors(issues):
                raise GeneratedAssetError(issues)
            spec = GeneratedAssetSpec.model_validate(spec)
        else:
            issues = validate_generated(spec)
            if has_errors(issues):
                raise GeneratedAssetError(issues)
        self.spec = spec
        self.issues = issues  # warnings only
        self.id = spec.id
        self.description = spec.description
        self.tags = frozenset(spec.tags) | {"generated", spec.body}
        self.example = None
        mass = _total_mass(spec) if spec.body in ("dynamic", "articulated") else None
        self._mass = mass
        self.physics = PhysicsInfo(spec.body, spec.physics_material, PHYSICAL[spec.physics_material][0], mass,
                                   "primitives" if any(p.collide for _, _, p in _iter_parts(spec)) else "none")
        self._fp = _footprint(spec)
        self._anchors = _anchors(spec)

    # -- planning
    def params(self, raw=None, rng=None):
        if raw:
            from .base import ParamError

            keys = raw.keys() if isinstance(raw, dict) else raw.model_fields_set
            raise ParamError(self.id, [{"loc": (k,), "msg": "generated assets take no params", "type": "extra_forbidden"} for k in keys])
        return _NoParams()

    def footprint(self, p=None) -> Footprint:
        return Footprint(list(self._fp.polygon), self._fp.z0, self._fp.z1)

    def anchors(self, p=None) -> list[Anchor]:
        return list(self._anchors)

    def joints(self, p=None) -> list[JointInfo]:
        out = []
        for ln in self.spec.links:
            j = ln.joint
            out.append(JointInfo(ln.name, j.type, j.axis, j.limits[0], j.limits[1], "deg" if j.type == "revolute" else "m", j.default,
                                 j.drive.model_dump() if j.drive else None, ln.name, ln.name))
        return out

    def describe(self, max_anchors: int = 24) -> dict:
        d = super().describe(max_anchors)
        d["spec"] = self.spec.model_dump(mode="json", exclude_none=True)
        d["warnings"] = [i.model_dump(mode="json") for i in self.issues]
        return d

    # -- authoring
    def _decorate(self, ctx: BuildContext, root: str, parts, offset=(0.0, 0.0, 0.0)):
        """Apply inline materials; add UV'd visual meshes for textured boxes."""
        k = ctx.kit
        for p in parts:
            prim = k.stage.GetPrimAtPath(f"{root}/{p.name}")
            if p.shape == "box" and p.texture is not None:
                mat = texture_material(ctx, p.texture)
                pos = tuple(float(a) - float(b) for a, b in zip(p.pos, offset))
                k.mesh_box(f"{root}/{p.name}_vis", tuple(p.size), pos, _quat(p), vmat=mat)
                if prim:
                    from pxr import UsdGeom

                    UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
            elif isinstance(p.material, InlineMaterial) and prim:
                m = p.material
                k.bind(prim, ctx.mats.color(m.rgb, m.roughness, m.metallic))

    def build(self, ctx: BuildContext, path: str, pose: Pose, p, rng) -> Built:
        spec, k = self.spec, ctx.kit
        pm = spec.physics_material
        mass = self._mass
        pos, quat = tuple(float(c) for c in pose.pos), tuple(pose.quat)
        if spec.body == "dynamic":
            pos = (pos[0], pos[1], pos[2] + SETTLE_GAP)

        def shapes(parts, offset=(0.0, 0.0, 0.0)):
            out = []
            for q in parts:
                s = _sh(q, offset)
                s.mat = _preset(q)
                if q.shape == "box" and q.texture is not None and not q.collide:
                    continue  # visual-only textured box: only the UV'd mesh is authored
                out.append(s)
            return out

        bodies: list[str] = []
        if spec.body == "static":
            _static(ctx, path, pos, quat, shapes(spec.parts), pm, category="generated", asset=spec.id)
            self._decorate(ctx, path, spec.parts)
        elif spec.body in ("dynamic", "kinematic"):
            m = mass if mass else 1.0
            prim = _body(ctx, path, pos, quat, shapes(spec.parts), m, pm, kinematic=True) if spec.body == "kinematic" else _body(ctx, path, pos, quat, shapes(spec.parts), m, pm)
            k.tag(prim, "generated", asset=spec.id)
            self._decorate(ctx, path, spec.parts)
            bodies = [path]
        else:
            vols = {b: sum(_volume(q) for q in ps if q.collide) for b, ps in _bodies(spec).items()}
            tot = sum(vols.values())
            bm = {b: mass * v / tot for b, v in vols.items()}
            yaw = float(pose.yaw)
            art = Art(ctx, path, pos, yaw, shapes(spec.parts), bm["root"], "generated", pm, quat=quat, asset=spec.id)
            self._decorate(ctx, f"{path}/root", spec.parts)
            pending = list(spec.links)
            done = {"root"}
            while pending:  # parents before children
                ln = next(l for l in pending if l.joint.parent in done)
                pending.remove(ln)
                j = ln.joint
                dr = j.drive or Drive(stiffness=0.0, damping=2.0)
                art.link(ln.name, j.origin, shapes(ln.parts, j.origin), bm[ln.name], j.type, axis=j.axis, lower=j.limits[0], upper=j.limits[1],
                         parent=j.parent, stiffness=dr.stiffness, damping=dr.damping, max_force=dr.max_force, target=j.default, role=ln.name,
                         meta={"asset": spec.id})
                self._decorate(ctx, f"{path}/{ln.name}", ln.parts, j.origin)
                done.add(ln.name)
            bodies = [v["path"] for v in art.links.values()]
        fp = self._fp
        rec_kw = dict(size=fp.size, mass=mass, friction=PHYSICAL[pm][0], asset_id=spec.id, subtype=spec.id)
        if spec.body == "articulated":
            rec_kw["articulation"] = art.name
        ctx.record(path, "generated", pos, quat, spec.body, **rec_kw)
        built = Built(path, anchors=[a.to_world(pose) for a in self._anchors], info={"mass_kg": mass, "asset": spec.id}, bodies=bodies)
        return built


# --------------------------------------------------------------------------- promotion / registry
def _as_spec(spec) -> GeneratedAssetSpec:
    issues = validate_generated(spec)
    if has_errors(issues):
        raise GeneratedAssetError(issues)
    return spec if isinstance(spec, GeneratedAssetSpec) else GeneratedAssetSpec.model_validate(spec)


def promote(spec, library_dir) -> Path:
    """Validate ``spec`` and write it to ``library_dir/generated/<id without 'gen/'>.yaml`` (overwrites).  Raises
    :class:`GeneratedAssetError` when the spec has errors."""
    s = _as_spec(spec)
    out = Path(library_dir) / "generated" / (s.id.split("/", 1)[1] + ".yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(s.model_dump(mode="json", exclude_none=True), sort_keys=False))
    return out


def load_promoted(directory=None) -> list[GeneratedAsset]:
    """Load every ``*.yaml`` in ``directory`` (default ``assets/library/generated``); broken files are skipped with a warning."""
    d = Path(directory) if directory else LIBRARY_GENERATED_DIR
    out = []
    for f in sorted(d.glob("*.yaml")) if d.is_dir() else []:
        try:
            out.append(GeneratedAsset(yaml.safe_load(f.read_text())))
        except Exception as e:  # noqa: BLE001 - one bad promoted file must not break the registry
            warnings.warn(f"skipping promoted generated asset {f.name}: {e}", stacklevel=2)
    return out


@add_loader
def _load_promoted_assets():
    return load_promoted()
