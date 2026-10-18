"""Scene specification v1.0 (pydantic v2).  See docs/SPEC.md for the narrative + examples.

Conventions: metres, kilograms, **degrees** for every angle in the spec, Z up, right-handed, x east / y north.
Every object has a stable ``id``; relations and constraints refer to objects, instances, anchors and zones with
*references* (grammar in ``Ref`` below).  Unknown fields are rejected (``extra="forbid"``) so typos surface as errors.

Distribution literals are accepted wherever a param value is: ``{"uniform": [a, b]}``, ``{"randint": [a, b]}``,
``{"choice": [...], "p": [...]}``, ``{"normal": [mu, sigma], "clip": [lo, hi]}``.  They are sampled per instance.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, StringConstraints, Tag, model_validator

SPEC_VERSION = "1.0"

Id = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]
Ref = Annotated[str, StringConstraints(min_length=1, max_length=200)]
"""Reference grammar:
  zone            ``packing``                       a zone id (also layout-generated zones like ``racks.aisle_2``)
  object          ``belt_a``                        all instances of a spec object
  instance        ``racks.3``                       one instance (0-based)
  anchor / port   ``belt_a.surface``, ``racks.3.slot_s0_b1_p0_L2``, ``racks.*.slot_*``  (``*`` wildcards)
  environment     ``env.opening_dock_0``, ``env.floor``
  wall            ``wall:south``
  robot           ``robot.spawn``, goal ids
"""
Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
Rect = Annotated[tuple[float, float, float, float], Field(description="[x0, y0, x1, y1] in metres")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------------------------------------- assets
class FileAssetRef(Strict):
    file: str = Field(description="Path to a user-provided USD / GLB / OBJ / STL file. Always preferred when given.")
    id: str | None = Field(None, description="Asset id to register it under (default: from the file name)")
    mass_kg: float | None = Field(None, gt=0)
    tags: list[str] = Field(default_factory=list)
    body: Literal["static", "dynamic", "kinematic"] | None = None
    material: str | None = Field(None, description="Physics material preset (cardboard, steel, concrete, rubber, wood, plastic, grating)")
    description: str | None = None
    up: Literal["y", "z"] | None = None
    units: Literal["m", "cm", "mm", "in"] | None = None


class CatalogAssetRef(Strict):
    catalog: Literal["omniverse"] = "omniverse"
    query: str | None = Field(None, description="Search text; resolved once and pinned in the resolved spec")
    path: str | None = Field(None, description="Exact catalog asset path relative to the asset root, e.g. Isaac/Props/Pallet/pallet.usd (pinned)")
    rev: str | None = Field(None, description="Content revision (ETag) of `path` from the catalog index; set when a query is pinned")
    tags: list[str] = Field(default_factory=list)
    body: Literal["static", "dynamic", "kinematic"] | None = Field(None, description="Physics body type (default dynamic). Use static for furniture, "
                                                                    "desks, tables and anything that must not move or tip over")
    mass_kg: float | None = Field(None, gt=0, description="Mass; default is estimated from the collision hull volume")
    up_axis: Literal["Y", "Z"] | None = Field(None, description="Override the up axis when the file declares it wrongly (the YCB props are authored "
                                                                "Z-up but modelled Y-up: use Y)")

    @model_validator(mode="after")
    def _one(self):
        if not (self.query or self.path):
            raise ValueError("catalog asset needs `query` or `path`")
        return self


class GenerateAssetRef(Strict):
    generate: dict[str, Any] = Field(description="Declarative GeneratedAssetSpec (see get_spec_schema('GeneratedAssetSpec')); "
                                                 "only when no registry / catalog asset fits")


def _asset_tag(v):
    if isinstance(v, str):
        return "id"
    if isinstance(v, dict):
        for k in ("file", "catalog", "generate"):
            if k in v:
                return k
    for k, cls in (("file", FileAssetRef), ("catalog", CatalogAssetRef), ("generate", GenerateAssetRef)):
        if isinstance(v, cls):
            return k
    return None


AssetRef = Annotated[
    Union[Annotated[str, Tag("id")], Annotated[FileAssetRef, Tag("file")], Annotated[CatalogAssetRef, Tag("catalog")],
          Annotated[GenerateAssetRef, Tag("generate")]],
    Discriminator(_asset_tag, custom_error_type="asset_ref", custom_error_message="asset must be a registry id string or an object with one of: file, catalog, generate"),
]


# ---------------------------------------------------------------------------------------------------------- environment
class SlopeZoneSpec(Strict):
    id: Id
    rect: Rect
    slope: float = Field(0.025, ge=0, le=0.1, description="Ramp gradient")
    max_height: float = Field(0.04, ge=0, le=0.3)


class MarkingSpec(Strict):
    """Painted floor marking (world metres)."""
    type: Literal["line", "dashed", "rect", "fill", "hatch", "text"]
    points: list[Point2] | None = Field(None, description="line / dashed: polyline vertices")
    rect: Rect | None = Field(None, description="rect / fill / hatch")
    color: tuple[int, int, int] = (255, 200, 20)
    width: float = Field(0.1, gt=0, le=1.0)
    text: str | None = None
    kind: str | None = Field(None, description="Semantic label, e.g. aisle_edge, pedestrian_walkway")


class FloorSpec(Strict):
    material: Literal["concrete_worn", "concrete", "asphalt", "grating", "epoxy", "oak_laminate"] = "concrete_worn"
    wear: float | dict = Field({"uniform": [0.3, 1.0]}, description="0 = new, 1 = heavily worn (or a distribution)")
    markings: Literal["auto", "none"] | list[MarkingSpec] = Field(
        "auto", description="auto: derive lane/zone/dock/agent-path markings from zones, openings and agents")
    extra_markings: list[MarkingSpec] = Field(default_factory=list)
    slope_zones: list[SlopeZoneSpec] = Field(default_factory=list)


class OpeningSpec(Strict):
    id: Id
    wall: Literal["south", "north", "east", "west"]
    center: float = Field(description="Coordinate along the wall (x for south/north, y for east/west), metres")
    width: float = Field(gt=0.3, le=20)
    height: float = Field(gt=0.5, le=20, description="Height of the opening's top edge above the floor, metres")
    sill: float = Field(0.0, ge=0, le=3, description="Height of the opening's bottom edge (windows); 0 for doors")
    door: AssetRef | None = Field(None, description="Asset auto-attached to this opening (door: rollup_door, swing_door; window: window)")
    door_params: dict[str, Any] = Field(default_factory=dict)


class WallsSpec(Strict):
    thickness: float = Field(0.3, ge=0.05, le=1.0)
    material: str = "wall"
    openings: list[OpeningSpec] = Field(default_factory=list)


class RectFootprint(Strict):
    type: Literal["rect"] = "rect"
    size: Point2 = Field(description="Interior size x, y (m)")


class Environment(Strict):
    kind: Literal["indoor", "outdoor"] = "indoor"
    footprint: RectFootprint
    height: float | None = Field(8.0, ge=2.5, le=40, description="Indoor ceiling height (ignored outdoors)")
    floor: FloorSpec = Field(default_factory=FloorSpec)
    walls: WallsSpec = Field(default_factory=WallsSpec, description="Indoor only")
    roof: bool = Field(True, description="Indoor only: roof slab + trusses")
    apron_depth: float = Field(0.0, ge=0, le=30, description="Indoor only: outdoor apron outside the south wall")
    ground_margin: float = Field(10.0, ge=0, le=200, description="Outdoor only: ground collider extends this far beyond the footprint")


# ---------------------------------------------------------------------------------------------------------- zones
class ZoneSpec(Strict):
    id: Id
    rect: Rect
    tags: list[str] = Field(default_factory=list)
    keep_clear: bool = Field(False, description="No objects may be placed in it (corridor / walkway)")
    marking: MarkingSpec | Literal["outline", "none"] = Field("none", description="Floor marking for this zone")


# ---------------------------------------------------------------------------------------------------------- placement
class PoseSpec(Strict):
    xy: Point2
    z: float | None = Field(None, description="Default: support height (floor or slope zone) at xy")
    yaw: float | None = Field(None, description="Degrees; overrides the object's yaw rule")
    rpy: Point3 | None = Field(None, description="Roll, pitch, yaw in degrees (full orientation escape hatch)")
    support: Ref | None = Field(None, description="Support anchor this pose rests on ('<object>.<i>.<anchor>'): collisions with it are "
                                                  "ignored and z (if omitted) comes from the anchor's built height")
    zone: str | None = Field(None, description="Semantic zone label recorded for this instance")


class PosePlacement(Strict):
    pose: PoseSpec | list[PoseSpec] = Field(description="Explicit pose(s): one per instance (escape hatch)")


class InsidePlacement(Strict):
    inside: Ref | list[Ref] = Field(description="Zone id(s); instances are sampled uniformly inside (footprint fully contained)")
    margin: float = Field(0.1, ge=0, description="Distance kept from the zone edge (m)")


class OnPlacement(Strict):
    on: Ref = Field(description="Support anchor reference, e.g. 'belt_a.surface', 'table.top', 'pallets.*.top'")
    arrange: Literal["random", "center", "grid"] = "random"
    margin: float = Field(0.02, ge=0)


class NextToSpec(Strict):
    target: Ref
    side: Literal["left", "right", "front", "back", "any"] = "any"
    gap: float = Field(0.2, ge=0)
    align: bool = Field(True, description="Match the target's yaw")


class NextToPlacement(Strict):
    next_to: NextToSpec


class AgainstPlacement(Strict):
    against: Ref = Field(description="'wall:south' | 'wall:north' | 'wall:east' | 'wall:west' | '<zone>.<edge>'")
    along: float | Point2 | None = Field(None, description="Coordinate (or [min, max] range) along the wall; default anywhere")
    gap: float = Field(0.05, ge=0, description="Distance between the object's back and the wall (m)")


class AttachPlacement(Strict):
    attach_to: Ref = Field(description="Port reference, e.g. 'env.opening_dock_0', 'pump.nozzle_discharge'")


class FillProbability(Strict):
    by: str = Field(description="Anchor meta key, e.g. 'level'")
    values: dict[str, float] = Field(description="Meta value (as a string) -> fill probability")
    default: float = Field(0.0, ge=0, le=1, description="Probability for meta values not listed")

    @model_validator(mode="before")
    @classmethod
    def _keys(cls, v):
        if isinstance(v, dict) and isinstance(v.get("values"), dict):
            v = {**v, "values": {str(k): x for k, x in v["values"].items()}}
        return v


class FillSpec(Strict):
    anchors: Ref = Field(description="Anchor pattern, e.g. 'racks.*.slot_*'")
    where: dict[str, Any] = Field(default_factory=dict, description="Anchor meta filters: {key: value | [values]}")
    probability: float | FillProbability = Field(1.0, description="Fill probability, or per-meta-value probabilities")
    force: list[Ref] = Field(default_factory=list, description="Anchors that are always filled")
    max: int | None = Field(None, ge=0)


class FillPlacement(Strict):
    """One instance per selected anchor; the object's ``count`` is ignored (use ``fill.max`` to cap)."""
    fill: FillSpec


class RackAislesLayout(Strict):
    type: Literal["rack_aisles"] = "rack_aisles"
    aisles: int = Field(ge=1, le=40)
    aisle_width: float = Field(3.2, ge=1.0, le=8.0)
    start_x: float = Field(description="x of the first row's outer face (west edge)")
    back_gap: float = Field(0.1, ge=0, le=1.0, description="Gap between back-to-back rows")
    row_depth: float = Field(0.9, gt=0)
    y: float = Field(0.0, description="Local y origin of the rows")


class RowLayout(Strict):
    type: Literal["row"] = "row"
    start: Point2
    direction: float = Field(0.0, description="Degrees")
    spacing: float = Field(gt=0)


class GridLayout(Strict):
    type: Literal["grid"] = "grid"
    origin: Point2
    rows: int = Field(ge=1)
    cols: int = Field(ge=1)
    spacing: Point2


class AlongPathLayout(Strict):
    type: Literal["along_path"] = "along_path"
    path: list[Point2] = Field(min_length=2)
    spacing: float = Field(gt=0)
    offset: float = Field(0.0, description="Lateral offset to the left of the path (m)")


Layout = Annotated[Union[RackAislesLayout, RowLayout, GridLayout, AlongPathLayout], Field(discriminator="type")]


class LayoutPlacement(Strict):
    layout: Layout


class NearSpec(Strict):
    target: Ref | Point2
    radius: float = Field(1.0, gt=0)


class NearPlacement(Strict):
    near: NearSpec


_PLACEMENTS = {"pose": PosePlacement, "inside": InsidePlacement, "on": OnPlacement, "next_to": NextToPlacement,
               "against": AgainstPlacement, "attach_to": AttachPlacement, "fill": FillPlacement, "layout": LayoutPlacement,
               "near": NearPlacement}


def _placement_tag(v):
    if isinstance(v, dict):
        keys = [k for k in _PLACEMENTS if k in v]
        return keys[0] if len(keys) == 1 else None
    for k, cls in _PLACEMENTS.items():
        if isinstance(v, cls):
            return k
    return None


Placement = Annotated[
    Union[tuple(Annotated[cls, Tag(k)] for k, cls in _PLACEMENTS.items())],
    Discriminator(_placement_tag, custom_error_type="placement",
                  custom_error_message="placement must have exactly one of: " + ", ".join(_PLACEMENTS)),
]


class YawRange(Strict):
    range: Point2 = Field(description="Uniform yaw range, degrees")


class YawFace(Strict):
    face: Ref = Field(description="Turn the object's +y (front) towards this reference")


Yaw = Union[float, Literal["random", "aligned"], YawRange, YawFace]


class Jitter(Strict):
    xy: float = Field(0.0, ge=0, le=1.0, description="Max random xy offset (m)")
    yaw: float = Field(0.0, ge=0, le=180, description="Max random yaw offset (deg)")


# ---------------------------------------------------------------------------------------------------------- constraints
class _C(Strict):
    id: Id | None = None
    hard: bool = True


class NoOverlap(_C):
    type: Literal["no_overlap"] = "no_overlap"
    between: tuple[Ref, Ref] | None = Field(None, description="Restrict to these two groups (default: everything)")
    margin: float = Field(0.0, ge=0)


class MinClearance(_C):
    type: Literal["min_clearance"] = "min_clearance"
    between: tuple[Ref, Ref]
    value: float = Field(gt=0)


class Clearance(_C):
    type: Literal["clearance"] = "clearance"
    zone: Ref = Field(description="Zone (e.g. an aisle) across which the free gap is measured")
    axis: Literal["x", "y"] = Field("x", description="Gap measured along this axis; scanned along the other")
    span: Point2 | None = Field(None, description="Scan range along the other axis (default: zone extent)")
    target: float = Field(gt=0, description="Required narrowest free gap (m)")
    side: Literal["any", "min", "max"] = Field("any", description="Which zone edge the free gap must touch: 'max' = the +axis edge (e.g. the "
                                                               "opposite rack), 'min' = the -axis edge, 'any' = the largest gap anywhere")
    tolerance: float = Field(0.03, ge=0)
    adjust: list[Ref] = Field(min_length=1, description="Objects the solver may slide to hit the target")


class Reachable(_C):
    type: Literal["reachable"] = "reachable"
    from_: Ref = Field("robot.spawn", alias="from")
    to: list[Ref] = Field(description="Goal ids / patterns ('*' = every nav goal)")
    robot_radius: float | None = Field(None, gt=0)
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class KeepClear(_C):
    type: Literal["keep_clear"] = "keep_clear"
    zone: Ref | None = None
    path: list[Point2] | None = None
    width: float = Field(1.0, gt=0, description="Corridor width for `path` (m)")


class Stable(_C):
    type: Literal["stable"] = "stable"
    objects: list[Ref] = Field(default_factory=lambda: ["*"])


class CountAtLeast(_C):
    type: Literal["count_at_least"] = "count_at_least"
    object: Ref
    n: int = Field(ge=0)


Constraint = Annotated[Union[NoOverlap, MinClearance, Clearance, Reachable, KeepClear, Stable, CountAtLeast], Field(discriminator="type")]


# ---------------------------------------------------------------------------------------------------------- objects
class ObjectSpec(Strict):
    id: Id
    asset: AssetRef
    count: int | tuple[int, int] = Field(1, description="Exact count or inclusive [min, max]; instances above min are optional")
    params: dict[str, Any] = Field(default_factory=dict, description="Asset params (values may be distribution literals)")
    instance_params: list[dict[str, Any]] | None = Field(None, description="Per-instance param overrides (index-aligned)")
    placement: Placement
    yaw: Yaw | None = Field(None, description="Degrees | 'random' | 'aligned' (to wall / support) | {range} | {face}")
    jitter: Jitter | None = None
    zone_label: str | None = Field(None, description="Semantic zone label written to the scene graph (default: inferred)")
    tags: list[str] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _count(self):
        if isinstance(self.count, tuple):
            lo, hi = self.count
            if not 0 <= lo <= hi:
                raise ValueError(f"count range [{lo}, {hi}] must satisfy 0 <= min <= max")
        elif self.count < 0:
            raise ValueError("count must be >= 0")
        return self


# ---------------------------------------------------------------------------------------------------------- agents / robot
class AgentPathSpec(Strict):
    waypoints: list[Point2 | Ref] = Field(min_length=2, description="[x, y] points or references (zone / anchor centres)")
    mode: Literal["loop", "pingpong"] = "pingpong"
    speed: float = Field(0.8, gt=0, le=5, description="m/s")
    phase: float = Field(0.0, ge=0, description="Start offset along the path (m)")


class AgentSpec(Strict):
    id: Id
    asset: AssetRef = "human"
    params: dict[str, Any] = Field(default_factory=dict)
    path: AgentPathSpec
    keep_clear_width: float | None = Field(None, gt=0, description="Corridor kept free along the path (default from asset size)")


class NavGoalSpec(Strict):
    id: Id
    type: str = Field(description="Task label, e.g. pick, place, dock, parking, inspect, open_door, chokepoint")
    target: Ref | Point2 = Field(description="Object / instance / anchor / port reference, 'clearance:<constraint id>' (the narrowest gap), or an [x, y] point")
    offset: Point2 | None = Field(None, description="World-frame [dx, dy] added to the target point before choosing the stand pose")
    approach: Literal["front", "any", "port"] = Field("any", description="front: along the target's +y / anchor face; port: stand on the port")
    distance: float = Field(0.9, ge=0, le=5, description="Stand-off distance from the target (m)")
    yaw: float | None = Field(None, description="Degrees; default faces the target")
    extra: dict[str, Any] = Field(default_factory=dict)


class RobotSpec(Strict):
    asset: AssetRef = Field("nova_carter", description="A real Isaac Sim robot: an alias from list_assets(tags=['robot']) or "
                            "{catalog: omniverse, path: Isaac/Robots/...}. Generated or file robots are not allowed.")
    params: dict[str, Any] = Field(default_factory=dict)
    spawn: PosePlacement | InsidePlacement | NearPlacement
    footprint_radius: float | None = Field(None, gt=0, le=3, description="Navigation radius in metres (default: the robot's measured footprint radius)")
    nav_goals: list[NavGoalSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------- cameras / lighting / physics
class CameraSpec(Strict):
    id: Id
    mode: Literal["perspective", "top_down"] = "perspective"
    position: Point3 | None = None
    look_at: Point3 | None = None
    mount: Ref | None = Field(None, description="'robot' or an object id; position/look_at are then in its frame")
    focal_mm: float = Field(18.0, gt=1, le=500)
    aperture_mm: Point2 = (20.955, 11.79)
    clip: Point2 = (0.05, 200.0)
    hide: list[str] = Field(default_factory=list, description="Groups hidden when rendering from this camera, e.g. ['roof']")


class LightingSpec(Strict):
    preset: Literal["bright_day", "mixed_warm", "dim_night", "sun", "overcast", "indoor_industrial"] = "bright_day"
    scale: float = Field(1.0, gt=0, le=20)
    color_temperature: float | None = Field(None, ge=1500, le=12000)
    sun_azimuth_deg: float | None = Field(None, ge=-360, le=360, description="sun/overcast only: direction the sun is in, degrees CCW from +x (0 = east, 90 = north); default random")
    sun_elevation_deg: float | None = Field(None, ge=5, le=90, description="sun only: sun height above the horizon, degrees")


class PhysicsSpec(Strict):
    hz: int = Field(120, ge=30, le=1000)
    solver: Literal["TGS", "PGS"] = "TGS"
    gravity: float = Field(9.81, ge=0, le=30)


class OutputSpec(Strict):
    usd_name: str = Field("scene.usd", pattern=r"^[\w\-]+\.usda?$")
    namespace: str = Field("scene", pattern=r"^[a-z][a-z0-9_]*$", description="USD attribute namespace for semantic tags")


# ---------------------------------------------------------------------------------------------------------- top level
class SceneSpec(Strict):
    """A complete scene description. Only ``environment`` is required."""
    spec_version: Literal["1.0"] = SPEC_VERSION
    name: str = Field("scene", max_length=100)
    description: str | None = Field(None, description="Free-text description (not interpreted)")
    seed: int | None = Field(None, description="Default seed; the compiler's seed argument overrides it")
    environment: Environment
    zones: list[ZoneSpec] = Field(default_factory=list)
    objects: list[ObjectSpec] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    agents: list[AgentSpec] = Field(default_factory=list)
    robot: RobotSpec | None = None
    cameras: list[CameraSpec] = Field(default_factory=list)
    lighting: LightingSpec = Field(default_factory=LightingSpec)
    physics: PhysicsSpec = Field(default_factory=PhysicsSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Free-form (recipe name, overrides, provenance)")
