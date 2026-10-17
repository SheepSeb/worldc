"""Industrial assets salvaged from the former ``plant`` generator: valves, gauges, enclosures, control stations, an
instrument panel, a pump set, flexible hoses, pipes / pipe runs, a static reaching worker, loose tools and I-beams.

Geometry, masses, joint types / limits / drives, PhysX mimic coupling and collider behaviour are the plant values; what
changed is the plumbing: no module globals, every instance authors under the ``path`` it is given, per-instance
bookkeeping that used to live in ``Ctx`` lists (valves, gauges, lines) is returned in ``Built.info``.

Conventions
  * ``pose`` is the instance frame.  For *pipe-mounted* equipment (gate_valve, control_valve, gauge) the origin is the
    point on the pipe axis (not the floor); for floor equipment it is the bottom centre; for loose tools pose z is the
    height of the supporting surface.  Geometric params that name points (``a``/``b`` of pipes, hoses, beams) are in the
    instance frame - with the identity pose they are world coordinates.
  * Articulated assets use ``path`` as the ``Art`` group (root body at ``path/root``); rigid bodies are never nested.
  * Each asset builds its geometry from one pure layout function that ``footprint``/``anchors``/``joints`` share with
    ``build`` so the planning data cannot drift from what is authored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Literal

import numpy as np
from pxr import Sdf, UsdPhysics
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from ...geom import Q_IDENT, qaxis, qfrom_mat, qmat, qmul, qyaw
from ...materials import VISUAL
from ...usdkit import T, _q, _v3
from ..base import Anchor, Asset, Built, Footprint, JointInfo, Keepout, Params, PhysicsInfo, Pose
from ..registry import register
from ._assembly import Art, Sh, body, cap, cube, cyl, esph, qalign, sph, static

Vec3 = tuple[float, float, float]


def _mat(v: str) -> str:
    if v not in VISUAL:
        raise ValueError(f"unknown material preset {v!r}")
    return v


MatName = Annotated[str, AfterValidator(_mat)]


class Sub(BaseModel):
    """Nested parameter object (strict, frozen)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- small helpers
def _r(v, n=4):
    return [round(float(c), n) for c in v]


def _W(pose: Pose, local) -> np.ndarray:
    return np.array(pose.world(tuple(float(c) for c in local))[0], float)


def _rect(lo, hi):
    return [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]


_HALF = {
    "cube": lambda s: np.asarray(s.size, float) / 2,
    "cyl": lambda s: np.array([s.size[0], s.size[0], s.size[1] / 2]),
    "cap": lambda s: np.array([s.size[0], s.size[0], s.size[1] / 2 + s.size[0]]),
    "sph": lambda s: np.array([s.size[0]] * 3),
    "cone": lambda s: np.array([s.size[0], s.size[0], s.size[1] / 2]),
    "esph": lambda s: np.asarray(s.size, float),
}


def sh_bounds(shapes, R=None, off=(0, 0, 0)):
    """Axis-aligned bounds (lo, hi) of shapes given in a frame that is rotated by R and shifted by off."""
    R = np.eye(3) if R is None else np.asarray(R, float)
    off = np.asarray(off, float)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for s in shapes:
        Rs = R @ qmat(s.quat)
        ext = np.abs(Rs) @ _HALF[s.kind](s)
        c = off + R @ np.asarray(s.pos, float)
        lo, hi = np.minimum(lo, c - ext), np.maximum(hi, c + ext)
    return lo, hi


def _merge(*boxes):
    return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)


def _pad(box, m):
    return box[0] - m, box[1] + m


def _fp(box) -> Footprint:
    lo, hi = box
    return Footprint.rect(float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]), float((lo[0] + hi[0]) / 2),
                          float((lo[1] + hi[1]) / 2), float(lo[2]))


def _seg_box(a, b, m):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.minimum(a, b) - m, np.maximum(a, b) + m


# --------------------------------------------------------------------------- articulated assembly specs
@dataclass
class LinkSpec:
    name: str
    lpos: tuple
    shapes: list
    mass: float
    jtype: str
    axis: str = "Z"
    lower: float | None = 0.0
    upper: float | None = 1.0
    parent: str = "root"
    jrot: tuple = Q_IDENT
    stiffness: float = 0.0
    damping: float = 0.0
    target: float = 0.0
    friction: float = 0.0
    unit: str | None = None
    mimic: tuple | None = None
    role: str = ""
    meta: dict | None = None


def _add_links(art: Art, links: list[LinkSpec]) -> dict:
    recs = {}
    for l in links:
        recs[l.name] = art.link(l.name, l.lpos, l.shapes, l.mass, l.jtype, axis=l.axis, lower=l.lower, upper=l.upper,
                                parent=l.parent, jrot=l.jrot, stiffness=l.stiffness, damping=l.damping, target=l.target,
                                friction=l.friction, unit=l.unit, mimic=l.mimic, role=l.role, meta=l.meta)
    return recs


def _joint_infos(links: list[LinkSpec]) -> list[JointInfo]:
    out = []
    for l in links:
        drive = None if l.mimic is not None else dict(stiffness=l.stiffness, damping=l.damping, target=l.target)
        if l.mimic is not None:
            drive = dict(mimic_of=l.mimic[0], gearing_per_deg=l.mimic[1])
        out.append(JointInfo(l.name, l.jtype, l.axis, l.lower, l.upper, l.unit or ("deg" if l.jtype == "revolute" else "m"), 0.0,
                             drive, l.name, l.role or l.name))
    return out


def _assembly_box(root_shapes, links: list[LinkSpec], R=None, sweep=True):
    """Bounds of the closed assembly plus the full travel of prismatic links (revolute doors are keep-outs instead)."""
    R = np.eye(3) if R is None else np.asarray(R, float)
    boxes = [sh_bounds(root_shapes, R)]
    for l in links:
        boxes.append(sh_bounds(l.shapes, R, R @ np.asarray(l.lpos, float)))
        if sweep and l.jtype == "prismatic" and l.upper:
            ax = R @ qmat(l.jrot) @ np.asarray({"X": (1.0, 0, 0), "Y": (0, 1.0, 0), "Z": (0, 0, 1.0)}[l.axis]) * float(l.upper)
            boxes.append(sh_bounds(l.shapes, R, R @ np.asarray(l.lpos, float) + ax))
    return _merge(*boxes)


def _record_art(ctx, path, category, pose, art: Art, fp: Footprint, **extra):
    return ctx.record(path, category, pose.pos, pose.quat, "articulated", size=fp.size, mass=art.mass,
                      articulation={"root": f"{path}/root", "joints": list(art.joint_names)}, **extra)


def _art_bodies(art: Art):
    return [l["path"] for l in art.links.values()]


def _world_anchors(pose, anchors):
    return [a.to_world(pose) for a in anchors]


# =========================================================================== gate valve
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


class GateValveParams(Params):
    pipe_radius: float = Field(0.05, ge=0.02, le=0.2, description="Radius of the pipe the valve sits on (m)")
    pipe_dir: Literal["x", "y", "z"] = Field("x", description="Pipe axis in the instance frame ('z' = vertical riser; give stem_dir)")
    wheel_diameter: float = Field(0.3, ge=0.2, le=0.4, description="Handwheel diameter (m)")
    reach: float | None = Field(None, ge=0.15, le=1.5, description="Distance from the pipe axis to the wheel centre along the stem (m); "
                                "default 0.6 for a vertical stem, wheel_diameter/2 + 0.2 when stem_dir is given")
    turns: int = Field(6, ge=1, le=12, description="Wheel turns from open to closed (joint range = turns * 360 deg)")
    stroke: float = Field(0.08, ge=0.02, le=0.2, description="Rising-stem travel (m), PhysX-mimic coupled to the wheel")
    stem_dir: tuple[float, ...] | None = Field(None, description="Stem direction (nx, ny[, nz]) in the instance frame, perpendicular to the "
                                               "pipe; None = vertical (+Z). Required semantics for pipe_dir 'z' (default (1, 0))")
    wheel_material: MatName = Field("steel_dark", description="Visual preset of the handwheel (e.g. steel_dark, wheel_blue, valve_red)")
    line: str = Field("L-1", description="Label of the process line this valve controls")

    @model_validator(mode="after")
    def _check(self):
        if self.stem_dir is not None:
            if len(self.stem_dir) not in (2, 3) or not np.linalg.norm(self.stem_dir) > 1e-6:
                raise ValueError("stem_dir must be a non-zero 2- or 3-vector")
            ez = np.array([*self.stem_dir, 0.0][:3], float)
            ez /= np.linalg.norm(ez)
            if abs(float(ez @ np.asarray(AXES[self.pipe_dir]))) > 1e-6:
                raise ValueError("stem_dir must be perpendicular to the pipe axis")
        elif self.pipe_dir == "z":
            pass  # defaulted to (1, 0, 0) in the layout
        hw = _gv_hw(self)
        if hw - 0.04 - self.pipe_radius * 1.7 < 0.02:
            raise ValueError(f"reach {hw:.3f} m too short for pipe_radius {self.pipe_radius} (needs > {0.06 + 1.7 * self.pipe_radius:.3f})")
        return self


def _gv_stem(p):
    if p.stem_dir is None:
        return np.array([1.0, 0, 0]) if p.pipe_dir == "z" else np.array([0.0, 0, 1.0]), p.pipe_dir == "z"
    ez = np.array([*p.stem_dir, 0.0][:3], float)
    return ez / np.linalg.norm(ez), True


def _gv_hw(p):
    if p.reach is not None:
        return float(p.reach)
    _, explicit = _gv_stem(p)
    return p.wheel_diameter / 2 + 0.2 if explicit else 0.6


@dataclass
class GVLayout:
    q: tuple
    R: np.ndarray
    root: list
    links: list
    L: float
    hw: float
    zb: float
    gearing: float
    horizontal_stem: bool
    rim_local: np.ndarray
    wheel_local: np.ndarray


def _gv_layout(p) -> GVLayout:
    ex = np.array(AXES[p.pipe_dir])
    ez, _ = _gv_stem(p)
    ey = np.cross(ez, ex)
    Rm = np.column_stack([ex, ey, ez])
    q = qfrom_mat(Rm)
    horizontal = abs(ez[2]) < 0.5
    hw = _gv_hw(p)
    pr = p.pipe_radius
    L = 0.14 + 1.5 * pr
    zb = pr * 1.2
    bt = pr * 1.2 + pr * 0.5
    R = p.wheel_diameter / 2
    wm = p.wheel_material
    root = [
        cyl(pr * 1.45, 0.03, (-(L / 2 - 0.015), 0, 0), "steel", axis="x"),
        cyl(pr * 1.45, 0.03, ((L / 2 - 0.015), 0, 0), "steel", axis="x"),
        cyl(pr * 1.25, L - 0.06, (0, 0, 0), "valve_body", axis="x"),
        cyl(pr * 0.95, pr * 1.2, (0, 0, zb), "valve_body"),
        cyl(pr * 1.15, 0.025, (0, 0, bt - 0.0125), "steel"),
        cyl(0.012, hw - 0.04 - bt, (pr * 0.8, 0, (bt + hw - 0.04) / 2), "steel_dark"),
        cyl(0.012, hw - 0.04 - bt, (-pr * 0.8, 0, (bt + hw - 0.04) / 2), "steel_dark"),
        cyl(0.05, 0.02, (0, 0, hw - 0.04 - 0.01), "steel_dark"),  # collar the hub sits on
    ]
    for sgn in (-1, 1):  # bolt heads (visual)
        for k in range(6):
            a = k * math.pi / 3
            root.append(cyl(0.008, 0.012, (sgn * (L / 2 - 0.03 - 0.004), pr * 1.3 * math.cos(a), pr * 1.3 * math.sin(a)), "steel_dark", col=False, axis="x"))
    ws = [cyl(0.035, 0.06, (0, 0, 0), wm), cyl(0.045, 0.012, (0, 0, 0.034), wm, col=False)]
    N = 16
    for k in range(N):
        a = 2 * math.pi * k / N
        ws.append(cube((2 * math.pi * R / N * 1.08, 0.026, 0.022), (R * math.cos(a), R * math.sin(a), 0), wm, quat=qyaw(a + math.pi / 2)))
    for k in range(3):
        ws.append(cube((2 * R - 0.02, 0.015, 0.012), (0, 0, 0), wm, quat=qyaw(k * math.pi / 3)))
    Ls = hw - zb + 0.06
    stem = [cyl(0.011, Ls, (0, 0, (zb - hw + 0.06) / 2), "steel"), cyl(0.02, 0.02, (0, 0, 0.06 - 0.01), "steel_dark")]
    gearing = -p.stroke / (p.turns * 360.0)
    links = [
        LinkSpec("wheel", (0, 0, hw), ws, 1.2 + 6.0 * R, "revolute", "Z", 0.0, p.turns * 360.0, friction=0.1, stiffness=0.0, damping=0.8,
                 role="handwheel", unit="deg", meta=dict(turns=p.turns, wheel_diameter=p.wheel_diameter, controls=p.line)),
        LinkSpec("stem", (0, 0, hw), stem, 0.7, "prismatic", "Z", 0.0, p.stroke + 0.002, mimic=("wheel", gearing), role="stem", unit="m",
                 meta=dict(stroke=p.stroke, driven_by="wheel")),
    ]
    wc = Rm @ np.array([0, 0, hw])
    rim = wc + (np.array([0, 0, R]) if horizontal else np.array([R, 0, 0]))
    return GVLayout(q, Rm, root, links, L, hw, zb, gearing, horizontal, rim, wc)


@register
class GateValve(Asset):
    id = "gate_valve"
    description = ("Flanged handwheel gate valve for a pipe run: multi-turn revolute handwheel (0..turns*360 deg) coupled by a "
                   "PhysX mimic joint to a rising stem (prismatic). Origin = body centre on the pipe axis; pipe_dir is the pipe axis, "
                   "stem_dir the stem/wheel direction (vertical by default). Ports pipe_in/pipe_out (pipe flanges) and "
                   "handwheel_rim (manipulation goal point on the wheel rim).")
    tags = frozenset({"industrial", "valve", "gate_valve", "handwheel", "articulated", "pipe_mounted", "manipulation"})
    Params = GateValveParams
    physics = PhysicsInfo("articulated", "steel", 0.6, None, "primitives", "mass 18 + 80*pipe_radius kg body + wheel + stem; fixed base")
    example = {"id": "wall_valve", "asset": "gate_valve", "params": {"pipe_radius": 0.06, "pipe_dir": "x", "stem_dir": [0, -1], "turns": 5},
               "placement": {"at": [1.0, 1.9, 1.2]}}

    def footprint(self, p):
        L = _gv_layout(p)
        return _fp(_assembly_box(L.root, L.links, L.R))

    def anchors(self, p):
        L = _gv_layout(p)
        ex = np.array(AXES[p.pipe_dir])
        R = p.wheel_diameter / 2
        return [
            Anchor("pipe_in", "port", tuple(-ex * L.L / 2), qalign(-ex), port_kind="pipe", radius=p.pipe_radius, meta={"flange_radius": round(p.pipe_radius * 1.45, 4)}),
            Anchor("pipe_out", "port", tuple(ex * L.L / 2), qalign(ex), port_kind="pipe", radius=p.pipe_radius, meta={"flange_radius": round(p.pipe_radius * 1.45, 4)}),
            Anchor("handwheel_rim", "port", tuple(L.rim_local), port_kind="handwheel", radius=R,
                   meta={"joint": "J_wheel", "turns": p.turns, "unit": "deg", "lower": 0.0, "upper": p.turns * 360.0}),
        ]

    def joints(self, p):
        return _joint_infos(_gv_layout(p).links)

    def build(self, ctx, path, pose, p, rng):
        L = _gv_layout(p)
        q = qmul(pose.quat, L.q)
        art = Art(ctx, path, pose.pos, 0.0, L.root, 18.0 + 80 * p.pipe_radius, "gate_valve", quat=q, line=p.line)
        recs = _add_links(art, L.links)
        wj, sj = recs["wheel"]["name"], recs["stem"]["name"]
        recs["stem"]["driven_by"] = wj
        wc = _W(pose, L.wheel_local)
        rim = _W(pose, L.rim_local)
        fp = self.footprint(p)
        info = dict(name=art.name, kind="gate_valve", pos=_r(pose.pos), pipe_dir=p.pipe_dir, length=L.L, wheel_height=float(wc[2]),
                    wheel_diameter=p.wheel_diameter, turns=p.turns, stroke=p.stroke, wheel_joint=wj, stem_joint=sj, line=p.line,
                    wheel_center=[float(c) for c in wc], rim_point=[float(c) for c in rim], gearing_m_per_deg=L.gearing, pipe_r=p.pipe_radius,
                    stem_axis_world=_r(qmat(art.q) @ np.array([0, 0, 1.0])), wheel_reach=float(L.hw))
        _record_art(ctx, path, "gate_valve", pose, art, fp, line=p.line, wheel_joint=wj, stem_joint=sj)
        return Built(path, _world_anchors(pose, self.anchors(p)), info=info, bodies=_art_bodies(art))


# =========================================================================== control valve
class ControlValveParams(Params):
    pipe_radius: float = Field(0.05, ge=0.02, le=0.2, description="Radius of the pipe the valve sits on (m)")
    pipe_dir: Literal["x", "y"] = Field("x", description="Horizontal pipe axis in the instance frame")
    stroke: float = Field(0.03, ge=0.005, le=0.1, description="Actuator stem travel (m), spring-return position drive")
    line: str = Field("L-1", description="Label of the process line this valve controls")


def _cv_layout(p):
    pr = p.pipe_radius
    L = 0.3 + 1.7 * pr
    zy0 = pr * 1.5 + 0.02
    act_z = zy0 + 0.30
    shapes = [
        cyl(pr * 1.5, 0.03, (-(L / 2 - 0.015), 0, 0), "valve_red", axis="x"),
        cyl(pr * 1.5, 0.03, ((L / 2 - 0.015), 0, 0), "valve_red", axis="x"),
        cyl(pr * 1.2, L - 0.06, (0, 0, 0), "valve_red", axis="x"),
        sph(pr * 1.5, (0, 0, pr * 0.2), "valve_red"),
        cyl(pr * 0.7, 0.14, (0, 0, pr * 1.5 + 0.06), "valve_red"),
        cyl(0.02, act_z - zy0, (0.07, 0, (zy0 + act_z) / 2), "valve_red"),
        cyl(0.02, act_z - zy0, (-0.07, 0, (zy0 + act_z) / 2), "valve_red"),
        cyl(0.17, 0.07, (0, 0, act_z + 0.035), "valve_red"),
        cyl(0.165, 0.06, (0, 0, act_z + 0.10), "valve_red"),
        esph(0.165, 0.165, 0.075, (0, 0, act_z + 0.13), "valve_red"),
        cube((0.09, 0.07, 0.11), (0.0, -0.12, act_z - 0.1), "gray_box"),
        cyl(0.012, 0.15, (0.0, -0.12, act_z - 0.19), "steel", col=False),
    ]
    for sgn in (-1, 1):
        for k in range(6):
            a = k * math.pi / 3
            shapes.append(cyl(0.009, 0.012, (sgn * (L / 2 - 0.034), pr * 1.38 * math.cos(a), pr * 1.38 * math.sin(a)), "steel_dark", col=False, axis="x"))
    stem = [cyl(0.008, act_z - zy0 + 0.04, (0, 0, -(act_z - zy0) / 2 + 0.02 - 0.0), "steel"),
            cube((0.05, 0.012, 0.035), (0.0, -0.035, 0.0), "yellow", col=False)]
    links = [LinkSpec("stem", (0, 0, act_z), stem, 0.6, "prismatic", "Z", 0.0, p.stroke, stiffness=2.0e4, damping=300.0, target=0.0,
                      role="control_stem", unit="m", meta=dict(stroke=p.stroke, controls=p.line))]
    return shapes, links, L, act_z


@register
class ControlValve(Asset):
    id = "control_valve"
    description = ("Red globe control valve with a diaphragm actuator on a yoke; the actuator stem is a prismatic joint with a "
                   "spring-return position drive (stroke param). Origin = body centre on the pipe axis. Ports pipe_in/pipe_out.")
    tags = frozenset({"industrial", "valve", "control_valve", "actuator", "articulated", "pipe_mounted"})
    Params = ControlValveParams
    physics = PhysicsInfo("articulated", "steel", 0.6, None, "primitives", "mass 60 + 100*pipe_radius kg; fixed base")
    example = {"id": "cv1", "asset": "control_valve", "params": {"pipe_radius": 0.05, "pipe_dir": "y"}, "placement": {"at": [-2.8, 0.0, 0.9]}}

    def _q(self, p):
        return qyaw(0.0 if p.pipe_dir == "x" else math.pi / 2)

    def footprint(self, p):
        shapes, links, _, _ = _cv_layout(p)
        return _fp(_assembly_box(shapes, links, qmat(self._q(p))))

    def anchors(self, p):
        _, _, L, act_z = _cv_layout(p)
        R = qmat(self._q(p))
        ex = R @ np.array([1.0, 0, 0])
        return [Anchor("pipe_in", "port", tuple(-ex * L / 2), qalign(-ex), port_kind="pipe", radius=p.pipe_radius),
                Anchor("pipe_out", "port", tuple(ex * L / 2), qalign(ex), port_kind="pipe", radius=p.pipe_radius),
                Anchor("stem_point", "port", (0.0, 0.0, act_z - 0.1), port_kind="actuator", meta={"joint": "J_stem", "unit": "m", "upper": p.stroke})]

    def joints(self, p):
        return _joint_infos(_cv_layout(p)[1])

    def build(self, ctx, path, pose, p, rng):
        shapes, links, L, act_z = _cv_layout(p)
        art = Art(ctx, path, pose.pos, 0.0, shapes, 60 + 100 * p.pipe_radius, "control_valve", quat=qmul(pose.quat, self._q(p)), line=p.line)
        recs = _add_links(art, links)
        stem_pt = pose.world((0, 0, act_z - 0.1))[0]
        info = dict(name=art.name, kind="control_valve", pos=_r(pose.pos), pipe_dir=p.pipe_dir, length=L, stem_joint=recs["stem"]["name"], stroke=p.stroke,
                    line=p.line, top_height=float(pose.pos[2] + act_z + 0.2), stem_point=[float(c) for c in stem_pt], pipe_r=p.pipe_radius)
        _record_art(ctx, path, "control_valve", pose, art, self.footprint(p), line=p.line, stem_joint=info["stem_joint"])
        return Built(path, _world_anchors(pose, self.anchors(p)), info=info, bodies=_art_bodies(art))


# =========================================================================== gauge
class GaugeParams(Params):
    pipe_radius: float = Field(0.05, ge=0.0, le=0.3, description="Radius of the pipe it is tapped into (sets the stem length)")
    mount: Literal["top", "side"] = Field("top", description="'top': vertical stem over a horizontal pipe; 'side': horizontal stem along the dial normal (vertical pipes)")
    dial_yaw_deg: float = Field(0.0, ge=-360, le=360, description="Horizontal direction the dial faces, degrees CCW from +X in the instance frame")
    line: str = Field("L-1", description="Label of the process line it measures")


def _gauge_layout(p):
    ny = math.radians(p.dial_yaw_deg)
    n = np.array([math.cos(ny), math.sin(ny), 0.0])
    u = np.array([0.0, 0.0, 1.0]) if p.mount == "top" else n
    stem_len = p.pipe_radius + 0.07
    qu, qn = qalign(u), qalign(n)
    c = u * stem_len
    shapes = [
        cyl(0.012, stem_len, tuple(u * stem_len / 2), "steel", quat=qu),
        cyl(0.062, 0.03, tuple(c), "black", quat=qn),
        cyl(0.052, 0.004, tuple(c + n * 0.016), "gauge_white", col=False, quat=qn),
        cube((0.004, 0.04, 0.004) if p.mount == "top" else (0.004, 0.004, 0.04), tuple(c + n * 0.019 + np.array([0, 0, 0.01])), "black", col=False, quat=qyaw(ny)),
    ]
    return shapes, n, c + n * 0.02


@register
class Gauge(Asset):
    id = "gauge"
    description = ("Pressure gauge: stem + black case + white dial (static colliders). Origin = tap point on the pipe axis. "
                   "Port 'dial' (reading viewpoint, +Z of the frame = dial normal) and 'pipe_tap'.")
    tags = frozenset({"industrial", "instrument", "gauge", "pipe_mounted", "static", "inspection"})
    Params = GaugeParams
    physics = PhysicsInfo("static", "steel", 0.6, None, "primitives")
    example = {"id": "g1", "asset": "gauge", "params": {"pipe_radius": 0.06, "mount": "side", "dial_yaw_deg": -90}, "placement": {"at": [1.0, 1.9, 1.6]}}

    def footprint(self, p):
        return _fp(sh_bounds(_gauge_layout(p)[0]))

    def anchors(self, p):
        _, n, dial = _gauge_layout(p)
        return [Anchor("dial", "port", tuple(dial), qalign(n), port_kind="dial", radius=0.052),
                Anchor("pipe_tap", "port", (0.0, 0.0, 0.0), port_kind="pipe_tap", radius=p.pipe_radius)]

    def build(self, ctx, path, pose, p, rng):
        shapes, n, dial = _gauge_layout(p)
        static(ctx, path, pose.pos, pose.quat, shapes, category="gauge", line=p.line)
        fp = self.footprint(p)
        ctx.record(path, "gauge", pose.pos, pose.quat, "static", size=fp.size, line=p.line)
        nw = qmat(pose.quat) @ n
        info = dict(name=Sdf.Path(path).name, path=path, position=_r(_W(pose, dial)), normal=_r(nw), line=p.line)
        return Built(path, _world_anchors(pose, self.anchors(p)), info=info)


# =========================================================================== instrument box
class InstrumentBoxParams(Params):
    size: Vec3 = Field((0.18, 0.12, 0.22), description="Box width, depth, height (m), each 0.04-0.6")
    material: MatName = Field("gray_box", description="Visual preset of the housing")

    @model_validator(mode="after")
    def _c(self):
        if any(not 0.04 <= s <= 0.6 for s in self.size):
            raise ValueError("size components must be within 0.04-0.6 m")
        return self


def _ib_shapes(p):
    return [cube(p.size, (0, 0, p.size[2] / 2), p.material), cube((0.05, 0.05, 0.03), (0, 0, -0.015), "steel")]


@register
class InstrumentBox(Asset):
    id = "instrument_box"
    description = "Small instrument / terminal box on a short stub (static). Origin = underside of the box (stub reaches 3 cm below)."
    tags = frozenset({"industrial", "instrument", "box", "static", "wall_mounted"})
    Params = InstrumentBoxParams
    physics = PhysicsInfo("static", "steel", 0.6, None, "primitives")
    example = {"id": "jb", "asset": "instrument_box", "placement": {"at": [0.5, 1.9, 1.0]}}

    def footprint(self, p):
        return _fp(sh_bounds(_ib_shapes(p)))

    def anchors(self, p):
        return [Anchor("front", "port", (0.0, p.size[1] / 2, p.size[2] / 2), qalign((0, 1, 0)), port_kind="front")]

    def build(self, ctx, path, pose, p, rng):
        static(ctx, path, pose.pos, pose.quat, _ib_shapes(p), category="instrument_box")
        ctx.record(path, "instrument_box", pose.pos, pose.quat, "static", size=self.footprint(p).size)
        return Built(path, _world_anchors(pose, self.anchors(p)), info=dict(name=Sdf.Path(path).name, size=list(p.size)))


# =========================================================================== enclosure
class DrawerSpec(Sub):
    x: float = Field(0.0, description="Drawer centre across the front (m, housing frame)")
    w: float = Field(0.2, ge=0.05, le=1.0, description="Drawer front width (m)")
    h: float = Field(0.12, ge=0.04, le=0.4, description="Drawer front height (m)")
    z: float = Field(0.2, ge=0.0, le=2.5, description="Drawer centre height above the enclosure base (m)")
    stroke: float = Field(0.18, ge=0.02, le=0.4, description="Pull-out travel (m)")


class EnclosureParams(Params):
    size: Vec3 = Field((0.5, 0.35, 0.8), description="Width, depth, height of the housing (m), each 0.1-3")
    material: MatName = Field("gray_box", description="Visual preset of the housing (gray_box, red_jb, cab_blue, stainless, ...)")
    door: bool = Field(True, description="Hinged front door (revolute, 0..110 deg, hinge on the left, spring return)")
    latch: bool = Field(False, description="Rotary latch on the door (revolute 0..90 deg); needs door")
    drawer: DrawerSpec | None = Field(None, description="Optional pull-out drawer in the front face (prismatic along +Y)")
    door_material: MatName | None = Field(None, description="Door preset (default: same as material)")
    louvers: bool = Field(False, description="Visual vent louvers on the right side")
    door_lo: float = Field(0.02, ge=0.0, le=1.5, description="Gap between base and the door's lower edge (m)")
    category: str = Field("enclosure", description="Scene-graph category tag (enclosure, cabinet, electrical_box, ...)")
    note: str = Field("", description="Free-text note stored on the root prim")

    @model_validator(mode="after")
    def _c(self):
        w, d, h = self.size
        if not (0.1 <= w <= 3 and 0.1 <= d <= 3 and 0.1 <= h <= 3):
            raise ValueError("size components must be within 0.1-3 m")
        if self.latch and not self.door:
            raise ValueError("latch requires door=true")
        if self.door and h - 0.02 - self.door_lo <= 0.05:
            raise ValueError("door_lo too large for the enclosure height")
        return self


_T = 0.018  # door / drawer-front thickness


def _enc_layout(p):
    w, d, h = p.size
    t = _T
    mat = p.material
    shapes = [cube((w, d, h), (0, 0, h / 2), mat)]
    if p.louvers:
        for k in range(6):
            shapes.append(cube((0.003, d * 0.6, 0.012), (w / 2 + 0.0015, 0, h * 0.2 + k * 0.03), "steel_dark", col=False))
    yf = d / 2 + 0.002
    links = []
    hd = h - 0.02 - p.door_lo
    if p.door:
        door_mat = p.door_material or mat
        ds = [cube((w, t, hd), (w / 2, -t / 2, 0), door_mat), cube((0.02, 0.02, min(0.14, hd * 0.3)), (w - 0.05, 0.011, 0), "steel_dark"),
              cyl(0.009, hd, (0, -t / 2, 0), "steel_dark", col=False)]
        links.append(LinkSpec("door", (-w / 2, yf + t, p.door_lo + hd / 2), ds, 1.0 + 9.0 * w * hd, "revolute", "Z", 0.0, 110.0, stiffness=6.0,
                              damping=3.0, friction=0.05, role="door", unit="deg", meta=dict(door_width=w)))
        if p.latch:
            links.append(LinkSpec("latch", (-w / 2 + w - 0.07, yf + t + 0.016, p.door_lo + hd / 2),
                                  [cube((0.07, 0.012, 0.018), (0.02, 0, 0), "steel_dark"), cyl(0.014, 0.012, (0, 0, 0), "steel", axis="y")],
                                  0.15, "revolute", "Y", 0.0, 90.0, parent="door", stiffness=0.3, damping=0.1, friction=0.02, role="latch", unit="deg"))
    if p.drawer:
        dw, dh = p.drawer.w, p.drawer.h
        tray_len = min(0.22, d - 0.03)
        sd = [cube((dw, t, dh), (0, t / 2, 0), "steel_dark" if mat != "stainless" else "stainless"),
              cube((dw - 0.02, tray_len, dh - 0.03), (0, -tray_len / 2 - 0.0, 0), "steel"),
              cyl(0.008, dw * 0.6, (0, t + 0.02, 0), "steel", axis="x"),
              cyl(0.006, 0.02, (-dw * 0.25, t + 0.01, 0), "steel", axis="y", col=False),
              cyl(0.006, 0.02, (dw * 0.25, t + 0.01, 0), "steel", axis="y", col=False)]
        links.append(LinkSpec("drawer", (p.drawer.x, yf, p.drawer.z), sd, 1.5, "prismatic", "Y", 0.0, p.drawer.stroke, stiffness=300.0, damping=40.0,
                              friction=0.05, role="drawer", unit="m"))
    return shapes, links, yf, hd


@register
class Enclosure(Asset):
    id = "enclosure"
    description = ("Housing (cabinet / junction box / electrical box) whose front faces local +Y, with optional hinged door "
                   "(0-110 deg, spring return), rotary latch on the door and pull-out drawer (prismatic +Y). Origin = bottom centre. "
                   "Ports: front, door_handle, drawer_pull; support 'top'. Keep-out: door swing arc.")
    tags = frozenset({"industrial", "enclosure", "cabinet", "junction_box", "door", "drawer", "articulated", "manipulation"})
    Params = EnclosureParams
    physics = PhysicsInfo("articulated", "steel", 0.6, None, "primitives", "root mass 30 + 120*w*d*h kg; door/latch/drawer links; fixed base")
    example = {"id": "cabinet", "asset": "enclosure", "params": {"size": [0.6, 0.4, 1.7], "material": "cab_blue", "door": True, "latch": True,
                                                                 "louvers": True, "category": "cabinet"},
               "placement": {"against_wall": "north"}}

    def footprint(self, p):
        shapes, links, _, _ = _enc_layout(p)
        return _fp(_assembly_box(shapes, links))

    def anchors(self, p):
        w, d, h = p.size
        t = _T
        yf = d / 2 + 0.002
        out = [Anchor("front", "port", (0.0, d / 2, h / 2), qalign((0, 1, 0)), port_kind="front"),
               Anchor("top", "support", (0.0, 0.0, h), polygon=_rect((-w / 2, -d / 2), (w / 2, d / 2)))]
        if p.door:
            hd = h - 0.02 - p.door_lo
            out.append(Anchor("door_handle", "port", (w / 2 - 0.05, yf + t + 0.011, p.door_lo + hd / 2), qalign((0, 1, 0)), port_kind="handle",
                              meta={"joint": "J_door", "unit": "deg", "upper": 110.0}))
        if p.drawer:
            out.append(Anchor("drawer_pull", "port", (p.drawer.x, yf + t + 0.03, p.drawer.z), qalign((0, 1, 0)), port_kind="handle",
                              meta={"joint": "J_drawer", "unit": "m", "upper": p.drawer.stroke}))
        return out

    def keepouts(self, p):
        if not p.door:
            return []
        w, d, h = p.size
        hx, hy = -w / 2, d / 2 + 0.002 + _T
        poly = [(hx, hy)] + [(hx + w * math.cos(math.radians(a)), hy + w * math.sin(math.radians(a))) for a in (0, 22, 44, 66, 88, 110)]
        return [Keepout(poly, "door_swing", (0.0, h))]

    def joints(self, p):
        return _joint_infos(_enc_layout(p)[1])

    def build(self, ctx, path, pose, p, rng):
        w, d, h = p.size
        shapes, links, yf, hd = _enc_layout(p)
        art = Art(ctx, path, pose.pos, 0.0, shapes, 30 + 120 * w * d * h, p.category, quat=pose.quat, note=p.note)
        recs = _add_links(art, links)
        joints = {k: recs[k]["name"] for k in ("door", "latch", "drawer") if k in recs}
        a = {x.name: x for x in self.anchors(p)}
        info = dict(name=art.name, kind=p.category, joints=joints,
                    drawer_point=_r(_W(pose, a["drawer_pull"].pos)) if p.drawer else None,
                    front_normal=_r(qmat(pose.quat) @ np.array([0.0, 1.0, 0.0])),
                    door_handle=_r(_W(pose, a["door_handle"].pos)) if p.door else None)
        _record_art(ctx, path, p.category, pose, art, self.footprint(p), housing=art.name)
        built_keepouts = [(k.tag, pose.world_xy(k.polygon)) for k in self.keepouts(p)]
        return Built(path, _world_anchors(pose, self.anchors(p)), built_keepouts, info, _art_bodies(art))


# =========================================================================== control station
class ControlStationParams(Params):
    n_buttons: int = Field(3, ge=1, le=4, description="Spring-return push buttons (green, red, black, green) in a 2-column grid")


def _cs_layout(p):
    w, d, h = 0.22, 0.12, 0.34
    shapes = [cube((w, d, h), (0, 0, h / 2), "gray_box"), cube((w - 0.02, 0.006, h - 0.03), (0, d / 2 + 0.003, h / 2), "steel_dark", col=False)]
    yf = d / 2 + 0.002
    cols = ["btn_green", "btn_red", "btn_black", "btn_green"]
    links = []
    btn = []
    for i in range(p.n_buttons):
        x = (-0.05, 0.05)[i % 2]
        z = h * 0.78 - 0.07 * (i // 2)
        btn.append((x, yf + 0.02, z))
        links.append(LinkSpec(f"btn{i}", (x, yf + 0.008, z), [cyl(0.016, 0.02, (0, 0, 0), cols[i], axis="y")], 0.03, "prismatic", "Y", 0.0, 0.008,
                              jrot=qyaw(math.pi), stiffness=800.0, damping=12.0, role="push_button", unit="m", meta=dict(button_index=i)))
    links.append(LinkSpec("selector", (0.0, yf + 0.012, h * 0.3), [cyl(0.026, 0.024, (0, 0, 0), "black", axis="y"), cube((0.008, 0.01, 0.03), (0, 0.012, 0.012), "white", col=False)],
                          0.08, "revolute", "Y", 0.0, 270.0, stiffness=0.4, damping=0.1, friction=0.03, role="selector_knob", unit="deg"))
    return shapes, links, btn, (w, d, h)


@register
class ControlStation(Asset):
    id = "control_station"
    description = ("Wall/post-mounted push-button station: housing (front +Y) with 1-4 spring-return push buttons (prismatic, 8 mm) "
                   "and a 270 deg selector knob. Origin = bottom centre. Ports button0.., selector.")
    tags = frozenset({"industrial", "control_station", "buttons", "articulated", "manipulation", "instrument"})
    Params = ControlStationParams
    physics = PhysicsInfo("articulated", "steel", 0.6, None, "primitives", "root 4 kg + 0.03 kg per button + 0.08 kg knob")
    example = {"id": "stop_start", "asset": "control_station", "params": {"n_buttons": 3}, "placement": {"against_wall": "north"}}

    def footprint(self, p):
        shapes, links, _, _ = _cs_layout(p)
        return _fp(_assembly_box(shapes, links))

    def anchors(self, p):
        shapes, links, btn, (w, d, h) = _cs_layout(p)
        out = [Anchor("front", "port", (0.0, d / 2, h / 2), qalign((0, 1, 0)), port_kind="front")]
        out += [Anchor(f"button{i}", "port", pt, qalign((0, 1, 0)), port_kind="push_button", radius=0.016, meta={"joint": f"J_btn{i}", "unit": "m", "upper": 0.008})
                for i, pt in enumerate(btn)]
        out.append(Anchor("selector", "port", (0.0, d / 2 + 0.02, h * 0.3), qalign((0, 1, 0)), port_kind="knob", radius=0.026, meta={"joint": "J_selector", "unit": "deg", "upper": 270.0}))
        return out

    def joints(self, p):
        return _joint_infos(_cs_layout(p)[1])

    def build(self, ctx, path, pose, p, rng):
        shapes, links, btn, _ = _cs_layout(p)
        art = Art(ctx, path, pose.pos, 0.0, shapes, 4.0, "control_station", quat=pose.quat)
        recs = _add_links(art, links)
        info = dict(name=art.name, kind="control_station", buttons=[recs[f"btn{i}"]["name"] for i in range(p.n_buttons)], selector=recs["selector"]["name"],
                    front_normal=_r(qmat(pose.quat) @ np.array([0.0, 1.0, 0.0])), button_world=[_r(_W(pose, b)) for b in btn])
        _record_art(ctx, path, "control_station", pose, art, self.footprint(p))
        return Built(path, _world_anchors(pose, self.anchors(p)), info=info, bodies=_art_bodies(art))


# =========================================================================== instrument panel
class InstrumentPanelParams(Params):
    manifold: bool = Field(True, description="Add the small-bore tubing manifold board to the left of the frame")


_PW = 0.72  # half width of the panel frame (posts at +-PW)
_YB = 0.058  # plane of the backplate front, enclosure backs sit here


def _panel_frame():
    PW = _PW
    return [
        cube((0.08, 0.06, 2.5), (-PW, 0, 1.25), "steel_dark"), cube((0.08, 0.06, 2.5), (PW, 0, 1.25), "steel_dark"),
        cube((2 * PW + 0.08, 0.05, 0.06), (0, 0.0, 0.45), "steel_dark"), cube((2 * PW + 0.08, 0.05, 0.06), (0, 0.0, 2.45), "steel_dark"),
        cube((2 * PW - 0.06, 0.025, 1.45), (0, 0.045, 1.6), "panel_plate"),
        cube((2 * PW - 0.06, 0.03, 0.05), (0, 0.085, 0.84), "steel_dark"), cube((2 * PW - 0.06, 0.03, 0.05), (0, 0.085, 2.14), "steel_dark"),
        cube((0.08, 0.25, 0.06), (-PW, -0.125, 0.1), "steel_dark"), cube((0.08, 0.25, 0.06), (PW, -0.125, 0.1), "steel_dark"),
    ]


def _panel_manifold_board():
    mx = -_PW - 0.6
    board = [cube((0.07, 0.06, 2.0), (mx - 0.42, 0, 1.0), "steel_dark"), cube((0.07, 0.06, 2.0), (mx + 0.42, 0, 1.0), "steel_dark"),
             cube((0.92, 0.025, 1.3), (mx, 0.045, 1.35), "panel_plate")]
    ms = []
    for k in range(5):
        z = 0.85 + 0.22 * k
        ms.append(cyl(0.011, 0.7, (mx, _YB, z), "tube", axis="x"))
        for j in range(5):
            x = mx - 0.3 + 0.15 * j
            ms.append(cyl(0.0055, 0.2, (x, _YB + (0.015 if k % 2 else 0.0), z - 0.1), "tube", col=False))
            ms.append(cube((0.022, 0.022, 0.022), (x, _YB, z), "steel", col=False))
            if (j + k) % 3 == 0:
                ms.append(cyl(0.018, 0.04, (x, _YB, z + 0.03), "valve_red" if (k + j) % 2 else "steel", col=False))
    return board, ms


@register
class InstrumentPanel(Asset):
    id = "instrument_panel"
    description = ("Unistrut frame + backplate (2.5 m tall, ~1.5 m wide) carrying six enclosures in a seeded random order (junction "
                   "boxes with hinged doors / latches, boxes with drawers, a 4-button control station), terminal boxes, gauges, "
                   "cable bundles and an optional tubing manifold board. Front faces local +Y. Origin = bottom centre of the frame. "
                   "Info lists every child's articulation info.")
    tags = frozenset({"industrial", "panel", "instrument", "enclosure", "articulated", "composite", "manipulation"})
    Params = InstrumentPanelParams
    physics = PhysicsInfo("mixed", "steel", 0.6, None, "primitives", "static frame + articulated enclosures / control station")
    example = {"id": "panel", "asset": "instrument_panel", "params": {"manifold": False}, "placement": {"against_wall": "north"}}

    def footprint(self, p):
        x0 = -1.82 if p.manifold else -(_PW + 0.06)
        return _fp((np.array([x0, -0.26, 0.0]), np.array([_PW + 0.06, 0.42, 2.5])))

    def anchors(self, p):
        return [Anchor("front", "port", (0.0, _YB + 0.04, 1.5), qalign((0, 1, 0)), port_kind="front")]

    def build(self, ctx, path, pose, p, rng):
        ctx.kit.xform(path)
        leaf = Sdf.Path(path).name

        def W(x, y, z):
            return tuple(float(c) for c in pose.world((x, y, z))[0])

        static(ctx, f"{path}/Frame", pose.pos, pose.quat, _panel_frame(), category="panel_frame")
        items = {}
        slots = [(-0.42, 1.5), (0.0, 1.5), (0.42, 1.5), (-0.42, 0.95), (0.0, 0.95), (0.42, 0.95)]
        kinds = ["red_jb", "gray_jb", "ctrl", "drawer_box", "gray_jb2", "latch_box"]
        order = list(rng.permutation(len(kinds)))
        encl_asset, cs_asset = Enclosure(), ControlStation()
        encl, bodies = [], []
        for si, ki in enumerate(order):
            kind = kinds[ki]
            x0, zb = slots[si]
            x0 += float(rng.uniform(-0.015, 0.015))
            sub = lambda nm: f"{path}/{leaf}_{nm}"
            if kind == "red_jb":
                w, d, h = 0.28 + float(rng.uniform(0, 0.03)), 0.2, 0.4
                ep = EnclosureParams(size=(w, d, h), material="red_jb", door=True, latch=True, note="red junction box")
                nm = "RedJB"
            elif kind == "gray_jb":
                w, d, h = 0.3, 0.18, 0.3
                ep = EnclosureParams(size=(w, d, h), material="gray_box", door=True)
                nm = "GrayJB_A"
            elif kind == "gray_jb2":
                w, d, h = 0.26, 0.16, 0.26
                ep = EnclosureParams(size=(w, d, h), material="gray_box", door=True)
                nm = "GrayJB_B"
            elif kind == "drawer_box":
                w, d, h = 0.32, 0.26, 0.3
                ep = EnclosureParams(size=(w, d, h), material="gray_box", door=False,
                                     drawer=DrawerSpec(x=0.0, w=w - 0.04, h=0.12, z=h * 0.62, stroke=0.16), note="instrument box with drawer")
                nm = "DrawerBox_A"
            elif kind == "latch_box":
                w, d, h = 0.3, 0.24, 0.3
                ep = EnclosureParams(size=(w, d, h), material="gray_box", door=True, latch=True, door_lo=0.16,
                                     drawer=DrawerSpec(x=0.0, w=w - 0.04, h=0.12, z=0.08, stroke=0.15), note="instrument box with latch + drawer")
                nm = "DrawerBox_B"
            else:
                w, d, h = 0.22, 0.12, 0.34
                b = cs_asset.build(ctx, sub("ControlStation"), Pose(W(x0, _YB + 0.06, zb), pose.quat), ControlStationParams(n_buttons=4), rng)
                b.info.update(slot=(x0, zb), w=w, d=d, h=h)
                bodies += b.bodies
                encl.append(b.info)
                items[b.info["name"]] = b.info
                continue
            b = encl_asset.build(ctx, sub(nm), Pose(W(x0, _YB + d / 2, zb), pose.quat), ep, rng)
            b.info.update(slot=(x0, zb), w=w, d=d, h=h)
            bodies += b.bodies
            encl.append(b.info)
            items[b.info["name"]] = b.info
        ib, ga = InstrumentBox(), Gauge()
        for i, (bx, bz) in enumerate([(-0.655, 1.1), (0.655, 1.1), (-0.655, 1.6), (0.655, 1.6)]):
            ib.build(ctx, f"{path}/InstBox_P{i}", Pose(W(bx, _YB, bz), pose.quat),
                     InstrumentBoxParams(size=(0.08, 0.07, 0.14), material="gray_box" if i % 2 else "cab_blue"), rng)
        for i, (gx, gz) in enumerate([(-0.21, 2.12), (0.21, 2.12), (0.0, 2.25), (-0.42, 2.2), (0.42, 2.2)]):
            ga.build(ctx, f"{path}/Gauge_P{i}", Pose(W(gx, _YB, gz), pose.quat),
                     GaugeParams(pipe_radius=0.0, mount="side", dial_yaw_deg=90.0, line="L-INST"), rng)
        detail = []
        for zrow in (1.38, 2.2, 0.9):
            for k in range(14):
                x = -0.62 + 0.095 * k
                detail.append(cube((0.07, 0.03, 0.05), (x, _YB, zrow), ["gray_box", "steel_dark", "yellow", "cable_orange", "steel_c"][(k + int(zrow * 10)) % 5], col=False))
            detail.append(cube((1.35, 0.012, 0.012), (0, _YB - 0.012, zrow - 0.03), "steel_dark", col=False))
        for k, (lx, lz) in enumerate([(-0.5, 1.28), (0.5, 1.28), (-0.62, 2.0), (0.62, 2.0), (0.0, 1.28)]):
            detail.append(cube((0.1, 0.004, 0.035), (lx, _YB, lz), "yellow" if k % 2 == 0 else "white", col=False))
        static(ctx, f"{path}/PlateDetail", pose.pos, pose.quat, detail, category="panel_detail")
        palette = ["black", "cable_red", "cable_orange", "cable_green", "black", "cable_blue"]
        for i, e_ in enumerate(encl):
            x0, zb = e_["slot"]
            n = int(rng.integers(2, 5))
            top_row = zb > 1.2
            for k in range(n):
                m = palette[int(rng.integers(len(palette)))]
                r = float(rng.uniform(0.008, 0.02))
                xc = x0 + (k - (n - 1) / 2) * 0.035 * (1 if e_["w"] > 0.3 else 0.5)
                yc = _YB + 0.02 + 0.03 * (k % 2) + r
                if top_row:
                    z0, z1 = zb + e_["h"], 2.4
                else:
                    z0, z1 = zb, (0.55 if rng.random() < 0.6 else 0.9)
                static(ctx, f"{path}/Cable_{i}_{k}", W(xc, yc, (z0 + z1) / 2), pose.quat, [cyl(r, abs(z1 - z0), mat=m, col=r > 0.01)])
        if p.manifold:
            board, ms = _panel_manifold_board()
            static(ctx, f"{path}/ManifoldBoard", pose.pos, pose.quat, board, category="manifold_board")
            static(ctx, f"{path}/Manifold", pose.pos, pose.quat, ms, category="tubing_manifold")
        fp = self.footprint(p)
        ctx.record(path, "instrument_panel", pose.pos, pose.quat, "mixed", size=fp.size, manifold=p.manifold)
        return Built(path, _world_anchors(pose, self.anchors(p)), info=dict(items=items, manifold=p.manifold), bodies=bodies)


# =========================================================================== pump set
class PumpSetParams(Params):
    scale: float = Field(1.0, ge=0.5, le=1.5, description="Uniform scale of the pump set (mass scales with scale^3)")
    spin_deg_s: float = Field(360.0, ge=0.0, le=1440.0, description="Velocity-drive target of the free-end coupling disc (deg/s); 0 = parked")


def _pump_layout(s):
    sh = [
        cube((2.9, 1.3, 0.25), (0, 0, 0.125), "brown"),
        cube((2.7, 1.1, 0.04), (0, 0, 0.27), "steel_dark"),
        cyl(0.34, 1.1, (-0.7, 0, 0.67), "motor", axis="x"),
        cyl(0.36, 0.06, (-1.2, 0, 0.67), "motor", axis="x", col=False),
        cube((0.3, 0.26, 0.2), (-0.75, 0, 1.1), "gray_box"),
        cube((0.75, 0.5, 0.18), (-0.7, 0, 0.36), "steel_dark"),
        cube((0.55, 0.42, 0.42), (0.15, 0, 0.53), "steel"),  # coupling guard
        cube((0.55, 0.012, 0.3), (0.15, 0.216, 0.55), "yellow", col=False),
        cyl(0.46, 0.55, (0.8, 0, 0.74), "pump_gray", axis="x"),  # volute casing
        cyl(0.11, 0.5, (0.95, 0, 1.25), "pump_gray"),  # discharge nozzle
        cyl(0.16, 0.03, (0.95, 0, 1.51), "steel_c"),  # discharge flange
        cyl(0.12, 0.5, (0.65, -0.55, 0.74), "pump_gray", axis="y"),  # suction nozzle
        cyl(0.17, 0.03, (0.65, -0.82, 0.74), "steel_c", axis="y"),  # suction flange
        cube((0.5, 0.6, 0.2), (0.8, 0, 0.35), "steel_dark"),
        cyl(0.05, 0.3, (1.19, 0, 0.74), "steel", axis="x"),
    ]
    for k in range(10):
        sh.append(cyl(0.355, 0.012, (-1.2 + 0.1 * k + 0.2, 0, 0.67), "motor", axis="x", col=False))
    for k in range(8):  # casing cover bolts (visual)
        a = k * math.pi / 4
        sh.append(cyl(0.02, 0.03, (1.09, 0.38 * math.cos(a), 0.74 + 0.38 * math.sin(a)), "steel_dark", col=False, axis="x"))
    rot = [cyl(0.05, 0.3, (0.15, 0, 0), "steel", axis="x"), cyl(0.27, 0.06, (0.32, 0, 0), "steel", axis="x")]
    for k in range(8):
        a = k * math.pi / 4
        rot.append(cyl(0.022, 0.03, (0.36, 0.2 * math.cos(a), 0.2 * math.sin(a)), "steel_dark", axis="x"))
    sh, rot = [_sc(x, s) for x in sh], [_sc(x, s) for x in rot]
    # seal-flush tubing manifold: pedestal beside the baseplate + three tubes to casing / motor ports (all in the pump frame)
    bp = np.array([0.45 * s, 0.78 * s, 0.0])
    ped = [cyl(0.025, 0.5 * s * 1.6, (0, 0, 0.4 * s), "steel_dark"), cube((0.16 * s * 1.6, 0.1 * s * 1.6, 0.12 * s * 1.6), (0, 0, 0.5 * s * 1.6 + 0.0), "steel_c")]
    top = np.array([bp[0], bp[1], 0.5 * s * 1.6 + 0.06 * s * 1.6])
    ports = [np.array(q, float) * s for q in ((0.8, 0.46, 0.74), (0.55, 0.46, 0.6), (-0.75, 0.13, 1.18))]
    tsh = []
    for i, pt in enumerate(ports):
        a = top + np.array([(i - 1) * 0.03, 0, 0.0])
        d = pt - a
        L = float(np.linalg.norm(d))
        tsh.append(cyl(0.006, L, tuple((a + pt) / 2), "tube", col=False, quat=qalign(d)))
        tsh.append(sph(0.011, tuple(pt), "steel", col=False))
        tsh.append(sph(0.011, tuple(a), "steel", col=False))
        tsh.append(cyl(0.017, 0.03, tuple(a + d / L * 0.1), "valve_red" if i == 0 else "steel_dark", col=False, quat=qalign(d)))
    return sh, rot, bp, ped, tsh, top


def _sc(sh, s):
    return Sh(sh.kind, tuple(float(x) * s for x in sh.size), tuple(float(x) * s for x in sh.pos), sh.quat, sh.mat, sh.col, sh.name)


@register
class PumpSet(Asset):
    id = "pump_set"
    description = ("Centrifugal pump set on a baseplate: motor with fins and terminal box, coupling guard, volute casing, flanged "
                   "discharge (up) and suction (-Y) nozzles, a velocity-driven coupling disc on the free shaft end (continuous revolute, "
                   "360 deg/s) and a seal-flush tubing manifold. Origin = baseplate centre on the floor; shaft along +X. Ports "
                   "nozzle_discharge / nozzle_suction (port_kind 'pipe', with radius) for pipe_run.")
    tags = frozenset({"industrial", "pump", "rotating_machine", "articulated", "piping", "port_provider"})
    Params = PumpSetParams
    physics = PhysicsInfo("articulated", "steel", 0.6, None, "primitives", "900 kg*scale^3 base + 25 kg*scale^3 rotor")
    example = {"id": "pump", "asset": "pump_set", "params": {"scale": 1.0}, "placement": {"at": [1.6, 0.3]}}

    def _links(self, p):
        s = p.scale
        _, rot, _, _, _, _ = _pump_layout(s)
        return [LinkSpec("rotor", (1.28 * s, 0, 0.74 * s), rot, 25.0 * s ** 3, "revolute", "X", None, None, stiffness=0.0, damping=60.0 * s ** 2,
                         target=360.0, role="rotor", unit="deg", meta=dict(spin_deg_s=p.spin_deg_s))]

    def footprint(self, p):
        sh, _, _, ped, tsh, _ = _pump_layout(p.scale)
        links = self._links(p)
        bp = np.array([0.45 * p.scale, 0.78 * p.scale, 0.0])
        return _fp(_merge(_assembly_box(sh, links), sh_bounds(ped, off=bp), sh_bounds(tsh)))

    def anchors(self, p):
        s = p.scale
        return [Anchor("nozzle_discharge", "port", (0.95 * s, 0.0, 1.52 * s), qalign((0, 0, 1)), port_kind="pipe", radius=0.11 * s),
                Anchor("nozzle_suction", "port", (0.65 * s, -0.84 * s, 0.74 * s), qalign((0, -1, 0)), port_kind="pipe", radius=0.12 * s),
                Anchor("rotor", "port", (1.28 * s, 0.0, 0.74 * s), port_kind="inspect", meta={"joint": "J_rotor", "continuous": True, "spin_deg_s": p.spin_deg_s})]

    def joints(self, p):
        j = _joint_infos(self._links(p))
        j[0].drive = dict(damping=60.0 * p.scale ** 2, target_velocity_deg_s=p.spin_deg_s)
        return j

    def build(self, ctx, path, pose, p, rng):
        s = p.scale
        sh, rot, bp, ped, tsh, top = _pump_layout(s)
        art = Art(ctx, path, pose.pos, 0.0, sh, 900.0 * s ** 3, "rotating_machine", quat=pose.quat, note="pump/compressor composite on baseplate")
        links = self._links(p)
        recs = _add_links(art, links)
        jp = ctx.kit.stage.GetPrimAtPath(f"{art.group}/J_rotor")
        drv = UsdPhysics.DriveAPI.Get(jp, "angular")
        drv.CreateTargetVelocityAttr(float(p.spin_deg_s))
        drv.GetTargetPositionAttr().Set(0.0)
        rec = recs["rotor"]
        rec["continuous"] = True
        rec["drive"]["target_velocity_deg_s"] = float(p.spin_deg_s)
        rec["drive"]["target"] = None
        # a continuous rotor has no limit to sweep: drop it from the sweep list the Isaac validator consumes
        ctx.scene["articulations"] = [a for a in ctx.scene["articulations"] if a.get("joint") != rec["path"]]
        mg = f"{path}/Manifold"
        ctx.kit.xform(mg)
        static(ctx, f"{mg}/pedestal", _W(pose, bp), pose.quat, ped, category="tubing_manifold")
        static(ctx, f"{mg}/tubes", pose.pos, pose.quat, tsh, category="tubing_manifold")
        fp = self.footprint(p)
        Wf = lambda q: [float(c) for c in _W(pose, np.array(q, float) * s)]
        info = dict(name=art.name, nozzle_top=Wf((0.95, 0, 1.52)), suction=Wf((0.65, -0.84, 0.74)), side=Wf((0.65, 0.2, 1.0)),
                    manifold_gauge_pt=[float(c) for c in _W(pose, top)], s=s, discharge_r=0.11 * s, suction_r=0.12 * s, size=(2.9 * s, 1.3 * s),
                    rotor_joint=rec["name"], rotor_center=Wf((1.28, 0, 0.74)))
        _record_art(ctx, path, "rotating_machine", pose, art, fp, scale=s)
        return Built(path, _world_anchors(pose, self.anchors(p)), info=info, bodies=_art_bodies(art))


# =========================================================================== flexible hose
class FlexibleHoseParams(Params):
    a: Vec3 = Field((0.0, 0.0, 1.0), description="First end (instance frame; identity pose = world)")
    b: Vec3 = Field((1.2, 0.0, 1.0), description="Second end (instance frame)")
    radius: float = Field(0.026, ge=0.008, le=0.08, description="Hose radius (m)")
    material: MatName = Field("green_band", description="Visual preset of the hose")
    n_links: int = Field(9, ge=3, le=20, description="Number of capsule segments (spherical joints with 55 deg cone limit)")
    slack: float = Field(1.18, ge=1.04, le=1.6, description="Hose length / end distance; reduced automatically until the sag clears the floor")
    floor_z: float = Field(0.0, description="Height of the floor below the hose (instance frame); the chain is kept 0.12 m above it")
    relax_iterations: int = Field(30000, ge=100, le=100000, description="Position-based relaxation steps for the rest shape")

    @model_validator(mode="after")
    def _c(self):
        if np.linalg.norm(np.subtract(self.a, self.b)) < 0.1:
            raise ValueError("hose ends must be at least 0.1 m apart")
        return self


def _relax_chain_py(a, b, n, slack, iters):
    """Equilibrium of an n-link pinned chain (equal link length) between fixed ends: position-based relaxation."""
    ax, ay, az = a
    bx, by, bz = b
    d0 = math.dist(a, b)
    ell = d0 * slack / n
    P = [[ax + (bx - ax) * i / n, ay + (by - ay) * i / n, az + (bz - az) * i / n] for i in range(n + 1)]
    for i in range(1, n):
        P[i][2] -= 0.15 * d0 * math.sin(math.pi * i / n)
    for _ in range(iters):
        for i in range(1, n):
            P[i][2] -= 1e-4
        for _ in range(4):
            for i in range(n):
                p0, p1 = P[i], P[i + 1]
                dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
                L = math.sqrt(dx * dx + dy * dy + dz * dz)
                k = (L - ell) / L * 0.5
                cx, cy, cz = k * dx, k * dy, k * dz
                if i > 0:
                    p0[0] += cx
                    p0[1] += cy
                    p0[2] += cz
                if i + 1 < n:
                    p1[0] -= cx
                    p1[1] -= cy
                    p1[2] -= cz
            P[0] = [ax, ay, az]
            P[n] = [bx, by, bz]
    return P


@lru_cache(maxsize=32)
def _relax_cached(a, b, n, slack, iters):
    return tuple(tuple(p) for p in _relax_chain_py(a, b, n, slack, iters))


def _hose_rest(a, b, r, n, slack, floor_z, iters):
    """Relaxed rest shape; slack is lowered until the chain clears floor_z + r + 0.12 (or reaches 1.04)."""
    a, b = tuple(float(c) for c in a), tuple(float(c) for c in b)
    while True:
        P = np.array(_relax_cached(a, b, n, round(slack, 6), min(iters, 6000)))
        if P[:, 2].min() > floor_z + r + 0.12 or slack < 1.04:
            break
        slack -= 0.03
    return np.array(_relax_cached(a, b, n, round(slack, 6), iters))


@register
class FlexibleHose(Asset):
    id = "flexible_hose"
    description = ("Flexible hose as a dynamic chain of capsule segments joined by spherical joints (55 deg cone), pinned at both "
                   "ends to static clamps; the rest shape is a relaxed catenary between the two end points a and b (instance frame). "
                   "Ports end_a / end_b (port_kind 'hose'). The first segment is a world-pinned joint, so place hoses by their end points.")
    tags = frozenset({"industrial", "hose", "flexible", "dynamic", "chain", "piping"})
    Params = FlexibleHoseParams
    physics = PhysicsInfo("dynamic", "rubber", 0.9, None, "primitives", "one rigid body per segment, mass 0.35 + 64*r^2*L kg, damped; last joint pinned to the world")
    example = {"id": "hose1", "asset": "flexible_hose", "params": {"a": [0.2, 0.3, 0.6], "b": [-0.4, 1.3, 0.5], "material": "green_band"}}

    def footprint(self, p):
        a, b = np.asarray(p.a, float), np.asarray(p.b, float)
        d = float(np.linalg.norm(b - a))
        ell = d * p.slack
        m = 2 * p.radius + 0.1
        # every chain point is within ell/2 of its nearer end and inside the ellipsoid |P-a|+|P-b| <= ell
        e = (ell - d) / 2
        lo, hi = np.minimum(a, b) - e - m, np.maximum(a, b) + e + m
        lo[2] = min(lo[2], min(a[2], b[2]) - ell / 2 - m)
        return _fp((lo, hi))

    def anchors(self, p):
        a, b = np.asarray(p.a, float), np.asarray(p.b, float)
        u = (b - a) / np.linalg.norm(b - a)
        return [Anchor("end_a", "port", tuple(a), qalign(-u), port_kind="hose", radius=p.radius),
                Anchor("end_b", "port", tuple(b), qalign(u), port_kind="hose", radius=p.radius)]

    def joints(self, p):
        return [JointInfo(f"J_{i}", "spherical", "X", 0.0, 55.0, "deg", 0.0, None, f"seg{min(i, p.n_links - 1)}", "hose_joint") for i in range(p.n_links + 1)]

    def build(self, ctx, path, pose, p, rng):
        r, n = p.radius, p.n_links
        a, b = _W(pose, p.a), _W(pose, p.b)
        P = _hose_rest(a, b, r, n, p.slack, pose.pos[2] + p.floor_z, p.relax_iterations)
        mat = p.material
        k = ctx.kit
        k.xform(path)
        static(ctx, f"{path}/anchor_a", a, Q_IDENT, [cyl(r * 1.5, 0.04, (0, 0, 0), "steel_dark", col=False)])
        static(ctx, f"{path}/anchor_b", b, Q_IDENT, [cyl(r * 1.5, 0.04, (0, 0, 0), "steel_dark", col=False)])
        paths, quats, poss = [], [], []
        for i in range(n):
            d = P[i + 1] - P[i]
            L = float(np.linalg.norm(d))
            q = qalign(d)
            mid = (P[i] + P[i + 1]) / 2
            sp = f"{path}/seg{i}"
            h = max(L - r, 0.02)
            mass = 0.35 + 8 * r * r * L * 8.0
            body(ctx, sp, mid, q, [cap(r, h, (0, 0, 0), mat), cyl(r * 1.12, 0.012, (0, 0, h * 0.25), mat, col=False), cyl(r * 1.12, 0.012, (0, 0, -h * 0.25), mat, col=False)],
                 mass, "rubber", linear_damping=0.5, angular_damping=4.0, iters=(16, 4))
            k.attr(k.stage.GetPrimAtPath(sp), f"{k.ns}:hose", T.String, Sdf.Path(path).name)
            paths.append(sp)
            quats.append(q)
            poss.append(mid)
        Rm = qaxis((0, 1, 0), -math.pi / 2)

        def sph_joint(jpath, b0, b1, q_frame0, q_frame1, lp0, lp1):
            j = UsdPhysics.SphericalJoint.Define(k.stage, jpath)
            if b0:
                j.CreateBody0Rel().SetTargets([Sdf.Path(b0)])
            j.CreateBody1Rel().SetTargets([Sdf.Path(b1)])
            j.CreateLocalPos0Attr(_v3(lp0))
            j.CreateLocalRot0Attr(_q(q_frame0))
            j.CreateLocalPos1Attr(_v3(lp1))
            j.CreateLocalRot1Attr(_q(q_frame1))
            j.CreateAxisAttr("X")
            j.CreateConeAngle0LimitAttr(55.0)
            j.CreateConeAngle1LimitAttr(55.0)

        for i in range(n):
            node = P[i]
            l1 = qmat(quats[i]).T @ (node - poss[i])
            if i == 0:
                sph_joint(f"{path}/J_{i}", None, paths[0], qmul(quats[0], Rm), Rm, node, l1)
            else:
                l0 = qmat(quats[i - 1]).T @ (node - poss[i - 1])
                sph_joint(f"{path}/J_{i}", paths[i - 1], paths[i], Rm, Rm, l0, l1)
        node = P[n]
        l0 = qmat(quats[n - 1]).T @ (node - poss[n - 1])
        j = UsdPhysics.SphericalJoint.Define(k.stage, f"{path}/J_{n}")  # last joint: segment n-1 -> world anchor
        j.CreateBody0Rel().SetTargets([Sdf.Path(paths[n - 1])])
        j.CreateLocalPos0Attr(_v3(l0))
        j.CreateLocalRot0Attr(_q(Rm))
        j.CreateLocalPos1Attr(_v3(node))
        j.CreateLocalRot1Attr(_q(qmul(quats[n - 1], Rm)))
        j.CreateAxisAttr("X")
        j.CreateConeAngle0LimitAttr(55.0)
        j.CreateConeAngle1LimitAttr(55.0)
        length = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        info = dict(name=Sdf.Path(path).name, segments=paths, anchor_a=[float(c) for c in a], anchor_b=[float(c) for c in b], radius=r, length=length,
                    joints=[f"{path}/J_{i}" for i in range(n + 1)])
        ctx.record(path, "hose", pose.pos, pose.quat, "dynamic", size=self.footprint(p).size, mass=sum(0.35 + 8 * r * r * float(np.linalg.norm(P[i + 1] - P[i])) * 8.0 for i in range(n)),
                   length=round(length, 4), n_links=n)
        return Built(path, _world_anchors(pose, self.anchors(p)), info=info, bodies=paths)


# =========================================================================== pipes
class Band(Sub):
    at: float = Field(0.5, ge=0.0, description="Distance of the band centre from the first end along the run (m)")
    material: MatName = Field("yellow", description="Visual preset of the band (e.g. yellow, orange_band, green_band, blue_band)")


class Gap(Sub):
    start: float = Field(ge=0.0, description="Start of an inline-equipment gap, distance from the first end (m)")
    end: float = Field(gt=0.0, description="End of the gap (m); flanged pipe faces are generated on both sides")


class PipeParams(Params):
    a: Vec3 = Field((-0.75, 0.0, 0.3), description="First end (instance frame)")
    b: Vec3 = Field((0.75, 0.0, 0.3), description="Second end (instance frame)")
    radius: float = Field(0.05, ge=0.01, le=0.5, description="Pipe radius (m)")
    material: MatName = Field("steel", description="Visual preset (steel, insul, stainless, ...)")
    bands: list[Band] = Field(default_factory=list, description="Colour identification bands")
    gaps: list[Gap] = Field(default_factory=list, description="Intervals left free for inline valves; the pipe is split there")
    flanges: bool = Field(True, description="Flange discs at the faces of every gap")
    end_flanges: bool = Field(True, description="Blind / mating flange at both free ends so no pipe end is bare")

    @model_validator(mode="after")
    def _c(self):
        if np.linalg.norm(np.subtract(self.a, self.b)) < 0.05:
            raise ValueError("pipe ends must be at least 5 cm apart")
        for g in self.gaps:
            if g.end <= g.start:
                raise ValueError("gap end must exceed start")
        return self


def _end_fitting(ctx, nm, p, u, r, equip):
    """Flange closing a free pipe end; ``u`` points from the end into the pipe."""
    static(ctx, nm, p + u * 0.0125, qalign(u), [cyl(r * 1.45, 0.025, mat="steel_c"), cyl(r * 1.1, 0.012, (0, 0, 0.018 if not equip else -0.0), "steel_dark", col=False)],
           category="fitting", kind="equipment_flange" if equip else "blind_flange")


@register
class Pipe(Asset):
    id = "pipe"
    description = ("Straight static pipe from a to b (instance frame) with optional colour bands, inline gaps for valves (flanged "
                   "faces) and end flanges. Ports end_a / end_b (port_kind 'pipe').")
    tags = frozenset({"industrial", "pipe", "piping", "static"})
    Params = PipeParams
    physics = PhysicsInfo("static", "steel", 0.6, None, "primitives")
    example = {"id": "feed", "asset": "pipe", "params": {"a": [0, 0, 1], "b": [3, 0, 1], "radius": 0.06, "bands": [{"at": 1.0, "material": "yellow"}]}}

    def footprint(self, p):
        return _fp(_seg_box(p.a, p.b, p.radius * 1.5 + 0.02))

    def anchors(self, p):
        a, b = np.asarray(p.a, float), np.asarray(p.b, float)
        u = (b - a) / np.linalg.norm(b - a)
        return [Anchor("end_a", "port", tuple(a), qalign(-u), port_kind="pipe", radius=p.radius),
                Anchor("end_b", "port", tuple(b), qalign(u), port_kind="pipe", radius=p.radius)]

    def build(self, ctx, path, pose, p, rng):
        ctx.kit.xform(path)
        a, b = _W(pose, p.a), _W(pose, p.b)
        r = p.radius
        d = b - a
        L = float(np.linalg.norm(d))
        u = d / L
        q = qalign(u)
        cuts = sorted((g.start, g.end) for g in p.gaps)
        spans, t = [], 0.0
        for c0, c1 in cuts:
            if c0 > t + 0.02:
                spans.append((t, c0))
            t = max(t, c1)
        if L > t + 0.02:
            spans.append((t, L))
        cut_edges = [x for cc in cuts for x in cc]
        n_flanges = 0
        for i, (s0, s1) in enumerate(spans):
            mid = a + u * (s0 + s1) / 2
            static(ctx, f"{path}/seg{i}", mid, q, [cyl(r, s1 - s0, mat=p.material)], category="pipe_segment", radius=float(r))
            if p.flanges:
                for s in (s0, s1):
                    if (s <= 0.01 or s >= L - 0.01) and not any(abs(s - c) < 0.01 for c in cut_edges):
                        continue
                    pos = a + u * (s + (0.0125 if s == s0 else -0.0125))
                    static(ctx, f"{path}/seg{i}_fl{int(s * 100)}", pos, q, [cyl(r * 1.4, 0.025, mat=p.material)])
                    n_flanges += 1
        for j, bd in enumerate(p.bands):
            static(ctx, f"{path}/band{j}", a + u * bd.at, q, [cyl(r * 1.04, 0.12, mat=bd.material, col=False)])
        if p.end_flanges:
            if spans and spans[0][0] <= 0.01:
                _end_fitting(ctx, f"{path}/fit_a", a, u, r, False)
                n_flanges += 1
            if spans and spans[-1][1] >= L - 0.01:
                _end_fitting(ctx, f"{path}/fit_b", b, -u, r, False)
                n_flanges += 1
        ctx.record(path, "pipe", pose.pos, pose.quat, "static", size=self.footprint(p).size, radius=float(r), length=round(L, 4))
        return Built(path, _world_anchors(pose, self.anchors(p)), info=dict(name=Sdf.Path(path).name, length=L, radius=float(r), n_segments=len(spans), n_flanges=n_flanges))


def _key(p):
    return tuple(int(round(float(c) * 1000)) for c in p)


class _PipeNet:
    """Connected pipe network (graph edges = pipes); every node gets a fitting (elbow / tee / reducer / flanged joint /
    blind or equipment flange), so no pipe can end without one.  Works in world coordinates."""

    def __init__(self, ctx, group):
        self.ctx, self.group = ctx, group
        self.nodes: dict = {}
        self.edges: list = []
        self.supports = 0
        self.fittings: dict = {}
        self._n = 0

    def _node(self, p):
        return self.nodes.setdefault(_key(p), dict(p=np.asarray(p, float), edges=[], equip=None))

    def equip(self, p, label="equipment"):
        self._node(p)["equip"] = label

    def pipe(self, a, b, r, mat="steel", bands=(), name=None, support="none", support_ref=0.0, spacing=1.3):
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if L < 0.02:
            return None
        u = (b - a) / L
        nm = name or f"p{self._n}"
        self._n += 1
        sh = [cyl(r, L, mat=mat)] + [cyl(r * 1.05, 0.14, (0, 0, (t - L / 2)), c, col=False) for t, c in bands if 0.07 < t < L - 0.07]
        static(self.ctx, f"{self.group}/{nm}", (a + b) / 2, qalign(u), sh, category="pipe_segment", radius=float(r))
        self.edges.append(dict(name=nm, a=_key(a), b=_key(b), r=float(r), length=L, path=f"{self.group}/{nm}"))
        self._node(a)["edges"].append((u, float(r), nm))
        self._node(b)["edges"].append((-u, float(r), nm))
        if support != "none" and abs(u[2]) < 0.2 and L > 0.6:
            n = max(1, int(L / spacing))
            for i in range(n):
                self._support(a + u * ((i + 0.5) * L / n), r, support, support_ref, f"{nm}_s{i}")
        return nm

    def _support(self, p, r, kind, ref, name):
        ctx = self.ctx
        if kind == "down":
            h = p[2] - r - ref
            if h > 0.06:
                static(ctx, f"{self.group}/{name}", (p[0], p[1], ref), Q_IDENT, [cyl(0.02, h, (0, 0, h / 2), "steel_dark"), cube((0.12, 0.12, 0.012), (0, 0, 0.006), "steel_dark", col=False)])
                self.supports += 1
        elif kind == "up":  # hanger rod to the beam above
            h = ref - (p[2] + r)
            if h > 0.05:
                static(ctx, f"{self.group}/{name}", (p[0], p[1], p[2] + r + h / 2), Q_IDENT,
                       [cyl(0.012, h, (0, 0, 0), "steel_dark", col=False), cube((0.06, 0.06, 0.01), (0, 0, -h / 2 + 0.005), "steel_dark", col=False)])
                self.supports += 1

    def finish(self):
        counts = dict(elbow=0, tee=0, reducer=0, flanged_joint=0, blind_flange=0, equipment_flange=0, cross=0)
        ctx = self.ctx
        for i, (k, n) in enumerate(self.nodes.items()):
            p, E = n["p"], n["edges"]
            if not E:
                continue
            rmax = max(e[1] for e in E)
            nm = f"{self.group}/fit{i}"
            if len(E) == 1:
                u, r, _ = E[0]
                t = "equipment_flange" if n["equip"] else "blind_flange"
                static(ctx, nm, p + u * 0.0125, qalign(u), [cyl(r * 1.45, 0.025, mat="steel_c"), cyl(r * 1.1, 0.012, (0, 0, 0.018 if not n["equip"] else -0.0), "steel_dark", col=False)],
                       category="fitting", kind=t)
                counts[t] += 1
            elif len(E) == 2:
                (u1, r1, _), (u2, r2, _) = E
                if float(np.dot(u1, u2)) < -0.97:
                    ax = qalign(u1)
                    if abs(r1 - r2) > 0.002:
                        rm = (r1 + r2) / 2
                        static(ctx, nm, p, ax, [cyl(rm, 0.12, mat="steel"), cyl(max(r1, r2) * 1.4, 0.025, (0, 0, 0.06), "steel_c"), cyl(min(r1, r2) * 1.4, 0.025, (0, 0, -0.06), "steel_c")],
                               category="fitting", kind="reducer")
                        counts["reducer"] += 1
                    else:
                        static(ctx, nm, p, ax, [cyl(r1 * 1.4, 0.03, mat="steel_c")], category="fitting", kind="flanged_joint")
                        counts["flanged_joint"] += 1
                else:
                    static(ctx, nm, p, Q_IDENT, [sph(rmax * 1.08, mat="steel"), cyl(rmax * 1.3, 0.02, tuple(E[0][0] * (rmax * 1.0 + 0.02)), "steel_c", col=False, quat=qalign(E[0][0]))],
                           category="fitting", kind="elbow")
                    counts["elbow"] += 1
            else:
                kind = "tee" if len(E) == 3 else "cross"
                rb = min(e[1] for e in E)
                static(ctx, nm, p, Q_IDENT, [sph(max(rb * 1.5, rmax * 1.03), mat="steel")], category="fitting", kind=kind)
                counts[kind] += 1
        self.fittings = counts
        n_nodes = sum(1 for n in self.nodes.values() if n["edges"])
        if sum(counts.values()) != n_nodes:  # pragma: no cover - guard for the "no bare pipe end" invariant
            raise RuntimeError(f"pipe network has {n_nodes} nodes but {sum(counts.values())} fittings")
        return counts

    def report(self):
        diam = [2 * e["r"] for e in self.edges]
        return dict(n_pipe_segments=len(self.edges), n_nodes=len(self.nodes), fittings=self.fittings, n_supports=self.supports,
                    unfitted_pipe_ends=sum(1 for n in self.nodes.values() if n["edges"]) - sum(self.fittings.values()),
                    diameter_range_m=[round(min(diam), 3), round(max(diam), 3)] if diam else None)


_ORDERS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")


class PipeRunParams(Params):
    a: Vec3 = Field((0.0, 0.0, 0.5), description="Start point on the pipe axis (instance frame; use a pump nozzle / valve port position)")
    b: Vec3 = Field((1.5, 1.0, 1.5), description="End point on the pipe axis (instance frame)")
    radius: float = Field(0.06, ge=0.01, le=0.5, description="Pipe radius (m)")
    material: MatName = Field("steel", description="Visual preset (steel, insul, ...)")
    order: Literal["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"] = Field("xyz", description="Axis order of the legs; offsets below 2 cm are ignored")
    bands: list[Band] = Field(default_factory=list, description="Colour bands, distance from the start along the whole route")
    support: Literal["none", "down", "up"] = Field("down", description="Horizontal runs > 0.6 m get posts down to support_ref ('down') or hanger rods up to it ('up')")
    support_ref: float = Field(0.0, description="Height (instance frame) of the floor for 'down' or the beam underside for 'up'")
    support_spacing: float = Field(1.3, ge=0.5, le=4.0, description="Maximum support spacing (m)")
    end_a: Literal["equipment", "blind"] = Field("equipment", description="Fitting at the start: flange mating with equipment, or blind flange")
    end_b: Literal["equipment", "blind"] = Field("equipment", description="Fitting at the end")

    @model_validator(mode="after")
    def _c(self):
        if sum(abs(x - y) >= 0.02 for x, y in zip(self.a, self.b)) == 0:
            raise ValueError("a and b coincide (offsets below 2 cm are ignored)")
        return self


def _route(a, b, order, tol=0.02):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pts = [a.copy()]
    cur = a.copy()
    for ax in order:
        i = "xyz".index(ax)
        d = b[i] - cur[i]
        if abs(d) >= tol:
            cur = cur.copy()
            cur[i] += d
            pts.append(cur)
    return pts


@register
class PipeRun(Asset):
    id = "pipe_run"
    description = ("Axis-aligned pipe route between two points (legs in the chosen axis order) built as a small pipe network: elbows at "
                   "every corner, a flange at both ends (equipment or blind) so no pipe end is bare, optional colour bands and "
                   "post / hanger supports for long horizontal legs. Info carries the fitting counts. Use port positions (pump "
                   "nozzles, valve pipe_in/out) as a/b.")
    tags = frozenset({"industrial", "pipe", "piping", "pipe_run", "static", "route"})
    Params = PipeRunParams
    physics = PhysicsInfo("static", "steel", 0.6, None, "primitives")
    example = {"id": "pump_out", "asset": "pipe_run", "params": {"a": [0.95, 0.0, 1.52], "b": [0.95, 2.0, 2.2], "radius": 0.11, "order": "zyx", "material": "insul"}}

    def footprint(self, p):
        box = _seg_box(p.a, p.b, p.radius * 1.5 + 0.02)
        lo, hi = box
        if p.support == "down":
            lo = lo.copy()
            lo[2] = min(lo[2], p.support_ref)
        elif p.support == "up":
            hi = hi.copy()
            hi[2] = max(hi[2], p.support_ref)
        return _fp((lo, hi))

    def anchors(self, p):
        pts = _route(p.a, p.b, p.order)
        if len(pts) < 2:
            pts = [np.asarray(p.a, float), np.asarray(p.b, float)]
        u0 = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
        u1 = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
        return [Anchor("end_a", "port", tuple(pts[0]), qalign(-u0), port_kind="pipe", radius=p.radius),
                Anchor("end_b", "port", tuple(pts[-1]), qalign(u1), port_kind="pipe", radius=p.radius)]

    def build(self, ctx, path, pose, p, rng):
        ctx.kit.xform(path)
        pts = _route(p.a, p.b, p.order)
        if len(pts) < 2:
            raise ValueError("pipe_run needs at least one leg of 2 cm or more")
        net = _PipeNet(ctx, path)
        zref = float(pose.pos[2] + p.support_ref)  # supports are referenced to the instance frame (yaw-only poses)
        wp = [_W(pose, q) for q in pts]
        cum = np.concatenate([[0.0], np.cumsum([np.linalg.norm(wp[i + 1] - wp[i]) for i in range(len(wp) - 1)])])
        for i in range(len(wp) - 1):
            bands = [(bd.at - cum[i], bd.material) for bd in p.bands if cum[i] < bd.at < cum[i + 1]]
            net.pipe(wp[i], wp[i + 1], p.radius, p.material, bands, name=f"leg{i}", support=p.support, support_ref=zref, spacing=p.support_spacing)
        if p.end_a == "equipment":
            net.equip(wp[0])
        if p.end_b == "equipment":
            net.equip(wp[-1])
        counts = net.finish()
        rep = net.report()
        length = float(cum[-1])
        end_err = float(np.linalg.norm(wp[-1] - _W(pose, p.b)))
        ctx.record(path, "pipe_run", pose.pos, pose.quat, "static", size=self.footprint(p).size, radius=float(p.radius), length=round(length, 4))
        info = dict(name=Sdf.Path(path).name, length=length, radius=float(p.radius), waypoints=[_r(w) for w in wp], fittings=counts, report=rep, end_error_m=round(end_err, 4))
        anchors = [Anchor("end_a", "port", tuple(pts[0]), qalign(-(pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])), port_kind="pipe", radius=p.radius),
                   Anchor("end_b", "port", tuple(pts[-1]), qalign((pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])), port_kind="pipe", radius=p.radius)]
        return Built(path, _world_anchors(pose, anchors), info=info)


# =========================================================================== worker
class WorkerParams(Params):
    reach_target: Vec3 = Field((0.75, 0.2, 1.4), description="Point the left hand reaches towards, in the worker frame (+X forward, +Y left, origin at the feet)")


def _worker_layout(reach_target):
    sh_l = np.array([0.0, 0.2, 1.42])
    d = np.asarray(reach_target, float) - sh_l
    d = d / np.linalg.norm(d)
    elbow_pt = sh_l + d * 0.30 + np.array([0, 0.03, -0.03])
    hand_pt = sh_l + d * 0.58

    def lim(p0, p1, r, mat="coverall_red", col=True):
        dd = np.asarray(p1, float) - np.asarray(p0, float)
        L = float(np.linalg.norm(dd))
        return cap(r, max(L - 2 * r, 0.02), tuple((np.asarray(p0, float) + np.asarray(p1, float)) / 2), mat, col=col, quat=qalign(dd))

    shapes = []
    for sgn in (-1, 1):
        hip, knee, ankle = (0.0, sgn * 0.095, 0.93), (0.02, sgn * 0.10, 0.50), (0.0, sgn * 0.10, 0.09)
        shapes += [lim(hip, knee, 0.085), lim(knee, ankle, 0.07), cube((0.28, 0.105, 0.085), (0.06, sgn * 0.10, 0.043), "boot")]
    shapes += [
        cap(0.16, 0.12, (0, 0, 0.98), "coverall_red"), cap(0.165, 0.24, (0, 0, 1.2), "coverall_red"),
        esph(0.2, 0.15, 0.2, (0.0, 0, 1.32), "coverall_red"),
        cyl(0.172, 0.06, (0, 0, 1.0), "black", col=False), cyl(0.17, 0.035, (0, 0, 1.15), "white", col=False),
        cyl(0.05, 0.1, (0.0, 0, 1.52), "skin", col=False), esph(0.095, 0.085, 0.115, (0.02, 0, 1.64), "skin"),
        cyl(0.105, 0.06, (0.02, 0, 1.70), "hardhat", col=False), esph(0.108, 0.108, 0.07, (0.02, 0, 1.72), "hardhat"),
        cyl(0.145, 0.012, (0.04, 0, 1.675), "hardhat", col=False), cube((0.24, 0.02, 0.03), (0.02, 0, 1.76), "hardhat", col=False),
        cyl(0.05, 0.045, (0.02, 0.115, 1.64), "black", axis="y", col=False), cyl(0.05, 0.045, (0.02, -0.115, 1.64), "black", axis="y", col=False),
        cyl(0.012, 0.26, (0.02, 0, 1.76), "black", axis="y", col=False),
        sph(0.07, (0.0, 0.2, 1.42), "coverall_red", col=False), sph(0.07, (0.0, -0.2, 1.42), "coverall_red", col=False),
        lim((0.0, -0.2, 1.42), (0.03, -0.24, 1.12), 0.05), lim((0.03, -0.24, 1.12), (0.08, -0.24, 0.88), 0.045),
        sph(0.05, (0.08, -0.24, 0.82), "glove", col=False),
        lim(sh_l, elbow_pt, 0.052), lim(elbow_pt, hand_pt, 0.046), sph(0.052, tuple(hand_pt + d * 0.035), "glove", col=False),
    ]
    return shapes, hand_pt


@register
class WorkerStatic(Asset):
    id = "worker_static"
    description = ("Static humanoid placeholder, 1.78 m tall: red coveralls, hard hat, ear defenders, gloves, boots; the left arm reaches "
                   "towards reach_target (worker frame, +X forward). Static colliders (legs, torso, arm). Origin = between the feet. "
                   "Port 'hand' marks the fingertips.")
    tags = frozenset({"industrial", "human", "worker", "static", "agent", "obstacle"})
    Params = WorkerParams
    physics = PhysicsInfo("static", "steel", 0.6, None, "primitives", "static colliders only")
    example = {"id": "operator", "asset": "worker_static", "params": {"reach_target": [0.8, 0.25, 1.5]}, "placement": {"at": [0.0, -0.8]}}

    def footprint(self, p):
        return _fp(sh_bounds(_worker_layout(p.reach_target)[0]))

    def anchors(self, p):
        return [Anchor("hand", "port", tuple(_worker_layout(p.reach_target)[1]), port_kind="hand")]

    def build(self, ctx, path, pose, p, rng):
        shapes, hand = _worker_layout(p.reach_target)
        static(ctx, path, pose.pos, pose.quat, shapes, category="human_worker", note="humanoid placeholder, static colliders")
        ctx.record(path, "human_worker", pose.pos, pose.quat, "static", size=self.footprint(p).size, height_m=1.78)
        info = dict(pos=_r(pose.pos), yaw=float(pose.yaw), hand=_r(_W(pose, hand), 3), height_m=1.78)
        return Built(path, _world_anchors(pose, self.anchors(p)), info=info)


# =========================================================================== loose tools
TOOL_KINDS = ("wrench", "hammer", "bucket", "hardhat", "hardhat_blue", "toolbox", "coil", "spool", "glove_box")


def _tool(kind):
    """(shapes, mass, physics-material preset) of a loose tool resting on its own origin plane (z = 0 is the bottom)."""
    if kind == "wrench":
        return [cube((0.3, 0.032, 0.012), (0, 0, 0.006), "steel"), cyl(0.028, 0.012, (0.15, 0, 0.006), "steel"), cyl(0.026, 0.012, (-0.15, 0, 0.006), "steel")], 0.5, "steel"
    if kind == "hammer":
        return [cube((0.3, 0.03, 0.03), (0, 0, 0.015), "wood_dark"), cube((0.08, 0.05, 0.05), (0.14, 0, 0.025), "steel")], 0.9, "steel"
    if kind == "bucket":
        R, H, th = 0.15, 0.28, 0.006
        sh = [cyl(R, th, (0, 0, th / 2), "bucket_yellow")]
        for j in range(10):
            a = 2 * math.pi * j / 10
            sh.append(cube((2 * math.pi * R / 10 * 1.06, th, H), (R * math.cos(a), R * math.sin(a), H / 2), "bucket_yellow", quat=qyaw(a + math.pi / 2)))
        return sh, 0.9, "plastic"
    if kind in ("hardhat", "hardhat_blue"):
        hm = "hardhat" if kind == "hardhat" else "hardhat_white"
        return [cyl(0.17, 0.008, (0, 0, 0.004), hm), cyl(0.105, 0.075, (0, 0, 0.0455), hm), esph(0.105, 0.105, 0.06, (0, 0, 0.083), hm)], 0.35, "plastic"
    if kind == "toolbox":
        return [cube((0.42, 0.18, 0.16), (0, 0, 0.08), "valve_red"), cube((0.3, 0.02, 0.02), (0, 0, 0.17), "steel", col=False)], 4.0, "steel"
    if kind == "coil":
        return [cyl(0.2, 0.09, (0, 0, 0.045), "black"), cyl(0.09, 0.091, (0, 0, 0.046), "steel_dark", col=False)], 1.5, "plastic"
    if kind == "spool":
        return [cyl(0.06, 0.3, (0, 0, 0.09), "steel", axis="x"), cyl(0.09, 0.025, (0.14, 0, 0.09), "steel", axis="x"), cyl(0.09, 0.025, (-0.14, 0, 0.09), "steel", axis="x")], 3.0, "steel"
    if kind == "glove_box":
        return [cube((0.25, 0.18, 0.1), (0, 0, 0.05), "orange_box"), cube((0.15, 0.003, 0.05), (0, 0.091, 0.05), "white", col=False)], 1.0, "plastic"
    raise ValueError(f"unknown tool kind {kind!r}")


class LooseToolParams(Params):
    kind: Literal["wrench", "hammer", "bucket", "hardhat", "hardhat_blue", "toolbox", "coil", "spool", "glove_box"] = Field(
        "wrench", description="Which loose item to drop")


@register
class LooseTool(Asset):
    id = "loose_tool"
    description = ("Loose dynamic prop resting on a surface: wrench, hammer, bucket, hardhat, hardhat_blue, toolbox, coil (cable), spool "
                   "(pipe spool) or glove_box. One rigid body of analytic colliders (0.35-4 kg). Pose z = supporting surface height.")
    tags = frozenset({"industrial", "clutter", "tool", "loose", "dynamic", "obstacle"})
    Params = LooseToolParams
    physics = PhysicsInfo("dynamic", "steel", None, (0.35, 4.0), "primitives", "mass and material per kind; settles 4 mm above the support")
    example = {"id": "tools", "asset": "loose_tool", "count": [2, 4], "params": {"kind": {"choice": ["wrench", "bucket", "hardhat"]}}, "placement": {"inside": "deck"}}

    def footprint(self, p):
        return _fp(sh_bounds(_tool(p.kind)[0]))

    def build(self, ctx, path, pose, p, rng):
        sh, m, pm = _tool(p.kind)
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + 0.004)
        pr = body(ctx, path, pos, pose.quat, sh, m, pm, angular_damping=0.3, iters=(16, 4))
        k = ctx.kit
        k.attr(pr, f"{k.ns}:loose", T.String, p.kind)
        k.tag(pr, "loose_item", kind=p.kind)
        ctx.record(path, "loose_item", pos, pose.quat, "dynamic", size=self.footprint(p).size, mass=m, kind=p.kind)
        return Built(path, [Anchor("top", "support", (pos[0], pos[1], pos[2] + self.footprint(p).z1), pose.quat)], info=dict(kind=p.kind, mass=m, path=path), bodies=[path])


# =========================================================================== I-beam
class IBeamParams(Params):
    a: Vec3 = Field((-1.0, 0.0, 2.0), description="First end (instance frame); the beam must be axis-aligned")
    b: Vec3 = Field((1.0, 0.0, 2.0), description="Second end (instance frame)")
    flange_width: float = Field(0.2, ge=0.05, le=0.6, description="Flange width w (m)")
    depth: float = Field(0.2, ge=0.05, le=0.8, description="Section depth h (m): vertical for X / Y beams, along X for columns")
    flange_thickness: float = Field(0.014, ge=0.003, le=0.05)
    web_thickness: float = Field(0.012, ge=0.003, le=0.05)
    material: MatName = Field("steel_d", description="Visual preset (steel_d, steel_b, steel_c, ...)")

    @model_validator(mode="after")
    def _c(self):
        d = np.abs(np.subtract(self.b, self.a))
        if np.linalg.norm(d) < 0.05:
            raise ValueError("beam ends must be at least 5 cm apart")
        if sorted(d)[1] > 1e-3:
            raise ValueError("beam must be axis-aligned (a and b differ along one axis only)")
        if 2 * self.flange_thickness >= self.depth:
            raise ValueError("flange_thickness too large for depth")
        return self


def _beam_layout(p):
    a, b = np.asarray(p.a, float), np.asarray(p.b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    ax = int(np.argmax(np.abs(d)))
    q = Q_IDENT if ax == 2 else (qaxis((0, 1, 0), math.pi / 2) if ax == 0 else qfrom_mat([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    w, h, tf, tw = p.flange_width, p.depth, p.flange_thickness, p.web_thickness
    sh = [cube((tf, w, L), (h / 2 - tf / 2, 0, 0), p.material), cube((tf, w, L), (-(h / 2 - tf / 2), 0, 0), p.material), cube((h - 2 * tf, tw, L), (0, 0, 0), p.material)]
    return sh, q, (a + b) / 2


@register
class IBeam(Asset):
    id = "ibeam"
    description = ("Static I-beam / H-column between two axis-aligned end points (instance frame): two flanges and a web, colliders on all "
                   "three plates. Compose several for a steel frame (columns along Z, beams along X / Y).")
    tags = frozenset({"industrial", "structure", "steel", "beam", "column", "static"})
    Params = IBeamParams
    physics = PhysicsInfo("static", "steel", 0.6, None, "primitives")
    example = {"id": "col1", "asset": "ibeam", "params": {"a": [0, 0, 0], "b": [0, 0, 4], "flange_width": 0.2, "depth": 0.2}}

    def footprint(self, p):
        sh, q, c = _beam_layout(p)
        return _fp(sh_bounds(sh, qmat(q), c))

    def anchors(self, p):
        sh, q, c = _beam_layout(p)
        lo, hi = sh_bounds(sh, qmat(q), c)
        return [Anchor("top", "support", (float(c[0]), float(c[1]), float(hi[2])),
                       polygon=[(-(hi[0] - lo[0]) / 2, -(hi[1] - lo[1]) / 2), ((hi[0] - lo[0]) / 2, -(hi[1] - lo[1]) / 2),
                                ((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2), (-(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2)])]

    def build(self, ctx, path, pose, p, rng):
        sh, q, c = _beam_layout(p)
        static(ctx, path, _W(pose, c), qmul(pose.quat, q), sh, category="steel_beam")
        ctx.record(path, "steel_beam", pose.pos, pose.quat, "static", size=self.footprint(p).size, length=round(float(np.linalg.norm(np.subtract(p.b, p.a))), 4))
        return Built(path, _world_anchors(pose, self.anchors(p)), info=dict(name=Sdf.Path(path).name, length=float(np.linalg.norm(np.subtract(p.b, p.a)))))
