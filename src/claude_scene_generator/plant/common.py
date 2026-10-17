"""Shared helpers for the plant generator: context, primitive shape specs, articulated assemblies."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from ..geom import Part, Q_IDENT, compose, qaxis, qfrom_mat, qmat, qmul, qyaw, yaw_of
from ..usdkit import Kit, T, _q, _v3

Q_X = qaxis((0, 1, 0), math.pi / 2)  # rotates +Z onto +X
Q_Y = qaxis((1, 0, 0), -math.pi / 2)  # rotates +Z onto +Y


def qalign(v):
    """Quaternion rotating +Z onto direction v."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    c = float(v[2])
    if c > 0.999999:
        return Q_IDENT
    if c < -0.999999:
        return qaxis((1, 0, 0), math.pi)
    ax = np.cross((0, 0, 1), v)
    return qaxis(ax, math.acos(c))


@dataclass
class Sh:
    kind: str  # cube (sx,sy,sz) | cyl (r,h) | cap (r,h) | sph (r,) | esph (rx,ry,rz) visual-only ellipsoid
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = Q_IDENT
    mat: str = "steel"
    col: bool = True
    name: str = ""


def cube(size, pos=(0, 0, 0), mat="steel", col=True, quat=Q_IDENT):
    return Sh("cube", tuple(size), tuple(pos), quat, mat, col)


def cyl(r, h, pos=(0, 0, 0), mat="steel", col=True, axis="z", quat=None):
    q = quat if quat is not None else {"z": Q_IDENT, "x": Q_X, "y": Q_Y}[axis]
    return Sh("cyl", (r, h), tuple(pos), q, mat, col)


def cap(r, h, pos=(0, 0, 0), mat="steel", col=True, quat=Q_IDENT):
    return Sh("cap", (r, h), tuple(pos), quat, mat, col)


def sph(r, pos=(0, 0, 0), mat="steel", col=True):
    return Sh("sph", (r,), tuple(pos), Q_IDENT, mat, col)


def esph(rx, ry, rz, pos=(0, 0, 0), mat="steel"):
    return Sh("esph", (rx, ry, rz), tuple(pos), Q_IDENT, mat, False)


def rot_world(yaw, v):
    return qmat(qyaw(yaw)) @ np.asarray(v, float)


@dataclass
class Ctx:
    kit: Kit
    rng: np.random.Generator
    seed: int
    dirt: float
    mats: dict = field(default_factory=dict)
    pm: dict = field(default_factory=dict)
    joints: list = field(default_factory=list)
    arts: list = field(default_factory=list)
    gauges: list = field(default_factory=list)
    valves: list = field(default_factory=list)
    lines: dict = field(default_factory=dict)
    goal_targets: list = field(default_factory=list)
    keepout: list = field(default_factory=list)  # (cx, cy, hx, hy) rectangles that pipes/props must avoid
    n_shapes: int = 0

    # -- materials
    def m(self, name):
        return self.mats[name]

    def phys(self, name):
        return self.pm[name]

    # -- shape emission
    def emit(self, root, shapes, pmat="steel"):
        """Create the prims for ``shapes`` under ``root``; returns Parts (collision shapes only) for inertia."""
        k = self.kit
        parts = []
        for i, s in enumerate(shapes):
            p = f"{root}/{s.name or f'{s.kind}{i}'}"
            vm = self.mats[s.mat]
            col = s.col
            pm = self.pm[pmat] if col else None
            if s.kind == "cube":
                k.cube(p, s.size, s.pos, s.quat, vmat=vm, pmat=pm, collide=col)
                parts.append(Part("box", s.size, s.pos, s.quat)) if col else None
            elif s.kind == "cyl":
                k.cylinder(p, s.size[0], s.size[1], s.pos, s.quat, vmat=vm, pmat=pm, collide=col)
                parts.append(Part("cyl", s.size, s.pos, s.quat)) if col else None
            elif s.kind == "cap":
                k.capsule(p, s.size[0], s.size[1], s.pos, s.quat, vmat=vm, pmat=pm, collide=col)
                parts.append(Part("cyl", (s.size[0], s.size[1] + 2 * s.size[0]), s.pos, s.quat)) if col else None
            elif s.kind == "sph":
                k.sphere(p, s.size[0], s.pos, vmat=vm, pmat=pm, collide=col)
                parts.append(Part("sph", s.size, s.pos, s.quat)) if col else None
            elif s.kind == "esph":
                prim = k.sphere(p, 1.0, s.pos, vmat=vm)
                k.set_pose(prim, s.pos, s.quat, s.size)
            self.n_shapes += 1
        return parts

    def static(self, path, pos, quat, shapes, pmat="steel", category=None, **tag):
        x = self.kit.xform(path, pos, quat)
        self.emit(path, shapes, pmat)
        if category:
            self.kit.tag(x, category, **tag)
        return x

    def body(self, path, pos, quat, shapes, mass, pmat="steel", category=None, art=None, **rb):
        parts = self.emit_parts_first(path, pos, quat, shapes, pmat, mass, **rb)
        return parts

    def emit_parts_first(self, path, pos, quat, shapes, pmat, mass, **rb):
        # inertia needs parts, but prims must live under the body prim: build a throw-away Part list first
        parts = []
        for s in shapes:
            if not s.col:
                continue
            if s.kind == "cube":
                parts.append(Part("box", s.size, s.pos, s.quat))
            elif s.kind == "cyl":
                parts.append(Part("cyl", s.size, s.pos, s.quat))
            elif s.kind == "cap":
                parts.append(Part("cyl", (s.size[0], s.size[1] + 2 * s.size[0]), s.pos, s.quat))
            elif s.kind == "sph":
                parts.append(Part("sph", s.size, s.pos, s.quat))
        prim = self.kit.rigid_body(path, pos, quat, parts=parts, mass=mass, **rb)
        self.emit(path, shapes, pmat)
        return prim


class Art:
    """A fixed-base articulation: a root link welded to the world plus child links joined by revolute/prismatic joints."""

    def __init__(self, ctx: Ctx, name, pos, yaw, shapes, mass, category, pmat="steel", quat=None, **meta):
        self.ctx, self.name, self.category = ctx, name, category
        self.group = f"/World/Equipment/{name}"
        ctx.kit.xform(self.group)
        self.q = tuple(quat) if quat is not None else qyaw(yaw)
        self.pos = np.asarray(pos, float)
        self.yaw = yaw if quat is None else yaw_of(self.q)
        path = f"{self.group}/root"
        prim = ctx.body(path, self.pos, self.q, shapes, mass, pmat, disable_gravity=True, iters=(16, 2))
        ctx.kit.attr(prim, "plant:art", T.String, name)
        ctx.kit.attr(prim, "plant:role", T.String, "root")
        ctx.kit.tag(prim, category, **meta)
        fj = UsdPhysics.FixedJoint.Define(ctx.kit.stage, f"{self.group}/J_world")
        fj.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        fj.CreateLocalPos0Attr(_v3(self.pos))
        fj.CreateLocalRot0Attr(_q(self.q))
        fj.CreateLocalPos1Attr(_v3((0, 0, 0)))
        fj.CreateLocalRot1Attr(_q(Q_IDENT))
        ctx.kit.articulation_root(prim)
        self.links = {"root": dict(path=path, pos=self.pos, mass=mass)}
        self.pmat = pmat
        ctx.arts.append(self)
        self.joint_names = []

    def world(self, local):
        """World position of a point expressed in the assembly frame (all links share the root orientation)."""
        return self.pos + qmat(self.q) @ np.asarray(local, float)

    def link(self, name, lpos, shapes, mass, jtype, axis="Z", lower=0.0, upper=1.0, parent="root", jrot=Q_IDENT,
             stiffness=0.0, damping=0.0, max_force=1e6, target=0.0, friction=0.0, unit=None, mimic=None, pmat=None,
             role="", meta=None, parent_lpos=None):
        """Add a child link whose origin sits at ``lpos`` (assembly frame) and is joined to ``parent``."""
        ctx = self.ctx
        P = self.links[parent]
        off = np.asarray(lpos, float) - (P["lpos"] if "lpos" in P else np.zeros(3))
        wpos = self.world(lpos)
        path = f"{self.group}/{name}"
        prim = ctx.body(path, wpos, self.q, shapes, mass, pmat or self.pmat, iters=(16, 2), angular_damping=0.5)
        ctx.kit.attr(prim, "plant:art", T.String, self.name)
        ctx.kit.attr(prim, "plant:role", T.String, role or name)
        jp = f"{self.group}/J_{name}"
        j = ctx.kit.joint(jp, jtype, P["path"], path, off, (0, 0, 0), axis=axis, lower=lower, upper=upper,
                          target=target, stiffness=stiffness, damping=damping, max_force=max_force, rot0=jrot, rot1=jrot,
                          friction=friction)
        if mimic is not None:
            ref, gearing = mimic
            j.AddAppliedSchema("PhysxMimicJointAPI:rotZ")
            j.CreateRelationship("physxMimicJoint:rotZ:referenceJoint").SetTargets([Sdf.Path(f"{self.group}/J_{ref}")])
            j.CreateAttribute("physxMimicJoint:rotZ:referenceJointAxis", T.Token, False).Set("rotZ")
            j.CreateAttribute("physxMimicJoint:rotZ:gearing", T.Float, False).Set(float(gearing))
            j.CreateAttribute("physxMimicJoint:rotZ:offset", T.Float, False).Set(0.0)
        self.links[name] = dict(path=path, pos=wpos, lpos=np.asarray(lpos, float), mass=mass)
        ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis]
        axis_w = qmat(self.q) @ qmat(jrot) @ np.asarray(ax, float)
        rec = dict(name=f"{self.name}/J_{name}", path=jp, art=self.name, art_root=self.links["root"]["path"], link=path,
                   parent=P["path"], type=jtype, axis_local=axis, axis_world=[round(float(c), 4) for c in axis_w],
                   lower=lower, upper=upper, unit=unit or ("deg" if jtype == "revolute" else "m"),
                   drive=dict(stiffness=stiffness, damping=damping, max_force=max_force, target=target),
                   friction=friction, mimic=None if mimic is None else dict(reference=f"{self.group}/J_{mimic[0]}", gearing_per_deg=mimic[1]),
                   role=role or name, origin_world=[round(float(c), 4) for c in wpos], mass=mass)
        if meta:
            rec.update(meta)
        ctx.joints.append(rec)
        self.joint_names.append(rec["name"])
        return rec
