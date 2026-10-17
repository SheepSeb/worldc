"""Shape lists and fixed-base articulated assemblies (generalised from the former plant ``common.py``).

``Sh`` is a primitive with a material preset name; ``emit`` turns a list of them into prims (+ inertia ``Part``s for the
colliding ones).  ``Art`` builds a fixed-base articulation: a root link welded to the world plus child links joined by
revolute / prismatic joints (optionally PhysX-mimic coupled).  Joints are recorded in ``ctx.scene["joints"]`` (detailed)
and ``ctx.scene["articulations"]`` (the per-joint sweep list the Isaac validator consumes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pxr import Sdf, UsdPhysics

from ...geom import Part, Q_IDENT, qaxis, qmat, qyaw, yaw_of
from ...usdkit import T, _q, _v3
from ..base import BuildContext

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
    kind: str  # cube (sx,sy,sz) | cyl (r,h) | cap (r,h) | sph (r,) | cone (r,h) | esph (rx,ry,rz) visual-only ellipsoid
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = Q_IDENT
    mat: str = "steel"
    col: bool = True
    name: str = ""


def cube(size, pos=(0, 0, 0), mat="steel", col=True, quat=Q_IDENT, name=""):
    return Sh("cube", tuple(size), tuple(pos), quat, mat, col, name)


def cyl(r, h, pos=(0, 0, 0), mat="steel", col=True, axis="z", quat=None, name=""):
    q = quat if quat is not None else {"z": Q_IDENT, "x": Q_X, "y": Q_Y}[axis]
    return Sh("cyl", (r, h), tuple(pos), q, mat, col, name)


def cap(r, h, pos=(0, 0, 0), mat="steel", col=True, quat=Q_IDENT, name=""):
    return Sh("cap", (r, h), tuple(pos), quat, mat, col, name)


def sph(r, pos=(0, 0, 0), mat="steel", col=True, name=""):
    return Sh("sph", (r,), tuple(pos), Q_IDENT, mat, col, name)


def cone(r, h, pos=(0, 0, 0), mat="steel", col=False, quat=Q_IDENT, name=""):
    return Sh("cone", (r, h), tuple(pos), quat, mat, col, name)


def esph(rx, ry, rz, pos=(0, 0, 0), mat="steel", name=""):
    return Sh("esph", (rx, ry, rz), tuple(pos), Q_IDENT, mat, False, name)


def shape_part(s: Sh) -> Part | None:
    """Inertia proxy of a colliding shape (None for visual-only shapes)."""
    if not s.col:
        return None
    if s.kind == "cube":
        return Part("box", s.size, s.pos, s.quat)
    if s.kind in ("cyl", "cone"):
        return Part("cyl", s.size, s.pos, s.quat)
    if s.kind == "cap":
        return Part("cyl", (s.size[0], s.size[1] + 2 * s.size[0]), s.pos, s.quat)
    if s.kind == "sph":
        return Part("sph", s.size, s.pos, s.quat)
    return None


def emit(ctx: BuildContext, root: str, shapes: list[Sh], pmat: str = "steel") -> list[Part]:
    """Create prims for ``shapes`` under ``root``; returns the inertia Parts of the colliding ones."""
    from pxr import UsdPhysics

    k = ctx.kit
    parts = []
    for i, s in enumerate(shapes):
        p = f"{root}/{s.name or f'{s.kind}{i}'}"
        vm = ctx.mats.get(s.mat)
        pm = ctx.mats.phys(pmat) if s.col else None
        if s.kind == "cube":
            k.cube(p, s.size, s.pos, s.quat, vmat=vm, pmat=pm, collide=s.col)
        elif s.kind == "cyl":
            k.cylinder(p, s.size[0], s.size[1], s.pos, s.quat, vmat=vm, pmat=pm, collide=s.col)
        elif s.kind == "cap":
            k.capsule(p, s.size[0], s.size[1], s.pos, s.quat, vmat=vm, pmat=pm, collide=s.col)
        elif s.kind == "sph":
            k.sphere(p, s.size[0], s.pos, vmat=vm, pmat=pm, collide=s.col)
        elif s.kind == "cone":
            prim = k.cone(p, s.size[0], s.size[1], s.pos, s.quat, vmat=vm)
            if s.col:
                UsdPhysics.CollisionAPI.Apply(prim)
                k.bind(prim, pm, "physics")
        elif s.kind == "esph":
            prim = k.sphere(p, 1.0, s.pos, vmat=vm)
            k.set_pose(prim, s.pos, s.quat, s.size)
        else:
            raise ValueError(f"unknown shape kind {s.kind!r}")
        part = shape_part(s)
        if part is not None:
            parts.append(part)
        ctx.scene["stats"]["n_shapes"] = ctx.scene["stats"].get("n_shapes", 0) + 1
    return parts


def static(ctx: BuildContext, path, pos, quat, shapes, pmat="steel", category=None, **tag):
    x = ctx.kit.xform(path, pos, quat)
    emit(ctx, path, shapes, pmat)
    if category:
        ctx.kit.tag(x, category, **tag)
    return x


def body(ctx: BuildContext, path, pos, quat, shapes, mass, pmat="steel", **rb):
    """Rigid body whose inertia comes from the colliding shapes; prims are authored under the body prim."""
    parts = [p for p in (shape_part(s) for s in shapes) if p is not None]
    prim = ctx.kit.rigid_body(path, pos, quat, parts=parts, mass=mass, **rb)
    emit(ctx, path, shapes, pmat)
    return prim


class Art:
    """A fixed-base articulation rooted at ``group/root`` and welded to the world."""

    def __init__(self, ctx: BuildContext, group: str, pos, yaw, shapes, mass, category, pmat="steel", quat=None, **meta):
        self.ctx, self.category = ctx, category
        self.group = group
        self.name = Sdf.Path(group).name
        ctx.kit.xform(group)
        self.q = tuple(quat) if quat is not None else qyaw(yaw)
        self.pos = np.asarray(pos, float)
        self.yaw = yaw if quat is None else yaw_of(self.q)
        path = f"{group}/root"
        prim = body(ctx, path, self.pos, self.q, shapes, mass, pmat, disable_gravity=True, iters=(16, 2))
        ns = ctx.kit.ns
        ctx.kit.attr(prim, f"{ns}:art", T.String, self.name)
        ctx.kit.attr(prim, f"{ns}:role", T.String, "root")
        ctx.kit.tag(prim, category, **meta)
        fj = UsdPhysics.FixedJoint.Define(ctx.kit.stage, f"{group}/J_world")
        fj.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        fj.CreateLocalPos0Attr(_v3(self.pos))
        fj.CreateLocalRot0Attr(_q(self.q))
        fj.CreateLocalPos1Attr(_v3((0, 0, 0)))
        fj.CreateLocalRot1Attr(_q(Q_IDENT))
        ctx.kit.articulation_root(prim)
        self.links = {"root": dict(path=path, pos=self.pos, mass=mass, shapes=shapes)}
        self.pmat = pmat
        self.joint_names: list[str] = []

    @property
    def mass(self) -> float:
        return float(sum(l["mass"] for l in self.links.values()))

    def world(self, local):
        """World position of a point expressed in the assembly frame (all links share the root orientation)."""
        return self.pos + qmat(self.q) @ np.asarray(local, float)

    def link(self, name, lpos, shapes, mass, jtype, axis="Z", lower=0.0, upper=1.0, parent="root", jrot=Q_IDENT,
             stiffness=0.0, damping=0.0, max_force=1e6, target=0.0, friction=0.0, unit=None, mimic=None, pmat=None,
             role="", meta=None):
        """Add a child link whose origin sits at ``lpos`` (assembly frame) and is joined to ``parent``."""
        ctx = self.ctx
        P = self.links[parent]
        off = np.asarray(lpos, float) - (P["lpos"] if "lpos" in P else np.zeros(3))
        wpos = self.world(lpos)
        path = f"{self.group}/{name}"
        prim = body(ctx, path, wpos, self.q, shapes, mass, pmat or self.pmat, iters=(16, 2), angular_damping=0.5)
        ns = ctx.kit.ns
        ctx.kit.attr(prim, f"{ns}:art", T.String, self.name)
        ctx.kit.attr(prim, f"{ns}:role", T.String, role or name)
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
        self.links[name] = dict(path=path, pos=wpos, lpos=np.asarray(lpos, float), mass=mass, shapes=shapes)
        ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis]
        axis_w = qmat(self.q) @ qmat(jrot) @ np.asarray(ax, float)
        unit = unit or ("deg" if jtype == "revolute" else "m")
        rec = dict(name=f"{self.name}/J_{name}", path=jp, art=self.name, art_root=self.links["root"]["path"], link=path,
                   parent=P["path"], type=jtype, axis_local=axis, axis_world=[round(float(c), 4) for c in axis_w],
                   lower=lower, upper=upper, unit=unit,
                   drive=dict(stiffness=stiffness, damping=damping, max_force=max_force, target=target),
                   friction=friction, mimic=None if mimic is None else dict(reference=f"{self.group}/J_{mimic[0]}", gearing_per_deg=mimic[1]),
                   role=role or name, origin_world=[round(float(c), 4) for c in wpos], mass=mass)
        if meta:
            rec.update(meta)
        ctx.scene["joints"].append(rec)
        if mimic is None:  # mimic-driven joints are swept through their reference joint
            ctx.scene["articulations"].append(dict(
                name=f"{self.name}_{name}", root=self.links["root"]["path"], link=path, joint=jp, kind=jtype,
                lower=lower, upper=upper, unit=unit, parent=self.group,
                moving_collider_paths=[f"{path}/{s.name or f'{s.kind}{i}'}" for i, s in enumerate(shapes) if s.col]))
        self.joint_names.append(rec["name"])
        return rec
