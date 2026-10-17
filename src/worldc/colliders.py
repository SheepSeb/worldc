"""Extract world-space oriented-box approximations of every collider in a stage (works with any pxr)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .geom import OBB

NAMESPACES = ("scene", "warehouse", "plant")  # attribute namespaces written by Kit.tag (current + legacy)


@dataclass
class Collider:
    path: str
    body: str | None  # owning rigid body prim path (None => static)
    kinematic: bool
    obb: OBB
    shape: str
    category: str
    is_mesh: bool = False
    group: str = ""  # articulation id (<ns>:art); colliders of one group never collide with each other


def _owner(prim):
    p = prim
    while p and p.GetPath().pathString != "/":
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


def _obb_for(prim, xf: Gf.Matrix4d):
    t = prim.GetTypeName()
    M = np.array(xf).T  # Gf is row-vector convention -> column-vector matrix
    ax = M[:3, :3]
    scale = np.linalg.norm(ax, axis=0)
    R = ax / scale
    c = M[:3, 3]
    if t == "Cube":
        s = UsdGeom.Cube(prim).GetSizeAttr().Get()
        half = 0.5 * s * scale
    elif t == "Cylinder":
        cy = UsdGeom.Cylinder(prim)
        r, h = cy.GetRadiusAttr().Get(), cy.GetHeightAttr().Get()
        half = np.array([r, r, h / 2]) * scale
    elif t == "Capsule":
        cp = UsdGeom.Capsule(prim)
        r, h = cp.GetRadiusAttr().Get(), cp.GetHeightAttr().Get()
        half = np.array([r, r, h / 2 + r]) * scale
    elif t == "Cone":
        cn = UsdGeom.Cone(prim)
        r, h = cn.GetRadiusAttr().Get(), cn.GetHeightAttr().Get()
        half = np.array([r, r, h / 2]) * scale
    elif t == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        half = np.array([r, r, r]) * scale
    elif t == "Mesh":
        pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        lo, hi = pts.min(0), pts.max(0)
        half = (hi - lo) / 2 * scale
        c = c + R @ ((hi + lo) / 2 * scale)
    else:
        return None
    return OBB(c, half, R)


def collect(stage: Usd.Stage, only_enabled=True) -> list[Collider]:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if only_enabled:
            a = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
            if a and a.HasAuthoredValue() and not a.Get():
                continue
        obb = _obb_for(prim, cache.GetLocalToWorldTransform(prim))
        if obb is None:
            continue
        own = _owner(prim)
        kin = False
        if own is not None:
            ka = UsdPhysics.RigidBodyAPI(own).GetKinematicEnabledAttr()
            kin = bool(ka.Get()) if ka else False
        cat, grp = "", ""
        p = own or prim
        for ns in NAMESPACES:
            a = p.GetAttribute(f"{ns}:category")
            if a and a.HasValue():
                cat = a.Get()
            a = p.GetAttribute(f"{ns}:art")
            if a and a.HasValue():
                grp = a.Get()
        out.append(Collider(prim.GetPath().pathString, own.GetPath().pathString if own else None, kin, obb,
                            prim.GetTypeName(), cat, prim.GetTypeName() == "Mesh", grp))
    return out


_SHRINK = {"Cylinder": np.array([0.85, 0.85, 1.0]), "Capsule": np.array([0.85, 0.85, 1.0]), "Sphere": np.array([0.85, 0.85, 0.85])}


def shrunk(c: Collider) -> OBB:
    """Oriented box of a collider with round shapes shrunk towards their inscribed box (the bounding box of a
    cylinder/capsule/sphere over-reports penetration)."""
    return OBB(c.obb.c, c.obb.h * _SHRINK.get(c.shape, 1.0), c.obb.R)
