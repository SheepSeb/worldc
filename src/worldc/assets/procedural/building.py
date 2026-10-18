"""Building assets: rectangular warehouse shell (slab collider, textured floor, apron, walls with openings, roof with
trusses) and slope zones (raised floor crowns).  Ported from the former ``Structures.shell`` / ``_crown`` / ``wall``.

Local frame of the shell: origin at the floor centre, +x east, +y north, z = 0 floor top.  The instance root is an Xform
carrying the pose; parts are authored in local coordinates under it.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pxr import UsdPhysics
from pydantic import BaseModel, ConfigDict, Field

from ...geom import Q_IDENT, qyaw
from ..base import Anchor, Asset, Built, Footprint, Keepout, Params, PhysicsInfo
from ..registry import register

WALLS = ("south", "north", "east", "west")


class Opening(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(description="Opening id; exposed as port 'opening_<id>'")
    wall: Literal["south", "north", "east", "west"]
    center: float = Field(description="Coordinate along the wall axis (x for south/north walls, y for east/west), local frame")
    width: float = Field(gt=0.3, le=20)
    height: float = Field(gt=0.5, le=20, description="Top edge of the opening above the floor")
    sill: float = Field(0.0, ge=0, le=3, description="Bottom edge of the opening above the floor (windows)")


class ShellParams(Params):
    size: tuple[float, float] = Field((40.0, 25.0), description="Interior floor size x, y (m)")
    height: float = Field(8.0, ge=2.5, le=40, description="Wall / ceiling height (m)")
    wall_thickness: float = Field(0.3, ge=0.05, le=1.0)
    openings: list[Opening] = Field(default_factory=list, description="Door / dock openings cut into the walls")
    roof: bool = Field(True, description="Roof slab (collider) + visual trusses")
    truss_spacing: float = Field(5.0, ge=1.0, le=20)
    apron_depth: float = Field(6.5, ge=0, le=30, description="Outdoor apron (slab + visual) depth outside the south wall")
    floor_texture: str | None = Field(None, description="Stage-relative texture path for the floor (set by the compiler)")
    floor_material: str = Field("apron", description="Visual material preset when no floor_texture")
    wall_material: str = "wall"
    roof_material: str = "roof"


def _wall_geom(name, size, T_):
    """(p0, p1, inward normal) of a wall's interior line, matching the pre-refactor orientation conventions."""
    X0, X1, Y0, Y1 = -size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2
    return {
        "south": ((X0, Y0), (X1, Y0), (0, 1)),
        "north": ((X1, Y1), (X0, Y1), (0, -1)),
        "east": ((X1, Y0), (X1, Y1), (-1, 0)),
        "west": ((X0, Y1), (X0, Y0), (1, 0)),
    }[name]


def _along(name, size, center):
    """Distance from the wall's p0 to ``center`` (coordinate along the wall axis)."""
    X0, X1, Y0, Y1 = -size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2
    return {"south": center - X0, "north": X1 - center, "east": center - Y0, "west": Y1 - center}[name]


def opening_frame(o: Opening, size) -> tuple[tuple, float]:
    """Local (x, y) on the wall's interior line + yaw whose local +y points into the building."""
    X0, X1, Y0, Y1 = -size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2
    p0, p1, n = _wall_geom(o.wall, size, 0)
    if o.wall in ("south", "north"):
        xy = (o.center, p0[1])
    else:
        xy = (p0[0], o.center)
    return xy, math.atan2(n[1], n[0]) - math.pi / 2


@register
class BuildingShell(Asset):
    id = "building_shell"
    description = ("Rectangular indoor building: concrete slab collider, textured floor quad, walls with door/dock openings, "
                   "roof slab + trusses, optional outdoor apron on the south side. Openings are exposed as ports "
                   "'opening_<id>' (local +y of the port points into the building) for doors to attach to.")
    tags = frozenset({"building", "structure", "static", "indoor", "walls"})
    Params = ShellParams
    physics = PhysicsInfo("static", "concrete", 0.7, None, "primitives")

    def footprint(self, p):
        T_ = p.wall_thickness
        return Footprint.rect(p.size[0] + 2 * T_, p.size[1] + 2 * T_, p.height + 0.3)

    def anchors(self, p):
        out = [Anchor("floor", "support", (0, 0, 0), polygon=[(-p.size[0] / 2, -p.size[1] / 2), (p.size[0] / 2, -p.size[1] / 2),
                                                               (p.size[0] / 2, p.size[1] / 2), (-p.size[0] / 2, p.size[1] / 2)])]
        for o in p.openings:
            xy, yaw = opening_frame(o, p.size)
            out.append(Anchor(f"opening_{o.id}", "port", (xy[0], xy[1], 0.0), qyaw(yaw), port_kind="opening",
                              meta={"wall": o.wall, "width": o.width, "height": o.height, "sill": o.sill, "center": o.center}))
        return out

    def build(self, ctx, path, pose, p, rng):
        k, m = ctx.kit, ctx.mats
        W, D = p.size
        T_ = p.wall_thickness
        X0, X1, Y0, Y1 = -W / 2, W / 2, -D / 2, D / 2
        k.xform(path, pose.pos, pose.quat)

        def rec(sub, cat, lp, q, **kw):
            wp, wq = pose.world(lp, q)
            ctx.record(f"{path}/{sub}", cat, wp, wq, "static", **kw)

        # collider slab (top at z=0) covering building + apron; render geometry is separate
        ys0, ys1 = Y0 - T_ - p.apron_depth, Y1 + T_
        slab_size = (W + 2 * T_, ys1 - ys0, 0.5)
        slab_c = (0, (ys0 + ys1) / 2, -0.25)
        slab = k.cube(f"{path}/FloorSlab", slab_size, slab_c, Q_IDENT, None, m.phys("concrete"), collide=True, guide=True)
        k.tag(slab, "floor")
        rec("FloorSlab", "floor", slab_c, Q_IDENT, size=slab_size, friction=0.7)
        fmat = m.textured("floor", p.floor_texture, rough=0.55) if p.floor_texture else m.get(p.floor_material)
        pts = [(X0, Y0, 0), (X1, Y0, 0), (X1, Y1, 0), (X0, Y1, 0)]
        k.mesh(f"{path}/FloorVisual", pts, [4], [0, 1, 2, 3], uvs=[(0, 0), (1, 0), (1, 1), (0, 1)], vmat=fmat)
        if p.apron_depth > 0:
            ap = [(X0 - T_, ys0, 0), (X1 + T_, ys0, 0), (X1 + T_, Y0 - T_, 0), (X0 - T_, Y0 - T_, 0)]
            k.mesh(f"{path}/ApronVisual", ap, [4], [0, 1, 2, 3], uvs=[(0, 0), (1, 0), (1, 1), (0, 1)], vmat=m.get("apron"))
        # walls
        k.scope(f"{path}/Walls")
        for name in WALLS:
            ops = [(_along(o.wall, p.size, o.center), o.width, o.height, o.sill) for o in p.openings if o.wall == name]
            self._wall(ctx, path, name, p, ops, rec)
        # roof slab (hidden for overhead renders) + visual trusses
        if p.roof:
            k.scope(f"{path}/Roof")
            roof = k.cube(f"{path}/Roof/Slab", (W + 2 * T_, D + 2 * T_, 0.3), (0, 0, p.height + 0.15), Q_IDENT,
                          m.get(p.roof_material), m.phys("steel"), collide=True)
            k.tag(roof, "roof")
            rec("Roof/Slab", "building", (0, 0, p.height + 0.15), Q_IDENT, size=(W + 2 * T_, D + 2 * T_, 0.3), friction=0.6, subtype="roof")
            for i, x in enumerate(np.arange(X0 + p.truss_spacing / 2, X1, p.truss_spacing) if p.height >= 4.0 else []):  # no trusses under a domestic ceiling
                k.cube(f"{path}/Roof/Truss_{i}", (0.25, D, 0.45), (x, 0, p.height - 0.225), Q_IDENT, m.get("truss"))
        anchors = [a.to_world(pose) for a in self.anchors(p)]
        return Built(path, anchors, info={"interior_bbox_local": [X0, Y0, X1, Y1], "roof_path": f"{path}/Roof" if p.roof else None})

    def _wall(self, ctx, path, name, p, openings, rec):
        """Wall along p0->p1 (the interior line); thickness extends outward.  openings = (s, width, height)."""
        k = ctx.kit
        T_ = p.wall_thickness
        p0, p1, n_in = _wall_geom(name, p.size, T_)
        t = np.array([p1[0] - p0[0], p1[1] - p0[1]], float)
        L = float(np.linalg.norm(t))
        t /= L
        yaw = math.atan2(t[1], t[0])
        base = np.array(p0, float) - np.array(n_in, float) * (T_ / 2)
        cur, pieces = -T_, []
        for s, w, h, sill in sorted(openings):
            pieces.append((cur, s - w / 2, 0.0, p.height))
            pieces.append((s - w / 2, s + w / 2, h, p.height))
            if sill > 0:
                pieces.append((s - w / 2, s + w / 2, 0.0, sill))
            cur = s + w / 2
        pieces.append((cur, L + T_, 0.0, p.height))
        for j, (s0, s1, z0, z1) in enumerate(pieces):
            if s1 - s0 < 1e-3 or z1 - z0 < 1e-3:
                continue
            ctr = base + t * ((s0 + s1) / 2)
            pos = (ctr[0], ctr[1], (z0 + z1) / 2)
            size = (s1 - s0, T_, z1 - z0)
            sub = f"Walls/{name.capitalize()}_{j}"
            prim = k.cube(f"{path}/{sub}", size, pos, qyaw(yaw), ctx.mats.get(p.wall_material), ctx.mats.phys("concrete"), collide=True)
            k.tag(prim, "wall")
            rec(sub, "building", pos, qyaw(yaw), size=size, friction=0.7, subtype="wall")


class SlopeParams(Params):
    size: tuple[float, float] = Field((8.5, 2.6), description="Zone size x, y (m)")
    slope: float = Field(0.025, ge=0.0, le=0.1, description="Ramp gradient (rise / run)")
    max_height: float = Field(0.04, ge=0.0, le=0.3, description="Crown peak height (m)")
    floor_texture: str | None = Field(None, description="Stage-relative floor texture (UVs continue the building floor)")


@register
class SlopeZone(Asset):
    id = "slope_zone"
    description = ("Raised floor crown: ramps up at `slope` from every edge to a flat top of `max_height` (static triangle-mesh "
                   "collider). Use for uneven-floor / ramp robustness tests.")
    tags = frozenset({"floor", "terrain", "ramp", "static"})
    Params = SlopeParams
    physics = PhysicsInfo("static", "concrete", 0.7, None, "triangle_mesh_static")

    def footprint(self, p):
        return Footprint.rect(p.size[0], p.size[1], p.max_height)

    def keepouts(self, p):
        return [Keepout(Footprint.rect(p.size[0] + 0.6, p.size[1] + 0.6, 0.1).polygon, "slope", (0.0, p.max_height))]

    @staticmethod
    def height_fn(cx, cy, p):
        x0, x1, y0, y1 = cx - p.size[0] / 2, cx + p.size[0] / 2, cy - p.size[1] / 2, cy + p.size[1] / 2

        def H(x, y):
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                return 0.0
            return min(p.max_height, p.slope * min(x - x0, x1 - x, y - y0, y1 - y))

        return H

    def build(self, ctx, path, pose, p, rng):
        k = ctx.kit
        cx, cy = pose.pos[0], pose.pos[1]
        x0, x1, y0, y1 = cx - p.size[0] / 2, cx + p.size[0] / 2, cy - p.size[1] / 2, cy + p.size[1] / 2
        s, mh = max(p.slope, 1e-6), p.max_height
        r = mh / s

        def lines(a, b):
            r2 = min(r, (b - a) / 2)
            return sorted({a, a + r2, b - r2, b})

        xs, ys = lines(x0, x1), lines(y0, y1)
        H = self.height_fn(cx, cy, p)
        (ex0, ex1), (ey0, ey1) = ctx.env.x_range, ctx.env.y_range
        pts, uvs = [], []
        for y in ys:
            for x in xs:
                pts.append((x, y, H(x, y) + 0.0005))
                uvs.append(((x - ex0) / (ex1 - ex0), (y - ey0) / (ey1 - ey0)))
        nx = len(xs)
        counts, idx = [], []
        for j in range(len(ys) - 1):
            for i2 in range(nx - 1):
                a, b, c2, d = j * nx + i2, j * nx + i2 + 1, (j + 1) * nx + i2 + 1, (j + 1) * nx + i2
                counts += [3, 3]
                idx += [a, b, c2, a, c2, d]
        vmat = ctx.mats.textured("floor", p.floor_texture, rough=0.55) if p.floor_texture else ctx.mats.get("apron")
        prim = k.mesh(path, pts, counts, idx, uvs=uvs, vmat=vmat)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")  # static triangle mesh
        k.bind(prim, ctx.mats.phys("concrete"), "physics")
        peak = max(q[2] for q in pts)
        ctx.record(path, "floor", (x0, y0, 0), Q_IDENT, "static", size=(x1 - x0, y1 - y0, peak), friction=0.7,
                   subtype="slope_zone", gradient=round(p.slope, 4), peak_height_m=round(peak, 4), bbox=[x0, y0, x1, y1])
        return Built(path, info={"bbox": [x0, y0, x1, y1], "peak": peak})
