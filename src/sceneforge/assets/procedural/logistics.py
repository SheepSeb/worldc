"""Logistics assets: EUR pallet, cardboard box, loaded pallet (boxes / stretch-wrapped / totes / empty), tote with items,
tipped pallet jack.  Ported from the former ``objects.Factory`` with behaviour preserved.

Every dynamic body is an ``Xform`` rigid body whose children are analytic colliders (convex compound shapes only, no
triangle meshes on dynamic bodies).  Mass and the full inertia tensor are authored explicitly from the compound shape.

Convention: ``pose.pos`` z is the height of the supporting surface; assets add their own small settle gap.
"""
from __future__ import annotations

import itertools
import math
from typing import Literal

import numpy as np
from pydantic import Field

from ...geom import Part, Q_IDENT, composite_inertia, qfrom_mat, qmat, qmul, qyaw, rect_overlap_2d
from ..base import Anchor, Asset, BuildContext, Built, Footprint, Params, PhysicsInfo, Pose
from ..registry import register

PAL_L, PAL_W, PAL_H = 1.2, 0.8, 0.144
PAL_EPS = 0.0006  # pallet settle gap above its support
BOX_EPS = 0.0008  # gap between stacked layers
BOX_MASS_RANGE = (0.2, 15.0)
JIT_XY, JIT_YAW = 0.03, math.radians(10.0)
N_CARDBOARD = 5
TOTE = (0.6, 0.4, 0.28)
TWALL = 0.012

Vec3 = tuple[float, float, float]


def pallet_parts():
    parts = [Part("box", (1.2, 0.8, 0.022), (0, 0, 0.133))]
    for x in (-0.5275, 0.0, 0.5275):
        for y in (-0.3275, 0.0, 0.3275):
            parts.append(Part("box", (0.145, 0.145, 0.1), (x, y, 0.072)))
    for y in (-0.3275, 0.0, 0.3275):
        parts.append(Part("box", (1.2, 0.1, 0.022), (0, y, 0.011)))
    return parts


def _rect(sx, sy):
    return [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]


# --------------------------------------------------------------------------- low-level builders (shared by assets)
def build_pallet(ctx: BuildContext, path, pos, yaw, rng, mass=None) -> dict:
    k = ctx.kit
    q = qyaw(yaw)
    mass = float(rng.uniform(20.0, 25.0)) if mass is None else float(mass)
    tint = rng.uniform(0.85, 1.12)
    mat = ctx.mats.get("wood" if tint < 1.0 else "wood_light")
    parts = pallet_parts()
    body = k.rigid_body(path, pos, q, parts=parts, mass=mass)
    for j, p in enumerate(parts):
        k.cube(f"{path}/c{j:02d}", p.size, p.pos, Q_IDENT, mat, ctx.mats.phys("wood"), collide=True)
    k.tag(body, "pallet", zone=ctx.zone or "")
    ctx.record(path, "pallet", pos, q, "dynamic", size=(PAL_L, PAL_W, PAL_H), mass=mass, friction=0.5,
               standard="EUR1", n_collision_shapes=len(parts))
    return dict(path=path, pos=tuple(pos), yaw=yaw, deck_top=pos[2] + PAL_H, load_top=pos[2] + PAL_H, n_boxes=0, kind="empty")


def box_mass(rng, vol):
    return float(np.clip(vol * rng.uniform(60.0, 220.0), *BOX_MASS_RANGE))


def build_box(ctx: BuildContext, path, pos, quat, size, variant, rng, mass=None, **extra):
    """One cardboard carton: rigid body + box collider (guide) + textured visual mesh with per-face UVs."""
    k = ctx.kit
    mass = box_mass(rng, size[0] * size[1] * size[2]) if mass is None else float(mass)
    body = k.rigid_body(path, pos, quat, parts=[Part("box", tuple(size))], mass=mass)
    k.cube(f"{path}/Collision", size, (0, 0, 0), Q_IDENT, None, ctx.mats.phys("cardboard"), collide=True, guide=True)
    v = variant % N_CARDBOARD
    mat = ctx.mats.textured(f"cardboard_{v}", ctx.textures.cardboard(v), rough=0.9)
    k.mesh_box(f"{path}/Visual", size, vmat=mat)
    k.tag(body, "box", zone=ctx.zone or "")
    ctx.record(path, "box", pos, quat, "dynamic", size=size, mass=mass, friction=0.5, **extra)
    return path, mass


def load_boxes(ctx: BuildContext, root, pal: dict, dims, max_h, rng, full=False, max_boxes=18, min_layers=1):
    """Stack cardboard layers on a pallet with random <=3 cm / <=10 deg misalignment per box."""
    a, b, c = dims
    if rng.random() < 0.5:
        a, b = b, a
    gap = 0.03
    nx = max(1, int((PAL_L + 0.04) // (a + gap)))
    ny = max(1, int((PAL_W + 0.04) // (b + gap)))
    while nx * ny > max_boxes and (nx > 1 or ny > 1):
        if nx >= ny and nx > 1:
            nx -= 1
        else:
            ny -= 1
    per_layer = nx * ny
    cap = int(np.clip(int(max_h // (c + 0.002)), 1, 6))
    cap = max(1, min(cap, max(1, max_boxes // per_layer)))
    layers = cap if full else int(rng.integers(min_layers if min_layers <= cap else 1, cap + 1))
    pitch_x = max(PAL_L / nx, a + gap)
    pitch_y = max(PAL_W / ny, b + gap)
    px = (np.arange(nx) - (nx - 1) / 2) * pitch_x
    py = (np.arange(ny) - (ny - 1) / 2) * pitch_y
    variant = int(rng.integers(N_CARDBOARD))
    q0 = qyaw(pal["yaw"])
    R0 = qmat(q0)
    n = 0
    top = pal["deck_top"]
    dropped: set = set()
    for layer in range(layers):
        placed: list[tuple] = []
        cells = [(x, y) for x in px for y in py if (x, y) not in dropped]
        if layer == layers - 1 and layers > 1 and not full:
            keep = max(1, int(round(len(cells) * rng.uniform(0.3, 1.0))))
            cells = [cells[i] for i in rng.permutation(len(cells))[:keep]]
        zc = pal["deck_top"] + layer * c + c / 2 + BOX_EPS * (layer + 1)
        for (cx, cy) in cells:
            best = None
            for attempt in range(12):
                s = 1.0 if attempt < 6 else (11 - attempt) / 6.0  # shrink jitter, last attempt = none
                rad, ang = JIT_XY * math.sqrt(rng.random()) * s, rng.uniform(0, math.tau)
                dx, dy = rad * math.cos(ang), rad * math.sin(ang)
                th = rng.uniform(-JIT_YAW, JIT_YAW) * s
                x, y = cx + dx, cy + dy
                ex = abs(math.cos(th)) * a / 2 + abs(math.sin(th)) * b / 2
                ey = abs(math.sin(th)) * a / 2 + abs(math.cos(th)) * b / 2
                if abs(x) + ex > PAL_L / 2 + 0.03 or abs(y) + ey > PAL_W / 2 + 0.03:
                    continue
                if any(rect_overlap_2d((x, y), (a / 2, b / 2), th, p[:2], (a / 2, b / 2), p[2], 0.003) for p in placed):
                    continue
                best = (x, y, th)
                break
            if best is None:
                dropped.add((cx, cy))
                continue
            placed.append(best)
            x, y, th = best
            wp = np.asarray(pal["pos"]) + R0 @ np.array([x, y, 0.0])
            wp[2] = zc
            mis_xy, mis_yaw = 100 * math.hypot(x - cx, y - cy), math.degrees(th)
            build_box(ctx, ctx.alloc(root, "Box", 3), tuple(wp), qmul(q0, qyaw(th)), (a, b, c), variant, rng,
                      parent=pal["path"], layer=layer, misalign_xy_cm=round(mis_xy, 2), misalign_yaw_deg=round(mis_yaw, 2))
            ctx.stat("misalign_xy_cm", mis_xy)
            ctx.stat("misalign_yaw_deg", abs(mis_yaw))
            n += 1
        top = pal["deck_top"] + (layer + 1) * c + BOX_EPS * (layer + 1)
    pal.update(load_top=top, n_boxes=pal["n_boxes"] + n, kind="boxes")
    return pal


def build_wrapped(ctx: BuildContext, path, pos, yaw, dims, max_h, rng, layers=None) -> dict:
    """Stretch-wrapped pallet = ONE rigid body: pallet shapes + consolidated load block."""
    k = ctx.kit
    a, b, c = dims
    n_layers = layers or int(np.clip(int(max_h // c), 1, 5))
    h = n_layers * c
    L, Wd = PAL_L - 0.02, PAL_W - 0.02
    q = qyaw(yaw)
    parts = pallet_parts()
    nbox = max(1, int(round(L / (a + 0.01))) * int(round(Wd / (b + 0.01))) * n_layers)
    load_mass = sum(box_mass(rng, a * b * c) for _ in range(min(nbox, 24))) * nbox / min(nbox, 24)
    pallet_mass = float(rng.uniform(20.0, 25.0))
    load_part = Part("box", (L, Wd, h), (0, 0, PAL_H + h / 2 + BOX_EPS), Q_IDENT, mass=load_mass)
    total = pallet_mass + load_mass
    vol = sum(p.volume() for p in parts)
    for p in parts:
        p.mass = pallet_mass * p.volume() / vol
    parts.append(load_part)
    body = k.rigid_body(path, pos, q, parts=parts, mass=total)
    for j, p in enumerate(parts[:-1]):
        k.cube(f"{path}/c{j:02d}", p.size, p.pos, Q_IDENT, ctx.mats.get("wood"), ctx.mats.phys("wood"), collide=True)
    lp = load_part.pos
    k.cube(f"{path}/Load", load_part.size, lp, Q_IDENT, None, ctx.mats.phys("cardboard"), collide=True, guide=True)
    wrap_sz = (L + 0.016, Wd + 0.016, h + 0.012)
    k.mesh_box(f"{path}/Wrap", wrap_sz, (lp[0], lp[1], lp[2] + 0.004), vmat=ctx.mats.get("wrap"), uv=False)
    k.tag(body, "pallet_wrapped", zone=ctx.zone or "")
    ctx.record(path, "pallet", pos, q, "dynamic", size=(PAL_L, PAL_W, PAL_H + h), mass=total, friction=0.5,
               standard="EUR1", wrapped=True, equivalent_boxes=nbox, n_collision_shapes=len(parts))
    return dict(path=path, pos=tuple(pos), yaw=yaw, deck_top=pos[2] + PAL_H, load_top=pos[2] + PAL_H + h + BOX_EPS,
                n_boxes=nbox, kind="wrapped")


def build_tote(ctx: BuildContext, root, pos, yaw, rng, n_items=None, parent=None, color=None) -> str:
    """Open plastic tote (5 wall/base shapes, one body) + n rigid items on a jittered grid on its floor.
    Tote at ``root/Tote``, items at ``root/Item_NN`` (siblings: rigid bodies are never nested)."""
    k = ctx.kit
    L, Wd, Hh = TOTE
    t = TWALL
    i = k.next_id("tote")
    path = f"{root}/Tote"
    k.xform(root)
    q = qyaw(yaw)
    mass = float(rng.uniform(1.4, 1.9))
    col = color or ("tote_blue", "tote_grey", "tote_red")[i % 3]
    parts = [
        Part("box", (L, Wd, t), (0, 0, t / 2)),
        Part("box", (L, t, Hh - t), (0, (Wd - t) / 2, t + (Hh - t) / 2)),
        Part("box", (L, t, Hh - t), (0, -(Wd - t) / 2, t + (Hh - t) / 2)),
        Part("box", (t, Wd - 2 * t, Hh - t), ((L - t) / 2, 0, t + (Hh - t) / 2)),
        Part("box", (t, Wd - 2 * t, Hh - t), (-(L - t) / 2, 0, t + (Hh - t) / 2)),
    ]
    body = k.rigid_body(path, pos, q, parts=parts, mass=mass)
    for nm, p in zip(("Base", "WallN", "WallS", "WallE", "WallW"), parts):
        k.cube(f"{path}/{nm}", p.size, p.pos, Q_IDENT, ctx.mats.get(col), ctx.mats.phys("plastic"), collide=True)
    k.tag(body, "tote", zone=ctx.zone or "")
    n_items = int(n_items if n_items is not None else rng.integers(3, 11))
    ctx.record(path, "tote", pos, q, "dynamic", size=TOTE, mass=mass, friction=0.4, parent=parent, open_top=True, n_items=n_items)
    cols, rows_ = 4, 3
    inner_x, inner_y = L - 2 * t - 0.02, Wd - 2 * t - 0.02
    cells = [(ix, iy) for ix in range(cols) for iy in range(rows_)]
    order = rng.permutation(len(cells))[:n_items]
    R0 = qmat(q)
    for n_i, ci in enumerate(order):
        ix, iy = cells[ci]
        cw, ch = inner_x / cols, inner_y / rows_
        lx = -inner_x / 2 + (ix + 0.5) * cw
        ly = -inner_y / 2 + (iy + 0.5) * ch
        is_cyl = rng.random() < 0.35
        if is_cyl:
            r = float(rng.uniform(0.02, 0.032))
            hh = float(rng.uniform(0.05, 0.12))
            size = (r, hh)
            vol = math.pi * r * r * hh
        else:
            lim = (min(cw, ch) - 0.026) / math.sqrt(2)  # rotated footprint stays inside its own cell
            size = (float(rng.uniform(0.04, lim)), float(rng.uniform(0.04, lim)), float(rng.uniform(0.04, 0.11)))
            vol = size[0] * size[1] * size[2]
        m_i = float(np.clip(vol * rng.uniform(250, 700), 0.02, 0.9))
        jx, jy = float(rng.uniform(-0.006, 0.006)), float(rng.uniform(-0.006, 0.006))
        hz = size[1] if is_cyl else size[2]
        lz = t + hz / 2 + BOX_EPS
        wp = np.asarray(pos) + R0 @ np.array([lx + jx, ly + jy, lz])
        _item(ctx, f"{root}/Item_{n_i:02d}", tuple(wp), qmul(q, qyaw(float(rng.uniform(0, math.tau)))), is_cyl, size, m_i, path, rng)
    return path


def _item(ctx: BuildContext, path, pos, quat, is_cyl, size, mass, tote_path, rng):
    k = ctx.kit
    part = Part("cyl", size) if is_cyl else Part("box", size)
    body = k.rigid_body(path, pos, quat, parts=[part], mass=mass, iters=(8, 1))
    mat = ctx.mats.get(("item_a", "item_b", "item_c", "item_d")[int(rng.integers(4))])
    if is_cyl:
        k.cylinder(f"{path}/Shape", size[0], size[1], (0, 0, 0), Q_IDENT, mat, ctx.mats.phys("plastic"), collide=True)
    else:
        k.cube(f"{path}/Shape", size, (0, 0, 0), Q_IDENT, mat, ctx.mats.phys("plastic"), collide=True)
    k.tag(body, "tote_item", zone=ctx.zone or "")
    ctx.record(path, "tote_item", pos, quat, "dynamic", size=size, mass=mass, friction=0.4, parent=tote_path,
               shape="cylinder" if is_cyl else "box")


def totes_on_pallet(ctx: BuildContext, root, pal: dict, rng, n_items_each=None):
    """Single layer of 4 totes (2 x 2) on a pallet deck."""
    q = qyaw(pal["yaw"])
    R0 = qmat(q)
    L, Wd, Hh = TOTE
    for ix in (-1, 1):
        for iy in (-1, 1):
            lp = np.array([ix * (L / 2 + 0.004), iy * (Wd / 2 + 0.004), 0.0])
            wp = np.asarray(pal["pos"]) + R0 @ lp
            wp[2] = pal["deck_top"] + BOX_EPS
            build_tote(ctx, ctx.alloc(root, "Tote", 2), tuple(wp), pal["yaw"], rng, n_items_each, parent=pal["path"])
    pal.update(load_top=pal["deck_top"] + Hh, kind="totes", n_boxes=4)
    return pal


# --------------------------------------------------------------------------- assets
class PalletParams(Params):
    mass_kg: float | None = Field(None, ge=5, le=60, description="Pallet mass; default: uniform 20-25 kg")


@register
class Pallet(Asset):
    id = "pallet"
    description = "Empty EUR1 wooden pallet (1.2 x 0.8 x 0.144 m), 15-shape compound collider, dynamic."
    tags = frozenset({"logistics", "pallet", "dynamic", "wood", "support"})
    Params = PalletParams
    physics = PhysicsInfo("dynamic", "wood", 0.5, (20.0, 25.0), "primitives")
    example = {"id": "spare_pallets", "asset": "pallet", "count": 3, "placement": {"inside": "dock_staging"}}

    def footprint(self, p):
        return Footprint.rect(PAL_L, PAL_W, PAL_H)

    def anchors(self, p):
        return [Anchor("deck", "support", (0, 0, PAL_H), polygon=_rect(PAL_L, PAL_W))]

    def build(self, ctx, path, pose, p, rng):
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + PAL_EPS)
        pal = build_pallet(ctx, path, pos, pose.yaw, rng, p.mass_kg)
        return Built(path, [Anchor("deck", "support", (pos[0], pos[1], pal["deck_top"]), pose.quat, _rect(PAL_L, PAL_W))],
                     info=pal, bodies=[path])


class BoxParams(Params):
    size: Vec3 = Field((0.4, 0.3, 0.25), description="Carton dimensions x, y, z (m), each 0.05-1.2")
    variant: int | None = Field(None, ge=0, lt=N_CARDBOARD, description="Cardboard texture variant; default random")
    mass_kg: float | None = Field(None, ge=0.05, le=40, description="Default: volume x U(60, 220) kg/m3, clipped to 0.2-15 kg")


@register
class CardboardBox(Asset):
    id = "cardboard_box"
    description = "Closed cardboard carton with a textured label/tape visual; single box collider, dynamic."
    tags = frozenset({"logistics", "box", "carton", "dynamic", "cardboard", "clutter", "support"})
    Params = BoxParams
    physics = PhysicsInfo("dynamic", "cardboard", 0.5, BOX_MASS_RANGE, "primitives")
    example = {"id": "floor_boxes", "asset": "cardboard_box", "count": [4, 8],
               "params": {"size": {"choice": [[0.3, 0.2, 0.2], [0.5, 0.4, 0.3]]}}, "placement": {"inside": "packing"}}

    def footprint(self, p):
        return Footprint.rect(p.size[0], p.size[1], p.size[2])

    def anchors(self, p):
        return [Anchor("top", "support", (0, 0, p.size[2]), polygon=_rect(p.size[0], p.size[1]))]

    def build(self, ctx, path, pose, p, rng):
        if any(not 0.05 <= s <= 1.2 for s in p.size):
            raise ValueError(f"cardboard_box size {p.size} outside 0.05-1.2 m")
        variant = int(rng.integers(N_CARDBOARD)) if p.variant is None else p.variant
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + p.size[2] / 2 + 0.001)
        build_box(ctx, path, pos, pose.quat, p.size, variant, rng, p.mass_kg)
        top = Anchor("top", "support", (pos[0], pos[1], pos[2] + p.size[2] / 2), pose.quat, _rect(p.size[0], p.size[1]))
        return Built(path, [top], info={"size": list(p.size)}, bodies=[path])


class PalletLoadParams(Params):
    variant: Literal["boxes", "wrapped", "totes", "empty"] = Field("boxes", description="What sits on the pallet")
    box_size: Vec3 = Field((0.35, 0.3, 0.25), description="Carton size for boxes/wrapped loads (m), each 0.1-0.6")
    box_size_scale: tuple[float, float] = Field((1.0, 1.0), description="Uniform random scale applied to box_size per load, e.g. [0.94, 1.06]")
    min_box_dim: float = Field(0.1, ge=0.05, le=0.6, description="Scaled carton dims are clipped to [min_box_dim, 0.6]")
    max_load_height: float = Field(1.12, ge=0.1, le=2.0, description="Maximum load height above the deck (m)")
    max_boxes: int = Field(12, ge=1, le=48)
    full: bool = Field(False, description="Fill every layer up to max_load_height (else a random number of layers)")
    min_layers: int = Field(1, ge=1, le=6)
    tote_items: int | None = Field(None, ge=0, le=12, description="Items per tote (totes variant); default random 3-10")
    footprint_pad: float = Field(0.03, ge=0.0, le=0.1, description="Planning footprint margin per side for carton overhang (max real overhang 0.03 m)")


@register
class PalletLoad(Asset):
    id = "pallet_load"
    description = ("EUR pallet carrying a load: stacked cartons with realistic misalignment (<=3 cm / 10 deg), a single "
                   "stretch-wrapped block, 4 totes with items, or nothing. Exposes 'deck' and 'top' (load top) supports.")
    tags = frozenset({"logistics", "pallet", "dynamic", "boxes", "totes", "support", "stackable"})
    Params = PalletLoadParams
    physics = PhysicsInfo("mixed", "wood", 0.5, None, "primitives", "pallet + one rigid body per carton/tote/item; wrapped = one body")
    example = {"id": "rack_pallets", "asset": "pallet_load",
               "params": {"variant": {"choice": ["boxes", "wrapped", "totes", "empty"], "p": [0.55, 0.28, 0.08, 0.09]}},
               "placement": {"fill": {"anchors": "racks.*.slot*", "probability": 0.5}}}

    def footprint(self, p):
        h = PAL_H + (0 if p.variant == "empty" else (TOTE[2] if p.variant == "totes" else p.max_load_height))
        return Footprint.rect(PAL_L + 2 * p.footprint_pad, PAL_W + 2 * p.footprint_pad, h)

    def anchors(self, p):
        h = self.footprint(p).z1
        return [Anchor("deck", "support", (0, 0, PAL_H), polygon=_rect(PAL_L, PAL_W), meta={"occupied": p.variant != "empty"}),
                Anchor("top", "support", (0, 0, h), polygon=_rect(PAL_L, PAL_W), meta={"height_is_upper_bound": True})]

    def build(self, ctx, path, pose, p, rng):
        if any(not 0.1 <= s <= 0.6 for s in p.box_size):
            raise ValueError(f"pallet_load box_size {p.box_size} outside 0.1-0.6 m")
        ctx.kit.xform(path)
        if p.box_size_scale != (1.0, 1.0):
            sc = float(rng.uniform(*p.box_size_scale))
            p = p.model_copy(update={"box_size": tuple(float(np.clip(d * sc, p.min_box_dim, 0.6)) for d in p.box_size)})
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + PAL_EPS)
        yaw = pose.yaw
        bodies = []
        if p.variant == "wrapped":
            pal = build_wrapped(ctx, f"{path}/Pallet", pos, yaw, p.box_size, p.max_load_height, rng)
        else:
            pal = build_pallet(ctx, f"{path}/Pallet", pos, yaw, rng)
            if p.variant == "boxes":
                load_boxes(ctx, path, pal, p.box_size, p.max_load_height, rng, full=p.full, max_boxes=p.max_boxes, min_layers=p.min_layers)
            elif p.variant == "totes":
                totes_on_pallet(ctx, path, pal, rng, p.tote_items)
        bodies.append(pal["path"])
        anchors = [Anchor("deck", "support", (pos[0], pos[1], pal["deck_top"]), pose.quat, _rect(PAL_L, PAL_W)),
                   Anchor("top", "support", (pos[0], pos[1], pal["load_top"]), pose.quat, _rect(PAL_L, PAL_W))]
        return Built(path, anchors, info=pal, bodies=bodies)


class ToteParams(Params):
    n_items: int | None = Field(None, ge=0, le=12, description="Rigid items inside; default random 3-10")
    color: Literal["blue", "grey", "red"] | None = None


@register
class Tote(Asset):
    id = "tote"
    description = "Open-top plastic tote 0.6 x 0.4 x 0.28 m with small rigid items (boxes/cylinders) on its floor."
    tags = frozenset({"logistics", "tote", "bin", "container", "dynamic", "plastic"})
    Params = ToteParams
    physics = PhysicsInfo("mixed", "plastic", 0.4, (1.4, 1.9), "primitives", "tote body + one body per item")

    def footprint(self, p):
        return Footprint.rect(*TOTE)

    def build(self, ctx, path, pose, p, rng):
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] + 0.001)
        col = None if p.color is None else f"tote_{p.color}"
        tote = build_tote(ctx, path, pos, pose.yaw, rng, p.n_items, color=col)
        top = Anchor("top", "support", (pos[0], pos[1], pos[2] + TOTE[2]), pose.quat, _rect(TOTE[0], TOTE[1]))
        return Built(path, [top], info={"tote": tote}, bodies=[tote])


# --------------------------------------------------------------------------- tipped pallet jack
def _axis_rotations():
    """The 24 proper rotations of the cube group (axis permutations with signs)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (pp, s) in enumerate(zip(perm, signs)):
                R[i, pp] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def stable_orientation(parts, com):
    """Axis-aligned orientation whose support polygon contains the COM with max margin -> (R, zmin, margin_m)."""
    from scipy.spatial import ConvexHull

    best = None
    for R in _axis_rotations():
        pts = []
        for p in parts:
            c = R @ np.asarray(p.pos)
            h = np.abs(R) @ (np.asarray(p.size) / 2)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(c + np.array([sx, sy, sz]) * h)
        pts = np.array(pts)
        zmin = pts[:, 2].min()
        contact = pts[pts[:, 2] < zmin + 0.004][:, :2]
        if len(contact) < 3:
            continue
        try:
            hull = ConvexHull(contact)
        except Exception:
            continue
        cxy = (R @ com)[:2]
        eq = hull.equations
        margin = float(-(eq[:, :2] @ cxy + eq[:, 2]).max())
        score = margin - 0.1 * ((R @ com)[2] - zmin)
        if margin > 0.02 and (best is None or score > best[0]):
            best = (score, R, zmin, margin)
    if best is None:
        raise RuntimeError("no stable resting orientation found")
    return best[1], best[2], best[3]


JACK_PARTS = [
    Part("box", (1.15, 0.16, 0.06), (0.575, 0.26, 0.03)),
    Part("box", (1.15, 0.16, 0.06), (0.575, -0.26, 0.03)),
    Part("box", (0.22, 0.68, 0.08), (0.05, 0, 0.1)),  # pump housing / yoke
    Part("box", (0.14, 0.10, 0.9), (0.0, 0, 0.55)),  # handle column
    Part("box", (0.08, 0.38, 0.05), (-0.1, 0, 1.05)),  # handle bar
]


@register
class PalletJackTipped(Asset):
    id = "pallet_jack_tipped"
    description = "Manual pallet jack lying on its side in a statically stable pose (support-polygon analysis), dynamic."
    tags = frozenset({"logistics", "clutter", "obstacle", "dynamic", "steel"})
    physics = PhysicsInfo("dynamic", "steel", 0.6, 72.0, "primitives")

    @staticmethod
    def _rest():
        parts = [Part(pp.kind, pp.size, pp.pos) for pp in JACK_PARTS]
        com, _, _, _ = composite_inertia(parts, 72.0)
        return parts, stable_orientation(parts, com)

    def footprint(self, p):
        """Exact bounding box of the jack in its resting pose (origin = body origin projected to the floor)."""
        parts, (R, zmin, _) = self._rest()
        pts = np.array([R @ (np.asarray(q.pos) + np.array([sx, sy, sz]) * np.asarray(q.size) / 2)
                        for q in parts for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        lo, hi = pts.min(0), pts.max(0)
        return Footprint.rect(hi[0] - lo[0] + 0.02, hi[1] - lo[1] + 0.02, hi[2] - zmin, (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)

    def build(self, ctx, path, pose, p, rng):
        k = ctx.kit
        mass = 72.0
        parts = [Part(pp.kind, pp.size, pp.pos) for pp in JACK_PARTS]
        com, _, _, _ = composite_inertia(parts, mass)
        R_rest, zmin, margin = stable_orientation(parts, com)
        q = qmul(pose.quat, qfrom_mat(R_rest))
        pos = (pose.pos[0], pose.pos[1], pose.pos[2] - zmin + PAL_EPS)
        body = k.rigid_body(path, pos, q, parts=parts, mass=mass)
        for j, pp in enumerate(parts):
            k.cube(f"{path}/p{j}", pp.size, pp.pos, Q_IDENT, ctx.mats.get("jack"), ctx.mats.phys("steel"), collide=True)
        k.tag(body, "pallet_jack", zone=ctx.zone or "")
        ctx.record(path, "clutter", pos, q, "dynamic", size=(1.2, 1.1, 0.9), mass=mass, friction=0.6,
                   subtype="pallet_jack_tipped", rest_stability_margin_m=round(float(margin), 3))
        return Built(path, info={"stability_margin_m": float(margin)}, bodies=[path])
