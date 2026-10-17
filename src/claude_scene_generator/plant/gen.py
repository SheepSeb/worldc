"""Industrial process-plant scene generator (Isaac Sim 5.x / Isaac Lab compatible USD).

    uv run python -m claude_scene_generator.plant.gen --seed 3 --out plant_out

Units: metres, Z up. Deck top = z 0, deck 12 x 8 m centred at the origin, ground at z = -1.5.
Only ``pxr`` (usd-core) is needed to generate; Isaac Sim is only needed for validation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from ..colliders import collect
from ..geom import Q_IDENT, qaxis, qfrom_mat, qmat, qmul, qyaw
from ..usdkit import Kit, T
from . import equipment as EQ
from . import nav as NV
from . import textures as TX
from .pipenet import PipeNet
from .common import Art, Ctx, Sh, cap, cube, cyl, esph, qalign, sph

LIGHTING = ["bright_day", "overcast", "indoor_industrial"]
SERVICES = ["Process water", "Steam LP", "Nitrogen", "Lube oil", "Fuel gas", "Condensate", "Instrument air", "Cooling water", "Flare header", "Seal oil"]

MAT_DEFS = {  # name: (rgb, roughness, metallic)
    "steel": ((0.68, 0.70, 0.72), 0.42, 0.4), "steel_b": ((0.62, 0.64, 0.66), 0.45, 0.4), "steel_c": ((0.76, 0.77, 0.78), 0.38, 0.45),
    "steel_d": ((0.56, 0.58, 0.60), 0.5, 0.4), "steel_dark": ((0.34, 0.35, 0.37), 0.5, 0.45), "valve_body": ((0.62, 0.64, 0.66), 0.45, 0.45),
    "valve_red": ((0.62, 0.06, 0.05), 0.45, 0.15), "red_jb": ((0.66, 0.05, 0.04), 0.45, 0.15), "gray_box": ((0.58, 0.60, 0.62), 0.5, 0.2),
    "panel_plate": ((0.45, 0.47, 0.5), 0.55, 0.35), "cab_blue": ((0.06, 0.17, 0.50), 0.45, 0.2), "brown": ((0.26, 0.12, 0.07), 0.65, 0.1),
    "motor": ((0.55, 0.57, 0.58), 0.5, 0.3), "pump_gray": ((0.45, 0.47, 0.49), 0.5, 0.4), "white": ((0.9, 0.9, 0.87), 0.6, 0.0),
    "insul": ((0.9, 0.9, 0.88), 0.45, 0.2), "yellow": ((0.98, 0.8, 0.05), 0.45, 0.0), "orange_band": ((0.95, 0.42, 0.05), 0.5, 0.0),
    "green_band": ((0.08, 0.50, 0.20), 0.5, 0.0), "white_band": ((0.92, 0.92, 0.9), 0.5, 0.0), "blue_band": ((0.1, 0.3, 0.75), 0.5, 0.0),
    "black": ((0.03, 0.03, 0.035), 0.6, 0.0), "tube": ((0.62, 0.63, 0.65), 0.35, 0.8), "cable_red": ((0.7, 0.05, 0.05), 0.6, 0.0),
    "cable_orange": ((0.95, 0.45, 0.05), 0.6, 0.0), "cable_green": ((0.06, 0.5, 0.15), 0.6, 0.0), "cable_blue": ((0.08, 0.2, 0.7), 0.6, 0.0),
    "btn_green": ((0.05, 0.6, 0.15), 0.4, 0.0), "btn_red": ((0.8, 0.04, 0.04), 0.4, 0.0), "btn_black": ((0.04, 0.04, 0.04), 0.4, 0.0),
    "gauge_white": ((0.92, 0.92, 0.88), 0.4, 0.0), "coverall_red": ((0.74, 0.06, 0.05), 0.8, 0.0), "boot": ((0.25, 0.14, 0.08), 0.7, 0.0),
    "glove": ((0.95, 0.5, 0.05), 0.7, 0.0), "skin": ((0.78, 0.58, 0.45), 0.7, 0.0), "hardhat": ((0.96, 0.42, 0.03), 0.4, 0.0),
    "hardhat_white": ((0.9, 0.9, 0.88), 0.4, 0.0), "bucket_yellow": ((0.9, 0.72, 0.06), 0.5, 0.0), "wood": ((0.5, 0.33, 0.18), 0.8, 0.0),
    "orange_box": ((0.9, 0.4, 0.05), 0.6, 0.0), "stainless": ((0.72, 0.73, 0.74), 0.32, 0.9), "wheel_blue": ((0.08, 0.22, 0.6), 0.5, 0.1),
    "wheel_white": ((0.85, 0.85, 0.84), 0.5, 0.1), "wheel_red": ((0.65, 0.07, 0.06), 0.5, 0.1), "dark_void": ((0.02, 0.02, 0.02), 1.0, 0.0),
    "tank_white": ((0.78, 0.8, 0.82), 0.6, 0.1),
}
STEELS = ["steel", "steel_b", "steel_c", "steel_d"]
WHEEL_MATS = ["steel_dark", "steel_dark", "steel_c", "steel_c", "wheel_white", "steel_dark", "wheel_blue", "valve_red"]


def u(rng, a, b):
    return float(rng.uniform(a, b))


def nid(v, scale=100):
    """Valid USD name fragment for a possibly negative number."""
    return f"{'m' if v < 0 else 'p'}{int(round(abs(v) * scale))}"


# --------------------------------------------------------------------------- materials / textures
def build_materials(ctx: Ctx, out: Path, light_idx: int):
    rng, k = ctx.rng, ctx.kit
    d = ctx.dirt
    for name, (rgb, rough, metal) in MAT_DEFS.items():
        c = np.array(rgb) * rng.uniform(0.93, 1.07, 3)
        if not name.startswith(("coverall", "skin", "glove", "hardhat", "btn_", "cable", "gauge")):
            c = c * (1 - 0.30 * d * rng.uniform(0.4, 1.0)) + np.array([0.20, 0.16, 0.11]) * 0.12 * d  # grime/rust tint
            rough = min(1.0, rough + 0.25 * d)
        ctx.mats[name] = k.material(name, tuple(np.clip(c, 0, 1)), rough, metal)
    tdir = out / "textures"
    tdir.mkdir(exist_ok=True)
    TX.make_grating(tdir / "grating.jpg", rng, 6.0, 4.0, d, ppm=320)
    TX.make_concrete(tdir / "concrete.jpg", rng, d)
    ctx.mats["grating"] = k.material("grating", (0.5, 0.5, 0.5), 0.45, 0.6, tex="textures/grating.jpg")
    ctx.mats["concrete"] = k.material("concrete", (0.5, 0.5, 0.5), 0.9, 0.0, tex="textures/concrete.jpg")
    if light_idx != 2:
        TX.make_sky(tdir / "sky.jpg", rng, "clear" if light_idx == 0 else "overcast")
    for n, mu in (("steel", 0.6), ("grating", 0.7), ("rubber", 0.8), ("plastic", 0.5), ("concrete", 0.7)):
        ctx.pm[n] = k.phys_material(n, mu, 0.05)


# --------------------------------------------------------------------------- beams
def ibeam(ctx, path, a, b, w=0.2, h=0.2, mat="steel_d", tf=0.014, tw=0.012):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    ax = np.argmax(np.abs(d))
    if ax == 2:
        q = Q_IDENT
    elif ax == 0:
        q = qaxis((0, 1, 0), math.pi / 2)
    else:
        q = qfrom_mat([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    sh = [cube((tf, w, L), (h / 2 - tf / 2, 0, 0), mat), cube((tf, w, L), (-(h / 2 - tf / 2), 0, 0), mat), cube((h - 2 * tf, tw, L), (0, 0, 0), mat)]
    ctx.static(path, (a + b) / 2, q, sh, category="rack_structure")


def rod(ctx, path, a, b, r=0.025, mat="steel_d", col=True):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ctx.static(path, (a + b) / 2, qalign(b - a), [cyl(r, float(np.linalg.norm(b - a)), mat=mat, col=col)])


def label_for(ctx, bands_cols):
    return bands_cols[int(ctx.rng.integers(len(bands_cols)))]


# --------------------------------------------------------------------------- deck, ground, structure
DX0, DX1, DY0, DY1 = -3.0, 3.0, -2.0, 2.0  # deck footprint (6 x 4 m)
YW = 1.9  # plane of the valve-wall pipes
XL, XR = -2.8, 2.0  # valve-wall header span


def build_deck(ctx: Ctx, drain):
    k = ctx.kit
    dx, dy, hs = drain[0], drain[1], 0.3
    rects = [(DX0, DY0, dx - hs, DY1), (dx + hs, DY0, DX1, DY1), (dx - hs, DY0, dx + hs, dy - hs), (dx - hs, dy + hs, dx + hs, DY1)]
    k.xform("/World/Deck")
    for i, (x0, y0, x1, y1) in enumerate(rects):
        c = ((x0 + x1) / 2, (y0 + y1) / 2, -0.05)
        p = k.cube(f"/World/Deck/collision_{i}", (x1 - x0, y1 - y0, 0.1), c, vmat=None, pmat=ctx.pm["grating"], collide=True, guide=True)
        k.tag(p, "deck_floor")
        pts = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
        uv = [((x - DX0) / 6.0, 1 - (y - DY0) / 4.0) for x, y, _ in pts]
        k.mesh(f"/World/Deck/grating_visual_{i}", pts, [4], [0, 1, 2, 3], uv, ctx.mats["grating"])
    k.cube("/World/Deck/underside_visual", (6, 4, 0.02), (0, 0, -0.12), vmat=ctx.mats["dark_void"])
    k.cube("/World/Deck/sump_floor", (0.9, 0.9, 0.05), (dx, dy, -0.6), vmat=ctx.mats["dark_void"], pmat=ctx.pm["steel"], collide=True)
    k.cylinder("/World/Deck/sump_pipe", 0.11, 1.0, (dx, dy, -1.0), vmat=ctx.mats["steel_dark"], pmat=ctx.pm["steel"], collide=True)
    ring = [cube((2 * math.pi * 0.27 / 16 * 1.1, 0.025, 0.05), (0.27 * math.cos(a), 0.27 * math.sin(a), 0.025), "steel_c", col=False, quat=qyaw(a + math.pi / 2)) for a in (2 * math.pi * i / 16 for i in range(16))]
    ring.append(cyl(0.2, 0.01, (0, 0, -0.45), "dark_void", col=False))
    ctx.static("/World/Deck/drain_lip", (dx, dy, 0), Q_IDENT, ring)
    for i, x in enumerate((-2.9, 0.0, 2.9)):
        ibeam(ctx, f"/World/Deck/beam_y{i}", (x, -1.95, -0.17), (x, 1.95, -0.17), 0.15, 0.2)
        for j, y in enumerate((-1.9, 1.9)):
            ibeam(ctx, f"/World/Deck/leg_{i}_{j}", (x, y, -1.5), (x, y, -0.3), 0.18, 0.18)
    for i, y in enumerate((-1.9, 0.0, 1.9)):
        ibeam(ctx, f"/World/Deck/beam_x{i}", (-2.95, y, -0.3), (2.95, y, -0.3), 0.15, 0.2)
    ctx.static("/World/Deck/kerb", (0, 0, 0), Q_IDENT, [
        cube((6, 0.04, 0.1), (0, -1.98, 0.05), "steel_dark"), cube((6, 0.04, 0.1), (0, 1.98, 0.05), "steel_dark"),
        cube((0.04, 4, 0.1), (-2.98, 0, 0.05), "steel_dark"), cube((0.04, 4, 0.1), (2.98, 0, 0.05), "steel_dark")])


def build_ground(ctx: Ctx):
    k = ctx.kit
    p = k.cube("/World/Ground/collision", (60, 60, 0.5), (0, 0, -1.75), vmat=None, pmat=ctx.pm["concrete"], collide=True, guide=True)
    k.tag(p, "ground")
    k.mesh_box("/World/Ground/visual", (44, 34, 0.04), (0, 0, -1.52), vmat=ctx.mats["concrete"])


def build_structure(ctx: Ctx):
    """Steel frame around the deck: columns, beams at the pipe-rack level (3.5 m) and top (5 m), cross-bracing."""
    cx = (-3.1, -1.0, 1.0, 3.1)
    ys = (-2.15, 2.15)
    for i, x in enumerate(cx):
        for j, y in enumerate(ys):
            ibeam(ctx, f"/World/Structure/col_{i}_{j}", (x, y, -1.5), (x, y, 5.0), 0.2, 0.2, "steel_d")
    for j, y in enumerate(ys):
        for z, nm in ((3.5, "rack"), (5.0, "top"), (0.0, "base")):
            if nm == "base":
                continue
            ibeam(ctx, f"/World/Structure/{nm}_long_{j}", (-3.2, y, z), (3.2, y, z), 0.2, 0.22, "steel_d")
    for i, x in enumerate((-3.1, -1.55, 0.0, 1.55, 3.1)):
        for z, nm in ((3.5, "rack"), (5.0, "top")):
            ibeam(ctx, f"/World/Structure/{nm}_cross_{i}", (x, -2.15, z), (x, 2.15, z), 0.18, 0.2, "steel_b")
    for i, x in enumerate(cx):
        for j in range(1):
            rod(ctx, f"/World/Structure/vb_{i}", (x, -2.15, 0.2), (x, 2.15, 0.2), 0.0, "steel_d", col=False) if False else None
    # X bracing in the back plane (y = 2.15) and top bays, and in the side planes
    def xbrace(nm, p00, p01, p10, p11, r=0.028):
        rod(ctx, nm + "a", p00, p11, r)
        rod(ctx, nm + "b", p01, p10, r)

    for i in range(3):
        x0, x1 = cx[i], cx[i + 1]
        xbrace(f"/World/Structure/bk_{i}", (x0, 2.15, 0.1), (x0, 2.15, 3.4), (x1, 2.15, 0.1), (x1, 2.15, 3.4))
        xbrace(f"/World/Structure/tp_{i}", (x0, 2.15, 3.6), (x0, 2.15, 4.9), (x1, 2.15, 3.6), (x1, 2.15, 4.9))
    for s, x in (("l", -3.1), ("r", 3.1)):
        xbrace(f"/World/Structure/sd_{s}", (x, -2.15, 3.6), (x, -2.15, 4.9), (x, 2.15, 3.6), (x, 2.15, 4.9))
        xbrace(f"/World/Structure/sdl_{s}", (x, -2.15, 0.1), (x, -2.15, 3.4), (x, 2.15, 0.1), (x, 2.15, 3.4))
    # back wall channels (unistrut) the wall pipes are clamped to, plus a vertical stile every 1 m
    ctx.static("/World/Structure/wall_channels", (0, 0, 0), Q_IDENT,
               [cube((6.0, 0.05, 0.045), (0, 2.06, z), "steel_dark") for z in (0.5, 0.85, 1.2, 1.55, 1.9, 2.9, 3.25)] + [cube((0.05, 0.05, 3.5), (x, 2.12, 1.75), "steel_dark") for x in (-2.95, -1.5, 0.0, 1.5, 2.95)],
               category="rack_structure")
    # cable tray in the roof of the area (above robot height), crossing the deck
    ctx.static("/World/Structure/cable_tray", (0.0, 0.0, 2.55), Q_IDENT,
               [cube((0.45, 3.8, 0.012), (x, 0, 0), "steel_c") for x in (-2.2, 2.2)] + [cube((0.012, 3.8, 0.07), (x + s * 0.225, 0, 0.04), "steel_c") for x in (-2.2, 2.2) for s in (-1, 1)] +
               [cube((0.45, 0.02, 0.012), (x, y, 0.012), "steel_dark") for x in (-2.2, 2.2) for y in np.arange(-1.8, 1.9, 0.4)], category="rack_structure")


def build_background(ctx: Ctx):
    """Visual-only plant behind and around the deck: steel frames, pipe racks, vessels, columns, bracing - no empty sky."""
    rng, k = ctx.rng, ctx.kit
    g = "/World/Background"
    k.xform(g)
    n = 0
    # three bays of steel frames behind the valve wall (y = 4.5, 8, 12) with beams at two levels, columns, bracing
    for bi, y in enumerate((4.6, 9.0)):
        for xi in range(-5, 6):
            x = xi * 2.2
            h = 7.5 if (xi + bi) % 3 else 10.5
            k.cube(f"{g}/col_{n}", (0.28, 0.28, h + 1.5), (x, y, h / 2 - 0.75), vmat=ctx.mats["steel_d"])
            n += 1
        for z in (3.6, 6.4):
            k.cube(f"{g}/beam_{n}", (24.0, 0.22, 0.24), (0, y, z), vmat=ctx.mats["steel_b"])
            n += 1
        for xi in range(-5, 5, 2):
            x0, x1 = xi * 2.2, (xi + 1) * 2.2
            for a, b in (((x0, 0.3), (x1, 3.5)), ((x1, 0.3), (x0, 3.5))):
                c = k.cylinder(f"{g}/br_{n}", 0.03, float(np.hypot(b[0] - a[0], b[1] - a[1])), (0, 0, 0), vmat=ctx.mats["steel_d"])
                n += 1
                ctx.kit.set_pose(c, ((a[0] + b[0]) / 2, y, (a[1] + b[1]) / 2), qalign((b[0] - a[0], 0, b[1] - a[1])))
        # pipe racks on the frames
        for pi in range(7):
            r = u(rng, 0.06, 0.3)
            z = u(rng, 3.8, 7.0)
            c = k.cylinder(f"{g}/rk_{n}", r, 24.0, (0, 0, 0), vmat=ctx.mats[str(rng.choice(STEELS + ["insul", "white"]))])
            n += 1
            ctx.kit.set_pose(c, (0, y + u(rng, -0.8, 0.8), z), qalign((1, 0, 0)))
    for i in range(12):  # vessels / columns / tanks
        x, y = u(rng, -14, 14), u(rng, 5.5, 15)
        r, h = u(rng, 0.3, 1.6), u(rng, 3.0, 11.0)
        k.cylinder(f"{g}/vessel_{i}", r, h, (x, y, -1.5 + h / 2), vmat=ctx.mats[str(rng.choice(["white", "insul", "steel_c", "steel_b", "tank_white"]))])
        k.cylinder(f"{g}/vband_{i}", r * 1.02, 0.16, (x, y, -1.5 + h * 0.6), vmat=ctx.mats["yellow"])
        k.sphere(f"{g}/vdome_{i}", r, (x, y, -1.5 + h), vmat=ctx.mats["white"])
    # side structure (left/right of the deck) so the lateral views are not empty either
    for si, x in enumerate((-8.0, 8.0)):
        for yi in range(-2, 3):
            y = yi * 3.0
            k.cube(f"{g}/side_col_{si}_{yi + 2}", (0.3, 0.3, 9.0), (x, y, 3.0), vmat=ctx.mats["steel_d"])
        for z in (3.6, 6.4):
            k.cube(f"{g}/side_beam_{si}_{int(z * 10)}", (0.22, 14.0, 0.24), (x, 0, z), vmat=ctx.mats["steel_b"])
    k.cylinder(f"{g}/stack", 0.9, 22.0, (-12, 9, 9.5), vmat=ctx.mats["steel_c"])


# --------------------------------------------------------------------------- pipe network: valve wall, left cluster, rack, pump tie-ins
def build_wall(ctx: Ctx, net: PipeNet, panel_cx, label_cols, valves_out):
    """Seven stacked horizontal headers (the 'main valve wall') with horizontal-stem gate valves whose handwheels face the
    camera, red control valves on row 4, gauges, junction boxes, vertical collectors and inter-row risers."""
    rng = ctx.rng
    rows_v = [0.5, 0.85, 1.2, 1.55, 1.9, 2.9, 3.25]
    CR = 4  # control-valve row (tall actuators rise above it; rows 5/6 are clear of them)
    branch_x = (-1.55, 0.0, 1.55)  # overhead Y branches hang from the rack beams at these x on row 5
    slots = [XL + 0.3 + 0.36 * j for j in range(13)]
    avail = [x for x in slots if abs(x - panel_cx) > 0.87 and x < XR - 0.15]
    ctl = []
    cand = [x for x in avail if abs(x - panel_cx) > 1.25 and x < XR - 0.45]
    rng.shuffle(cand)
    for x in cand:
        if all(abs(x - c) > 1.0 for c in ctl) and len(ctl) < 3:
            ctl.append(x)
    ctl.sort()
    ctx.lines["L-W"] = "Process manifold"
    info = dict(valves=0, gauges=0, boxes=0, risers=0)
    nrows = len(rows_v)
    rr = {k: float(rng.choice([0.04, 0.05, 0.06])) for k in range(nrows)}
    used = {}
    for k in range(nrows):
        used[k] = [x for x in avail if not (k == CR and any(abs(x - c) < 0.5 for c in ctl)) and not (k == 5 and any(abs(x - b) < 0.22 for b in branch_x))
                   and not (k >= 5 and any(abs(x - cx_) < 0.34 for cx_ in (-3.1, -1.55, 0.0, 1.55, 3.1)))]  # wheels of the top rows must clear the rack cross beams (z 3.5)
    mids = [x + 0.18 for x in slots[:-1] if abs(x + 0.18 - panel_cx) > 0.87 and x + 0.18 < XR - 0.2]
    riser_taps = {k: [] for k in range(nrows)}
    risers = []
    for k in range(nrows - 1):
        ok = [m for m in mids if not (CR in (k, k + 1) and any(abs(m - c) < 0.5 for c in ctl)) and not (5 in (k, k + 1) and any(abs(m - b) < 0.3 for b in branch_x))]
        rng.shuffle(ok)
        for m in ok[:2]:
            risers.append((k, m))
            riser_taps[k].append(m)
            riser_taps[k + 1].append(m)
    for k, z in enumerate(rows_v):
        r = rr[k]
        a, b = np.array([XL, YW, z]), np.array([XR, YW, z])
        vlist, vspecs = [], []
        for x in used[k]:
            vlist.append((x - XL, 0.14 + 1.5 * r))
            vspecs.append(("gate", x, z, r))
        if k == CR:
            for x in ctl:
                vlist.append((x - XL, 0.3 + 1.7 * r))
                vspecs.append(("ctl", x, z, r))
        taps = [m - XL for m in riser_taps[k]] + ([x - XL for x in branch_x] if k == 5 else [])
        bands = [(u(rng, 0.3, 4.4), label_for(ctx, label_cols)) for _ in range(2)]
        net.run(a, b, r, valves=vlist, bands=bands, name=f"row{k}", support=("back", 2.04), service="Process manifold", taps=taps)
        for kind, x, zz, rr_ in vspecs:
            if kind == "gate":
                R = u(rng, 0.1, 0.145)
                v = EQ.gate_valve(ctx, f"GateValve_W{k}_{len(valves_out)}", (x, YW, zz), "x", rr_, zz, 2 * R, int(rng.integers(4, 8)), u(rng, 0.05, 0.09), "L-W", str(rng.choice(WHEEL_MATS)), stem_dir=(0, -1, 0))
            else:
                v = EQ.control_valve(ctx, f"ControlValve_W{len(valves_out)}", (x, YW, zz), "x", rr_, "L-W")
            v["header"] = f"W{k}"
            v["on_wall"] = True
            valves_out.append(v)
            info["valves"] += 1
    # collectors (vertical) with tees at every row, and the back lane at 3.6 m
    zs_l = sorted([0.3] + rows_v + [3.6])
    zs_r = sorted([0.3] + rows_v)
    for i in range(len(zs_l) - 1):
        net.pipe((XL, YW, zs_l[i]), (XL, YW, zs_l[i + 1]), 0.07, "steel_b", name=f"colL{i}")
    for i in range(len(zs_r) - 1):
        net.pipe((XR, YW, zs_r[i]), (XR, YW, zs_r[i + 1]), 0.05, "steel_b", name=f"colR{i}")
    net.run(np.array([-3.0, YW, 3.6]), np.array([3.0, YW, 3.6]), 0.1, bands=[(u(rng, 0.5, 5.5), "yellow")], name="lane_back", taps=[XL + 3.0, XR + 3.0])
    for k, m in risers:
        net.pipe((m, YW, rows_v[k]), (m, YW, rows_v[k + 1]), float(rng.choice([0.015, 0.02, 0.025])), "steel", name=f"rs{k}_{nid(m)}")
        info["risers"] += 1
    # gauges (dial facing the camera) and junction boxes at the mid-points between valves, per row
    gi = bi = 0
    for k, z in enumerate(rows_v):
        r = rr[k]
        for m in mids:
            if any(abs(m - t) < 0.05 for t in riser_taps[k]) or (abs(k - CR) <= 1 and any(abs(m - c) < 0.5 for c in ctl)) or (k == 5 and any(abs(m - b) < 0.12 for b in branch_x)):
                continue
            q = rng.random()
            if q < 0.6:
                EQ.gauge(ctx, f"Gauge_W{gi}", (m, YW, z), -math.pi / 2, "L-W", 0.09, mount="top")
                gi += 1
            elif q < 0.97:
                EQ.instrument_box(ctx, f"InstBox_W{bi}", (m, YW, z + r + 0.1), 0.0, (0.1, 0.08, 0.16), str(rng.choice(["gray_box", "cab_blue", "steel_c", "red_jb"])))
                bi += 1
    info["gauges"], info["boxes"] = gi, bi
    # instruments on the vertical collectors (dials / boxes facing the camera)
    for zc in (0.9, 1.45, 2.05, 2.6):
        EQ.instrument_box(ctx, f"InstBox_CL{int(zc * 10)}", (XL, YW - 0.07 - 0.08, zc), 0.0, (0.1, 0.08, 0.14), str(rng.choice(["gray_box", "cab_blue"])))
    for zc in (0.65, 0.95, 1.3, 1.6, 1.95, 2.25, 2.6):
        EQ.gauge(ctx, f"Gauge_CR{int(zc * 10)}", (XR, YW, zc), -math.pi / 2, "L-W", 0.05, mount="side")
    # overhead Y-direction branch pipes over the aisle (hung from the rack cross beams at x = -1.55 / 0 / 1.55)
    for x in branch_x:
        net.pipe((x, YW, 2.9), (x, -0.9, 2.9), 0.045, "steel", bands=[(0.8, label_for(ctx, label_cols))], name=f"yb{nid(x)}", support=("up", 3.4))
        net.pipe((x, -0.9, 2.9), (x, -0.9, 3.3), 0.045, "steel", name=f"yb{nid(x)}_up")
    return info, rows_v, rr


def build_left(ctx: Ctx, net: PipeNet, label_cols, valves_out):
    """Left cluster: a header along -Y from the wall's bottom row with vertical-stem gate valves (wheels 0.3-0.4 m at
    1.0-1.4 m) and two red control valves, gauges and boxes."""
    rng = ctx.rng
    r = 0.08
    z = 0.5
    a, b = np.array([XL, YW, z]), np.array([XL, -1.55, z])
    spec = [("gate", 1.25), ("ctl", 0.7), ("gate", 0.12), ("ctl", -0.45), ("gate", -1.0)]
    vl = []
    for kind, y in spec:
        y += u(rng, -0.05, 0.05)
        L = (0.14 + 1.5 * r) if kind == "gate" else (0.3 + 1.7 * r)
        vl.append((kind, y, L))
    net.run(a, b, r, valves=[(YW - y, L) for _, y, L in vl], bands=[(u(rng, 0.2, 1.0), label_for(ctx, label_cols)), (u(rng, 1.3, 2.4), label_for(ctx, label_cols))], name="hdrL", support="down", service="Left manifold")
    ctx.lines["L-L"] = "Left manifold"
    for i, (kind, y, L) in enumerate(vl):
        if kind == "gate":
            wh = u(rng, 1.0, 1.4)
            v = EQ.gate_valve(ctx, f"GateValve_L{i}", (XL, y, z), "y", r, wh, u(rng, 0.3, 0.4), int(rng.integers(5, 9)), u(rng, 0.08, 0.13), "L-L", str(rng.choice(WHEEL_MATS)))
        else:
            v = EQ.control_valve(ctx, f"ControlValve_L{i}", (XL, y, z), "y", r, "L-L")
        v["header"] = "L"
        valves_out.append(v)
    net.pipe((XL, -1.55, z), (XL, -1.55, 2.3), r, name="hdrL_up")
    # gauges / boxes between valves (only where the gap clears both neighbouring valve bodies)
    vs = sorted((y, L) for _, y, L in vl)
    for i in range(len(vs) - 1):
        ym = (vs[i][0] + vs[i + 1][0]) / 2
        gap = (vs[i + 1][0] - vs[i][0]) - (vs[i][1] + vs[i + 1][1]) / 2
        if gap < 0.28:
            continue
        if i % 2 == 0:
            EQ.gauge(ctx, f"Gauge_L{i}", (XL, ym, z + r - 0.01), 0.0, "L-L", 0.09, mount="top")
        else:
            EQ.instrument_box(ctx, f"InstBox_L{i}", (XL, ym, z + r + 0.1), 0.0, (0.1, 0.1, 0.16), "gray_box")


def build_rack(ctx: Ctx, net: PipeNet, special, label_cols):
    """Horizontal rack runs at 3.6-4.6 m (diameters 0.1 - 0.5 m) on tiers; special lanes get taps for equipment drops."""
    rng = ctx.rng
    sizes = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
    lanes = [dict(y=s["y"], r=s["r"], mat=s.get("mat", "insul"), taps=s["taps"]) for s in special] + [dict(y=-1.45, r=0.25, mat="insul", taps=[])]
    y = -1.75
    while y < 1.55 and len(lanes) < 11:
        r = float(rng.choice(sizes, p=[0.2, 0.2, 0.2, 0.2, 0.12, 0.08]))
        y += r + u(rng, 0.02, 0.18)
        if y + r > 1.6:
            break
        if any(abs(y - l["y"]) < r + l["r"] + 0.07 for l in lanes):
            y += 0.05
            continue
        lanes.append(dict(y=y, r=r, mat=str(rng.choice(STEELS)), taps=[]))
        y += r
    lanes.sort(key=lambda l: l["y"])
    for i, l in enumerate(lanes):
        tier = i % 3 == 1
        l["z"] = (4.35 + l["r"]) if tier else (3.5 + l["r"] + 0.02)
        nb = int(rng.integers(1, 4))
        bands = [(u(rng, 0.4, 5.6), label_for(ctx, label_cols)) for _ in range(nb)]
        net.run(np.array([-3.15, l["y"], l["z"]]), np.array([3.15, l["y"], l["z"]]), l["r"], mat=l["mat"], bands=bands, name=f"rack{i}", taps=[x + 3.15 for x in l["taps"]] + [0.15], service=SERVICES[i % len(SERVICES)])
        for xs in (-1.55, 0.0, 1.55):  # saddle posts for the lower tier
            if not tier:
                continue
            ctx.static(f"{net.group}/rack{i}_post{nid(xs)}", (xs, l["y"], 3.5 + (l["z"] - l["r"] - 3.5) / 2), Q_IDENT, [cyl(0.03, l["z"] - l["r"] - 3.5, mat="steel_dark")])
        if l["r"] >= 0.1:
            ctx.static(f"{net.group}/rack{i}_tag", (u(rng, -2.5, 2.5), l["y"] - l["r"] - 0.003, l["z"]), Q_IDENT, [cube((0.22, 0.005, min(0.1, l["r"] * 1.2)), (0, 0, 0), label_cols[0], col=False)])
    # backbone header along Y at x = -3.0 (z 3.3, below the lanes) tying every lane (and the back lane) into one network
    ys_ = [l["y"] for l in lanes] + [YW]
    for i in range(len(ys_) - 1):
        net.pipe((-3.0, ys_[i], 3.3), (-3.0, ys_[i + 1], 3.3), 0.06, "steel_b", name=f"bb{i}", service="backbone")
    for l in lanes:
        net.pipe((-3.0, l["y"], 3.3), (-3.0, l["y"], l["z"]), min(0.04, l["r"]), "steel_b", name=f"bbs{nid(l['y'])}")
    net.pipe((-3.0, YW, 3.3), (-3.0, YW, 3.6), 0.05, "steel_b", name="bb_back")
    return lanes


def build_tubing_bundles(ctx: Ctx, rows_v, n=44):
    """Visual-only small-bore tubing/cable bundles along the wall between rows (3-5 parallel tubes, tie rings, end fittings)."""
    rng, k = ctx.rng, ctx.kit
    pal = ["tube", "black", "cable_red", "cable_orange", "cable_green", "cable_blue", "tube"]
    for i in range(n):
        kz = int(rng.integers(0, 5))
        z0 = rows_v[kz] + 0.17 + u(rng, -0.02, 0.02)
        x0 = u(rng, XL + 0.3, XR - 1.3)
        L = u(rng, 0.7, 1.3)
        m = int(rng.integers(3, 6))
        sh = []
        for j in range(m):
            sh.append(cyl(0.006, L, (x0 + L / 2, YW - 0.075 - 0.0 + 0.012 * (j % 2), z0 + 0.012 * j), pal[int(rng.integers(len(pal)))], col=False, axis="x"))
        for tx in np.arange(x0 + 0.1, x0 + L, 0.3):
            sh.append(cyl(0.028, 0.012, (tx, YW - 0.075 + 0.006, z0 + 0.024), "black", col=False, axis="x"))
        sh.append(cyl(0.006, 0.17, (x0, YW - 0.075 + 0.006, z0 - 0.085), "tube", col=False))
        sh.append(cyl(0.006, 0.17, (x0 + L, YW - 0.075 + 0.006, z0 - 0.085), "tube", col=False))
        ctx.static(f"/World/Instruments/TubingBundle_{i}", (0, 0, 0), Q_IDENT, sh, category="tubing_bundle")


def lookat_quat(pos, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    if abs(f[2]) > 0.99:
        up = (0, 1, 0)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    uu = np.cross(r, f)
    return qfrom_mat(np.column_stack([r, uu, -f]))


def build_camera(kit, path, pos, target, focal, hap=36.0, vap=24.0, clip=(0.1, 500.0)):
    cam = UsdGeom.Camera.Define(kit.stage, path)
    cam.CreateFocalLengthAttr(float(focal))
    cam.CreateHorizontalApertureAttr(float(hap))
    cam.CreateVerticalApertureAttr(float(vap))
    cam.CreateClippingRangeAttr(Gf.Vec2f(*clip))
    kit.set_pose(cam.GetPrim(), pos, lookat_quat(pos, target))
    return cam


def build_lights(ctx: Ctx, idx, light_scale=1.0):
    k, rng = ctx.kit, ctx.rng
    k.scope("/World/Lights")
    info = {"setup_index": idx, "setup_name": LIGHTING[idx]}
    if idx in (0, 1):
        dome = UsdLux.DomeLight.Define(k.stage, "/World/Lights/Sky")
        dome.CreateTextureFileAttr(Sdf.AssetPath("textures/sky.jpg"))
        dome.CreateTextureFormatAttr("latlong")
        dome.CreateIntensityAttr(float((1800.0 if idx == 0 else 1900.0) * u(rng, 0.9, 1.1) * light_scale))
        dome.CreateEnableColorTemperatureAttr(False)
        sun = UsdLux.DistantLight.Define(k.stage, "/World/Lights/Sun")
        az = u(rng, -2.5, -1.9)  # sun in front-left of the deck: lights the valve wall, shadows fall backwards
        el = u(rng, 0.7, 0.95) if idx == 0 else 1.2
        d = -np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        k.set_pose(sun.GetPrim(), (0, 0, 10), qalign(-d))
        sun.CreateAngleAttr(1.0 if idx == 0 else 6.0)
        sun.CreateIntensityAttr(float((4200.0 if idx == 0 else 500.0) * light_scale))
        sun.CreateColorTemperatureAttr(5900.0 if idx == 0 else 6800.0)
        sun.CreateEnableColorTemperatureAttr(True)
        info.update(dome_intensity=dome.GetIntensityAttr().Get(), sun_intensity=sun.GetIntensityAttr().Get(), sun_dir=[round(float(c), 3) for c in d])
    else:
        dome = UsdLux.DomeLight.Define(k.stage, "/World/Lights/Ambient")
        dome.CreateColorAttr(Gf.Vec3f(0.55, 0.6, 0.7))
        dome.CreateIntensityAttr(float(260.0 * light_scale))
        n = 0
        for i, x in enumerate((-4.0, 0.0, 4.0)):
            for j, y in enumerate((-2.0, 2.0)):
                L = UsdLux.RectLight.Define(k.stage, f"/World/Lights/Fixture_{i}_{j}")
                L.CreateWidthAttr(1.2)
                L.CreateHeightAttr(0.3)
                L.CreateIntensityAttr(float(60000.0 * u(rng, 0.8, 1.1) * light_scale))
                L.CreateColorTemperatureAttr(4200.0)
                L.CreateEnableColorTemperatureAttr(True)
                k.set_pose(L.GetPrim(), (x, y, 5.3))
                n += 1
        info.update(dome_intensity=260.0, fixtures=n)
    return info


# --------------------------------------------------------------------------- main build
def build(seed: int, out_root: Path, lighting: int | None = None, light_scale: float = 1.0) -> Path:
    t0 = time.time()
    rng = np.random.default_rng(7000 + seed)
    out = Path(out_root) / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    usd_path = out / "plant.usd"
    if usd_path.exists():
        usd_path.unlink()
    idx = (seed % 3) if lighting is None else lighting
    dirt = u(rng, 0.2, 0.85)
    kit = Kit(usd_path)
    kit.ns = "plant"
    ctx = Ctx(kit, rng, seed, dirt)
    ctx.extra = {}
    kit.physics_scene()
    for s in ("Equipment", "Pipes", "Instruments", "Structure", "Loose", "Hoses", "Ground", "Cameras"):
        kit.xform(f"/World/{s}")
    build_materials(ctx, out, idx)
    label_cols = list(rng.choice(["yellow", "yellow", "yellow", "orange_band", "green_band", "white_band", "blue_band"], 3))
    label_cols[0] = "yellow"

    drain = (0.45 + u(rng, -0.3, 0.3), -1.2 + u(rng, -0.1, 0.1))  # left of the pump so the corridor in front of it stays open
    build_deck(ctx, drain)
    build_ground(ctx)
    build_structure(ctx)
    build_background(ctx)

    net = PipeNet(ctx, "/World/Pipes/net")
    valves = []
    # ---- machine side (right): pump set, cabinet, stainless box
    S_PUMP = 0.62
    pump_pos = (1.65 + u(rng, -0.1, 0.1), 0.25 + u(rng, -0.08, 0.08))
    pump = EQ.pump_set(ctx, pump_pos, 0.0, S_PUMP)
    W = pump["W"]
    cab_x = 2.45 + u(rng, -0.02, 0.02)
    cab = EQ.enclosure(ctx, "BlueCabinet", (cab_x, 1.55, 0.0), math.pi, (0.6, 0.4, 1.7), "cab_blue", door=True, latch=True, louvers=True, note="blue cabinet", category="cabinet")
    sb = (2.8, -0.9 + u(rng, -0.1, 0.1))
    for sy in (-0.2, 0.2):
        ctx.static(f"/World/Equipment/StainlessStand_{nid(sy, 10)}", (sb[0], sb[1] + sy, 0.0), Q_IDENT, [cyl(0.035, 0.85, (0, 0, 0.425), "steel_c")])
    sbox = EQ.enclosure(ctx, "StainlessBox", (sb[0], sb[1], 0.85), math.pi / 2, (0.5, 0.26, 0.65), "stainless", door=True, latch=True, note="stainless steel electrical box", category="electrical_box")
    ctx.static("/World/Equipment/StainlessConduit", (sb[0], sb[1], 1.82), Q_IDENT, [cyl(0.03, 0.62, (0, 0, 0), "steel_c")])

    # ---- centre: instrument panel + worker
    panel_cx = -0.15 + u(rng, -0.1, 0.1)
    panel = EQ.instrument_panel(ctx, (panel_cx, 1.75), math.pi, manifold=False)
    for i, gx in enumerate((-0.21, 0.21)):
        EQ.gauge(ctx, f"Gauge_panel_{i}", panel["W"](gx, 0.07, 1.4), -math.pi / 2, "L-INST", 0.0, mount="side")
    reach = panel["W"](-0.84, 0.3, 1.7)  # just right of the panel frame (panel +x is world -x), clear of every door swing
    wpos = (panel_cx + 1.15 + u(rng, -0.06, 0.06), 1.1 + u(rng, -0.03, 0.03))
    human = EQ.worker(ctx, wpos, math.atan2(reach[1] - wpos[1], reach[0] - wpos[0]), reach)

    # ---- connected pipe network
    wall_info, rows_v, rr = build_wall(ctx, net, panel_cx, label_cols, valves)
    build_left(ctx, net, label_cols, valves)
    nx, ny, nz = [float(c) for c in pump["nozzle_top"]]
    sx, sy, sz = [float(c) for c in pump["suction"]]
    sy2 = sy - 0.35
    lanes = build_rack(ctx, net, [dict(y=ny, r=pump["discharge_r"], taps=[nx]), dict(y=sy2, r=pump["suction_r"], taps=[sx])], label_cols)
    ld = next(l for l in lanes if abs(l["y"] - ny) < 1e-6)
    ls = next(l for l in lanes if abs(l["y"] - sy2) < 1e-6)
    nz += 0.006
    net.equip((nx, ny, nz), "pump_discharge_nozzle")
    net.equip((sx, sy, sz), "pump_suction_nozzle")
    net.pipe((nx, ny, nz), (nx, ny, ld["z"]), pump["discharge_r"], "insul", bands=[(t, "steel_c") for t in np.arange(0.3, 2.4, 0.4)] + [(1.2, label_cols[0])], name="pump_out", service="pump discharge")
    net.pipe((sx, sy, sz), (sx, sy2, sz), pump["suction_r"], "insul", name="pump_in_h", service="pump suction")
    net.pipe((sx, sy2, sz), (sx, sy2, ls["z"]), pump["suction_r"], "insul", bands=[(t, "steel_c") for t in np.arange(0.3, 2.6, 0.4)], name="pump_in_v", service="pump suction")
    fit = net.finish()
    build_tubing_bundles(ctx, rows_v)

    # ---- flexible hoses (dynamic chains): green from the baseplate to a post, black between the two insulated risers
    ha = W((0.5, 0.78, 0.6))
    hb = np.array([pump_pos[0] - 0.55, pump_pos[1] + 1.1, 0.5])
    dh = (hb - ha) * np.array([1, 1, 0])
    dh /= np.linalg.norm(dh)
    ha2, hb2 = ha + dh * 0.16, hb - dh * 0.16
    hoses = [EQ.hose(ctx, "GreenHose", ha2, hb2, 0.026, "green_band", n=9, slack=1.2)]
    for nm, p0_, p1_ in (("HoseStubA", ha, ha2), ("HoseStubB", hb, hb2)):
        ctx.static(f"/World/Hoses/{nm}", (p0_[0], p0_[1], p0_[2] / 2), Q_IDENT, [cyl(0.035, p0_[2], mat="steel_c")])
        ln = float(np.linalg.norm(p1_ - p0_))
        ctx.static(f"/World/Hoses/{nm}_arm", ((p0_[0] + p1_[0]) / 2, (p0_[1] + p1_[1]) / 2, p0_[2]), qalign(p1_ - p0_), [cyl(0.018, ln - 0.09, (0, 0, -0.045), mat="steel_c"), cyl(0.024, 0.09, (0, 0, (ln - 0.09) / 2), mat="steel_c", col=False)])
    rd, rs_ = pump["discharge_r"], pump["suction_r"]
    hc = (nx, ny - rd - 0.05, 1.55)
    hd = (sx, sy2 + rs_ + 0.05, 1.25)
    hoses.append(EQ.hose(ctx, "BlackHose", hc, hd, 0.02, "black", n=8, slack=1.3))
    EQ.gauge(ctx, "Gauge_pump_discharge", (nx, ny, 2.0), -math.pi / 2, "L-300", rd, mount="side")
    EQ.gauge(ctx, "Gauge_pump_suction", (sx, sy2, 1.7), math.pi, "L-301", rs_, mount="side")

    # ---- loose clutter on the deck (dynamic); each item must keep >= 97% of the walkable (robot-inflated) area connected
    from scipy.ndimage import label as _label

    cols0 = collect(kit.stage)
    occ0 = NV.build_occ(cols0, [(drain[0] - 0.4, drain[1] - 0.4, drain[0] + 0.4, drain[1] + 0.4)])
    xs_g = NV.X0 + (np.arange(NV.NX) + 0.5) * NV.RES
    ys_g = NV.Y1 - (np.arange(NV.NY) + 0.5) * NV.RES
    GX, GY = np.meshgrid(xs_g, ys_g)

    def largest(o):
        lab, _ = _label(NV.free_mask(o))
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        return int(sz.max())

    base = largest(occ0)
    spots, tries, occ_now = [], 0, occ0.copy()
    n_loose = int(rng.integers(6, 10))
    while len(spots) < n_loose and tries < 400:
        tries += 1
        x, y = u(rng, -2.0, 1.4), u(rng, -1.65, 0.7)
        if math.hypot(x - wpos[0], y - wpos[1]) < 0.65 or any(math.hypot(x - sx_, y - sy_) < 0.6 for sx_, sy_, _ in spots):
            continue
        if math.hypot(x - drain[0], y - drain[1]) < 0.7 or (x > 0.2 and y > -0.8):
            continue
        cand = occ_now | (np.hypot(GX - x, GY - y) < 0.22)
        if largest(cand) < 0.97 * base:
            continue
        occ_now = cand
        spots.append((x, y, u(rng, -math.pi, math.pi)))
    loose = EQ.loose_items(ctx, spots)

    # ---- lights, cameras
    lights = build_lights(ctx, idx, light_scale)
    cam_pos = np.array([-2.45, -1.95, 1.6])
    tgt = (0.35, 1.5, 1.6)
    FOCAL = 35.0
    build_camera(kit, "/World/Cameras/PhotoView", cam_pos, tgt, FOCAL)
    build_camera(kit, "/World/Cameras/Overhead", (0.0, 0.0, 8.0), (0.0, 0.0, 0.0), 24.0, clip=(1.0, 500.0))
    cams = [dict(name="photo_view", path="/World/Cameras/PhotoView", position=cam_pos.tolist(), target=list(tgt), focal_mm=FOCAL, eye_height_m=1.6, distance_to_valve_wall_m=round(YW - cam_pos[1], 2),
                 note="reference-photo viewpoint: 1.6 m eye height, 4 m from the valve-wall pipes; 35 mm full-frame-equivalent (50 mm cannot frame a 1.75 m worker at 3 m: vertical FOV 27 deg)"),
            dict(name="overhead", path="/World/Cameras/Overhead", position=[0, 0, 8.0], note="top-down; structure/rack hidden for a clear view")]

    rep_ = net.report()
    ctx.extra["pipe_network"] = rep_
    ctx.extra["wall"] = wall_info
    wheels = [v["wheel_diameter"] for v in valves if v["kind"] == "gate_valve"]
    ctx.extra["scale_check"] = dict(
        handwheel_diameter_m=[round(min(wheels), 3), round(max(wheels), 3)], handwheel_target_m=[0.2, 0.4], pipe_diameter_m=rep_["diameter_range_m"], pipe_target_m=[0.03, 0.5],
        worker_height_m=human["height_m"], worker_target_m=1.75, deck_m=[6, 4], camera=dict(eye_height_m=1.6, distance_to_valve_wall_m=round(YW - cam_pos[1], 2), focal_mm=FOCAL))
    recs = [dict(id=e["name"], category="pipe_segment", path=e["path"], radius=e["r"], length=round(e["length"], 3), from_xyz=[c / 1000 for c in e["a"]], to_xyz=[c / 1000 for c in e["b"]], service=e["service"]) for e in net.edges]
    kit.save()
    stage = Usd.Stage.Open(str(usd_path))
    cols = collect(stage)
    finish(ctx, stage, cols, out, seed, idx, dirt, lights, cams, recs, valves, pump, cab, sbox, panel, human, hoses, loose, drain, label_cols, t0, wpos, (panel_cx, 1.75))
    return out


def finish(ctx, stage, cols, out, seed, idx, dirt, lights, cams, recs, valves, pump, cab, sbox, panel, human, hoses, loose, drain, label_cols, t0, wpos, panel_xy):
    kit = ctx.kit
    drain_rect = [(drain[0] - 0.4, drain[1] - 0.4, drain[0] + 0.4, drain[1] + 0.4)]
    occ = NV.build_occ(cols, drain_rect)
    free = NV.free_mask(occ)
    spawn_cell, comp = NV.pick_spawn(free)
    sx, sy = NV.g2w(*spawn_cell)
    spawn = (round(sx, 3), round(sy, 3), round(math.pi / 2, 4))
    # ---- goal targets
    G = []
    n = 0

    def add(gtype, name, tgt_pt, joint=None, delta=None, unit=None, normal=None, target_path=None, **extra):
        nonlocal n
        pose = NV.approach_pose(free, comp, occ, tgt_pt[:2], normal)
        G.append(dict(id=n, name=name, type=gtype, target_point=[round(float(c), 4) for c in tgt_pt], target_path=target_path, joint=joint, joint_delta=delta, unit=unit,
                      base_pose=pose, **extra))
        n += 1

    for v in valves:
        if v["kind"] == "gate_valve" and v["wheel_height"] > 1.75:
            continue  # above comfortable manipulator reach: kept as scenery, no goal
        if v["kind"] == "gate_valve":
            a = np.array(v["wheel_center"], float)
            nrm = tuple(v["stem_axis_world"][:2]) if abs(v["stem_axis_world"][2]) < 0.5 else None
            add("turn_handwheel", f"turn_{v['name']}", np.array(v["rim_point"]), joint=v["wheel_joint"], delta=2.0 * 360.0, unit="deg", normal=nrm,
                target_path=f"/World/Equipment/{v['name']}/wheel", expected_stem_rise_m=round(abs(v["gearing_m_per_deg"]) * 720.0, 5), valve=v["name"], controls_line=v["line"],
                wheel_height=v["wheel_height"])
        else:
            add("actuate_control_valve", f"actuate_{v['name']}", np.array(v["stem_point"]), joint=v["stem_joint"], delta=v["stroke"], unit="m",
                target_path=f"/World/Equipment/{v['name']}/stem", valve=v["name"], controls_line=v["line"])
    for name, e in panel["items"].items():
        art = e["art"]
        nrm_xy = (e["front_normal"][0], e["front_normal"][1])
        if e["kind"] == "control_station":
            for bi, (bj, bw) in enumerate(zip(e["buttons"], e["button_world"])):
                add("press_button", f"press_{name}_btn{bi}", np.array(bw), joint=bj, delta=0.008, unit="m", normal=nrm_xy, target_path="/World/Equipment/" + bj.replace("/J_", "/"))
        else:
            if "door" in e["joints"]:
                add("open_door", f"open_{name}_door", np.array(e["door_handle"]), joint=e["joints"]["door"], delta=90.0, unit="deg", normal=nrm_xy, target_path=f"/World/Equipment/{name}/door")
            if "drawer" in e["joints"]:
                add("pull_drawer", f"pull_{name}_drawer", np.array(e["drawer_point"]), joint=e["joints"]["drawer"], delta=0.15, unit="m", normal=nrm_xy, target_path=f"/World/Equipment/{name}/drawer")
    for nm, e in (("BlueCabinet", cab), ("StainlessBox", sbox)):
        add("open_door", f"open_{nm}_door", np.array(e["door_handle"]), joint=e["joints"]["door"], delta=90.0, unit="deg", normal=(e["front_normal"][0], e["front_normal"][1]), target_path=f"/World/Equipment/{nm}/door")
    for g in ctx.gauges:
        add("inspect_gauge", f"inspect_{g['name']}", np.array(g["position"]), normal=(g["normal"][0], g["normal"][1]), target_path=g["path"], line=g["line"], view_height=g["position"][2],
            required_view_distance_m=0.8)
    add("inspect_machine", "inspect_pump_rotor", np.array(pump["W"]((1.28, 0, 0.74))), target_path="/World/Equipment/PumpSet/rotor", joint=None, note="check coupling spins (360 deg/s)")
    goals = [g for g in G if g["base_pose"] is not None]
    dropped = [g["name"] for g in G if g["base_pose"] is None]
    # reachability with paths (for the occupancy PNG)
    cells = [NV.w2g(g["base_pose"][0], g["base_pose"][1]) for g in goals]
    dist, prev = NV.dijkstra(free, spawn_cell, cells)
    paths = {}
    for g, c in zip(goals, cells):
        g["path_length_m"] = round(float(dist[c] * NV.RES), 2) if dist[c] < 1e17 else None
        if dist[c] < 1e17:
            p, cur = [], c
            while cur != spawn_cell:
                p.append(cur)
                cur = tuple(prev[cur])
            paths[g["name"]] = p
    NV.render_png(out / "nav_occupancy.png", occ, free, spawn_cell, [g["base_pose"] for g in goals], paths)
    for i, g in enumerate(goals):
        g["id"] = i
    # robot placeholder + camera (authored into the stage)
    robot = "/World/Robot"
    xf = UsdGeom.Xform.Define(kit.stage, robot)
    Kit.set_pose(xf.GetPrim(), (spawn[0], spawn[1], 0.0), qyaw(spawn[2]))
    kit.cube(f"{robot}/chassis_placeholder", (0.7, 0.5, 0.3), (0, 0, 0.25), vmat=ctx.mats["cab_blue"], guide=True)
    kit.cylinder(f"{robot}/mast_placeholder", 0.04, 0.9, (0.1, 0, 0.85), vmat=ctx.mats["steel_c"], guide=True)
    build_camera(kit, f"{robot}/RobotCamera", (0.3, 0.0, 1.25), (1.3, 0.0, 1.25), 18.0)
    kit.save()
    cams.append(dict(name="robot", path=f"{robot}/RobotCamera", mounted_on=robot, position_in_robot_frame=[0.3, 0.0, 1.25], focal_mm=18.0))
    graph = scene_graph(ctx, seed_dir=out, seed=ctx.seed, idx=idx, dirt=dirt, lights=lights, cams=cams, recs=recs, valves=valves, pump=pump, cab=cab, sbox=sbox, panel=panel, human=human,
                        hoses=hoses, loose=loose, drain=drain, label_cols=label_cols, spawn=spawn, goals=goals, dropped=dropped, t0=t0, n_cols=len(cols))
    (out / "scene_graph.json").write_text(json.dumps(graph, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return graph


def scene_graph(ctx, seed_dir, seed, idx, dirt, lights, cams, recs, valves, pump, cab, sbox, panel, human, hoses, loose, drain, label_cols, spawn, goals, dropped, t0, n_cols):
    stage = ctx.kit.stage
    objs = []
    for art in ctx.arts:
        objs.append(dict(id=art.name, category=art.category, path=art.group, body_type="articulation_fixed_base", position=[round(float(c), 4) for c in art.pos], yaw_deg=round(math.degrees(art.yaw), 2),
                         orientation_wxyz=[round(float(c), 5) for c in art.q], mass_kg=round(sum(l["mass"] for l in art.links.values()), 3),
                         links={k: v["path"] for k, v in art.links.items()}, joints=[j for j in art.joint_names]))
    for v in valves:
        o = next(o for o in objs if o["id"] == v["name"])
        o["valve"] = {k: v[k] for k in v if k not in ("name",)}
    for h in hoses:
        objs.append(dict(id=h["name"], category="flexible_hose", path=f"/World/Hoses/{h['name']}", body_type="dynamic_chain", segments=h["segments"], radius=h["radius"], length_m=round(h["length"], 3),
                         anchors=[h["anchor_a"], h["anchor_b"]], mass_kg=round(sum(float(stage.GetPrimAtPath(s).GetAttribute("physics:mass").Get() or 0) for s in h["segments"]), 3),
                         joint_info="spherical joints, cone limit 55 deg, anchored at both ends"))
    for l in loose:
        objs.append(dict(id=l["path"].split("/")[-1], category="loose_item", kind=l["kind"], path=l["path"], body_type="dynamic", mass_kg=l["mass"], position=l["pos"], yaw_rad=round(l["yaw"], 3)))
    objs.append(dict(id="Worker", category="human_worker", path="/World/Worker", body_type="static_collider", position=human["pos"], yaw_rad=round(human["yaw"], 3), reaching_hand=human["hand"]))
    for r in recs:
        r = dict(r)
        r["body_type"] = "static_collider"
        objs.append(r)
    for g in ctx.gauges:
        objs.append(dict(id=g["name"], category="gauge", path=g["path"], body_type="static_collider", position=g["position"], normal=g["normal"], line=g["line"]))
    objs.append(dict(id="Deck", category="grating_deck", path="/World/Deck", body_type="static_collider", size=[12, 8, 0.1], friction=0.7, drain_opening=dict(center=list(map(float, drain)), size=[0.6, 0.6]),
                     note="flat box collision + visual-only grating texture"))
    lines = {}
    for v in valves:
        lines.setdefault(v["line"], dict(service=ctx.lines.get(v["line"]), valves=[]))["valves"].append(v["name"])
    valve_map = []
    for v in valves:
        hy = v["pos"][1]
        down = [g["name"] for g in ctx.gauges if g["line"] == v["line"] and g["position"][1] > hy]
        valve_map.append(dict(valve=v["name"], kind=v["kind"], line=v["line"], service=ctx.lines.get(v["line"]), handwheel_joint=v.get("wheel_joint"), stem_joint=v["stem_joint"],
                              effect=("isolates" if v["kind"] == "gate_valve" else "throttles") + " flow in the line; downstream gauges: " + (", ".join(down) if down else "none"),
                              downstream_gauges=down, stem_rise_per_turn_m=None if v["kind"] != "gate_valve" else round(abs(v["gearing_m_per_deg"]) * 360, 5)))
    return {
        "seed": seed, "generator": "claude_scene_generator.plant.gen", "units": "m", "up_axis": "Z", "usd": "plant.usd", "generation_seconds": round(time.time() - t0, 1),
        "randomization": dict(dirt=round(dirt, 3), lighting=lights["setup_name"], label_colors=label_cols, n_gate_valves=sum(v["kind"] == "gate_valve" for v in valves),
                              n_control_valves=sum(v["kind"] == "control_valve" for v in valves), n_loose_items=len(loose), drain_xy=list(map(float, drain))),
        "lighting": lights, "robot": dict(spawn=list(spawn), spawn_note="x, y, yaw(rad); z=0 deck", prim="/World/Robot", camera="/World/Robot/RobotCamera", radius_m=NV.ROBOT_R),
        "cameras": cams, "goals": goals, "dropped_goals_no_reachable_pose": dropped, "valve_control_map": valve_map, "lines": ctx.lines,
        "objects": objs, "joints": ctx.joints, "articulations": [dict(name=a.name, root=a.links["root"]["path"], category=a.category, joints=a.joint_names) for a in ctx.arts],
        "pipe_network": ctx.extra.get("pipe_network"), "scale_check": ctx.extra.get("scale_check"), "valve_wall": ctx.extra.get("wall"),
        "physics": dict(steel_friction=0.6, grating_friction=0.7, hz=120, solver="TGS"), "n_colliders": n_cols, "n_shapes": ctx.n_shapes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="plant_out")
    ap.add_argument("--lighting", type=int, choices=[0, 1, 2], default=None, help="0 bright_day, 1 overcast, 2 indoor_industrial (default: seed %% 3)")
    ap.add_argument("--light-scale", type=float, default=1.0)
    a = ap.parse_args()
    d = build(a.seed, Path(a.out), a.lighting, a.light_scale)
    print("wrote", d)


if __name__ == "__main__":
    main()
