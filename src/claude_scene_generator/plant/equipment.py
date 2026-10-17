"""Plant equipment: valves, enclosures, panel, pump set, hoses, worker, loose items."""
from __future__ import annotations

import math

import numpy as np
from pxr import Sdf, UsdPhysics

from ..geom import Q_IDENT, compose, qaxis, qfrom_mat, qmat, qmul, qyaw
from ..usdkit import T, _q, _v3
from .common import Art, Ctx, Q_X, Q_Y, Sh, cap, cube, cyl, esph, qalign, sph

DEG = math.pi / 180.0


# --------------------------------------------------------------------------- pipes & small fittings
def pipe_segments(ctx: Ctx, group, a, b, r, mat="steel", cuts=(), flanges=True, bands=(), name="p"):
    """Static pipe a->b, skipping the (t0,t1) intervals (metres along the run) occupied by inline valves.
    bands: list of (t_centre, colour_material) yellow/colour bands."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    u = d / L
    q = qalign(u)
    spans, t = [], 0.0
    for c0, c1 in sorted(cuts):
        if c0 > t + 0.02:
            spans.append((t, c0))
        t = max(t, c1)
    if L > t + 0.02:
        spans.append((t, L))
    for i, (s0, s1) in enumerate(spans):
        mid = a + u * (s0 + s1) / 2
        ctx.static(f"{group}/{name}_{i}", mid, q, [cyl(r, s1 - s0, mat=mat)], category=None)
        if flanges:
            for s in (s0, s1):
                if (s <= 0.01 or s >= L - 0.01) and not any(abs(s - c) < 0.01 for c in [x for cc in cuts for x in cc]):
                    continue  # no flange at free pipe ends that meet elbows
                pos = a + u * (s + (0.0125 if s == s0 else -0.0125))
                ctx.static(f"{group}/{name}_{i}_fl{int(s * 100)}", pos, q, [cyl(r * 1.4, 0.025, mat=mat)])
    for j, (tc, bmat) in enumerate(bands):
        ctx.static(f"{group}/{name}_band{j}", a + u * tc, q, [cyl(r * 1.04, 0.12, mat=bmat, col=False)])
    return L


def elbow(ctx: Ctx, path, p, r, mat="steel"):
    ctx.static(path, p, Q_IDENT, [sph(r * 1.02, mat=mat)])


def support(ctx: Ctx, path, p, r, h_to_ground, mat="steel_dark"):
    """Saddle support: vertical post (static) from the pipe underside down to z = h_to_ground."""
    z0 = h_to_ground
    h = p[2] - r - z0
    if h > 0.05:
        ctx.static(path, (p[0], p[1], z0 + h / 2), Q_IDENT, [cyl(0.03, h, mat=mat)])


def gauge(ctx: Ctx, name, p, normal_yaw, line_id, r_pipe, mount="top"):
    """Pressure gauge. p = point on the pipe axis. mount='top': vertical stem over a horizontal pipe, dial faces the
    horizontal direction ``normal_yaw``; mount='side': horizontal stem along the dial normal (for vertical pipes)."""
    n = np.array([math.cos(normal_yaw), math.sin(normal_yaw), 0.0])
    u = np.array([0.0, 0.0, 1.0]) if mount == "top" else n
    stem_len = r_pipe + 0.07
    qu, qn = qalign(u), qalign(n)
    c = u * stem_len
    shapes = [
        cyl(0.012, stem_len, tuple(u * stem_len / 2), "steel", quat=qu),
        cyl(0.062, 0.03, tuple(c), "black", quat=qn),
        cyl(0.052, 0.004, tuple(c + n * 0.016), "gauge_white", col=False, quat=qn),
        cube((0.004, 0.04, 0.004) if mount == "top" else (0.004, 0.004, 0.04), tuple(c + n * 0.019 + np.array([0, 0, 0.01])), "black", col=False, quat=qyaw(normal_yaw)),
    ]
    ctx.static(f"/World/Instruments/{name}", p, Q_IDENT, shapes, category="gauge", line=line_id)
    dial = np.asarray(p, float) + c + n * 0.02
    ctx.gauges.append(dict(name=name, path=f"/World/Instruments/{name}", position=[round(float(x), 4) for x in dial], normal=[round(float(x), 4) for x in n], line=line_id))
    return dial


def instrument_box(ctx: Ctx, name, p, yaw, size=(0.18, 0.12, 0.22), mat="gray_box"):
    ctx.static(f"/World/Instruments/{name}", p, qyaw(yaw), [cube(size, (0, 0, size[2] / 2), mat), cube((0.05, 0.05, 0.03), (0, 0, -0.015), "steel")],
               category="instrument_box")


# --------------------------------------------------------------------------- handwheel gate valve
def gate_valve(ctx: Ctx, name, pos, pipe_dir, pipe_r, wheel_h, wheel_d, turns, stroke, line, wheel_mat="steel_dark", stem_dir=None, tag=None):
    """Gate valve on a pipe along ``pipe_dir`` ('x' | 'y' horizontal with a vertical stem, or 'z' = vertical riser with a
    horizontal stem pointing along ``stem_dir`` = (nx, ny)). ``pos`` = centre of the body on the pipe axis.
    wheel_h = absolute height of the wheel centre (for 'z' it equals pos z; the wheel sits ``hw`` away horizontally)."""
    ex = np.array({"x": (1.0, 0, 0), "y": (0, 1.0, 0), "z": (0, 0, 1.0)}[pipe_dir])
    ez = np.array([0, 0, 1.0]) if stem_dir is None else np.array([stem_dir[0], stem_dir[1], stem_dir[2] if len(stem_dir) > 2 else 0.0], float)
    ez = ez / np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    quat = qfrom_mat(np.column_stack([ex, ey, ez]))
    yaw = 0.0
    horizontal_stem = abs(ez[2]) < 0.5
    hw = (wheel_d / 2 + 0.2) if stem_dir is not None else (wheel_h - pos[2])
    pr = pipe_r
    L = 0.14 + 1.5 * pr
    zb = pr * 1.2  # bonnet centre
    bt = pr * 1.2 + pr * 0.5  # bonnet top
    R = wheel_d / 2
    shapes = [
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
            shapes.append(cyl(0.008, 0.012, (sgn * (L / 2 - 0.03 - 0.004), pr * 1.3 * math.cos(a), pr * 1.3 * math.sin(a)), "steel_dark", col=False, axis="x"))
    art = Art(ctx, name, pos, yaw, shapes, mass=18.0 + 80 * pr, category="gate_valve", quat=quat, line=line)
    mass_w = 1.2 + 6.0 * R
    ws = [cyl(0.035, 0.06, (0, 0, 0), wheel_mat), cyl(0.045, 0.012, (0, 0, 0.034), wheel_mat, col=False)]
    N = 16
    for k in range(N):
        a = 2 * math.pi * k / N
        ws.append(cube((2 * math.pi * R / N * 1.08, 0.026, 0.022), (R * math.cos(a), R * math.sin(a), 0), wheel_mat, quat=qyaw(a + math.pi / 2)))
    for k in range(3):
        ws.append(cube((2 * R - 0.02, 0.015, 0.012), (0, 0, 0), wheel_mat, quat=qyaw(k * math.pi / 3)))
    # joints: wheel revolute (multi-turn) + stem prismatic coupled by a PhysX mimic joint (rising stem)
    wj = art.link("wheel", (0, 0, hw), ws, mass_w, "revolute", axis="Z", lower=0.0, upper=turns * 360.0, friction=0.1,
                  stiffness=0.0, damping=0.8, role="handwheel", unit="deg",
                  meta=dict(turns=turns, wheel_diameter=wheel_d, controls=line))
    Ls = hw - zb + 0.06
    stem = [cyl(0.011, Ls, (0, 0, (zb - hw + 0.06) / 2), "steel"), cyl(0.02, 0.02, (0, 0, 0.06 - 0.01), "steel_dark")]
    gearing = -stroke / (turns * 360.0)
    sj = art.link("stem", (0, 0, hw), stem, 0.7, "prismatic", axis="Z", lower=0.0, upper=stroke + 0.002, stiffness=0.0, damping=0.0,
                  mimic=("wheel", gearing), role="stem", unit="m", meta=dict(stroke=stroke, driven_by=wj["name"]))
    wc = art.world((0, 0, hw))
    rim = wc + (np.array([0, 0, R]) if horizontal_stem else np.array([R, 0, 0]))
    if tag:
        info_tag = tag
    info = dict(name=name, kind="gate_valve", pos=[float(c) for c in pos], pipe_dir=pipe_dir, length=L, wheel_height=float(wc[2]), wheel_diameter=wheel_d,
                turns=turns, stroke=stroke, wheel_joint=wj["name"], stem_joint=sj["name"], line=line, wheel_center=[float(c) for c in wc], rim_point=[float(c) for c in rim],
                gearing_m_per_deg=gearing, pipe_r=pr, stem_axis_world=[round(float(c), 4) for c in (qmat(art.q) @ np.array([0, 0, 1.0]))], wheel_reach=float(hw))
    ctx.valves.append(info)
    return info


# --------------------------------------------------------------------------- control valve with prismatic stem
def control_valve(ctx: Ctx, name, pos, pipe_dir, pipe_r, line, stroke=0.03, height=0.9):
    yaw = 0.0 if pipe_dir == "x" else math.pi / 2
    pr = pipe_r
    L = 0.3 + 1.7 * pr
    zy0 = pr * 1.5 + 0.02  # top of globe body
    act_z = zy0 + 0.30  # yoke length 0.3
    shapes = [
        cyl(pr * 1.5, 0.03, (-(L / 2 - 0.015), 0, 0), "valve_red", axis="x"),
        cyl(pr * 1.5, 0.03, ((L / 2 - 0.015), 0, 0), "valve_red", axis="x"),
        cyl(pr * 1.2, L - 0.06, (0, 0, 0), "valve_red", axis="x"),
        sph(pr * 1.5, (0, 0, pr * 0.2), "valve_red"),
        cyl(pr * 0.7, 0.14, (0, 0, pr * 1.5 + 0.06), "valve_red"),
        cyl(0.02, act_z - zy0, (0.07, 0, (zy0 + act_z) / 2), "valve_red"),
        cyl(0.02, act_z - zy0, (-0.07, 0, (zy0 + act_z) / 2), "valve_red"),
        cyl(0.17, 0.07, (0, 0, act_z + 0.035), "valve_red"),  # lower diaphragm case
        cyl(0.165, 0.06, (0, 0, act_z + 0.10), "valve_red"),  # upper case
        esph(0.165, 0.165, 0.075, (0, 0, act_z + 0.13), "valve_red"),
        cube((0.09, 0.07, 0.11), (0.0, -0.12, act_z - 0.1), "gray_box"),  # positioner
        cyl(0.012, 0.15, (0.0, -0.12, act_z - 0.19), "steel", col=False),
    ]
    for sgn in (-1, 1):
        for k in range(6):
            a = k * math.pi / 3
            shapes.append(cyl(0.009, 0.012, (sgn * (L / 2 - 0.034), pr * 1.38 * math.cos(a), pr * 1.38 * math.sin(a)), "steel_dark", col=False, axis="x"))
    art = Art(ctx, name, pos, yaw, shapes, mass=60 + 100 * pr, category="control_valve", line=line)
    stem = [cyl(0.008, act_z - zy0 + 0.04, (0, 0, -(act_z - zy0) / 2 + 0.02 - 0.0), "steel"),
            cube((0.05, 0.012, 0.035), (0.0, -0.035, 0.0), "yellow", col=False)]
    sj = art.link("stem", (0, 0, act_z), stem, 0.6, "prismatic", axis="Z", lower=0.0, upper=stroke, stiffness=2.0e4, damping=300.0, target=0.0,
                  role="control_stem", unit="m", meta=dict(stroke=stroke, controls=line))
    info = dict(name=name, kind="control_valve", pos=[float(c) for c in pos], pipe_dir=pipe_dir, length=L, stem_joint=sj["name"], stroke=stroke, line=line,
                top_height=float(pos[2] + act_z + 0.2), stem_point=[float(pos[0]), float(pos[1]), float(pos[2] + act_z - 0.1)], pipe_r=pr)
    ctx.valves.append(info)
    return info


# --------------------------------------------------------------------------- enclosures (doors, latches, drawers, buttons)
def enclosure(ctx: Ctx, name, pos, yaw, size, mat, door=True, latch=False, drawer=None, hinge_mat=None, category="enclosure", note="", louvers=False, door_lo=0.02):
    """Box housing whose front faces the assembly's local +Y. pos = centre of the footprint on the floor/mount level."""
    w, d, h = size
    t = 0.018
    shapes = [cube((w, d, h), (0, 0, h / 2), mat)]
    if louvers:
        for k in range(6):
            shapes.append(cube((0.003, d * 0.6, 0.012), (w / 2 + 0.0015, 0, h * 0.2 + k * 0.03), "steel_dark", col=False))
    art = Art(ctx, name, pos, yaw, shapes, mass=30 + 120 * w * d * h, category=category, note=note)
    meta = dict(housing=name)
    out = dict(name=name, kind=category, joints={})
    yf = d / 2 + 0.002
    if door:
        hd = h - 0.02 - door_lo
        door_mat = hinge_mat or mat
        ds = [cube((w, t, hd), (w / 2, -t / 2, 0), door_mat),
              cube((0.02, 0.02, min(0.14, hd * 0.3)), (w - 0.05, 0.011, 0), "steel_dark"),
              cyl(0.009, hd, (0, -t / 2, 0), "steel_dark", col=False)]
        dj = art.link("door", (-w / 2, yf + t, door_lo + hd / 2), ds, mass=1.0 + 9.0 * w * hd, jtype="revolute", axis="Z", lower=0.0, upper=110.0,
                      stiffness=6.0, damping=3.0, friction=0.05, role="door", unit="deg", meta=dict(door_width=w))
        out["joints"]["door"] = dj["name"]
        if latch:
            lj = art.link("latch", (-w / 2 + w - 0.07, yf + t + 0.016, door_lo + hd / 2), [cube((0.07, 0.012, 0.018), (0.02, 0, 0), "steel_dark"), cyl(0.014, 0.012, (0, 0, 0), "steel", axis="y")],
                          mass=0.15, jtype="revolute", axis="Y", lower=0.0, upper=90.0, parent="door", stiffness=0.3, damping=0.1, friction=0.02, role="latch", unit="deg")
            out["joints"]["latch"] = lj["name"]
    if drawer:
        dw, dh = drawer["w"], drawer["h"]
        zc = drawer["z"]
        tray_len = min(0.22, d - 0.03)
        shapes_d = [cube((dw, t, dh), (0, t / 2, 0), "steel_dark" if mat != "stainless" else "stainless"),
                    cube((dw - 0.02, tray_len, dh - 0.03), (0, -tray_len / 2 - 0.0, 0), "steel"),
                    cyl(0.008, dw * 0.6, (0, t + 0.02, 0), "steel", axis="x"),
                    cyl(0.006, 0.02, (-dw * 0.25, t + 0.01, 0), "steel", axis="y", col=False),
                    cyl(0.006, 0.02, (dw * 0.25, t + 0.01, 0), "steel", axis="y", col=False)]
        pj = art.link("drawer", (drawer["x"], yf, zc), shapes_d, mass=1.5, jtype="prismatic", axis="Y", lower=0.0, upper=drawer.get("stroke", 0.18),
                      stiffness=300.0, damping=40.0, friction=0.05, role="drawer", unit="m")
        out["joints"]["drawer"] = pj["name"]
    out["art"] = art
    out["drawer_point"] = [round(float(c), 4) for c in art.world((drawer["x"], yf + t + 0.03, drawer["z"]))] if drawer else None
    out["front_normal"] = [float(-math.sin(yaw)), float(math.cos(yaw)), 0.0]
    out["door_handle"] = None
    if door:
        hnd = art.world((-w / 2 + w - 0.05, yf + t + 0.011, door_lo + (h - 0.02 - door_lo) / 2))
        out["door_handle"] = [round(float(c), 4) for c in hnd]
    return out


def control_station(ctx: Ctx, name, pos, yaw, n_buttons=3):
    w, d, h = 0.22, 0.12, 0.34
    shapes = [cube((w, d, h), (0, 0, h / 2), "gray_box"), cube((w - 0.02, 0.006, h - 0.03), (0, d / 2 + 0.003, h / 2), "steel_dark", col=False)]
    art = Art(ctx, name, pos, yaw, shapes, mass=4.0, category="control_station")
    yf = d / 2 + 0.002
    cols = ["btn_green", "btn_red", "btn_black", "btn_green"]
    out = dict(name=name, kind="control_station", buttons=[], art=art, front_normal=[float(-math.sin(yaw)), float(math.cos(yaw)), 0.0])
    for i in range(n_buttons):
        x = -0.05 + 0.05 * (i % 2) * 2 - 0.0
        x = (-0.05, 0.05)[i % 2]
        z = h * 0.78 - 0.07 * (i // 2)
        bj = art.link(f"btn{i}", (x, yf + 0.008, z), [cyl(0.016, 0.02, (0, 0, 0), cols[i], axis="y")], mass=0.03, jtype="prismatic", axis="Y", lower=0.0, upper=0.008,
                      jrot=qyaw(math.pi), stiffness=800.0, damping=12.0, role="push_button", unit="m", meta=dict(button_index=i))
        out["buttons"].append(bj["name"])
    sj = art.link("selector", (0.0, yf + 0.012, h * 0.3), [cyl(0.026, 0.024, (0, 0, 0), "black", axis="y"), cube((0.008, 0.01, 0.03), (0, 0.012, 0.012), "white", col=False)],
                  mass=0.08, jtype="revolute", axis="Y", lower=0.0, upper=270.0, stiffness=0.4, damping=0.1, friction=0.03, role="selector_knob", unit="deg")
    out["selector"] = sj["name"]
    out["button_world"] = [[round(float(c), 4) for c in art.world((-0.05 if i % 2 == 0 else 0.05, yf + 0.02, h * 0.78 - 0.07 * (i // 2)))] for i in range(n_buttons)]
    return out


# --------------------------------------------------------------------------- instrument panel
def instrument_panel(ctx: Ctx, center, yaw=math.pi, manifold=True):
    """Unistrut frame + backplate carrying junction boxes (some with hinged doors / drawers / latches), a push-button
    station, cable bundles and a small-bore tubing manifold. Front faces local +Y (yaw=pi -> faces world -Y)."""
    rng = ctx.rng
    cx, cy = center
    q = qyaw(yaw)
    Rm = qmat(q)

    def W(x, y, z):
        return np.array([cx, cy, 0.0]) + Rm @ np.array([x, y, z])

    grp = "/World/Panel"
    ctx.kit.xform(grp)
    PW = 0.72  # half width of the frame (posts at +-PW)
    ctx.static(f"{grp}/Frame", (cx, cy, 0), q, [
        cube((0.08, 0.06, 2.5), (-PW, 0, 1.25), "steel_dark"), cube((0.08, 0.06, 2.5), (PW, 0, 1.25), "steel_dark"),
        cube((2 * PW + 0.08, 0.05, 0.06), (0, 0.0, 0.45), "steel_dark"), cube((2 * PW + 0.08, 0.05, 0.06), (0, 0.0, 2.45), "steel_dark"),
        cube((2 * PW - 0.06, 0.025, 1.45), (0, 0.045, 1.6), "panel_plate"),
        cube((2 * PW - 0.06, 0.03, 0.05), (0, 0.085, 0.84), "steel_dark"), cube((2 * PW - 0.06, 0.03, 0.05), (0, 0.085, 2.14), "steel_dark"),
        cube((0.08, 0.25, 0.06), (-PW, -0.125, 0.1), "steel_dark"), cube((0.08, 0.25, 0.06), (PW, -0.125, 0.1), "steel_dark"),
    ], category="panel_frame")
    items = {}
    # enclosure slots (x centre, z bottom, room w) - shuffled per seed
    slots_top = [(-0.42, 1.5), (0.0, 1.5), (0.42, 1.5)]
    slots_bot = [(-0.42, 0.95), (0.0, 0.95), (0.42, 0.95)]
    kinds = ["red_jb", "gray_jb", "ctrl", "drawer_box", "gray_jb2", "latch_box"]
    order = list(rng.permutation(len(kinds)))
    slots = slots_top + slots_bot
    pf = 0.0  # panel plate front at y = +0.0575; enclosures back at that plane
    y_back = 0.058
    encl = []
    for si, ki in enumerate(order):
        kind = kinds[ki]
        x0, zb = slots[si]
        x0 += float(rng.uniform(-0.015, 0.015))
        if kind == "red_jb":
            w, d, h = 0.28 + float(rng.uniform(0, 0.03)), 0.2, 0.4
            e = enclosure(ctx, "Panel_RedJB", W(x0, y_back + d / 2, zb), yaw, (w, d, h), "red_jb", door=True, latch=True, note="red junction box")
        elif kind == "gray_jb":
            w, d, h = 0.3, 0.18, 0.3
            e = enclosure(ctx, "Panel_GrayJB_A", W(x0, y_back + d / 2, zb), yaw, (w, d, h), "gray_box", door=True)
        elif kind == "gray_jb2":
            w, d, h = 0.26, 0.16, 0.26
            e = enclosure(ctx, "Panel_GrayJB_B", W(x0, y_back + d / 2, zb), yaw, (w, d, h), "gray_box", door=True)
        elif kind == "drawer_box":
            w, d, h = 0.32, 0.26, 0.3
            e = enclosure(ctx, "Panel_DrawerBox_A", W(x0, y_back + d / 2, zb), yaw, (w, d, h), "gray_box", door=False,
                          drawer=dict(x=0.0, w=w - 0.04, h=0.12, z=h * 0.62, stroke=0.16), note="instrument box with drawer")
        elif kind == "latch_box":
            w, d, h = 0.3, 0.24, 0.3
            e = enclosure(ctx, "Panel_DrawerBox_B", W(x0, y_back + d / 2, zb), yaw, (w, d, h), "gray_box", door=True, latch=True, door_lo=0.16,
                          drawer=dict(x=0.0, w=w - 0.04, h=0.12, z=0.08, stroke=0.15), note="instrument box with latch + drawer")
            # door above the drawer would overlap: shrink the door by making the drawer sit in the lower strip only
        else:
            e = control_station(ctx, "Panel_ControlStation", W(x0, y_back + 0.06, zb), yaw, n_buttons=4)
            w, d, h = 0.22, 0.12, 0.34
        e.update(slot=(x0, zb), w=w, d=d, h=h)
        encl.append(e)
        items[e["name"]] = e
    # extra small instruments on the backplate: terminal boxes at the sides, dials above/between the enclosure rows
    for i, (bx, bz) in enumerate([(-0.655, 1.1), (0.655, 1.1), (-0.655, 1.6), (0.655, 1.6)]):
        instrument_box(ctx, f"InstBox_P{i}", W(bx, y_back + 0.0, bz), yaw, (0.08, 0.07, 0.14), "gray_box" if i % 2 else "cab_blue")
    for i, (gx, gz) in enumerate([(-0.21, 2.12), (0.21, 2.12), (0.0, 2.25), (-0.42, 2.2), (0.42, 2.2)]):
        gauge(ctx, f"Gauge_P{i}", W(gx, y_back + 0.0, gz), -math.pi / 2, "L-INST", 0.0, mount="side")
    # terminal strips, label plates and DIN-rail blocks on the plate (fine detail, visual-only)
    detail = []
    for zrow in (1.38, 2.2, 0.9):
        for k in range(14):
            x = -0.62 + 0.095 * k
            detail.append(cube((0.07, 0.03, 0.05), (x, y_back + 0.0, zrow), ["gray_box", "steel_dark", "yellow", "cable_orange", "steel_c"][(k + int(zrow * 10)) % 5], col=False))
        detail.append(cube((1.35, 0.012, 0.012), (0, y_back - 0.012, zrow - 0.03), "steel_dark", col=False))
    for k, (lx, lz) in enumerate([(-0.5, 1.28), (0.5, 1.28), (-0.62, 2.0), (0.62, 2.0), (0.0, 1.28)]):
        detail.append(cube((0.1, 0.004, 0.035), (lx, y_back + 0.0, lz), "yellow" if k % 2 == 0 else "white", col=False))
    ctx.static(f"{grp}/PlateDetail", (cx, cy, 0), q, detail, category="panel_detail")
    # cable bundles: bottom-row boxes drop to the floor tray, top-row boxes run up to the overhead tray
    palette = ["black", "cable_red", "cable_orange", "cable_green", "black", "cable_blue"]
    for i, e_ in enumerate(encl):
        x0, zb = e_["slot"]
        n = int(rng.integers(2, 5))
        top_row = zb > 1.2
        for k in range(n):
            m = palette[int(rng.integers(len(palette)))]
            r = float(rng.uniform(0.008, 0.02))
            xc = x0 + (k - (n - 1) / 2) * 0.035 * (1 if e_["w"] > 0.3 else 0.5)
            yc = y_back + 0.02 + 0.03 * (k % 2) + r
            if top_row:
                z0, z1 = zb + e_["h"], 2.4
            else:
                z0, z1 = zb, (0.55 if rng.random() < 0.6 else 0.9)
            ctx.static(f"{grp}/Cable_{i}_{k}", W(xc, yc, (z0 + z1) / 2), q, [cyl(r, abs(z1 - z0), mat=m, col=r > 0.01)])
    # tubing manifold on its own small board left of the frame (x -2.0 .. -1.7)
    if not manifold:
        return dict(items=items, center=(cx, cy), yaw=yaw, W=W)
    mx = -PW - 0.6
    ctx.static(f"{grp}/ManifoldBoard", (cx, cy, 0), q, [
        cube((0.07, 0.06, 2.0), (mx - 0.42, 0, 1.0), "steel_dark"), cube((0.07, 0.06, 2.0), (mx + 0.42, 0, 1.0), "steel_dark"),
        cube((0.92, 0.025, 1.3), (mx, 0.045, 1.35), "panel_plate")], category="manifold_board")
    mshapes = []
    for k in range(5):
        z = 0.85 + 0.22 * k
        mshapes.append(cyl(0.011, 0.7, (mx, y_back + 0.0, z), "tube", axis="x"))
        for j in range(5):
            x = mx - 0.3 + 0.15 * j
            mshapes.append(cyl(0.0055, 0.2, (x, y_back + (0.015 if k % 2 else 0.0), z - 0.1), "tube", col=False))
            mshapes.append(cube((0.022, 0.022, 0.022), (x, y_back, z), "steel", col=False))
            if (j + k) % 3 == 0:
                mshapes.append(cyl(0.018, 0.04, (x, y_back, z + 0.03), "valve_red" if (k + j) % 2 else "steel", col=False))
    ctx.static(f"{grp}/Manifold", (cx, cy, 0), q, mshapes, category="tubing_manifold")
    return dict(items=items, center=(cx, cy), yaw=yaw, W=W)


# --------------------------------------------------------------------------- rotating machine
def _sc(sh, s):
    from .common import Sh
    k = sh.kind
    size = tuple(float(x) * s for x in sh.size)
    return Sh(k, size, tuple(float(x) * s for x in sh.pos), sh.quat, sh.mat, sh.col, sh.name)


def pump_set(ctx: Ctx, pos, yaw=0.0, s=1.0):
    """Composite centrifugal pump set: baseplate, motor (fins, terminal box), coupling guard, volute casing, flanged inlet
    and outlet nozzles, spinning coupling disc on the free shaft end and a small-bore seal-flush tubing manifold."""
    cx, cy = pos
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
        sh.append(cyl(0.02, 0.03, (1.09, 0.38 * math.cos(a), 0.74 + 0.38 * math.sin(a)), "steel_dark", axis="x", col=False))
    sh = [_sc(x, s) for x in sh]
    art = Art(ctx, "PumpSet", (cx, cy, 0.0), yaw, sh, mass=900.0 * s ** 3, category="rotating_machine", note="pump/compressor composite on baseplate")
    rot = [cyl(0.05, 0.3, (0.15, 0, 0), "steel", axis="x"), cyl(0.27, 0.06, (0.32, 0, 0), "steel", axis="x")]
    for k in range(8):
        a = k * math.pi / 4
        rot.append(cyl(0.022, 0.03, (0.36, 0.2 * math.cos(a), 0.2 * math.sin(a)), "steel_dark", axis="x"))
    rot = [_sc(x, s) for x in rot]
    art.link("rotor", (1.28 * s, 0, 0.74 * s), rot, mass=25.0 * s ** 3, jtype="revolute", axis="X", lower=None, upper=None, stiffness=0.0, damping=60.0 * s ** 2,
             target=360.0, role="rotor", unit="deg", meta=dict(spin_deg_s=360.0))
    jp = ctx.kit.stage.GetPrimAtPath(f"{art.group}/J_rotor")
    drv = UsdPhysics.DriveAPI.Get(jp, "angular")
    drv.CreateTargetVelocityAttr(360.0)
    drv.GetTargetPositionAttr().Set(0.0)
    ctx.joints[-1]["continuous"] = True
    ctx.joints[-1]["drive"]["target_velocity_deg_s"] = 360.0
    ctx.joints[-1]["drive"]["target"] = None
    W = lambda p: np.array(art.world(np.array(p, float) * s))
    # ---- seal-flush tubing manifold: block on a pedestal beside the baseplate, three tubes to casing / motor ports (visual) + gauge + needle valves
    bp = W((0.45, 0.78, 0.0))
    mg = "/World/Equipment/PumpManifold"
    ctx.kit.xform(mg)
    ctx.static(f"{mg}/pedestal", (bp[0], bp[1], 0.0), Q_IDENT, [cyl(0.025, 0.5 * s * 1.6, (0, 0, 0.4 * s), "steel_dark"), cube((0.16 * s * 1.6, 0.1 * s * 1.6, 0.12 * s * 1.6), (0, 0, 0.5 * s * 1.6 + 0.0), "steel_c")], category="tubing_manifold")
    top = np.array([bp[0], bp[1], 0.5 * s * 1.6 + 0.06 * s * 1.6])
    ports = [W((0.8, 0.46, 0.74)), W((0.55, 0.46, 0.6)), W((-0.75, 0.13, 1.18))]
    tsh = []
    for i, pt in enumerate(ports):
        a = top + np.array([(i - 1) * 0.03, 0, 0.0])
        d = pt - a
        L = float(np.linalg.norm(d))
        tsh.append(cyl(0.006, L, tuple((a + pt) / 2), "tube", col=False, quat=qalign(d)))
        tsh.append(sph(0.011, tuple(pt), "steel", col=False))
        tsh.append(sph(0.011, tuple(a), "steel", col=False))
        tsh.append(cyl(0.017, 0.03, tuple(a + d / L * 0.1), "valve_red" if i == 0 else "steel_dark", col=False, quat=qalign(d)))
    ctx.static(f"{mg}/tubes", (0, 0, 0), Q_IDENT, tsh, category="tubing_manifold")
    nozzle_top = W((0.95, 0, 1.52))
    suction = W((0.65, -0.84, 0.74))
    return dict(art=art, W=W, nozzle_top=nozzle_top, suction=suction, side=W((0.65, 0.2, 1.0)), manifold_gauge_pt=top, s=s, discharge_r=0.11 * s, suction_r=0.12 * s,
                size=(2.9 * s, 1.3 * s))


# --------------------------------------------------------------------------- flexible hoses (dynamic chains)
def _relax_chain(a, b, n, slack, iters=30000):
    """Equilibrium of an n-link pinned chain (equal link length) between fixed ends: position-based relaxation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ell = np.linalg.norm(b - a) * slack / n
    P = np.array([a + (b - a) * i / n for i in range(n + 1)])
    P[1:-1, 2] -= 0.15 * np.linalg.norm(b - a) * np.sin(np.linspace(0, math.pi, n + 1))[1:-1]
    for it in range(iters):
        P[1:-1, 2] -= 1e-4
        for _ in range(4):
            for i in range(n):
                d = P[i + 1] - P[i]
                L = np.linalg.norm(d)
                c = (L - ell) / L * 0.5 * d
                if i > 0:
                    P[i] += c
                else:
                    pass
                if i + 1 < n:
                    P[i + 1] -= c
            P[0], P[-1] = a, b
    return P


def hose(ctx: Ctx, name, a, b, r, mat, n=9, slack=1.18, floor_z=0.0):
    while True:  # reduce slack until the hanging chain clears the floor (no ground contact -> small settling motion)
        P = _relax_chain(a, b, n, slack, iters=6000)
        if P[:, 2].min() > floor_z + r + 0.12 or slack < 1.04:
            break
        slack -= 0.03
    P = _relax_chain(a, b, n, slack)
    grp = f"/World/Hoses/{name}"
    ctx.kit.xform(grp)
    ctx.static(f"{grp}/anchor_a", a, Q_IDENT, [cyl(r * 1.5, 0.04, (0, 0, 0), "steel_dark", col=False)])
    ctx.static(f"{grp}/anchor_b", b, Q_IDENT, [cyl(r * 1.5, 0.04, (0, 0, 0), "steel_dark", col=False)])
    paths, quats, poss = [], [], []
    for i in range(n):
        d = P[i + 1] - P[i]
        L = float(np.linalg.norm(d))
        q = qalign(d)
        mid = (P[i] + P[i + 1]) / 2
        path = f"{grp}/seg{i}"
        h = max(L - r, 0.02)
        mass = 0.35 + 8 * r * r * L * 8.0
        ctx.body(path, mid, q, [cap(r, h, (0, 0, 0), mat), cyl(r * 1.12, 0.012, (0, 0, h * 0.25), mat, col=False), cyl(r * 1.12, 0.012, (0, 0, -h * 0.25), mat, col=False)], mass, "rubber",
                 linear_damping=0.5, angular_damping=4.0, iters=(16, 4))
        ctx.kit.attr(ctx.kit.stage.GetPrimAtPath(path), "plant:hose", T.String, name)
        paths.append(path)
        quats.append(q)
        poss.append(mid)
    Rm = qaxis((0, 1, 0), -math.pi / 2)

    def sph_joint(jpath, b0, b1, anchor, q_frame0, q_frame1, lp0, lp1):
        j = UsdPhysics.SphericalJoint.Define(ctx.kit.stage, jpath)
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
        # joint at node i between seg i-1 and seg i
        node = P[i]
        l1 = qmat(quats[i]).T @ (node - poss[i])
        if i == 0:
            sph_joint(f"{grp}/J_{i}", None, paths[0], node, qmul(quats[0], Rm), Rm, node, l1)
        else:
            l0 = qmat(quats[i - 1]).T @ (node - poss[i - 1])
            sph_joint(f"{grp}/J_{i}", paths[i - 1], paths[i], node, Rm, Rm, l0, l1)
    node = P[n]
    l0 = qmat(quats[n - 1]).T @ (node - poss[n - 1])
    # last joint: segment n-1 -> world anchor (body1 is the world side)
    j = UsdPhysics.SphericalJoint.Define(ctx.kit.stage, f"{grp}/J_{n}")
    j.CreateBody0Rel().SetTargets([Sdf.Path(paths[n - 1])])
    j.CreateLocalPos0Attr(_v3(l0))
    j.CreateLocalRot0Attr(_q(Rm))
    j.CreateLocalPos1Attr(_v3(node))
    j.CreateLocalRot1Attr(_q(qmul(quats[n - 1], Rm)))
    j.CreateAxisAttr("X")
    j.CreateConeAngle0LimitAttr(55.0)
    j.CreateConeAngle1LimitAttr(55.0)
    return dict(name=name, segments=paths, anchor_a=[float(c) for c in a], anchor_b=[float(c) for c in b], radius=r, length=float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))))


# --------------------------------------------------------------------------- worker
def worker(ctx: Ctx, pos, yaw, reach_target):
    """Humanoid placeholder, 1.75 m tall: red coveralls, orange hard hat, black ear defenders, orange gloves, boots; the left
    arm reaches towards ``reach_target``. Static colliders (legs, torso, raised arm)."""
    x, y = pos
    q = qyaw(yaw)
    Rm = qmat(q)
    tl = Rm.T @ (np.asarray(reach_target, float) - np.array([x, y, 0.0]))  # target in worker frame (forward = +X)
    sh_l = np.array([0.0, 0.2, 1.42])
    d = tl - sh_l
    d = d / np.linalg.norm(d)
    elbow_pt = sh_l + d * 0.30 + np.array([0, 0.03, -0.03])
    hand_pt = sh_l + d * 0.58

    def lim(p0, p1, r, mat="coverall_red", col=True):
        dd = np.asarray(p1, float) - np.asarray(p0, float)
        L = float(np.linalg.norm(dd))
        return cap(r, max(L - 2 * r, 0.02), tuple((np.asarray(p0, float) + np.asarray(p1, float)) / 2), mat, col=col, quat=qalign(dd))

    shapes = []
    for sgn in (-1, 1):  # legs: thigh + shin, boots
        hip, knee, ankle = (0.0, sgn * 0.095, 0.93), (0.02, sgn * 0.10, 0.50), (0.0, sgn * 0.10, 0.09)
        shapes += [lim(hip, knee, 0.085), lim(knee, ankle, 0.07), cube((0.28, 0.105, 0.085), (0.06, sgn * 0.10, 0.043), "boot")]
    shapes += [
        cap(0.16, 0.12, (0, 0, 0.98), "coverall_red"),  # pelvis
        cap(0.165, 0.24, (0, 0, 1.2), "coverall_red"),  # torso
        esph(0.2, 0.15, 0.2, (0.0, 0, 1.32), "coverall_red"),  # chest
        cyl(0.172, 0.06, (0, 0, 1.0), "black", col=False),  # belt
        cyl(0.17, 0.035, (0, 0, 1.15), "white", col=False),  # reflective stripe
        cyl(0.05, 0.1, (0.0, 0, 1.52), "skin", col=False),  # neck
        esph(0.095, 0.085, 0.115, (0.02, 0, 1.64), "skin"),  # head
        cyl(0.105, 0.06, (0.02, 0, 1.70), "hardhat", col=False), esph(0.108, 0.108, 0.07, (0.02, 0, 1.72), "hardhat"),  # hard hat
        cyl(0.145, 0.012, (0.04, 0, 1.675), "hardhat", col=False),
        cube((0.24, 0.02, 0.03), (0.02, 0, 1.76), "hardhat", col=False),  # hat ridge
        cyl(0.05, 0.045, (0.02, 0.115, 1.64), "black", axis="y", col=False), cyl(0.05, 0.045, (0.02, -0.115, 1.64), "black", axis="y", col=False),  # ear defenders
        cyl(0.012, 0.26, (0.02, 0, 1.76), "black", axis="y", col=False),
        sph(0.07, (0.0, 0.2, 1.42), "coverall_red", col=False), sph(0.07, (0.0, -0.2, 1.42), "coverall_red", col=False),  # shoulders
        lim((0.0, -0.2, 1.42), (0.03, -0.24, 1.12), 0.05), lim((0.03, -0.24, 1.12), (0.08, -0.24, 0.88), 0.045),  # right arm hanging
        sph(0.05, (0.08, -0.24, 0.82), "glove", col=False),
        lim(sh_l, elbow_pt, 0.052), lim(elbow_pt, hand_pt, 0.046), sph(0.052, tuple(hand_pt + d * 0.035), "glove", col=False),  # reaching arm
    ]
    ctx.static("/World/Worker", (x, y, 0.0), q, shapes, category="human_worker", note="humanoid placeholder, static colliders")
    return dict(pos=[x, y, 0.0], yaw=yaw, hand=[round(float(c), 3) for c in (np.array([x, y, 0]) + Rm @ hand_pt)], height_m=1.78)


# --------------------------------------------------------------------------- loose dynamic props
def loose_items(ctx: Ctx, spots):
    """spots: list of (x, y, yaw). Returns records. Items rest on the deck (z=0)."""
    rng = ctx.rng
    kinds = ["wrench", "hammer", "bucket", "hardhat", "toolbox", "coil", "spool", "wrench", "hardhat_blue", "glove_box"]
    out = []
    for i, (x, y, yaw) in enumerate(spots):
        k = kinds[i % len(kinds)] if i >= 3 else ["bucket", "hardhat", "wrench"][i]
        path = f"/World/Loose/{k}_{i}"
        if k == "wrench":
            sh = [cube((0.3, 0.032, 0.012), (0, 0, 0.006), "steel"), cyl(0.028, 0.012, (0.15, 0, 0.006), "steel"), cyl(0.026, 0.012, (-0.15, 0, 0.006), "steel")]
            m, z = 0.5, 0.0
        elif k == "hammer":
            sh = [cube((0.3, 0.03, 0.03), (0, 0, 0.015), "wood"), cube((0.08, 0.05, 0.05), (0.14, 0, 0.025), "steel")]
            m, z = 0.9, 0.0
        elif k == "bucket":
            R, H, th = 0.15, 0.28, 0.006
            sh = [cyl(R, th, (0, 0, th / 2), "bucket_yellow")]
            for j in range(10):
                a = 2 * math.pi * j / 10
                sh.append(cube((2 * math.pi * R / 10 * 1.06, th, H), (R * math.cos(a), R * math.sin(a), H / 2), "bucket_yellow", quat=qyaw(a + math.pi / 2)))
            m, z = 0.9, 0.0
        elif k in ("hardhat", "hardhat_blue"):
            hm = "hardhat" if k == "hardhat" else "hardhat_white"
            sh = [cyl(0.17, 0.008, (0, 0, 0.004), hm), cyl(0.105, 0.075, (0, 0, 0.0455), hm), esph(0.105, 0.105, 0.06, (0, 0, 0.083), hm)]
            m, z = 0.35, 0.0
        elif k == "toolbox":
            sh = [cube((0.42, 0.18, 0.16), (0, 0, 0.08), "valve_red"), cube((0.3, 0.02, 0.02), (0, 0, 0.17), "steel", col=False)]
            m, z = 4.0, 0.0
        elif k == "coil":
            sh = [cyl(0.2, 0.09, (0, 0, 0.045), "black"), cyl(0.09, 0.091, (0, 0, 0.046), "steel_dark", col=False)]
            m, z = 1.5, 0.0
        elif k == "spool":
            sh = [cyl(0.06, 0.3, (0, 0, 0.09), "steel", axis="x"), cyl(0.09, 0.025, (0.14, 0, 0.09), "steel", axis="x"), cyl(0.09, 0.025, (-0.14, 0, 0.09), "steel", axis="x")]
            m, z = 3.0, 0.0
        else:
            sh = [cube((0.25, 0.18, 0.1), (0, 0, 0.05), "orange_box"), cube((0.15, 0.003, 0.05), (0, 0.091, 0.05), "white", col=False)]
            m, z = 1.0, 0.0
        ctx.body(path, (x, y, z + 0.004), qyaw(yaw), sh, m, "steel" if k in ("wrench", "hammer", "spool", "toolbox") else "plastic", angular_damping=0.3, iters=(16, 4))
        ctx.kit.attr(ctx.kit.stage.GetPrimAtPath(path), "plant:loose", T.String, k)
        ctx.kit.tag(ctx.kit.stage.GetPrimAtPath(path), "loose_item", kind=k)
        out.append(dict(path=path, kind=k, mass=m, pos=[x, y, z], yaw=yaw))
    return out
