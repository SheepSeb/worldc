"""Procedural warehouse generator: ``python -m claude_scene_generator.generate --seed N --out warehouse_out``."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, Vt

from . import layout as LY
from .geom import Part, Q_IDENT, qaxis, qfrom_mat, qmul, qyaw, rect_overlap_2d
from .objects import Factory, PAL_H, PAL_L, PAL_W
from .structures import Ctx, Frame, Structures
from .textures import make_cardboard_texture, make_floor_texture
from .usdkit import Kit, T

VERSION = "1.0"
N_CARDBOARD = 5
LIGHT_BASE = 30000.0  # RectLight intensity (nits) of a full-power ceiling panel
LIGHT_SETUPS = ("bright_day", "mixed_warm", "dim_night")


def look_at_quat(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return qfrom_mat(np.stack([r, u, -f], axis=1))


# ----------------------------------------------------------------------------- materials
def make_materials(kit: Kit, rng, tex_dir: Path):
    pal = [
        # uprights, beams
        ((0.10, 0.25, 0.65), (0.95, 0.45, 0.08)),
        ((0.35, 0.37, 0.40), (0.80, 0.12, 0.10)),
        ((0.08, 0.40, 0.25), (0.95, 0.55, 0.05)),
        ((0.55, 0.10, 0.10), (0.20, 0.22, 0.25)),
    ][int(rng.integers(4))]
    m = {}
    mt = kit.material
    m["floor"] = mt("Floor", tex="./textures/floor.jpg", rough=0.55)
    m["apron"] = mt("Apron", (0.30, 0.30, 0.30), 0.85)
    m["wall"] = mt("Wall", (0.78, 0.78, 0.74), 0.9)
    m["roof"] = mt("Roof", (0.62, 0.64, 0.66), 0.8)
    m["truss"] = mt("Truss", (0.25, 0.28, 0.32), 0.6, 0.6)
    m["wood"] = mt("Wood", (0.58, 0.41, 0.24), 0.85)
    m["wood_light"] = mt("WoodLight", (0.72, 0.55, 0.34), 0.85)
    m["wrap"] = mt("StretchWrap", (0.80, 0.85, 0.90), 0.25, 0.0)
    for i in range(N_CARDBOARD):
        m[f"cardboard_{i}"] = mt(f"Cardboard_{i}", tex=f"./textures/cardboard_{i}.jpg", rough=0.9)
    m["tote_blue"] = mt("ToteBlue", (0.08, 0.28, 0.68), 0.5)
    m["tote_grey"] = mt("ToteGrey", (0.45, 0.47, 0.50), 0.5)
    m["tote_red"] = mt("ToteRed", (0.68, 0.12, 0.10), 0.5)
    for n, c in zip("abcd", [(0.9, 0.8, 0.1), (0.85, 0.85, 0.85), (0.15, 0.55, 0.3), (0.3, 0.3, 0.8)]):
        m[f"item_{n}"] = mt(f"Item_{n}", c, 0.6)
    m["safety_yellow"] = mt("SafetyYellow", (0.95, 0.75, 0.05), 0.5)
    m["rubber"] = mt("Rubber", (0.04, 0.04, 0.04), 0.9)
    m["belt"] = mt("BeltRubber", (0.06, 0.06, 0.065), 0.85)
    m["conv_frame"] = mt("ConveyorFrame", (0.45, 0.48, 0.52), 0.4, 0.7)
    m["roller"] = mt("Roller", (0.7, 0.72, 0.75), 0.3, 0.9)
    m["steel_bright"] = mt("SteelBright", (0.75, 0.76, 0.78), 0.25, 1.0)
    m["table_top"] = mt("TableTop", (0.75, 0.62, 0.42), 0.7)
    m["drawer"] = mt("Drawer", (0.55, 0.58, 0.62), 0.5, 0.4)
    m["door_grey"] = mt("DoorGrey", (0.62, 0.64, 0.67), 0.45, 0.6)
    m["door_slats"] = mt("DoorSlats", (0.80, 0.82, 0.84), 0.4, 0.5)
    m["door_frame"] = mt("DoorFrame", (0.30, 0.33, 0.38), 0.5, 0.5)
    m["door_leaf"] = mt("DoorLeaf", (0.18, 0.40, 0.62), 0.5, 0.3)
    m["leveler"] = mt("Leveler", (0.50, 0.52, 0.55), 0.5, 0.7)
    m["cage_mesh"] = mt("CageMesh", (0.75, 0.78, 0.8), 0.5, 0.0, opacity=0.3)
    m["cage_post"] = mt("CagePost", (0.90, 0.70, 0.05), 0.5)
    m["cage_gate"] = mt("CageGate", (0.88, 0.50, 0.08), 0.5, 0.4, opacity=0.6)
    m["cone"] = mt("ConeOrange", (0.95, 0.35, 0.05), 0.55)
    m["cone_base"] = mt("ConeBase", (0.08, 0.08, 0.08), 0.9)
    m["bin_green"] = mt("BinGreen", (0.10, 0.45, 0.20), 0.6)
    m["bin_dark"] = mt("BinDark", (0.03, 0.04, 0.03), 0.9)
    m["jack"] = mt("PalletJack", (0.80, 0.15, 0.10), 0.5, 0.4)
    m["forklift"] = mt("Forklift", (0.95, 0.72, 0.08), 0.5)
    m["human_a"] = mt("HumanVest", (0.95, 0.45, 0.05), 0.7)
    m["human_b"] = mt("HumanBlue", (0.10, 0.30, 0.65), 0.7)
    for i in range(3):
        m[f"rack_up_{i}"] = mt(f"RackUpright_{i}", tuple(np.clip(np.array(pal[0]) * rng.uniform(0.9, 1.1, 3), 0, 1)), 0.45, 0.5)
        m[f"rack_beam_{i}"] = mt(f"RackBeam_{i}", tuple(np.clip(np.array(pal[1]) * rng.uniform(0.9, 1.1, 3), 0, 1)), 0.45, 0.5)
    m["charger"] = mt("Charger", (0.10, 0.60, 0.30), 0.4, 0.3)
    m["charger_led"] = mt("ChargerLED", (0.1, 1.0, 0.3), 0.3, 0.0, emissive=(0.1, 0.9, 0.3))
    m["robot_body"] = mt("RobotBody", (0.85, 0.87, 0.90), 0.4, 0.3)
    m["robot_arm"] = mt("RobotArm", (0.20, 0.45, 0.80), 0.4, 0.4)
    m["robot_marker"] = mt("RobotMarker", (0.1, 0.8, 0.9), 0.5, 0.0, opacity=0.5)
    pm = {
        "cardboard": kit.phys_material("Cardboard", 0.5, 0.05),
        "steel": kit.phys_material("Steel", 0.6, 0.1),
        "concrete": kit.phys_material("Concrete", 0.7, 0.05),
        "rubber": kit.phys_material("RubberBelt", 0.9, 0.05),
        "wood": kit.phys_material("Wood", 0.5, 0.05),
        "plastic": kit.phys_material("Plastic", 0.4, 0.05),
    }
    return m, pm


# ----------------------------------------------------------------------------- markings
def floor_markings(L: LY.Layout):
    Y = (255, 200, 20)
    Wt = (235, 235, 230)
    G = (30, 150, 70)
    Bk = (25, 25, 25)
    mk = []
    for xa, xb, xc in L.aisles:
        for x in (xa + 0.12, xb - 0.12):
            mk.append({"type": "line", "pts": [(x, -7.5), (x, 9.7)], "color": Y, "width": 0.1, "kind": "aisle_edge"})
    (xa, ya), (xb, _), (_, yb), _ = L.loop
    mk.append({"type": "dashed", "pts": [(xa, ya), (xb, ya)], "color": Wt, "width": 0.1, "kind": "forklift_route"})
    mk.append({"type": "dashed", "pts": [(xa, yb), (xb, yb)], "color": Wt, "width": 0.1, "kind": "forklift_route"})
    mk.append({"type": "dashed", "pts": [(xb, ya), (xb, yb)], "color": Wt, "width": 0.1, "kind": "forklift_route"})
    mk.append({"type": "dashed", "pts": [(xa, ya), (xa, yb)], "color": Wt, "width": 0.1, "kind": "forklift_route"})
    mk.append({"type": "line", "pts": [(-18.8, 9.95), (10.0, 9.95)], "color": G, "width": 0.12, "kind": "pedestrian_walkway"})
    for i, x in enumerate(LY.DOCK_BAYS_X):
        mk.append({"type": "rect", "bbox": (x - 1.7, -12.4, x + 1.7, -9.6), "color": Wt, "width": 0.1, "kind": f"dock_bay_{i}"})
        mk.append({"type": "hatch", "bbox": (x - 1.7, -9.9, x + 1.7, -9.6), "color": Bk, "stripe": 0.2, "kind": "dock_hatch"})
    for i, (px, py, _) in enumerate(LY.PARKING):
        mk.append({"type": "rect", "bbox": (px - 0.8, py - 1.0, px + 0.8, py + 1.0), "color": (60, 140, 235), "width": 0.1, "kind": f"parking_{i}"})
        mk.append({"type": "line", "pts": [(px - 0.5, py - 0.7), (px, py + 0.7), (px + 0.5, py - 0.7)], "color": (60, 140, 235), "width": 0.08, "kind": "parking_arrow"})
    cz = LY.CHARGE_ZONE
    mk.append({"type": "rect", "bbox": cz, "color": G, "width": 0.12, "kind": "charging_zone"})
    pz = LY.PACK_ZONE
    mk.append({"type": "rect", "bbox": pz, "color": (240, 130, 20), "width": 0.12, "kind": "packing_zone"})
    cg = LY.CAGE
    mk.append({"type": "rect", "bbox": (cg[0] - 0.3, cg[1] - 1.6, cg[2] + 0.1, cg[3] + 0.3), "color": Y, "width": 0.1, "kind": "cage_keepout"})
    mk.append({"type": "dashed", "pts": [(-17.5, -9.45), (-3.6, -9.45)], "color": Wt, "width": 0.08, "kind": "staging_line"})
    for z in LY.SLOPE_ZONES:
        mk.append({"type": "rect", "bbox": z, "color": (240, 130, 20), "width": 0.06, "kind": "slope_zone"})
    return mk


# ----------------------------------------------------------------------------- lights / cameras
def build_lights(kit: Kit, rng, setup: int, light_scale: float):
    kit.scope("/World/Lights")
    name = LIGHT_SETUPS[setup]
    cfg = {
        0: dict(keep=1.0, inten=1.0, temp=5600, dome=450.0),
        1: dict(keep=0.8, inten=0.9, temp=3800, dome=250.0),
        2: dict(keep=0.55, inten=0.42, temp=3000, dome=130.0),
    }[setup]
    lights = []
    xs = np.arange(-17.0, 19.0, 6.0)
    ys = np.array([-9.5, -4.5, 2.2, 7.5, 11.0])
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if rng.random() > cfg["keep"]:
                continue
            inten = LIGHT_BASE * cfg["inten"] * float(rng.uniform(0.85, 1.1)) * light_scale
            lp = f"/World/Lights/Panel_{i}_{j}"
            L = UsdLux.RectLight.Define(kit.stage, lp)
            L.CreateWidthAttr(1.4)
            L.CreateHeightAttr(0.5)
            L.CreateIntensityAttr(inten)
            L.CreateEnableColorTemperatureAttr(True)
            L.CreateColorTemperatureAttr(float(cfg["temp"] + rng.uniform(-300, 300)))
            kit.set_pose(L.GetPrim(), (x, y, LY.H - 0.35))  # RectLight emits along -Z (down)
            lights.append({"path": lp, "position": [float(x), float(y), LY.H - 0.35], "intensity": inten})
    dome = UsdLux.DomeLight.Define(kit.stage, "/World/Lights/FillDome")
    dome.CreateIntensityAttr(cfg["dome"] * light_scale)
    dome.CreateColorTemperatureAttr(float(cfg["temp"]))
    dome.CreateEnableColorTemperatureAttr(True)
    return {"setup_index": setup, "setup_name": name, "n_panels": len(lights), "dome_intensity": cfg["dome"], "panels": lights}


def build_camera(kit, path, pos, quat, focal=18.0, hap=20.955, vap=11.79, clip=(0.05, 200.0)):
    cam = UsdGeom.Camera.Define(kit.stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(hap)
    cam.CreateVerticalApertureAttr(vap)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*clip))
    kit.set_pose(cam.GetPrim(), pos, quat)
    return cam.GetPrim()


# ----------------------------------------------------------------------------- hazards
def build_hazards(kit: Kit, m, pm, plan, L):
    kit.scope("/World/Hazards")
    out = []

    def hazard(name, kind, pts, speed, mode, size, mat, z, shape, mass, start_s=0.0):
        path = f"/World/Hazards/{name}"
        p0 = pts[0]
        pos = (p0[0], p0[1], z)
        body = kit.rigid_body(path, pos, Q_IDENT, parts=[Part("box", size)], mass=mass, kinematic=True)
        if shape == "capsule":
            r, h = size[0] / 2, size[2] - size[0]
            kit.capsule(f"{path}/Shape", r, h, (0, 0, 0), Q_IDENT, mat, pm["plastic"], collide=True)
        else:
            kit.cube(f"{path}/Shape", size, (0, 0, 0), Q_IDENT, mat, pm["steel"], collide=True)
            kit.cube(f"{path}/Forks", (1.0, 1.0, 0.08), (size[0] / 2 + 0.45, 0, -size[2] / 2 + 0.12), Q_IDENT, m["steel_bright"])
        pts3 = [(float(x), float(y), float(z)) for x, y in pts]
        kit.attr(body, "warehouse:waypoints", T.Double3Array, Vt.Vec3dArray([Gf.Vec3d(*p) for p in pts3]))
        kit.attr(body, "warehouse:speed_mps", T.Double, float(speed))
        kit.attr(body, "warehouse:path_mode", T.String, mode)
        kit.attr(body, "warehouse:path_phase_m", T.Double, float(start_s))
        kit.tag(body, "hazard", kind=kind)
        rec = kit.record(path, "hazard", pos, Q_IDENT, "kinematic", size=size, mass=mass, friction=0.5, subtype=kind,
                         waypoints=[[round(x, 3), round(y, 3)] for x, y in pts], z=z, speed_mps=speed, path_mode=mode,
                         path_phase_m=start_s, driven_by="runtime.PathFollower (set kinematic pose each physics step)")
        out.append(rec)

    loop = L.loop
    hazard("Forklift", "forklift_box", loop, 0.8, "loop", (2.6, 1.2, 2.1), m["forklift"], 1.05 + 0.001, "box", 3500.0, start_s=0.0)
    plan.add_path(loop, 1.05, "forklift", closed=True)
    hp = L.human_paths()
    hazard("Human_0", "human_capsule", hp[0], 0.9, "pingpong", (0.44, 0.44, 1.7), m["human_a"], 0.85 + 0.001, "capsule", 80.0)
    hazard("Human_1", "human_capsule", hp[1], 0.8, "pingpong", (0.44, 0.44, 1.7), m["human_b"], 0.85 + 0.001, "capsule", 75.0, start_s=6.0)
    plan.add_path(hp[0], 0.7, "human")
    plan.add_path(hp[1], 0.7, "human")
    return out


# ----------------------------------------------------------------------------- robot
def build_robot(kit: Kit, m, plan):
    x, y, yaw = LY.ROBOT_SPAWN
    q = qyaw(yaw)
    rp = "/World/Robot"
    kit.scope(rp)
    base = f"{rp}/MobileManipulator"
    kit.xform(base, (x, y, 0.0), q)
    kit.tag(kit.stage.GetPrimAtPath(base), "robot_spawn", placeholder=True, footprint_radius_m=0.38)
    kit.cylinder(f"{base}/Base", 0.38, 0.35, (0, 0, 0.2), Q_IDENT, m["robot_body"])
    kit.cube(f"{base}/Torso", (0.22, 0.3, 1.35), (-0.08, 0, 0.875), Q_IDENT, m["robot_body"])
    kit.cube(f"{base}/ArmLink", (0.5, 0.08, 0.08), (0.17, 0.0, 1.15), Q_IDENT, m["robot_arm"])
    kit.cube(f"{base}/Gripper", (0.1, 0.18, 0.06), (0.45, 0.0, 1.15), Q_IDENT, m["robot_arm"])
    cam_q = look_at_quat((0.2, 0, 1.5), (3.0, 0, 1.2), (0, 0, 1))
    build_camera(kit, f"{base}/RobotCamera", (0.2, 0, 1.5), cam_q, focal=15.0)
    kit.record(base, "robot_spawn", (x, y, 0.0), q, "none", size=(0.76, 0.76, 1.3), placeholder=True,
               note="visual-only placeholder (no physics); replace with the real robot USD / ArticulationCfg",
               footprint_radius_m=0.38)
    plan.add(x, y, 0.8, 0.8, 0.0, "spawn")
    parking = []
    for i, (px, py, pyaw) in enumerate(LY.PARKING):
        kit.cube(f"/World/Charging/Pad_{i}", (1.2, 1.6, 0.01), (px, py, 0.005), qyaw(pyaw - math.pi / 2), m["robot_marker"])
        sx = px
        cp = f"/World/Charging/Charger_{i}"
        kit.xform(cp, (sx, LY.Y0 + 0.3, 0.0))
        kit.cube(f"{cp}/Body", (0.5, 0.3, 1.0), (0, 0, 0.5), Q_IDENT, m["charger"], kit._pmats["Steel"], collide=True)
        kit.cube(f"{cp}/LED", (0.06, 0.02, 0.06), (0, 0.16, 0.75), Q_IDENT, m["charger_led"])
        kit.record(cp, "charger", (sx, LY.Y0 + 0.3, 0.5), Q_IDENT, "static", size=(0.5, 0.3, 1.0), friction=0.6, zone="charging")
        parking.append({"name": f"parking_{i}", "pose": [px, py, pyaw], "charger": f"/World/Charging/Charger_{i}"})
        plan.add(px, py, 0.9, 1.1, 0.0, f"parking_{i}")
        plan.add(sx, LY.Y0 + 0.3, 0.35, 0.25, 0.0, "charger")
    return {"spawn": [x, y, yaw], "prim": base, "camera": f"{base}/RobotCamera", "parking_spots": parking}


# ----------------------------------------------------------------------------- main build
def build(seed: int, out_root: Path, lighting: int | None = None, light_scale: float = 1.0) -> Path:
    t0 = time.time()
    out = Path(out_root) / f"seed_{seed}"
    (out / "textures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    usd_path = out / "warehouse.usd"
    if usd_path.exists():
        usd_path.unlink()
    L = LY.Layout()
    setup = (seed % 3) if lighting is None else lighting
    wear = float(rng.uniform(0.3, 1.0))
    slope, max_h = float(rng.uniform(0.02, 0.03)), 0.04

    kit = Kit(usd_path)
    kit.attr(kit.world.GetPrim(), "warehouse:seed", T.Int, seed)
    kit.attr(kit.world.GetPrim(), "warehouse:generator_version", T.String, VERSION)
    mk = floor_markings(L)
    make_floor_texture(out / "textures" / "floor.jpg", np.random.default_rng(seed + 1000), LY.W, LY.D, mk, wear)
    for i in range(N_CARDBOARD):
        make_cardboard_texture(out / "textures" / f"cardboard_{i}.jpg", np.random.default_rng(seed * 100 + i), i)
    m, pm = make_materials(kit, rng, out / "textures")
    kit.physics_scene(hz=120)

    plan = LY.Plan()
    ctx = Ctx(kit, rng, plan, m, pm, L, slope, max_h)
    S = Structures(ctx)
    S.shell()
    slots = S.racks()

    # ---- reserved zones / dynamic-hazard corridors
    hazards = build_hazards(kit, m, pm, plan, L)
    robot = build_robot(kit, m, plan)

    # ---- doors, dock, cage, packing equipment
    kit.scope("/World/Doors")
    kit.scope("/World/Dock")
    kit.scope("/World/Conveyors")
    kit.scope("/World/Charging")
    for i, x in enumerate(LY.DOCK_BAYS_X):
        S.rollup_door(i, x)
    S.swing_door(0, (LY.X0, 11.0), -math.pi / 2)  # west wall
    S.swing_door(1, (16.4, LY.Y1), math.pi)  # north wall
    S.cage()
    S.belt("BeltA", 15.2, 2.0, 6.0)
    S.belt("BeltB", 17.2, 2.0, 6.0)
    S.roller_conveyor("RollerConveyor", 15.2, 6.3, 9.3)
    table_top = S.work_table(17.2, 8.3)
    plan.add_aabb(LY.PACK_ZONE[0] - 0.1, 1.0, 13.9, 12.4, "packing_edge")

    # ---- box size classes (6 per seed) within 0.15-0.6 m
    def rr(a, b):
        return float(rng.uniform(a, b))

    classes = [
        (rr(0.15, 0.25), rr(0.15, 0.25), rr(0.15, 0.25)),
        (rr(0.2, 0.32), rr(0.2, 0.32), rr(0.18, 0.3)),
        (rr(0.3, 0.42), rr(0.25, 0.38), rr(0.2, 0.35)),
        (rr(0.4, 0.5), rr(0.3, 0.45), rr(0.25, 0.4)),
        (rr(0.5, 0.6), rr(0.4, 0.5), rr(0.28, 0.42)),
        (rr(0.45, 0.6), rr(0.3, 0.45), rr(0.15, 0.22)),
    ]
    F = Factory(kit, rng, m, pm, plan, classes, N_CARDBOARD)

    # ---- pick targets first (so they are always populated)
    goals = []
    pick_specs = [(1, 2, 0), (4, 2, 1), (3, 3, 0), (5, 3, 1)]  # (aisle, level index, seg)
    forced = {}
    for aisle, lvl, seg in pick_specs:
        # rack row on the east side of the aisle: faces -1; west row faces +1; alternate
        cand = [s for s in slots if s["aisle"] == aisle and s["level"] == lvl and s["seg"] == seg and s["bay"] == 0 and s["slot"] == 0]
        cand = [s for s in cand if (s["face"] == 1) == (len(forced) % 2 == 0)] or cand
        forced[(cand[0]["row"], cand[0]["seg"], cand[0]["bay"], cand[0]["slot"], cand[0]["level"])] = cand[0]

    # ---- populate racks
    max_load = 1.12
    n_rack_pallets = 0
    for s in slots:
        key = (s["row"], s["seg"], s["bay"], s["slot"], s["level"])
        fill = {0: 0.55, 1: 0.45, 2: 0.4, 3: 0.35}[s["level"]]
        if key not in forced and rng.random() > fill:
            continue
        pos = (s["x"] + rng.uniform(-0.01, 0.01), s["y"] + rng.uniform(-0.01, 0.01), s["z"] + 0.0006)
        yaw = math.pi / 2 + rng.uniform(-0.02, 0.02)
        kind = "boxes" if key in forced else rng.choice(["boxes", "wrapped", "totes", "empty"], p=[0.55, 0.28, 0.08, 0.09])
        zone = f"rack_row_{s['row']}"
        if kind == "boxes":
            pal = F.pallet(pos, yaw, zone)
            F.load_boxes(pal, zone, F.pick_dims(int(rng.integers(1, 6)), lo=0.15), max_load, max_boxes=12)
        elif kind == "wrapped":
            F.wrapped_load(pos, yaw, zone, F.pick_dims(int(rng.integers(2, 6))), max_load)
        elif kind == "totes":
            F.pallet_of_totes(pos, yaw, zone)
        else:
            F.pallet(pos, yaw, zone)
        n_rack_pallets += 1

    # ---- dock staging row: 2-high stacks and singles
    stage_y = -8.55
    xs = [-17.0 + i * 2.2 for i in range(7)]
    stacks = set(int(v) for v in rng.choice(len(xs), 3, replace=False))
    for i, x in enumerate(xs):
        yaw = float(rng.uniform(-0.05, 0.05))
        pos = (x, stage_y, 0.0006)
        zone = "dock_staging"
        plan.add(x, stage_y, 0.65, 0.45, yaw, "staging")
        if i in stacks:
            lower = F.pallet(pos, yaw, zone)
            F.load_boxes(lower, zone, F.pick_dims(int(rng.integers(2, 5))), 0.9, full=True, max_boxes=12)
            upper = F.pallet((x + rng.uniform(-0.015, 0.015), stage_y + rng.uniform(-0.015, 0.015), lower.load_top + 0.0008), yaw + rng.uniform(-0.03, 0.03), zone)
            F.load_boxes(upper, zone, F.pick_dims(int(rng.integers(2, 5))), 0.8, max_boxes=10)
            kit.stage.GetPrimAtPath(upper.path).CreateAttribute("warehouse:stacked_on", T.String, True).Set(lower.path)
        else:
            if rng.random() < 0.5:
                F.wrapped_load(pos, yaw, zone, F.pick_dims(int(rng.integers(2, 6))), 1.1)
            else:
                p = F.pallet(pos, yaw, zone)
                F.load_boxes(p, zone, F.pick_dims(int(rng.integers(1, 6))), 1.1, max_boxes=12)

    # ---- fallen pallet group in aisle 2, segment 2 -> 0.9 m chokepoint
    ax0, ax1, axc = L.aisles[2]
    yb = 7.0
    clear = 0.9
    left_edge, right_edge = aisle_free_edges(kit.records, ax0, ax1, (yb - 2.2, yb + 2.2))
    target_x = right_edge - clear  # blockage may extend to here, leaving exactly `clear` to the opposite rack
    plan.add_aabb(ax0, 4.3, ax1, 9.7, "choke_aisle")
    p1_yaw = math.radians(6.0)
    span1 = PAL_L * abs(math.cos(p1_yaw)) + PAL_W * abs(math.sin(p1_yaw))
    p1x = left_edge + span1 / 2 + 0.01
    p1 = F.pallet((p1x, yb - 0.05, 0.0006), p1_yaw, "chokepoint")
    F.load_boxes(p1, "chokepoint", F.pick_dims(2), 0.4, max_boxes=4)
    p2_yaw = math.pi / 2 + math.radians(12.0)
    span2 = PAL_L * abs(math.cos(p2_yaw)) + PAL_W * abs(math.sin(p2_yaw))
    p2x, p2y = target_x - span2 / 2, yb + 0.05
    for _ in range(60):  # keep both pallets on the same y rows so the squeeze is a true cross-section
        if not rect_overlap_2d((p1x, yb - 0.05), (PAL_L / 2, PAL_W / 2), p1_yaw, (p2x, p2y), (PAL_L / 2, PAL_W / 2), p2_yaw, 0.015):
            break
        p2y += 0.02
    # slide pallet 2 so the exact scanline gap to the opposite rack/load is `clear` (not just AABB-based)
    gap_now = right_squeeze(kit.records, (p2x, p2y), p2_yaw, ax1)
    p2x += gap_now - clear
    F.pallet((p2x, p2y, 0.0006), p2_yaw, "chokepoint")
    target_x = p2x + span2 / 2
    # cones flank the hazard on the blocked side
    F.cone(ax0 + 1.0, yb - 2.1, "chokepoint")
    F.cone(ax0 + 1.0, yb + 2.1, "chokepoint")
    spilled = 0
    for _ in range(200):
        if spilled >= 5:
            break
        d = tuple(float(rng.uniform(0.15, 0.4)) for _ in range(3))
        x = float(rng.uniform(left_edge + 0.2, target_x - 0.25))
        y = float(rng.uniform(yb - 1.9, yb + 1.9))
        yw = float(rng.uniform(0, math.pi))
        hx, hy = d[0] / 2, d[1] / 2
        ex = abs(math.cos(yw)) * hx + abs(math.sin(yw)) * hy
        if x + ex > target_x - 0.02:
            continue
        if any(rect_overlap_2d((x, y), (hx, hy), yw, (cx, cy), (PAL_L / 2, PAL_W / 2), cyaw, 0.03) for cx, cy, cyaw in
               [(p1x, yb - 0.05, p1_yaw), (p2x, p2y, p2_yaw)]):
            continue
        if not plan_free_local(F, x, y, hx, hy, yw):
            continue
        F.loose_box(x, y, yw, "chokepoint", dims=d)
        spilled += 1
        F.plan.add(x, y, hx, hy, yw, "spill")
    choke = {"aisle": 2, "aisle_x_range": [ax0, ax1], "blocked_until_x": target_x, "free_edge_right_x": right_edge,
             "target_clearance_m": clear, "y_center": yb, "y_range": [yb - 1.9, yb + 1.9]}
    goals = make_goals(L, slots, forced, choke, S)
    for gl in goals:  # keep goals and their approach free of floor clutter
        plan.add(gl["base_pose"][0], gl["base_pose"][1], 0.7, 0.7, 0.0, "goal")

    # ---- cage contents
    cx0, cy0, cx1, cy1 = LY.CAGE
    wl = F.wrapped_load((cx0 + 1.1, cy1 - 1.0, 0.0006), 0.1, "cage", F.pick_dims(3), 1.0)
    p = F.pallet((cx1 - 1.2, cy1 - 1.2, 0.0006), -0.2, "cage")
    F.load_boxes(p, "cage", F.pick_dims(4), 1.0, max_boxes=12)
    p = F.pallet((cx0 + 1.3, cy0 + 1.1, 0.0006), 1.4, "cage")
    F.load_boxes(p, "cage", F.pick_dims(2), 0.8, max_boxes=12)

    # ---- packing zone dynamics
    p = F.pallet_of_totes((18.9, 10.4, 0.0006), math.pi / 2, "packing", None)
    plan.add(18.9, 10.4, 0.45, 0.65, 0.0, "tote_pallet")
    tz = table_top
    for i, dx in enumerate((-0.4, 0.4)):
        F.tote((17.2 + dx, 8.3 + 0.0, tz + 0.001), 0.0, "packing")
    for bname, bx, ys in (("A", 15.2, (2.6, 3.6)), ("B", 17.2, (2.6, 3.8))):
        for y in ys:
            d = (float(rng.uniform(0.18, 0.45)), float(rng.uniform(0.18, 0.45)), float(rng.uniform(0.15, 0.35)))
            nm, mass = F.box((bx, y, 0.8 + d[2] / 2 + 0.0015), qyaw(float(rng.uniform(-0.1, 0.1))), d,
                             int(rng.integers(N_CARDBOARD)), "packing", extra={"on_conveyor": bname})
    d = (0.4, 0.3, 0.2)
    F.box((15.2, 6.6, 0.8 + 0.1 + 0.0015), Q_IDENT, d, 1, "packing", extra={"on_conveyor": "Roller"})
    F.trash_bin(19.2, 1.9, "packing")

    # ---- floor clutter
    regions = [
        ("dock_front", -17.0, -12.0, 1.0, -9.9),
        ("east_lane", 11.4, -7.2, 13.4, 2.0),
        ("north_lane", -18.5, 9.9, 10.0, 12.2),
        ("charging", 12.8, -12.0, 19.4, -8.6),
        ("packing", 13.9, 1.2, 19.4, 12.2),
        ("cross_aisle_1", -17.0, -7.4, 9.0, -4.4),
        ("cross_aisle_2", -17.0, 1.2, 9.0, 4.2),
    ]
    n_loose = 0
    for _ in range(600):
        if n_loose >= 16:
            break
        name, x0, y0, x1, y1 = regions[int(rng.integers(len(regions)))]
        d = tuple(float(rng.uniform(0.15, 0.6)) for _ in range(3))
        yw = float(rng.uniform(0, math.pi))
        hx, hy = d[0] / 2, d[1] / 2
        x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
        ex = abs(math.cos(yw)) * hx + abs(math.sin(yw)) * hy
        ey = abs(math.sin(yw)) * hx + abs(math.cos(yw)) * hy
        if not (x0 <= x - ex and x + ex <= x1 and y0 <= y - ey and y + ey <= y1):
            continue
        if not plan.free(x, y, ex, ey, 0.0, 0.2) or not plan.in_bounds(x, y, ex, ey, 0.0):
            continue
        (path, mass), dd = F.loose_box(x, y, yw, name, dims=d)
        plan.add(x, y, ex, ey, 0.0, "loose_box")
        n_loose += 1
        if n_loose % 5 == 0:  # a second, smaller carton stacked on top
            d2 = (min(d[0], float(rng.uniform(0.15, 0.4))), min(d[1], float(rng.uniform(0.15, 0.4))), float(rng.uniform(0.15, 0.3)))
            F.loose_box(x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02), yw + rng.uniform(-0.15, 0.15), name,
                        dims=d2, z0=d[2] + 0.0015)
    # tipped pallet jack + cones
    for _ in range(300):
        x, y = float(rng.uniform(-17, 9)), float(rng.uniform(-12.0, -9.9))
        yw = float(rng.uniform(0, math.pi))
        if plan.free(x, y, 0.8, 0.8, yw, 0.3) and plan.in_bounds(x, y, 0.8, 0.8, yw):
            F.pallet_jack_tipped(x, y, yw, "dock_front")
            break
    n_c = 0
    for _ in range(300):
        if n_c >= 3:
            break
        x, y = float(rng.uniform(-17, 12)), float(rng.uniform(-12.0, -9.9))
        if plan.free(x, y, 0.2, 0.2, 0.0, 0.2) and plan.in_bounds(x, y, 0.2, 0.2, 0.0):
            F.cone(x, y, "dock_front")
            n_c += 1


    # ---- lights / cameras
    lighting_info = build_lights(kit, rng, setup, light_scale)
    kit.scope("/World/Cameras")
    ov = build_camera(kit, "/World/Cameras/Overhead", (0, 0, 23.0), Q_IDENT, focal=10.5, hap=20.955, vap=11.79, clip=(1.0, 200.0))
    kit.attr(ov, "warehouse:note", T.String, "above the roof: hide /World/Building/Roof (UsdGeom.Imageable.MakeInvisible) before rendering")
    a1 = L.aisles[1][2]
    aq = look_at_quat((a1, -8.0, 1.7), (a1, 8.0, 2.6))
    build_camera(kit, "/World/Cameras/AisleLevel", (a1, -8.0, 1.7), aq, focal=14.0)
    cams = [
        {"name": "overhead", "path": "/World/Cameras/Overhead", "position": [0, 0, 23.0], "note": "hide roof for render"},
        {"name": "aisle_level", "path": "/World/Cameras/AisleLevel", "position": [a1, -8.0, 1.7]},
        {"name": "robot", "path": robot["camera"], "mounted_on": robot["prim"]},
    ]
    for cinfo in cams:
        kit.record(cinfo["path"], "camera", cinfo.get("position", robot["spawn"][:2] + [1.25]), Q_IDENT, "none")

    # ---- save USD + scene graph
    kit.stage.SetStartTimeCode(0)
    kit.stage.SetEndTimeCode(0)
    kit.stage.GetRootLayer().documentation = f"Procedural warehouse seed={seed} (generator {VERSION})"
    kit.save()
    graph = build_scene_graph(seed, kit, F, S, goals, hazards, robot, lighting_info, cams, choke, mk, wear, slope, classes, L, t0)
    (out / "scene_graph.json").write_text(json.dumps(graph, indent=1))
    return out



def aisle_free_edges(records, ax0, ax1, y_range, zr=(0.1, 1.8)):
    """Left/right free x-edges of an aisle over y_range, from the actual rack beams / pallet loads (not nominal bounds)."""
    from .geom import obb_from_pose

    left, right = ax0 - 0.5, ax1 + 0.5
    for r in records:
        boxes = []
        if r["category"] == "rack_row":
            boxes = [(tuple(p["position"]), (1, 0, 0, 0), tuple(p["size"])) for p in r["parts"] if p["type"] in ("beam", "upright", "guard")]
        elif r["category"] in ("pallet", "box", "tote", "tote_item") and "size" in r:
            pos = list(r["position"])
            if r["category"] == "pallet":
                pos[2] += r["size"][2] / 2
            size = tuple(r["size"]) if len(r["size"]) == 3 else (2 * r["size"][0], 2 * r["size"][0], r["size"][1])
            boxes = [(tuple(pos), tuple(r["orientation_wxyz"]), size)]
        for pos, q, size in boxes:
            lo, hi = obb_from_pose(pos, q, size).aabb()
            if hi[1] < y_range[0] or lo[1] > y_range[1] or hi[2] < zr[0] or lo[2] > zr[1]:
                continue
            if hi[0] <= ax0 + 0.6:
                left = max(left, hi[0])
            elif lo[0] >= ax1 - 0.6:
                right = min(right, lo[0])
    return left, right


def right_squeeze(records, p2_xy, p2_yaw, ax1, zr=(0.1, 1.8)):
    """Min over scanlines of (opposite-rack structure x - pallet-2 right edge) for rows where pallet 2 exists."""
    from .geom import footprint, obb_from_pose, scan_interval

    polys = []
    for r in records:
        boxes = []
        if r["category"] == "rack_row":
            boxes = [(tuple(p["position"]), (1, 0, 0, 0), tuple(p["size"])) for p in r["parts"] if p["type"] in ("beam", "upright", "guard")]
        elif r["category"] in ("pallet", "box", "tote", "tote_item") and "size" in r:
            pos = list(r["position"])
            size = tuple(r["size"]) if len(r["size"]) == 3 else (2 * r["size"][0], 2 * r["size"][0], r["size"][1])
            if r["category"] == "pallet":
                pos[2] += size[2] / 2
            boxes = [(tuple(pos), tuple(r["orientation_wxyz"]), size)]
        for pos, q, size in boxes:
            o = obb_from_pose(pos, q, size)
            lo, hi = o.aabb()
            if hi[2] < zr[0] or lo[2] > zr[1] or lo[0] < ax1 - 0.7 or lo[0] > ax1 + 0.6:
                continue
            polys.append(footprint(o))
    pal = footprint(obb_from_pose((p2_xy[0], p2_xy[1], 0.07), qyaw(p2_yaw), (PAL_L, PAL_W, 0.144)))
    ys = np.arange(p2_xy[1] - 0.7, p2_xy[1] + 0.7, 0.01)
    best = 1e9
    for y in ys:
        pi = scan_interval(pal, y)
        if not pi:
            continue
        xs = [iv[0] for iv in (scan_interval(p, y) for p in polys) if iv]
        if xs:
            best = min(best, min(xs) - pi[1])
    return best


def plan_free_local(F, x, y, hx, hy, yw):
    """Spill boxes only need to avoid each other: check against loose boxes registered with tag 'spill'."""
    for cx, cy, h2x, h2y, y2, tag in F.plan.items:
        if tag in ("spill", "cone") and rect_overlap_2d((x, y), (hx, hy), yw, (cx, cy), (h2x, h2y), y2, 0.03):
            return False
    return True


def make_goals(L, slots, forced, choke, S):
    goals = []

    def add(name, typ, x, y, yaw, **extra):
        goals.append({"id": len(goals), "name": name, "type": typ, "base_pose": [round(x, 3), round(y, 3), round(yaw, 4)], **extra})

    for i, (px, py, pyaw) in enumerate(LY.PARKING):
        add(f"parking_{i}", "parking", px, py, pyaw)
    for i, x in enumerate(LY.DOCK_BAYS_X):
        add(f"dock_bay_{i}", "dock", x, -10.5, -math.pi / 2)
    add("belt_A_infeed", "packing", 15.2, 0.9, math.pi / 2)
    add("belt_B_infeed", "packing", 17.2, 0.9, math.pi / 2)
    add("roller_end", "packing", 15.2, 10.3, -math.pi / 2)
    add("work_table", "packing", 17.2, 10.0, -math.pi / 2)
    add("cage_gate", "cage", 16.9, -8.7, math.pi / 2)
    for (row, seg, bay, slot, level), s in forced.items():
        face = s["face"]
        xfront = s["x"] + face * 0.45
        rx = xfront + face * 0.95
        yaw = math.pi if face == 1 else 0.0
        add(f"pick_{LY.LEVELS[level]:.1f}m_row{row}", "pick_high", rx, s["y"], yaw,
            pick={"target_xyz": [round(s["x"] + face * 0.4, 3), round(s["y"], 3), round(s["z"] + 0.35, 3)],
                  "shelf_level_m": LY.LEVELS[level], "rack_row": row, "segment": seg, "bay": bay, "slot": slot,
                  "needs_lift": True})
    ax0, ax1, axc = L.aisles[2]
    gap_c = (choke["blocked_until_x"] + choke["free_edge_right_x"]) / 2
    add("choke_before", "chokepoint", gap_c, choke["y_center"] - 2.6, math.pi / 2)
    add("choke_beyond", "chokepoint", gap_c, choke["y_center"] + 2.6, math.pi / 2)
    add("staging_pickup", "staging", -9.9, -10.2, 0.0)
    return goals


def build_scene_graph(seed, kit, F, S, goals, hazards, robot, lighting, cams, choke, mk, wear, slope, classes, L, t0):
    recs = kit.records
    counts = {}
    for r in recs:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    dyn = [r for r in recs if r["body_type"] == "dynamic"]
    mis = F.stats
    return {
        "seed": seed,
        "generator_version": VERSION,
        "units": {"length": "m", "mass": "kg", "up_axis": "Z"},
        "usd": "warehouse.usd",
        "generation_seconds": round(time.time() - t0, 2),
        "building": {"size_m": [LY.W, LY.D, LY.H], "x_range": [LY.X0, LY.X1], "y_range": [LY.Y0, LY.Y1]},
        "layout": {
            "aisles": [{"index": i, "x_range": [a[0], a[1]], "x_center": a[2], "width_m": LY.AISLE_W} for i, a in enumerate(L.aisles)],
            "cross_aisles": [{"y_range": list(c), "width_m": c[1] - c[0]} for c in LY.CROSS_AISLES],
            "rack_levels_m": list(LY.LEVELS),
            "rack_segments_y": [list(s) for s in LY.SEGS],
            "zones": {"dock": [LY.X0, LY.Y0, 0.0, -9.4], "packing": list(LY.PACK_ZONE), "charging": list(LY.CHARGE_ZONE), "cage": list(LY.CAGE)},
            "slope_zones": [{"bbox": list(z), "gradient": round(slope, 4), "max_height_m": 0.04} for z in LY.SLOPE_ZONES],
        },
        "chokepoint": choke,
        "floor": {"wear": round(wear, 3), "markings": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in mk_.items()} for mk_ in mk]},
        "lighting": lighting,
        "box_classes_m": [[round(d, 3) for d in c] for c in classes],
        "physics_materials": {"cardboard": 0.5, "steel": 0.6, "concrete": 0.7, "rubber_belt": 0.9, "wood": 0.5, "plastic": 0.4},
        "box_misalignment": {
            "max_xy_cm": round(max(mis["misalign_xy_cm"] or [0]), 2),
            "max_yaw_deg": round(max(mis["misalign_yaw_deg"] or [0]), 2),
            "mean_xy_cm": round(float(np.mean(mis["misalign_xy_cm"] or [0])), 2),
            "mean_yaw_deg": round(float(np.mean(mis["misalign_yaw_deg"] or [0])), 2),
        },
        "counts": counts,
        "n_dynamic_bodies": len(dyn),
        "conveyors": S.conveyors,
        "articulations": S.articulations,
        "hazards": [{"path": h["path"], "kind": h["subtype"], "waypoints": h["waypoints"], "z": h["z"], "speed_mps": h["speed_mps"],
                     "path_mode": h["path_mode"], "path_phase_m": h["path_phase_m"]} for h in hazards],
        "robot": robot,
        "cameras": cams,
        "goals": goals,
        "objects": recs,
    }


def main(argv=None):
    from .isaac_env import gate_or_exit

    gate_or_exit()
    ap = argparse.ArgumentParser(description="Generate a procedural Isaac Sim warehouse (USD + scene graph).")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="warehouse_out")
    ap.add_argument("--lighting", type=int, choices=[0, 1, 2], default=None, help="0 bright, 1 mixed warm, 2 dim (default: seed %% 3)")
    ap.add_argument("--light-scale", type=float, default=1.0)
    a = ap.parse_args(argv)
    for s in a.seed:
        p = build(s, Path(a.out), a.lighting, a.light_scale)
        print(f"seed {s}: wrote {p}/warehouse.usd")


if __name__ == "__main__":
    main()
