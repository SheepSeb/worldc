"""Isaac Sim (headless) validation of one generated scene.  Run with Isaac's python:

    ~/isaacsim/python.sh -m claude_scene_generator.isaac_validate warehouse_out/seed_0   (PYTHONPATH=src)

Stages: load -> schema check -> 10 s physics settle (+ hazards) -> conveyor speed -> camera renders ->
articulation sweeps (open/close every door/drawer, stuck + interpenetration checks).
Writes <seed_dir>/isaac_validation.json and renders/*.png.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("seed_dir")
ap.add_argument("--sim-seconds", type=float, default=10.0)
ap.add_argument("--skip-render", action="store_true")
ap.add_argument("--skip-art", action="store_true")
ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res[0], "height": args.res[1]})

import carb  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

from claude_scene_generator.colliders import collect  # noqa: E402
from claude_scene_generator.geom import OBB, obb_penetration, qmat  # noqa: E402
from claude_scene_generator.runtime import ArticulationSweep, hazard_followers, load_scene_graph, yaw_to_quat_wxyz  # noqa: E402

SEED_DIR = Path(args.seed_dir).resolve()
OUT = {"seed_dir": str(SEED_DIR), "stages": {}}
T0 = time.time()


def log(*a):
    print(f"[validate +{time.time() - T0:6.1f}s]", *a, flush=True)


def save():
    (SEED_DIR / "isaac_validation.json").write_text(json.dumps(OUT, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


graph = load_scene_graph(SEED_DIR / "scene_graph.json")

# ------------------------------------------------------------------ 1. load
ctx = omni.usd.get_context()
log("opening stage")
ok = ctx.open_stage(str(SEED_DIR / "warehouse.usd")); msg = ""
while ctx.get_stage_loading_status()[2] > 0:
    app.update()
for _ in range(10):
    app.update()
stage = ctx.get_stage()
n_prims = sum(1 for _ in stage.Traverse())
OUT["stages"]["load"] = {"opened": bool(ok), "message": str(msg), "n_prims": n_prims, "ok": bool(ok) and n_prims > 1000}
log("loaded", n_prims, "prims")

# ------------------------------------------------------------------ 2. schema check (raw attrs are understood by PhysxSchema)
sch = {"ok": True, "checked": []}
try:
    rb = stage.GetPrimAtPath(graph["objects"][[o["category"] for o in graph["objects"]].index("box")]["path"])
    pr = PhysxSchema.PhysxRigidBodyAPI(rb)
    sch["rigid_body_api_applied"] = rb.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
    sch["solver_position_iterations"] = pr.GetSolverPositionIterationCountAttr().Get()
    belt = stage.GetPrimAtPath(graph["conveyors"][0]["belt_path"])
    sv = PhysxSchema.PhysxSurfaceVelocityAPI(belt)
    sch["surface_velocity_api_applied"] = belt.HasAPI(PhysxSchema.PhysxSurfaceVelocityAPI)
    sch["belt_surface_velocity"] = list(sv.GetSurfaceVelocityAttr().Get())
    scn = stage.GetPrimAtPath("/World/PhysicsScene")
    ps = PhysxSchema.PhysxSceneAPI(scn)
    sch["scene_api_applied"] = scn.HasAPI(PhysxSchema.PhysxSceneAPI)
    sch["solver_type"] = ps.GetSolverTypeAttr().Get()
    sch["steps_per_second"] = ps.GetTimeStepsPerSecondAttr().Get()
    art = graph["articulations"][0]
    sch["articulation_root_api"] = stage.GetPrimAtPath(art["root"]).HasAPI(UsdPhysics.ArticulationRootAPI)
    sch["ok"] = bool(sch["rigid_body_api_applied"] and sch["surface_velocity_api_applied"] and sch["scene_api_applied"] and sch["articulation_root_api"]
                     and abs(sch["belt_surface_velocity"][1] - 0.5) < 1e-6 and sch["solver_type"] == "TGS")
except Exception as e:  # noqa: BLE001
    sch["ok"] = False
    sch["error"] = repr(e)
OUT["stages"]["schema"] = sch
log("schema", sch)

# initial colliders (before any simulation)
COLS = collect(stage)
log("collected", len(COLS), "colliders")

# ------------------------------------------------------------------ 3. physics
from isaacsim.core.api import SimulationContext  # noqa: E402

DT = 1.0 / 120.0
sim = SimulationContext(physics_dt=DT, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0, backend="numpy")
sim.initialize_physics()
sim.play()
sim.step(render=False)
import omni.physics.tensors as tensors  # noqa: E402

sv = tensors.create_simulation_view("numpy")
sv.initialize_kinematic_bodies()

objs = graph["objects"]
dyn = [o for o in objs if o["body_type"] == "dynamic"]
conv_payload = {o["path"] for o in dyn if o.get("on_conveyor")}
art_roots = {a["link"] for a in graph["articulations"]}
dyn_paths = [o["path"] for o in dyn]
kin_paths = [h["path"] for h in graph["hazards"]]
dview = sv.create_rigid_body_view(dyn_paths)
kview = sv.create_rigid_body_view(kin_paths)
log("views", dview.count, "dynamic bodies;", kview.count, "kinematic hazards")
OUT["stages"]["load"]["n_dynamic_bodies"] = int(dview.count)

followers = hazard_followers(graph)
kin_idx = np.arange(kview.count, dtype=np.int32)


def drive_hazards(t):
    data = np.zeros((kview.count, 7), np.float32)
    for i, p in enumerate(kin_paths):
        x, y, z, yaw = followers[p].pose(t)
        q = yaw_to_quat_wxyz(yaw)
        data[i] = [x, y, z, 0.0, 0.0, q[3], q[0]]  # xyzw
    kview.set_kinematic_targets(data, kin_idx)


p0 = dview.get_transforms()[:, :3].copy()
v_max = np.zeros(len(dyn_paths))
z_min = p0[:, 2].copy()
n_steps = int(round(args.sim_seconds / DT))
snap = {}
payload_idx = [i for i, p in enumerate(dyn_paths) if p in conv_payload]
payload_track = {i: [] for i in payload_idx}
log("stepping physics", n_steps, "steps")
for k in range(n_steps):
    t = k * DT
    drive_hazards(t)
    sim.step(render=False)
    tr = dview.get_transforms()
    vel = dview.get_velocities()
    sp = np.linalg.norm(vel[:, :3], axis=1)
    v_max = np.maximum(v_max, sp)
    z_min = np.minimum(z_min, tr[:, 2])
    if k % 12 == 0:
        for i in payload_idx:
            payload_track[i].append((t, float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2])))
    if k == n_steps // 2:
        snap["mid"] = tr[:, :3].copy()
    if k % 240 == 0:
        log(f"  t={t:5.2f}s  max|v|={sp.max():.3f}")
tr = dview.get_transforms()
pF = tr[:, :3]
disp = np.linalg.norm(pF - p0, axis=1)
disp_late = np.linalg.norm(pF - snap["mid"], axis=1)
mask = np.array([p not in conv_payload for p in dyn_paths])
frac = float((disp[mask] < 0.02).mean())
frac_late = float((disp_late[mask] < 0.02).mean())
fell = [dyn_paths[i] for i in range(len(dyn_paths)) if z_min[i] < min(p0[i, 2], 0.0) - 0.05 or pF[i, 2] < p0[i, 2] - 0.3]
fast = [{"body": dyn_paths[i], "max_speed_mps": round(float(v_max[i]), 3)} for i in range(len(dyn_paths)) if v_max[i] > 1.0 and mask[i]]
moved = [{"body": dyn_paths[i], "displacement_m": round(float(disp[i]), 3)} for i in np.argsort(-disp) if disp[i] >= 0.02 and mask[i]][:25]
# conveyor payload speed (steady phase 1.5 .. 3.5 s)
speeds = {}
for i, tl in payload_track.items():
    seg = [s for s in tl if 1.5 <= s[0] <= 3.5]
    if seg:
        speeds[dyn_paths[i]] = {"mean_vy_mps": round(float(np.mean([s[2] for s in seg])), 3),
                                "mean_vx_mps": round(float(np.mean([s[1] for s in seg])), 3)}
belt_ok = all(abs(v["mean_vy_mps"] - 0.5) <= 0.05 and abs(v["mean_vx_mps"]) <= 0.05 for v in speeds.values()) and len(speeds) > 0
OUT["stages"]["physics"] = {
    "sim_seconds": args.sim_seconds,
    "physics_dt": DT,
    "n_bodies_evaluated": int(mask.sum()),
    "excluded_conveyor_payload": len(conv_payload),
    "fraction_displacement_below_2cm": round(frac, 4),
    "fraction_late_drift_below_2cm_t5_to_t10": round(frac_late, 4),
    "target": 0.95,
    "max_displacement_m": round(float(disp[mask].max()), 4),
    "median_displacement_m": round(float(np.median(disp[mask])), 5),
    "fell_through_floor": fell,
    "faster_than_1mps": fast,
    "worst_movers": moved,
    "conveyor_payload_speeds": speeds,
    "conveyor_ok": bool(belt_ok),
    "hazards_final": {p: [round(float(c), 2) for c in kview.get_transforms()[i, :3]] for i, p in enumerate(kin_paths)},
    "ok": bool(frac >= 0.95 and not fell and not fast and belt_ok),
}
log("physics", {k: v for k, v in OUT["stages"]["physics"].items() if k not in ("worst_movers", "hazards_final")})
save()

# ------------------------------------------------------------------ 4. renders
RENDER = {"ok": True, "images": []}
if not args.skip_render:
    try:
        import omni.replicator.core as rep
        from PIL import Image

        (SEED_DIR / "renders").mkdir(exist_ok=True)
        roof = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Building/Roof"))
        for cam in graph["cameras"]:
            name = cam["name"]
            if name == "overhead":
                roof.MakeInvisible()
            rp = rep.create.render_product(cam["path"], tuple(args.res))
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            for _ in range(40):
                app.update()
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=32, delta_time=0.0, pause_timeline=False)
            data = ann.get_data()
            img = np.asarray(data)[..., :3].astype(np.uint8)
            out = SEED_DIR / "renders" / f"{name}.png"
            Image.fromarray(img).save(out)
            lum = img.astype(np.float32).mean(axis=2)
            magenta = float(((img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] > 200)).mean())
            st = {"camera": name, "file": str(out.relative_to(SEED_DIR)), "mean_luminance": round(float(lum.mean()), 1),
                  "std_luminance": round(float(lum.std()), 1), "frac_black": round(float((lum < 8).mean()), 4),
                  "frac_saturated": round(float((lum > 247).mean()), 4), "frac_magenta_missing_tex": round(magenta, 5)}
            st["ok"] = bool(st["mean_luminance"] > 10 and st["std_luminance"] > 8 and st["frac_black"] < 0.6 and st["frac_saturated"] < 0.35 and magenta < 0.001)
            RENDER["images"].append(st)
            RENDER["ok"] &= st["ok"]
            if name == "overhead":
                roof.MakeVisible()
            log("rendered", name, st)
    except Exception as e:  # noqa: BLE001
        RENDER["ok"] = False
        RENDER["error"] = traceback.format_exc()
        log("render failed", e)
else:
    RENDER["skipped"] = True
OUT["stages"]["render"] = RENDER
save()

# ------------------------------------------------------------------ 5. articulation sweeps
ART = {"ok": True, "joints": []}
if not args.skip_art:
    try:
        by_body = {}
        for i, c in enumerate(COLS):
            by_body.setdefault(c.body, []).append(i)
        lo_all = np.array([c.obb.aabb()[0] for c in COLS])
        hi_all = np.array([c.obb.aabb()[1] for c in COLS])
        for a in graph["articulations"]:
            root = a["link"]  # moving link; a["root"] is the world-fixed articulation base
            av = sv.create_articulation_view(a["root"])
            rv = sv.create_rigid_body_view(root)
            n_dof = int(av.max_dofs)
            lim = np.array([a["lower"], a["upper"]], float)
            if a["unit"] == "deg":
                lim = np.deg2rad(lim)
            sweep = ArticulationSweep(lim[0], lim[1], t_open=3.0, t_hold=0.5)
            # moving colliders relative to root body initial pose
            Tinit = rv.get_transforms()[0]
            R0 = qmat((Tinit[6], Tinit[3], Tinit[4], Tinit[5]))
            p_root0 = Tinit[:3].astype(float)
            movers = []
            for ci in by_body.get(root, []):
                c = COLS[ci]
                movers.append((ci, R0.T @ (c.obb.c - p_root0), R0.T @ c.obb.R, c.obb.h))
            others = np.ones(len(COLS), bool)
            others[by_body.get(root, [])] = False
            idx0 = np.array([0], np.int32)
            max_pen, pen_pair, max_err, samples = 0.0, None, 0.0, []
            steps = int((sweep.duration + 1.5) / DT)  # 1.5 s hold at the closed target before judging the return
            q_hist = []
            for k in range(steps):
                tt = k * DT
                tgt = sweep.target(tt)
                av.set_dof_position_targets(np.array([[tgt]], np.float32).reshape(1, n_dof), idx0)
                drive_hazards(args.sim_seconds + tt)
                sim.step(render=False)
                if k % 12 == 0:
                    q = float(av.get_dof_positions()[0, 0])
                    q_hist.append((tt, q, tgt))
                    max_err = max(max_err, abs(q - tgt))
                    Tn = rv.get_transforms()[0]
                    Rn, pn = qmat((Tn[6], Tn[3], Tn[4], Tn[5])), Tn[:3].astype(float)
                    for ci, pr, Rr, h in movers:
                        o = OBB(pn + Rn @ pr, h, Rn @ Rr)
                        lo, hi = o.aabb()
                        cand = np.nonzero(others & np.all(lo < hi_all, axis=1) & np.all(hi > lo_all, axis=1))[0]
                        for cj in cand:
                            if COLS[cj].is_mesh:
                                continue
                            pen = obb_penetration(o, COLS[cj].obb)
                            if pen > max_pen:
                                max_pen, pen_pair = pen, (COLS[ci].path, COLS[cj].path, round(tt, 2))
            qs = np.array([h[1] for h in q_hist])
            reached_open = float(qs.max()) >= lim[1] - 0.03 * (lim[1] - lim[0])
            returned_closed = abs(float(qs[-1]) - lim[0]) <= 0.03 * (lim[1] - lim[0])
            res = {"name": a["name"], "kind": a["kind"], "limits": [float(lim[0]), float(lim[1])], "unit_rad_or_m": True,
                   "max_position_reached": round(float(qs.max()), 4), "final_position": round(float(qs[-1]), 4),
                   "max_tracking_error": round(float(max_err), 4), "tracking_error_frac_of_range": round(float(max_err / (lim[1] - lim[0])), 3),
                   "reached_full_open": bool(reached_open), "returned_closed": bool(returned_closed),
                   "max_interpenetration_mm": round(max_pen * 1000, 2), "worst_pair": pen_pair}
            res["ok"] = bool(reached_open and returned_closed and max_err / (lim[1] - lim[0]) < 0.15 and max_pen < 0.005)
            ART["joints"].append(res)
            ART["ok"] &= res["ok"]
            log("articulation", res)
    except Exception:  # noqa: BLE001
        ART["ok"] = False
        ART["error"] = traceback.format_exc()
        log("articulation failed", ART["error"])
else:
    ART["skipped"] = True
OUT["stages"]["articulation"] = ART
save()
log("done")
app.close()
