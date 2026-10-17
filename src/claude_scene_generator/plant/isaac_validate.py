"""Isaac Sim (headless) validation of one generated plant seed.  Run with Isaac's python:

    PYTHONPATH=src ~/isaacsim/python.sh -m claude_scene_generator.plant.isaac_validate plant_out/seed_0

Stages: load -> schema -> 10 s physics settle -> actuate every joint through its full range (handwheels, valve
stems, doors, latches, drawers, buttons, knobs, rotor) -> camera renders.  Writes <seed>/isaac_validation.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("seed_dir")
ap.add_argument("--sim-seconds", type=float, default=10.0)
ap.add_argument("--skip-render", action="store_true")
ap.add_argument("--skip-art", action="store_true")
ap.add_argument("--res", type=int, nargs=2, default=[1200, 800])
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res[0], "height": args.res[1]})

import omni.usd  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

from claude_scene_generator.colliders import collect, shrunk  # noqa: E402
from claude_scene_generator.geom import OBB, obb_penetration, qmat  # noqa: E402
from claude_scene_generator.runtime import ArticulationSweep  # noqa: E402

SEED_DIR = Path(args.seed_dir).resolve()
OUT = {"seed_dir": str(SEED_DIR), "stages": {}}
T0 = time.time()
PEN_TOL = 0.005


def log(*a):
    print(f"[validate +{time.time() - T0:6.1f}s]", *a, flush=True)


def save():
    (SEED_DIR / "isaac_validation.json").write_text(json.dumps(OUT, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


graph = json.loads((SEED_DIR / "scene_graph.json").read_text())

# ------------------------------------------------------------------ 1. load
ctx = omni.usd.get_context()
ok = ctx.open_stage(str(SEED_DIR / "plant.usd"))
while ctx.get_stage_loading_status()[2] > 0:
    app.update()
for _ in range(10):
    app.update()
stage = ctx.get_stage()
n_prims = sum(1 for _ in stage.Traverse())
OUT["stages"]["load"] = {"opened": bool(ok), "n_prims": n_prims, "ok": bool(ok) and n_prims > 500}
log("loaded", n_prims, "prims")

# ------------------------------------------------------------------ 2. schema checks
sch = {"ok": True}
try:
    valves = [v for v in graph["valve_control_map"] if v["kind"] == "gate_valve"]
    mim = 0
    for v in valves:
        sj = stage.GetPrimAtPath("/World/Equipment/" + v["stem_joint"].replace("/J_", "/J_"))
        mim += int(sj.HasAPI(PhysxSchema.PhysxMimicJointAPI, "rotZ"))
    sch["gate_valves"] = len(valves)
    sch["with_mimic_joint_api"] = mim
    sch["scene_solver"] = PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath("/World/PhysicsScene")).GetSolverTypeAttr().Get()
    sch["articulation_roots"] = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    sch["ok"] = bool(mim == len(valves) and sch["scene_solver"] == "TGS" and sch["articulation_roots"] == len(graph["articulations"]))
except Exception as e:  # noqa: BLE001
    sch["ok"] = False
    sch["error"] = repr(e)
OUT["stages"]["schema"] = sch
log("schema", sch)

COLS = collect(stage)
_xc = UsdGeom.XformCache(Usd.TimeCode.Default())
P0_USD = {p.GetPath().pathString: np.array(_xc.GetLocalToWorldTransform(p).ExtractTranslation(), float) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}

# ------------------------------------------------------------------ 3. physics settle
from isaacsim.core.api import SimulationContext  # noqa: E402

DT = 1.0 / 120.0
sim = SimulationContext(physics_dt=DT, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0, backend="numpy")
sim.initialize_physics()
sim.play()
sim.step(render=False)
import omni.physics.tensors as tensors  # noqa: E402

sv = tensors.create_simulation_view("numpy")

rb_paths, rb_root, rb_kind = [], [], []
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        ka = UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr()
        if ka and ka.Get():
            continue
        rb_paths.append(p.GetPath().pathString)
        role = p.GetAttribute("plant:role")
        rb_root.append(bool(role and role.Get() == "root"))
        if p.GetAttribute("plant:loose") and p.GetAttribute("plant:loose").HasValue():
            rb_kind.append("loose")
        elif p.GetAttribute("plant:hose") and p.GetAttribute("plant:hose").HasValue():
            rb_kind.append("hose")
        elif role and role.Get() == "root":
            rb_kind.append("art_root")
        else:
            rb_kind.append("art_link")
rb_root = np.array(rb_root)
rb_kind = np.array(rb_kind)
dview = sv.create_rigid_body_view(rb_paths)
OUT["stages"]["load"]["n_rigid_bodies"] = len(rb_paths)
p0 = np.array([P0_USD[p] for p in rb_paths])  # authored poses, before any physics step (stricter than the post-step pose)
v_max = np.zeros(len(rb_paths))
z_min = p0[:, 2].copy()
n_steps = int(round(args.sim_seconds / DT))
mid = None
for k in range(n_steps):
    sim.step(render=False)
    tr = dview.get_transforms()
    vel = dview.get_velocities()
    v_max = np.maximum(v_max, np.linalg.norm(vel[:, :3], axis=1))
    z_min = np.minimum(z_min, tr[:, 2])
    if k == n_steps // 2:
        mid = tr[:, :3].copy()
tr = dview.get_transforms()
disp = np.linalg.norm(tr[:, :3] - p0, axis=1)
disp_late = np.linalg.norm(tr[:, :3] - mid, axis=1)
free = ~rb_root
frac_free = float((disp[free] < 0.02).mean())
frac_all = float((disp < 0.02).mean())
fell = [rb_paths[i] for i in range(len(rb_paths)) if z_min[i] < min(p0[i, 2], 0.0) - 0.05]
fast = [{"body": rb_paths[i], "max_speed_mps": round(float(v_max[i]), 3)} for i in range(len(rb_paths)) if v_max[i] > 1.0]
movers = [{"body": rb_paths[i], "displacement_m": round(float(disp[i]), 4), "kind": str(rb_kind[i])} for i in np.argsort(-disp) if disp[i] >= 0.02][:25]
by_kind = {k: dict(n=int((rb_kind == k).sum()), settled=round(float((disp[rb_kind == k] < 0.02).mean()), 4) if (rb_kind == k).any() else None,
                   max_disp_m=round(float(disp[rb_kind == k].max()), 4) if (rb_kind == k).any() else None) for k in ("loose", "hose", "art_link", "art_root")}
OUT["stages"]["physics"] = {
    "sim_seconds": args.sim_seconds, "physics_dt": DT, "n_rigid_bodies": len(rb_paths), "n_free_or_moving_bodies": int(free.sum()),
    "fraction_settled_free_or_moving_below_2cm": round(frac_free, 4), "fraction_settled_all_bodies_below_2cm": round(frac_all, 4),
    "fraction_late_drift_below_2cm_t5_to_t10": round(float((disp_late[free] < 0.02).mean()), 4), "target": 0.95, "by_kind": by_kind,
    "max_displacement_m": round(float(disp[free].max()), 4), "fell_through_floor": fell, "faster_than_1mps": fast, "worst_movers": movers,
    "max_speed_mps_overall": round(float(v_max.max()), 3), "ok": bool(frac_free >= 0.95 and not fell and not fast),
}
log("physics", {k: v for k, v in OUT["stages"]["physics"].items() if k != "worst_movers"})
save()

# ------------------------------------------------------------------ 4. joint sweeps
GAINS = {  # role -> (stiffness, damping, max_force)
    "handwheel": (400.0, 25.0, 1e5), "door": (300.0, 30.0, 1e5), "latch": (2.0, 0.2, 1e3), "drawer": (3000.0, 120.0, 1e5), "push_button": (3000.0, 15.0, 1e4),
    "selector_knob": (2.0, 0.2, 1e3), "control_stem": (2.0e4, 300.0, 1e6),
}
ART = {"ok": True, "joints": []}
if not args.skip_art:
    try:
        jrec = {j["name"]: j for j in graph["joints"]}
        by_body = {}
        for i, c in enumerate(COLS):
            by_body.setdefault(c.body, []).append(i)
        lo_all = np.array([c.obb.aabb()[0] for c in COLS])
        hi_all = np.array([c.obb.aabb()[1] for c in COLS])

        arts = []
        for a in graph["articulations"]:
            av = sv.create_articulation_view(a["root"])
            names = list(av.shared_metatype.dof_names)
            recs = []
            for n in names:
                key = next(k for k in a["joints"] if k.endswith("/" + n))
                recs.append(jrec[key])
            n_dof = len(names)
            kp = np.zeros((1, n_dof), np.float32)
            kd = np.zeros((1, n_dof), np.float32)
            fm = np.full((1, n_dof), 1e6, np.float32)
            for i, r in enumerate(recs):
                if r.get("mimic"):
                    continue
                if r.get("continuous"):  # keep the authored velocity-drive damping
                    kd[0, i] = r["drive"]["damping"]
                    continue
                g = GAINS[r["role"]]
                kp[0, i], kd[0, i], fm[0, i] = g
            idx0 = np.array([0], np.int32)
            av.set_dof_stiffnesses(kp, idx0)
            av.set_dof_dampings(kd, idx0)
            av.set_dof_max_forces(fm, idx0)
            arts.append(dict(a=a, av=av, recs=recs, n=n_dof, idx0=idx0))
        mover_paths = sorted({r["link"] for A in arts for r in A["recs"]})
        mview = sv.create_rigid_body_view(mover_paths)
        mpos0 = mview.get_transforms().copy()
        mover_cols = []  # (link index, collider idx, local pos, local R)
        for li, lp in enumerate(mover_paths):
            T = mpos0[li]
            R0, pp = qmat((T[6], T[3], T[4], T[5])), T[:3].astype(float)
            for ci in by_body.get(lp, []):
                c = shrunk(COLS[ci])
                mover_cols.append((li, ci, R0.T @ (c.c - pp), R0.T @ c.R, c.h))
        grp_of = {i: COLS[i].group for i in range(len(COLS))}
        # sweep time base: wheels get longer ramps
        SW = {}
        for A in arts:
            for i, r in enumerate(A["recs"]):
                if r.get("mimic") or r.get("continuous"):
                    continue
                lim = np.array([r["lower"], r["upper"]], float)
                if r["unit"] == "deg":
                    lim = np.deg2rad(lim)
                SW[(A["a"]["name"], i)] = ArticulationSweep(lim[0], lim[1], t_open=5.0 if r["role"] == "handwheel" else 3.0, t_hold=0.7)
        T_total = max(s.duration for s in SW.values()) + 1.5
        steps = int(T_total / DT)
        hist = {k: [] for k in SW}
        mim_hist = {}
        maxpen = {}
        rot_speed = []
        for k_ in range(steps):
            tt = k_ * DT
            for A in arts:
                tg = np.zeros((1, A["n"]), np.float32)
                for i, r in enumerate(A["recs"]):
                    key = (A["a"]["name"], i)
                    if key in SW:
                        tg[0, i] = SW[key].target(min(tt, SW[key].duration))
                    elif r.get("mimic") or r.get("continuous"):
                        tg[0, i] = 0.0
                A["av"].set_dof_position_targets(tg, A["idx0"])
            sim.step(render=False)
            if k_ % 6 == 0:
                for A in arts:
                    q = A["av"].get_dof_positions()[0]
                    qd = A["av"].get_dof_velocities()[0]
                    for i, r in enumerate(A["recs"]):
                        key = (A["a"]["name"], i)
                        if key in SW:
                            hist[key].append((tt, float(q[i]), float(SW[key].target(min(tt, SW[key].duration)))))
                        elif r.get("mimic"):
                            ref = next(j for j, rr in enumerate(A["recs"]) if rr["path"] == r["mimic"]["reference"])
                            expect = -r["mimic"]["gearing_per_deg"] * math.degrees(float(q[ref]))
                            mim_hist.setdefault((A["a"]["name"], i), []).append((float(q[i]), expect))
                        elif r.get("continuous"):
                            rot_speed.append(math.degrees(float(qd[i])))
                T = mview.get_transforms()
                for (li, ci, pl, Rl, h) in mover_cols:
                    Rn, pn = qmat((T[li][6], T[li][3], T[li][4], T[li][5])), T[li][:3].astype(float)
                    o = OBB(pn + Rn @ pl, h, Rn @ Rl)
                    lo, hi = o.aabb()
                    cand = np.nonzero(np.all(lo < hi_all, axis=1) & np.all(hi > lo_all, axis=1))[0]
                    for cj in cand:
                        if COLS[cj].is_mesh or (grp_of[cj] and grp_of[cj] == grp_of[ci]) or COLS[cj].body == COLS[ci].body:
                            continue
                        pen = obb_penetration(o, shrunk(COLS[cj]))
                        if pen > maxpen.get(ci, (0, None))[0]:
                            maxpen[ci] = (pen, (COLS[ci].path, COLS[cj].path, round(tt, 2)))
        for A in arts:
            for i, r in enumerate(A["recs"]):
                key = (A["a"]["name"], i)
                res = {"name": r["name"], "role": r["role"], "art": A["a"]["name"], "type": r["type"]}
                lim = np.array([r["lower"], r["upper"]], float) if r["lower"] is not None else None
                if key in SW:
                    if r["unit"] == "deg":
                        lim = np.deg2rad(lim)
                    rng_ = lim[1] - lim[0]
                    qs = np.array([h[1] for h in hist[key]])
                    errs = np.array([abs(h[1] - h[2]) for h in hist[key]])
                    pens = [maxpen[ci] for (li, ci, *_r) in mover_cols if COLS[ci].body == r["link"] and ci in maxpen]
                    wp = max(pens, key=lambda x: x[0]) if pens else (0.0, None)
                    res.update(limits=[float(lim[0]), float(lim[1])], max_position_reached=round(float(qs.max()), 4), final_position=round(float(qs[-1]), 4),
                               max_tracking_error_frac=round(float(errs.max() / rng_), 3), reached_full_range=bool(qs.max() >= lim[1] - 0.03 * rng_),
                               returned_closed=bool(abs(qs[-1] - lim[0]) <= 0.03 * rng_), max_interpenetration_mm=round(wp[0] * 1000, 2), worst_pair=wp[1])
                    res["ok"] = bool(res["reached_full_range"] and res["returned_closed"] and res["max_tracking_error_frac"] < 0.2 and wp[0] < PEN_TOL)
                elif r.get("mimic"):
                    h = np.array(mim_hist[key])
                    err = np.abs(h[:, 0] - h[:, 1])
                    stroke = r["upper"] - r["lower"]
                    res.update(mimic_max_error_mm=round(float(err.max() * 1000), 3), stem_max_travel_m=round(float(h[:, 0].max()), 4), stem_stroke_m=round(float(r["upper"]), 4),
                               expected_max_travel_m=round(float(h[:, 1].max()), 4))
                    # the stem must follow the wheel linearly (full multi-turn travel) and come back to zero
                    res["ok"] = bool(err.max() < 0.002 and h[:, 0].max() >= 0.95 * float(h[:, 1].max()) and abs(h[-1, 0]) < 0.002)
                else:
                    sp = np.array(rot_speed[-200:]) if rot_speed else np.zeros(1)
                    res.update(mean_speed_deg_s=round(float(sp.mean()), 2), target_deg_s=r["drive"].get("target_velocity_deg_s"))
                    res["ok"] = bool(abs(sp.mean() - res["target_deg_s"]) < 0.1 * res["target_deg_s"])
                ART["joints"].append(res)
                ART["ok"] &= res["ok"]
        ART["n_joints"] = len(ART["joints"])
        ART["n_ok"] = sum(j["ok"] for j in ART["joints"])
        ART["valve_handwheels"] = sum(j["role"] == "handwheel" for j in ART["joints"])
        ART["control_valve_stems"] = sum(j["role"] == "control_stem" for j in ART["joints"])
        log("articulation", ART["n_ok"], "/", ART["n_joints"], "ok")
    except Exception:  # noqa: BLE001
        ART["ok"] = False
        ART["error"] = traceback.format_exc()
        log("articulation failed", ART["error"])
else:
    ART["skipped"] = True
OUT["stages"]["articulation"] = ART
save()

# ------------------------------------------------------------------ 5. renders (fresh stage state not required; cameras are static)
RENDER = {"ok": True, "images": []}
if not args.skip_render:
    try:
        import omni.replicator.core as rep
        from PIL import Image

        (SEED_DIR / "renders").mkdir(exist_ok=True)
        for cam in graph["cameras"]:
            name = cam["name"]
            hidden = []
            if name == "overhead":
                for pth in ("/World/Structure", "/World/Pipes"):
                    for ch in stage.GetPrimAtPath(pth).GetChildren():
                        if pth == "/World/Structure" or ch.GetName().startswith("rack_"):
                            UsdGeom.Imageable(ch).MakeInvisible()
                            hidden.append(ch)
            rp = rep.create.render_product(cam["path"], tuple(args.res))
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            seg_ann = None
            if name == "photo_view":
                seg_ann = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
                seg_ann.attach([rp])
            for _ in range(40):
                app.update()
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=48, delta_time=0.0, pause_timeline=False)
            img = np.asarray(ann.get_data())[..., :3].astype(np.uint8)
            out = SEED_DIR / "renders" / f"{name}.png"
            Image.fromarray(img).save(out)
            lum = img.astype(np.float32).mean(axis=2)
            magenta = float(((img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] > 200)).mean())
            st = {"camera": name, "file": str(out.relative_to(SEED_DIR)), "mean_luminance": round(float(lum.mean()), 1), "std_luminance": round(float(lum.std()), 1),
                  "frac_black": round(float((lum < 8).mean()), 4), "frac_saturated": round(float((lum > 247).mean()), 4), "frac_magenta_missing_tex": round(magenta, 5)}
            st["ok"] = bool(st["mean_luminance"] > 10 and st["std_luminance"] > 8 and st["frac_black"] < 0.6 and st["frac_saturated"] < 0.35 and magenta < 0.001)
            if seg_ann is not None:
                from claude_scene_generator.plant.viewstats import analyze

                sd = seg_ann.get_data()
                RENDER["reference_view_density"] = analyze(np.asarray(sd["data"]), sd["info"]["idToLabels"])
                (SEED_DIR / "renders" / "photo_view_instance_ids.npy").write_bytes(b"") if False else None
            RENDER["images"].append(st)
            RENDER["ok"] &= st["ok"]
            for ch in hidden:
                UsdGeom.Imageable(ch).MakeVisible()
            log("rendered", name, st)
    except Exception:  # noqa: BLE001
        RENDER["ok"] = False
        RENDER["error"] = traceback.format_exc()
        log("render failed", RENDER["error"])
else:
    RENDER["skipped"] = True
OUT["stages"]["render"] = RENDER
save()
log("done")
app.close()
