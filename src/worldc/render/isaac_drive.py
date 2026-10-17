"""Drive the scene's Isaac Sim robot along a planned route in Isaac Sim physics, optionally capturing frames.  Run with Isaac's python
(``worldc.drive.run_drive`` does this):

    PYTHONPATH=src ~/isaacsim/python.sh -m worldc.render.isaac_drive SCENE_DIR OUT_DIR [--fps 30 --res 1280 720 --max-seconds 120]

Reads OUT_DIR/route.json (waypoints, legs, wheel joint names, cameras); writes OUT_DIR/drive_result.json and OUT_DIR/frames_<cam>/*.png.

The robot is driven by wheel velocity commands only.  Its wheel radius / track are never assumed: a short calibration moves it
forward and then turns it in place, and the measured gains are used by the controller.  Moving hazards (forklifts, people) follow
their paths while it drives.
"""
from __future__ import annotations

import argparse
import functools
import builtins
import json
import math
import sys
import time
import traceback
from pathlib import Path

print = functools.partial(builtins.print, flush=True)  # noqa: A001 - Isaac's shutdown skips stdout flushing

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("out")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
ap.add_argument("--max-seconds", type=float, default=120.0)
ap.add_argument("--rt-subframes", type=int, default=2)
args = ap.parse_args()
SCENE, OUT = Path(args.scene).resolve(), Path(args.out).resolve()
ROUTE = json.loads((OUT / "route.json").read_text())

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res[0], "height": args.res[1]})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from worldc.runtime import hazard_followers, yaw_to_quat_wxyz  # noqa: E402

DT = 1.0 / 120.0
RESULT: dict = {"ok": False, "legs": []}
T0 = time.time()


def log(*a):
    print(f"[drive +{time.time() - T0:6.1f}s]", *a)


def save():
    (OUT / "drive_result.json").write_text(json.dumps(RESULT, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q) -> float:  # q = (x, y, z, w)
    return math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))


def up_z(q) -> float:
    return 1 - 2 * (q[0] ** 2 + q[1] ** 2)


def look_at(pos, target, up=(0, 0, 1)) -> Gf.Matrix4d:
    pos, target, up = (np.array(v, float) for v in (pos, target, up))
    back = pos - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross((0, 1, 0), back)
    right /= np.linalg.norm(right)
    upv = np.cross(back, right)
    m = Gf.Matrix4d(1.0)
    for r, v in enumerate((right, upv, back, pos)):
        m.SetRow(r, Gf.Vec4d(*v, 1.0 if r == 3 else 0.0))
    return m


try:
    graph = json.loads((SCENE / "scene_graph.json").read_text())
    ctx = omni.usd.get_context()
    ctx.open_stage(str(SCENE / graph.get("usd", "scene.usd")))
    while ctx.get_stage_loading_status()[2] > 0:
        app.update()
    for _ in range(10):
        app.update()
    stage = ctx.get_stage()
    VIDEO = bool(ROUTE.get("cameras"))
    if VIDEO:  # look through the roof
        for prim in stage.Traverse():
            if "/Roof" in str(prim.GetPath()) and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()

    from isaacsim.core.api import SimulationContext

    sim = SimulationContext(physics_dt=DT, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0, backend="numpy")
    sim.initialize_physics()
    sim.play()
    sim.step(render=False)
    import omni.physics.tensors as tensors

    sv = tensors.create_simulation_view("numpy")
    root_path = next(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(graph["robot"]["prim"]), Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    av = sv.create_articulation_view(root_path)
    names = list(av.shared_metatype.dof_names)
    left = [names.index(n) for n in ROUTE["drive"]["left"]]
    right = [names.index(n) for n in ROUTE["drive"]["right"]]
    n_dof, idx0 = len(names), np.array([0], np.int32)
    log("robot", graph["robot"].get("asset"), "dofs", names)

    hz_paths = [h["path"] for h in graph.get("hazards", [])]
    followers = hazard_followers(graph)
    kview = sv.create_rigid_body_view(hz_paths) if hz_paths else None
    kidx = np.arange(len(hz_paths), dtype=np.int32)
    sim_t = 0.0

    def physics_step(wl=0.0, wr=0.0):
        """One 1/120 s step: hazards on their paths, wheel velocity targets, step."""
        global sim_t
        if kview is not None:
            data = np.zeros((len(hz_paths), 7), np.float32)
            for i, p in enumerate(hz_paths):
                x, y, z, yaw = followers[p].pose(sim_t)
                q = yaw_to_quat_wxyz(yaw)
                data[i] = [x, y, z, 0.0, 0.0, q[3], q[0]]
            kview.set_kinematic_targets(data, kidx)
        tgt = np.zeros((1, n_dof), np.float32)
        tgt[0, left], tgt[0, right] = wl, wr
        av.set_dof_velocity_targets(tgt, idx0)
        sim.step(render=False)
        sim_t += DT

    def pose():
        t = np.asarray(av.get_root_transforms()[0], float)
        return t[0], t[1], t[2], (t[3], t[4], t[5], t[6])  # x, y, z, quat(xyzw)

    # ------------------------------------------------------------------ settle + calibrate (not filmed)
    for _ in range(int(1.0 / DT)):
        physics_step()
    x0, y0, z0, q0 = pose()
    yaw0 = yaw_of(q0)
    W0 = 4.0
    for k in range(int(0.9 / DT)):
        physics_step(W0, W0)
        if k == int(0.45 / DT):
            xa, ya, _, _ = pose()
    xb, yb, _, qb = pose()
    dt_cal = (int(0.9 / DT) - int(0.45 / DT)) * DT
    dx, dy = xb - xa, yb - ya
    speed_per_w = math.hypot(dx, dy) / dt_cal / W0  # m/s per rad/s on both wheels
    phi0 = wrap(math.atan2(dy, dx) - yaw_of(qb))  # direction of travel in the robot frame
    for _ in range(int(0.6 / DT)):
        physics_step()
    ya0 = yaw_of(pose()[3])
    for k in range(int(0.9 / DT)):
        physics_step(-W0, W0)
        if k == int(0.45 / DT):
            yaw_a = yaw_of(pose()[3])
    yaw_b = yaw_of(pose()[3])
    turn_per_w = wrap(yaw_b - yaw_a) / dt_cal / W0  # rad/s per rad/s of (right - left) / 2
    for _ in range(int(0.6 / DT)):
        physics_step()
    RESULT["calibration"] = {"forward_m_per_s_per_rad_s": round(speed_per_w, 4), "travel_heading_in_robot_frame_rad": round(phi0, 3),
                             "turn_rad_s_per_rad_s": round(turn_per_w, 4), "left_dofs": ROUTE["drive"]["left"], "right_dofs": ROUTE["drive"]["right"]}
    log("calibration", RESULT["calibration"])
    if speed_per_w < 0.01 or abs(turn_per_w) < 0.01:
        raise RuntimeError(f"the robot did not respond to wheel commands (forward {speed_per_w:.4f}, turn {turn_per_w:.4f}); check the drive joints")

    # ------------------------------------------------------------------ cameras
    VIDEO_CAMS = list(ROUTE.get("cameras") or [])
    cam_state: dict = {}
    frame_no = 0
    if VIDEO:
        import omni.replicator.core as rep

        # ONE render product, re-posed per stream: a second live render product renders black in Isaac Sim
        cam = UsdGeom.Camera.Define(stage, "/World/Cameras/drive_cam")
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 500.0))
        cam_op = UsdGeom.Xformable(cam).AddTransformOp()
        rp = rep.create.render_product("/World/Cameras/drive_cam", tuple(args.res))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        for name in VIDEO_CAMS:
            (OUT / f"frames_{name}").mkdir(exist_ok=True)
            for old in (OUT / f"frames_{name}").glob("*.png"):
                old.unlink()
        for _ in range(30):
            app.update()
        wide = ROUTE["wide_camera"]
        # Kinematic targets set through the tensor API move the bodies in physics but are not drawn by the renderer: write the same
        # pose into USD before each captured frame (translate + orient ops).
        hz_ops = {}
        for hp in hz_paths:
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(hp))
            tr = next((o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None) or xf.AddTranslateOp()
            ori = next((o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeOrient), None) or xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
            hz_ops[hp] = (tr, ori)

    def sync_hazards():
        for hp, (tr, ori) in hz_ops.items():
            x, y, z, yaw = followers[hp].pose(sim_t)
            tr.Set(Gf.Vec3d(float(x), float(y), float(z)))
            ori.Set(Gf.Quatf(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))

    def capture():
        global frame_no
        if not VIDEO:
            return
        sync_hazards()
        x, y, z, q = pose()
        th = yaw_of(q) + phi0
        want_p = np.array([x - 3.2 * math.cos(th), y - 3.2 * math.sin(th), 1.7])
        want_t = np.array([x + 1.2 * math.cos(th), y + 1.2 * math.sin(th), 0.35])
        if not cam_state:
            cam_state.update(p=want_p, t=want_t)
        cam_state["p"] += (want_p - cam_state["p"]) * 0.12
        cam_state["t"] += (want_t - cam_state["t"]) * 0.2
        for name in VIDEO_CAMS:
            if name == "chase":
                cam.GetFocalLengthAttr().Set(16.0)
                cam_op.Set(look_at(cam_state["p"], cam_state["t"]))
            else:
                cam.GetFocalLengthAttr().Set(float(wide.get("focal_mm", 18.0)))
                cam_op.Set(look_at(wide["position"], wide["look_at"]))
            rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=False)
            img = np.asarray(ann.get_data())[..., :3].astype(np.uint8)
            Image.fromarray(img).save(OUT / f"frames_{name}" / f"{frame_no:05d}.png")
        frame_no += 1

    # ------------------------------------------------------------------ drive the route
    pts = np.array(ROUTE["waypoints"], float)
    seg = np.hypot(*(pts[1:] - pts[:-1]).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    def closest(px, py, s_from):
        best = (1e9, s_from)
        for i in range(max(0, int(np.searchsorted(cum, s_from, side="right")) - 2), len(seg)):
            if cum[i] > s_from + 6.0:
                break
            a, b = pts[i], pts[i + 1]
            e = b - a
            t = float(np.clip(((px - a[0]) * e[0] + (py - a[1]) * e[1]) / max(e @ e, 1e-9), 0, 1))
            d = math.hypot(px - (a[0] + t * e[0]), py - (a[1] + t * e[1]))
            if d < best[0]:
                best = (d, cum[i] + t * seg[i])
        return best[1]

    def point_at(s):
        s = min(max(s, 0.0), cum[-1])
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
        t = (s - cum[i]) / max(seg[i], 1e-9)
        return pts[i] + t * (pts[i + 1] - pts[i])

    VMAX, WMAX, LOOK = float(ROUTE.get("speed", 1.0)), 1.4, 0.8
    steps_per_frame = max(1, int(round(1.0 / DT / args.fps)))
    v_cmd = w_cmd = 0.0
    s_prog, leg, state, state_t, step_i, min_z, max_tilt = 0.0, 0, "drive", 0.0, 0, 1e9, 0.0
    traj, t_start = [], sim_t
    arrivals = []
    while leg < len(ROUTE["legs"]) and sim_t - t_start < args.max_seconds:
        x, y, z, q = pose()
        th = yaw_of(q) + phi0
        min_z, max_tilt = min(min_z, z), max(max_tilt, math.degrees(math.acos(max(-1.0, min(1.0, up_z(q))))))
        L = ROUTE["legs"][leg]
        s_end, (gx, gy, gyaw) = cum[L["end_index"]], L["pose"]
        v_t = w_t = 0.0
        if state == "drive":
            s_prog = closest(x, y, s_prog)
            dist_end = math.hypot(gx - x, gy - y)
            tx, ty = point_at(min(s_prog + LOOK, s_end))
            alpha = wrap(math.atan2(ty - y, tx - x) - th)
            if dist_end < 0.3:
                state, state_t = "align", sim_t
            elif abs(alpha) > 0.7:
                v_t, w_t = 0.0, float(np.clip(2.0 * alpha, -WMAX, WMAX))
            else:
                v_t = VMAX * float(np.clip(dist_end / 1.5, 0.25, 1.0)) * (1.0 - 0.6 * abs(alpha))
                w_t = float(np.clip(2.0 * v_t * math.sin(alpha) / LOOK + 0.8 * alpha, -WMAX, WMAX))
        if state == "align":
            err = wrap(gyaw - th)
            w_t = float(np.clip(1.8 * err, -1.0, 1.0)) if abs(err) > 0.12 else 0.0
            if abs(err) <= 0.12 or sim_t - state_t > 4.0:
                arrivals.append({"goal": L["goal"], "reached": bool(math.hypot(gx - x, gy - y) < 0.4), "distance_m": round(math.hypot(gx - x, gy - y), 3),
                                 "yaw_error_rad": round(float(err), 3), "time_s": round(sim_t - t_start, 2)})
                log("arrived", arrivals[-1])
                state, state_t = "pause", sim_t
        if state == "pause" and sim_t - state_t > 0.9:
            leg += 1
            state = "drive"
        # slew limits keep the wheels from slipping
        v_cmd += float(np.clip(v_t - v_cmd, -1.5 * DT * 2, 1.0 * DT * 2))
        w_cmd += float(np.clip(w_t - w_cmd, -4.0 * DT * 2, 4.0 * DT * 2))
        w_avg, w_diff = v_cmd / speed_per_w, w_cmd / turn_per_w
        physics_step(w_avg - w_diff, w_avg + w_diff)
        step_i += 1
        if step_i % 24 == 0:
            traj.append([round(sim_t - t_start, 2), round(float(x), 3), round(float(y), 3), round(float(th), 3)])
        if step_i % steps_per_frame == 0:
            capture()
    x, y, z, q = pose()
    RESULT.update(legs=arrivals, trajectory=traj, time_s=round(sim_t - t_start, 2), frames=frame_no, fps=args.fps, min_z_m=round(float(min_z), 4),
                  max_tilt_deg=round(float(max_tilt), 2), upright=bool(max_tilt < 20.0), planned_legs=len(ROUTE["legs"]),
                  ok=bool(len(arrivals) == len(ROUTE["legs"]) and all(a["reached"] for a in arrivals) and max_tilt < 20.0))
    log("done", {k: RESULT[k] for k in ("ok", "time_s", "frames", "max_tilt_deg")})
except Exception:  # noqa: BLE001
    RESULT["error"] = traceback.format_exc()[-1800:]
    log("FAILED", RESULT["error"])
finally:
    save()
    app.close()
