"""Render one camera of a built scene with Isaac Sim RTX.  Run with Isaac's python (the MCP ``render`` tool does this):

    PYTHONPATH=src ~/isaacsim/python.sh -m claude_scene_generator.render.isaac_render SCENE_DIR --out img.png \
        [--camera overview | --camera-json '{"position": [..], "look_at": [..], "focal_mm": 18}'] [--width 1280 --height 720]

Prints ``RENDER_RESULT:{json}`` (file, luminance stats, ok) as its last line.  Roof prims are hidden for steeply
downward cameras, like the proxy renderer does.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--out", required=True)
ap.add_argument("--camera", default=None)
ap.add_argument("--camera-json", default=None)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--warmup", type=int, default=40)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402


def finish(res: dict, code: int = 0):
    print("RENDER_RESULT:" + json.dumps(res), flush=True)
    app.close()
    sys.exit(code)


def look_at_matrix(pos, target, up=(0, 0, 1)) -> Gf.Matrix4d:
    pos, target, up = (np.array(v, float) for v in (pos, target, up))
    back = pos - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    if np.linalg.norm(right) < 1e-6:  # looking straight along `up`
        right = np.cross((0, 1, 0), back)
    right /= np.linalg.norm(right)
    upv = np.cross(back, right)
    m = Gf.Matrix4d(1.0)
    for r, v in enumerate((right, upv, back, pos)):
        m.SetRow(r, Gf.Vec4d(*v, 1.0 if r == 3 else 0.0))
    return m


try:
    scene = Path(args.scene).resolve()
    graph = json.loads((scene / "scene_graph.json").read_text())
    ctx = omni.usd.get_context()
    ctx.open_stage(str(scene / graph.get("usd", "scene.usd")))
    while ctx.get_stage_loading_status()[2] > 0:
        app.update()
    for _ in range(10):
        app.update()
    stage = ctx.get_stage()

    if args.camera_json:
        d = json.loads(args.camera_json)
        cam_path = "/World/Cameras/_mcp_camera"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        ap_mm = d.get("aperture_mm", [20.955, 11.79])
        cam.GetFocalLengthAttr().Set(float(d.get("focal_mm", 18.0)))
        cam.GetHorizontalApertureAttr().Set(float(ap_mm[0]))
        cam.GetVerticalApertureAttr().Set(float(ap_mm[1]))
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(float(d.get("near", 0.05)), float(d.get("far", 1e4))))
        UsdGeom.Xformable(cam).AddTransformOp().Set(look_at_matrix(d["position"], d["look_at"], d.get("up", (0, 0, 1))))
        name = d.get("name", "custom")
    else:
        entry = next((c for c in graph.get("cameras", []) if args.camera in (c.get("name"), c.get("id"), c.get("path"))), None)
        if entry is None and not (args.camera or "").startswith("/"):
            finish({"ok": False, "error": f"unknown camera {args.camera!r}; available: {[c.get('name') for c in graph.get('cameras', [])]}"}, 2)
        cam_path = args.camera if (args.camera or "").startswith("/") else entry["path"]
        name = (entry or {}).get("name") or Path(cam_path).name
        if not stage.GetPrimAtPath(cam_path):
            finish({"ok": False, "error": f"camera prim {cam_path} not found"}, 2)

    M = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    forward_z = -M.GetRow3(2)[2] / max(float(Gf.Vec3d(*M.GetRow3(2)).GetLength()), 1e-9)
    hidden = []
    if forward_z < -0.8 or (args.camera_json and json.loads(args.camera_json).get("hide_roof")):  # overhead / explicit: look through the roof
        for prim in stage.Traverse():
            if "/Roof" in str(prim.GetPath()) and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden.append(str(prim.GetPath()))

    rp = rep.create.render_product(cam_path, (args.width, args.height))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    for _ in range(args.warmup):
        app.update()
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=32, delta_time=0.0, pause_timeline=False)
    img = np.asarray(ann.get_data())[..., :3].astype(np.uint8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)
    lum = img.astype(np.float32).mean(axis=2)
    magenta = float(((img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] > 200)).mean())
    res = {"camera": name, "file": str(out), "width": args.width, "height": args.height,
           "mean_luminance": round(float(lum.mean()), 1), "std_luminance": round(float(lum.std()), 1),
           "frac_black": round(float((lum < 8).mean()), 4), "frac_saturated": round(float((lum > 247).mean()), 4),
           "frac_magenta_missing_tex": round(magenta, 5), "hidden_roof_prims": len(hidden)}
    res["ok"] = bool(res["mean_luminance"] > 10 and res["std_luminance"] > 8 and res["frac_black"] < 0.6
                     and res["frac_saturated"] < 0.35 and magenta < 0.001)
    finish(res)
except SystemExit:
    raise
except Exception:  # noqa: BLE001
    finish({"ok": False, "error": traceback.format_exc()[-1500:]}, 1)
