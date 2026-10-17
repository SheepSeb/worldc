"""Render the named cameras of a generated seed (fast look-dev; no physics).  Run with Isaac's python.

    PYTHONPATH=src ~/isaacsim/python.sh -m claude_scene_generator.plant.quick_render plant_out/seed_0 [cam ...] [--res W H]
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("seed_dir")
ap.add_argument("cams", nargs="*", default=["photo_view"])
ap.add_argument("--res", type=int, nargs=2, default=[1200, 800])
ap.add_argument("--tag", default="")
ap.add_argument("--adhoc", nargs=8, action="append", metavar=("NAME", "X", "Y", "Z", "TX", "TY", "TZ", "FOCAL"), help="extra camera for look-dev")
ap.add_argument("--hide-structure-for", nargs="*", default=["overhead"])
a = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": a.res[0], "height": a.res[1]})
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom  # noqa: E402

d = Path(a.seed_dir).resolve()
g = json.loads((d / "scene_graph.json").read_text())
ctx = omni.usd.get_context()
ctx.open_stage(str(d / "plant.usd"))
while ctx.get_stage_loading_status()[2] > 0:
    app.update()
for _ in range(20):
    app.update()
stage = ctx.get_stage()
(d / "renders").mkdir(exist_ok=True)
from claude_scene_generator.plant.gen import build_camera  # noqa: E402


class _K:  # minimal Kit stand-in for build_camera
    def __init__(self, st):
        self.stage = st

    set_pose = staticmethod(__import__("claude_scene_generator.usdkit", fromlist=["Kit"]).Kit.set_pose)


cams = list(g["cameras"])
for nm, x, y, z, tx, ty, tz, f in a.adhoc or []:
    build_camera(_K(stage), f"/World/Cameras/{nm}", (float(x), float(y), float(z)), (float(tx), float(ty), float(tz)), float(f))
    cams.append(dict(name=nm, path=f"/World/Cameras/{nm}"))
for cam in cams:
    if cam["name"] not in a.cams and not any(cam["name"] == ad[0] for ad in a.adhoc or []):
        continue
    hide = cam["name"] in a.hide_structure_for
    if hide:
        for pth in ("/World/Structure", "/World/Pipes"):
            pr = stage.GetPrimAtPath(pth)
            if pr:
                for ch in pr.GetChildren():
                    if ch.GetName().startswith(("rack_", "top_", "col_", "cross_", "brace", "cable_tray")) or pth == "/World/Structure":
                        UsdGeom.Imageable(ch).MakeInvisible()
    rp = rep.create.render_product(cam["path"], tuple(a.res))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    for _ in range(30):
        app.update()
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=48, delta_time=0.0, pause_timeline=False)
    img = np.asarray(ann.get_data())[..., :3].astype(np.uint8)
    out = d / "renders" / f"{cam['name']}{a.tag}.png"
    Image.fromarray(img).save(out)
    print("RENDERED", out, img.mean(), flush=True)
    if hide:
        for pth in ("/World/Structure", "/World/Pipes"):
            for ch in stage.GetPrimAtPath(pth).GetChildren():
                UsdGeom.Imageable(ch).MakeVisible()
app.close()
