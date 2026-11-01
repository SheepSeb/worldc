"""Kinematic replay of a logged E3 trial: move the robot and the workers along recorded poses (no physics) and render frames.
Run with Isaac's python (``scripts/e3_replay.py`` does this):

    PYTHONPATH=src ~/isaacsim/python.sh -m worldc.render.isaac_replay SCENE_DIR REPLAY_JSON OUT_DIR [--cameras overhead,corner --res 1280 720]

REPLAY_JSON: {"fps", "label", "robot_prim", "worker_prims": [...], "frames": [{"t", "robot": [x, y, yaw], "workers": [[x, y, yaw], ...]}], "collisions": [t, ...]}.
Poses are set as translate + yaw orient ops (the prims' z stays as built); the timeline never plays, so nothing is simulated.  Writes
OUT_DIR/<camera>/%05d.png with the time, label and a collision flag drawn on each frame.
"""
from __future__ import annotations

import argparse
import functools
import builtins
import json
import math
import sys
import traceback
from pathlib import Path

print = functools.partial(builtins.print, flush=True)  # noqa: A001

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("replay")
ap.add_argument("out")
ap.add_argument("--cameras", default="overhead,corner")
ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
ap.add_argument("--warmup", type=int, default=40)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.res[0], "height": args.res[1]})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

try:
    scene, OUT = Path(args.scene).resolve(), Path(args.out).resolve()
    R = json.loads(Path(args.replay).read_text())
    graph = json.loads((scene / "scene_graph.json").read_text())
    ctx = omni.usd.get_context()
    ctx.open_stage(str(scene / graph.get("usd", "scene.usd")))
    while ctx.get_stage_loading_status()[2] > 0:
        app.update()
    for _ in range(10):
        app.update()
    stage = ctx.get_stage()

    def ops_of(path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"prim {path} not found")
        xf = UsdGeom.Xformable(prim)
        tr = next((o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None) or xf.AddTranslateOp()
        ori = next((o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeOrient), None) or xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
        z = float(tr.Get()[2]) if tr.Get() is not None else 0.0
        for op in (tr, ori):
            op.GetAttr().Clear()  # a prim with a recorded path (time samples) would ignore the default value we set: drop the samples
        return tr, ori, z

    robot = ops_of(R["robot_prim"])
    workers = [ops_of(p) for p in R["worker_prims"]]

    def place(ops, x, y, yaw):
        tr, ori, z = ops
        dbl = lambda op: op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble  # noqa: E731  (the ops keep the precision they were authored with)
        tr.Set((Gf.Vec3d if dbl(tr) else Gf.Vec3f)(float(x), float(y), z))
        ori.Set((Gf.Quatd if dbl(ori) else Gf.Quatf)(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))

    cams = []
    for name in args.cameras.split(","):
        entry = next((c for c in graph.get("cameras", []) if name in (c.get("name"), c.get("id"))), None)
        if entry is None:
            raise RuntimeError(f"camera {name} not in the scene: {[c.get('name') for c in graph.get('cameras', [])]}")
        cams.append((name, entry["path"]))
        M = UsdGeom.Xformable(stage.GetPrimAtPath(entry["path"])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if -M.GetRow3(2)[2] / max(float(Gf.Vec3d(*M.GetRow3(2)).GetLength()), 1e-9) < -0.8:  # overhead: look through the roof
            for prim in stage.Traverse():
                if "/Roof" in str(prim.GetPath()) and prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeInvisible()
    anns = {}
    for name, path in cams:
        rp = rep.create.render_product(path, tuple(args.res))
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        a.attach([rp])
        anns[name] = a
        (OUT / name).mkdir(parents=True, exist_ok=True)

    place(robot, *R["frames"][0]["robot"])
    for w, p in zip(workers, R["frames"][0]["workers"]):
        place(w, *p)
    for _ in range(args.warmup):
        app.update()
    hits = R.get("collisions", [])
    for n, fr in enumerate(R["frames"]):
        place(robot, *fr["robot"])
        for w, p in zip(workers, fr["workers"]):
            place(w, *p)
        app.update()
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
        for name, _ in cams:
            img = Image.fromarray(np.asarray(anns[name].get_data())[..., :3].astype(np.uint8))
            d = ImageDraw.Draw(img)
            hit = any(0.0 <= fr["t"] - h < 0.8 for h in hits)
            d.rectangle([0, 0, 470, 26], fill=(255, 255, 255))
            d.text((8, 7), f"{R['label']}   t = {fr['t']:5.1f} s" + ("   COLLISION" if hit else ""), fill=(200, 0, 0) if hit else (0, 0, 0))
            img.save(OUT / name / f"{n:05d}.png")
        if n % 50 == 0:
            print(f"frame {n}/{len(R['frames'])}")
    (OUT / "replay_done.json").write_text(json.dumps({"frames": len(R["frames"]), "cameras": [c[0] for c in cams]}))
    print("REPLAY_DONE")
    app.close()
except SystemExit:
    raise
except Exception:  # noqa: BLE001
    print("REPLAY_ERROR " + traceback.format_exc()[-2500:])
    app.close()
    sys.exit(1)
