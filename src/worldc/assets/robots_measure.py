"""Measure Isaac Sim robots with Isaac Sim itself (all payloads loaded).  Run under Isaac's python (``robots.build_table`` does this):

    PYTHONPATH=src ~/isaacsim/python.sh -m worldc.assets.robots_measure OUT.json alias=URL [alias=URL ...]

Writes {alias: {size_m, base_offset_z, footprint_radius_m, articulation, n_joints, n_dofs, collision_size_m, ok}}; bounds are in
metres, Z-up, in the robot's default pose, from the visual (default + render) geometry; ``collision_size_m`` from the collision
shapes is a cross-check.  Dependency-free on purpose (Isaac's python has no pydantic).
"""
import functools
import builtins
import json
import math
import sys

print = functools.partial(builtins.print, flush=True)  # noqa: A001 - Isaac's shutdown skips stdout flushing
OUT, ITEMS = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

res = {}
for alias, url in ITEMS:
    try:
        ctx = omni.usd.get_context()
        ctx.new_stage()
        for _ in range(3):
            app.update()
        stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
        root.GetReferences().AddReference(url)
        while ctx.get_stage_loading_status()[2] > 0:
            app.update()
        for _ in range(30):
            app.update()
        vis = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
        rng = vis.ComputeWorldBound(root).ComputeAlignedRange()
        lo, hi = rng.GetMin(), rng.GetMax()
        size = [float(hi[i] - lo[i]) for i in range(3)]
        n_joints = n_roots = 0
        col_lo, col_hi = [1e9] * 3, [-1e9] * 3
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            t = p.GetTypeName()
            n_joints += t.startswith("Physics") and "Joint" in t
            n_roots += p.HasAPI(UsdPhysics.ArticulationRootAPI)
            if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim):
                r = cache.ComputeWorldBound(p).ComputeAlignedRange()
                if not r.IsEmpty():
                    for i in range(3):
                        col_lo[i], col_hi[i] = min(col_lo[i], r.GetMin()[i]), max(col_hi[i], r.GetMax()[i])
        ok = not rng.IsEmpty() and all(math.isfinite(v) and 0.02 < v < 6.0 for v in size) and n_roots > 0
        res[alias] = {"size_m": [round(v, 4) for v in size], "base_offset_z": round(float(-lo[2]), 4),
                      "footprint_radius_m": round(math.hypot(size[0], size[1]) / 2, 3), "articulation": n_roots > 0, "n_joints": int(n_joints),
                      "collision_size_m": [round(col_hi[i] - col_lo[i], 4) for i in range(3)] if col_lo[0] < 1e8 else None, "ok": bool(ok)}
        print("MEASURED", alias, res[alias])
    except Exception as e:  # noqa: BLE001
        res[alias] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print("FAILED", alias, e)
json.dump(res, open(OUT, "w"), indent=1)
app.close()
