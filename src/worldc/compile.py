"""Compiler: spec + seed -> validate -> solve -> build USD + textures + scene_graph.json (+ resolved_spec.json, check.json).

    from worldc.compile import compile_scene
    res = compile_scene("examples/specs/warehouse_lite.yaml", seed=3, out_dir="out/lite_3")

Everything is deterministic in (spec, seed): every random draw comes from a generator keyed by (seed, purpose, object id,
instance index), so editing one object does not reshuffle the others.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, UsdGeom

from . import __name__ as _pkg
from .assets import registry
from .assets.base import BuildContext, Environment, Pose, sample_value
from .assets.resolve import pin_catalog
from .assets.procedural.agents import look_at_quat
from .geom import Q_IDENT, qmul, qyaw
from .lighting import build_lights
from .materials import Materials, TextureCache
from .solver.solve import Solution, rng_for, solve
from .spec.issues import Issue, error, has_errors
from .spec.models import SPEC_VERSION, SceneSpec
from .spec.refs import parse_ref
from .spec.validate import check as validate_check
from .textures import make_floor_texture, make_grating, make_sky
from .usdkit import Kit, T

GENERATOR_VERSION = "2.0"


@dataclass
class CompileResult:
    ok: bool
    out_dir: Path | None = None
    usd: Path | None = None
    scene_graph: Path | None = None
    issues: list = field(default_factory=list)
    check: dict | None = None
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0

    def summary(self) -> dict:
        return {"ok": self.ok, "out_dir": str(self.out_dir) if self.out_dir else None, "usd": str(self.usd) if self.usd else None,
                "issues": [i.model_dump() for i in self.issues], "check_passed": None if self.check is None else self.check.get("passed"),
                "stats": self.stats, "seconds": round(self.seconds, 2)}


def spec_hash(spec: SceneSpec) -> str:
    return hashlib.sha256(json.dumps(spec.model_dump(mode="json", by_alias=True), sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------------- floor markings
WHITE, YELLOW, GREEN, BLACK, BLUE, ORANGE = (235, 235, 230), (255, 200, 20), (30, 150, 70), (25, 25, 25), (60, 140, 235), (240, 130, 20)


def _markings(spec: SceneSpec, sol: Solution) -> list[dict]:
    fl = spec.environment.floor
    out = []

    def conv(m):
        d = {"type": m.type, "color": tuple(m.color), "width": m.width, "kind": m.kind or m.type}
        if m.points:
            d["pts"] = [tuple(p) for p in m.points]
        if m.rect:
            d["bbox"] = tuple(m.rect)
        if m.text:
            d["text"], d["pos"] = m.text, tuple(m.points[0]) if m.points else (m.rect[0], m.rect[1])
        return d

    if isinstance(fl.markings, list):
        out = [conv(m) for m in fl.markings]
    elif fl.markings == "auto":
        for zid, (x0, y0, x1, y1) in sol.zones.items():
            if ".aisle_" in zid:  # aisle edge lines
                for x in (x0 + 0.12, x1 - 0.12):
                    out.append({"type": "line", "pts": [(x, y0), (x, y1)], "color": YELLOW, "width": 0.1, "kind": "aisle_edge"})
        for z in spec.zones:
            if z.marking == "outline":
                col = ORANGE if "packing" in z.tags else GREEN if "charging" in z.tags else WHITE if "dock" in z.tags else YELLOW
                out.append({"type": "rect", "bbox": tuple(z.rect), "color": col, "width": 0.12, "kind": f"zone_{z.id}"})
            elif z.marking not in ("none", "outline"):
                out.append(conv(z.marking))
            if z.keep_clear:
                out.append({"type": "rect", "bbox": tuple(z.rect), "color": WHITE, "width": 0.08, "kind": f"keep_clear_{z.id}"})
        for ag in sol.agents:
            pts = list(ag["waypoints"]) + ([ag["waypoints"][0]] if ag["params"].path_mode == "loop" else [])
            dashed = "forklift" in ag["asset"].tags
            for a, b in zip(pts[:-1], pts[1:]):
                out.append({"type": "dashed" if dashed else "line", "pts": [a, b], "color": WHITE if dashed else GREEN, "width": 0.1,
                            "kind": "forklift_route" if dashed else "pedestrian_walkway"})
        for o in spec.environment.walls.openings:
            p = sol.env.ports.get(f"opening_{o.id}")
            if p is None or o.width < 2.0:
                continue
            # bay rectangle + hazard hatch just inside the opening
            n = np.array([0.0, 1.0]) if o.wall == "south" else np.array([0.0, -1.0]) if o.wall == "north" else np.array([1.0, 0.0]) if o.wall == "west" else np.array([-1.0, 0.0])
            t = np.array([-n[1], n[0]])
            c = np.asarray(p.pos[:2]) + n * 1.6
            corners = [c + t * s * (o.width / 2 + 0.2) + n * d for s in (-1, 1) for d in (-1.4, 1.4)]
            xs, ys = [q[0] for q in corners], [q[1] for q in corners]
            out.append({"type": "rect", "bbox": (min(xs), min(ys), max(xs), max(ys)), "color": WHITE, "width": 0.1, "kind": f"dock_bay_{o.id}"})
            h0 = np.asarray(p.pos[:2]) + n * 2.6
            hc = [h0 + t * s * (o.width / 2 + 0.2) + n * d for s in (-1, 1) for d in (-0.15, 0.15)]
            xs, ys = [q[0] for q in hc], [q[1] for q in hc]
            out.append({"type": "hatch", "bbox": (min(xs), min(ys), max(xs), max(ys)), "color": BLACK, "stripe": 0.2, "kind": "dock_hatch"})
        for inst in sol.instances:
            if inst.asset.id == "charging_station":
                a = next(a for a in inst.asset.anchors(inst.params) if a.name == "parking").to_world(inst.pose)
                x, y = a.pos[:2]
                out.append({"type": "rect", "bbox": (x - 0.8, y - 1.0, x + 0.8, y + 1.0), "color": BLUE, "width": 0.1, "kind": "parking"})
        for zs in fl.slope_zones:
            out.append({"type": "rect", "bbox": tuple(zs.rect), "color": ORANGE, "width": 0.06, "kind": "slope_zone"})
    out += [conv(m) for m in fl.extra_markings]
    return out


# --------------------------------------------------------------------------------------------------- cameras
def _cameras(kit, spec: SceneSpec, sol: Solution, robot_prim: str | None) -> list[dict]:
    out = []
    if not spec.cameras:
        return out
    kit.scope("/World/Cameras")
    (x0, x1), (y0, y1) = sol.env.x_range, sol.env.y_range
    for c in spec.cameras:
        parent = "/World/Cameras"
        if c.mount == "robot" and robot_prim:
            parent = robot_prim
        elif c.mount:
            keys = [p.key for p in sol.instances if p.obj_id == parse_ref(c.mount).target]
            parent = f"/World/{parse_ref(c.mount).target}/{keys[0].replace('.', '_')}" if keys else parent
        path = f"{parent}/{c.id}"
        if c.mode == "top_down":
            hfov = 2 * math.atan(c.aperture_mm[0] / (2 * c.focal_mm))
            vfov = 2 * math.atan(c.aperture_mm[1] / (2 * c.focal_mm))
            need = max((x1 - x0) / 2 / math.tan(hfov / 2), (y1 - y0) / 2 / math.tan(vfov / 2)) * 1.05
            pos = tuple(c.position) if c.position else ((x0 + x1) / 2, (y0 + y1) / 2, need)
            q = Q_IDENT
            look = (pos[0], pos[1], 0.0)
        else:
            pos = tuple(c.position)
            look = tuple(c.look_at) if c.look_at else (pos[0] + 1.0, pos[1], pos[2])
            q = look_at_quat(pos, look)
        cam = UsdGeom.Camera.Define(kit.stage, path)
        cam.CreateFocalLengthAttr(float(c.focal_mm))
        cam.CreateHorizontalApertureAttr(float(c.aperture_mm[0]))
        cam.CreateVerticalApertureAttr(float(c.aperture_mm[1]))
        cam.CreateClippingRangeAttr(Gf.Vec2f(*[float(v) for v in c.clip]))
        kit.set_pose(cam.GetPrim(), pos, q)
        if c.hide:
            kit.attr(cam.GetPrim(), f"{kit.ns}:hide", T.String, ",".join(c.hide))
        out.append({"name": c.id, "id": c.id, "path": path, "mode": c.mode, "position": [round(float(v), 3) for v in pos],
                    "look_at": [round(float(v), 3) for v in look], "focal_mm": c.focal_mm, "aperture_mm": list(c.aperture_mm),
                    "hide": list(c.hide), **({"mounted_on": parent} if c.mount else {})})
    return out


# --------------------------------------------------------------------------------------------------- clearance refinement
def _exact_gap(stage, chk) -> float:
    from .check.clearance import measure
    from .colliders import collect
    from .geom import footprint

    polys, boxes = [], []
    for c in collect(stage):
        lo, hi = c.obb.aabb()
        if c.is_mesh or c.category in ("hazard", "floor", "roof") or hi[2] < 0.10 or lo[2] > 1.80:
            continue
        P = footprint(c.obb)
        polys.append(P)
        boxes.append((P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()))
    return measure(polys, boxes, chk)[0]


def _refine_clearances(kit, sol: Solution, inst_info: dict, max_iter: int = 3) -> dict:
    """Planning uses conservative footprints; measure the exact gap on the built colliders and translate the `adjust`
    instances by the residual (a pure translation along the gap axis changes a side-anchored gap one-for-one)."""
    from pxr import UsdPhysics

    out = {}
    cons = {(k.id or f"clearance_{i}"): k for i, k in enumerate(sol.spec.constraints) if k.type == "clearance"}
    for c in sol.clearance:
        k = cons.get(c["id"])
        if k is None or not c.get("rect") or c.get("side", "any") == "any":
            continue
        adjust = {parse_ref(r).target for r in k.adjust}
        roots = [v["root"] for v in inst_info.values() if v["spec_id"] in adjust]
        chk = {"rect": c["rect"], "axis": c["axis"], "span": c["span"], "side": c["side"]}
        total = 0.0
        for _ in range(max_iter):
            err = _exact_gap(kit.stage, chk) - c["target"]
            if abs(err) <= min(0.005, c["tolerance"] / 3):
                break
            d = err if c["side"] == "max" else -err  # max: move toward the +edge to shrink the gap
            off = (d, 0.0) if c["axis"] == "x" else (0.0, d)
            for root in roots:
                rp = kit.stage.GetPrimAtPath(root)
                bodies = [rp] if rp.HasAPI(UsdPhysics.RigidBodyAPI) else [q for q in rp.GetChildren() if q.HasAPI(UsdPhysics.RigidBodyAPI)]
                for b in bodies:
                    op = UsdGeom.Xformable(b).GetOrderedXformOps()[0]
                    v = op.Get()
                    op.Set(Gf.Vec3d(v[0] + off[0], v[1] + off[1], v[2]))
                for r_ in kit.records:
                    if r_["path"] == root or r_["path"].startswith(root + "/"):
                        r_["position"][0] = round(r_["position"][0] + off[0], 4)
                        r_["position"][1] = round(r_["position"][1] + off[1], 4)
            total += d
        out[c["id"]] = round(total, 4)
    if out:
        kit.save()
    return out


# --------------------------------------------------------------------------------------------------- build
def _group(obj_id: str) -> str:
    return f"/World/{obj_id}"


def build(sol: Solution, out_dir: Path) -> tuple[Path, dict]:
    t0 = time.time()
    spec, seed = sol.spec, sol.seed
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    usd = out_dir / spec.output.usd_name
    if usd.exists():
        usd.unlink()
    kit = Kit(usd)
    kit.ns = spec.output.namespace
    kit.attr(kit.world.GetPrim(), f"{kit.ns}:seed", T.Int, seed)
    kit.attr(kit.world.GetPrim(), f"{kit.ns}:generator_version", T.String, GENERATOR_VERSION)
    kit.attr(kit.world.GetPrim(), f"{kit.ns}:spec_hash", T.String, spec_hash(spec))
    sc = kit.physics_scene(hz=spec.physics.hz)
    sc.GetAttribute("physxScene:solverType").Set(spec.physics.solver)
    sc.GetAttribute("physics:gravityMagnitude").Set(spec.physics.gravity)
    env = sol.env
    mats = Materials(kit)
    tex = TextureCache(out_dir, seed)
    ctx = BuildContext(kit, mats, tex, Environment(env.x_range, env.y_range, env.height, spec.environment.walls.thickness, env.floor_z, env.kind),
                       seed=seed, out_dir=out_dir)
    # ---- floor texture + environment
    wear = float(sample_value(spec.environment.floor.wear, rng_for(seed, "floor", "wear")))
    marks = _markings(spec, sol)
    W, D = env.x_range[1] - env.x_range[0], env.y_range[1] - env.y_range[0]
    fmat = spec.environment.floor.material
    if fmat == "grating":
        floor_tex = tex.get("floor", lambda p, r: make_grating(p, r, W, D, wear, ppm=100), legacy_seed=seed + 1000)
    else:
        floor_tex = tex.get("floor", lambda p, r: make_floor_texture(p, r, W, D, marks, wear, style=fmat), legacy_seed=seed + 1000)
    builts = {}
    if env.kind == "indoor":
        sp = env.shell.params({**env.shell_params.model_dump(), "floor_texture": floor_tex})
        ctx.meta = {"spec_id": "environment", "zone": "building"}
        builts["environment"] = env.shell.build(ctx, "/World/Building", Pose(), sp, rng_for(seed, "env", "build"))
    else:
        g = spec.environment.ground_margin
        kit.scope("/World/Ground")
        slab = kit.cube("/World/Ground/Collision", (W + 2 * g, D + 2 * g, 0.5), (0, 0, -0.25), Q_IDENT, None, mats.phys("concrete"), collide=True, guide=True)
        kit.tag(slab, "floor")
        ctx.record("/World/Ground/Collision", "floor", (0, 0, -0.25), Q_IDENT, "static", size=(W + 2 * g, D + 2 * g, 0.5), friction=0.7)
        pts = [(env.x_range[0], env.y_range[0], 0), (env.x_range[1], env.y_range[0], 0), (env.x_range[1], env.y_range[1], 0), (env.x_range[0], env.y_range[1], 0)]
        kit.mesh("/World/Ground/Visual", pts, [4], [0, 1, 2, 3], uvs=[(0, 0), (1, 0), (1, 1), (0, 1)], vmat=mats.textured("floor", floor_tex, 0.6))
        big = [(-W / 2 - g, -D / 2 - g, -0.002), (W / 2 + g, -D / 2 - g, -0.002), (W / 2 + g, D / 2 + g, -0.002), (-W / 2 - g, D / 2 + g, -0.002)]
        kit.mesh("/World/Ground/Surround", big, [4], [0, 1, 2, 3], vmat=mats.get("apron"))
    sz = registry.get("slope_zone")
    for zs, p in env.slope_zones:
        p = sz.params({**p.model_dump(), "floor_texture": floor_tex})
        ctx.meta = {"spec_id": f"slope:{zs.id}", "zone": "floor"}
        kit.scope("/World/SlopeZones")
        sz.build(ctx, f"/World/SlopeZones/{zs.id}", Pose(((zs.rect[0] + zs.rect[2]) / 2, (zs.rect[1] + zs.rect[3]) / 2, 0)), p, rng_for(seed, zs.id, "build"))
    # ---- instances (solver order: parents before children)
    inst_info = {}
    sol_keys = sol.by_key()
    for inst in sol.instances:
        pose = inst.pose
        if inst.parent and inst.anchor and inst.parent in builts:
            try:
                a = builts[inst.parent].anchor(inst.anchor)
                pose = Pose((pose.pos[0], pose.pos[1], a.pos[2]), pose.quat)  # actual (post-build) support height
            except KeyError:
                pass
        group = _group(inst.obj_id)
        ctx.scope(group)
        path = f"{group}/{inst.obj_id}_{inst.index:04d}"
        zone = inst.zone_label or (inst.meta.get("zone")) or (builts and inst.parent and inst.parent.split(".")[0]) or inst.obj_id
        extra = {"instance": inst.key}
        if "aisle" in inst.meta:
            extra["faces_aisle"] = inst.meta["aisle"]
        if inst.parent and inst.relation in ("on", "pose") and inst.parent in sol_keys and "conveyor" in sol_keys[inst.parent].asset.tags:
            extra["on_conveyor"] = sol_keys[inst.parent].obj_id
        ctx.meta = {"spec_id": inst.obj_id, "zone": zone, "record_extra": extra}
        b = inst.asset.build(ctx, path, pose, inst.params, rng_for(seed, inst.obj_id, inst.index, "build"))
        builts[inst.key] = b
        inst_info[inst.key] = {"spec_id": inst.obj_id, "index": inst.index, "asset": inst.asset.id, "source": inst.asset.source,
                               "root": b.root, "bodies": b.bodies, "parent": inst.parent, "anchor": inst.anchor, "relation": inst.relation,
                               "optional": inst.optional, "pose": {"xyz": [round(float(v), 4) for v in pose.pos], "yaw_deg": round(math.degrees(pose.yaw), 3),
                                                                   "quat_wxyz": [round(float(v), 6) for v in pose.quat]},
                               "params": inst.params.model_dump(mode="json")}
    # ---- agents, robot
    if sol.agents:
        ctx.scope("/World/Agents")
    for ag in sol.agents:
        x, y = ag["waypoints"][0]
        ctx.meta = {"spec_id": ag["id"], "zone": "agents"}
        ag["asset"].build(ctx, f"/World/Agents/{ag['id']}", Pose.xy_yaw(x, y, 0.0, env.floor_z(x, y)), ag["params"], rng_for(seed, ag["id"], "build"))
    robot = None
    if sol.robot:
        r = sol.robot
        ctx.meta = {"spec_id": "robot", "zone": "robot"}
        rb = r["asset"].build(ctx, "/World/Robot", r["pose"], r["params"], rng_for(seed, "robot", "build"))
        robot = {"spawn": [round(r["pose"].pos[0], 3), round(r["pose"].pos[1], 3), round(r["pose"].yaw, 4)], "prim": "/World/Robot",
                 "camera": rb.info.get("camera"), "radius_m": r["radius"], "asset": r["asset"].id}
    cams = _cameras(kit, spec, sol, "/World/Robot" if robot else None)
    for c in cams + ([{"path": robot["camera"], "position": robot["spawn"][:2] + [1.5]}] if robot and robot.get("camera") else []):
        kit.record(c["path"], "camera", c["position"], Q_IDENT, "none")
    sky = None
    if spec.lighting.preset in ("sun", "overcast"):
        kind = "clear" if spec.lighting.preset == "sun" else "overcast"
        sky = tex.get(f"sky_{kind}", lambda p, r: make_sky(p, r, kind))
    lighting = build_lights(kit, rng_for(seed, "lights"), spec.lighting.preset, env.x_range, env.y_range, env.height if env.kind == "indoor" else None,
                            spec.lighting.scale, spec.lighting.color_temperature, sky_texture=sky)
    kit.stage.SetStartTimeCode(0)
    kit.stage.SetEndTimeCode(0)
    kit.stage.GetRootLayer().documentation = f"{spec.name} seed={seed} (worldc {GENERATOR_VERSION}, spec {SPEC_VERSION})"
    kit.save()
    # ---- scene graph
    recs = kit.records
    counts: dict[str, int] = {}
    for r_ in recs:
        counts[r_["category"]] = counts.get(r_["category"], 0) + 1
    stats = ctx.scene["stats"]
    refine = _refine_clearances(kit, sol, inst_info)
    clearance_checks = [{"id": c["id"], "rect": c["rect"], "axis": c["axis"], "side": c.get("side", "any"), "span": c["span"], "target": c["target"],
                         "tolerance": c["tolerance"], "refined_shift_m": refine.get(c["id"]),
                         "gap_center": c["gap_center"], "planned_m": c["planned_m"]} for c in sol.clearance if c.get("rect")]
    graph = {
        "seed": seed, "generator": "worldc.compile", "generator_version": GENERATOR_VERSION, "spec_version": SPEC_VERSION,
        "spec_name": spec.name, "spec_hash": spec_hash(spec), "units": {"length": "m", "mass": "kg", "up_axis": "Z", "angles": "rad (yaw_deg fields in degrees)"},
        "usd": spec.output.usd_name, "namespace": kit.ns,
        "environment": {"kind": env.kind, "x_range": list(env.x_range), "y_range": list(env.y_range), "height": env.height, "size_m": [W, D]},
        "building": {"size_m": [W, D, env.height], "x_range": list(env.x_range), "y_range": list(env.y_range)},
        "layout": {"zones": {k: list(v) for k, v in sol.zones.items()},
                   "slope_zones": [{"id": zs.id, "bbox": list(zs.rect), "gradient": zs.slope, "max_height_m": zs.max_height} for zs, _ in env.slope_zones]},
        "floor": {"wear": round(wear, 3), "markings": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()} for m in marks]},
        "lighting": lighting,
        "counts": counts, "n_dynamic_bodies": sum(1 for r_ in recs if r_["body_type"] == "dynamic"),
        "box_misalignment": {"max_xy_cm": round(max(stats.get("misalign_xy_cm") or [0]), 2), "max_yaw_deg": round(max(stats.get("misalign_yaw_deg") or [0]), 2)},
        "conveyors": ctx.scene["conveyors"], "articulations": ctx.scene["articulations"], "joints": ctx.scene["joints"],
        "hazards": ctx.scene.get("hazards", []),
        "robot": robot, "cameras": cams + ([{"name": "robot", "path": robot["camera"], "mounted_on": "/World/Robot"}] if robot and robot.get("camera") else []),
        "goals": sol.goals, "clearance_checks": clearance_checks,
        "nav": {"resolution_m": 0.05, "z_band": [0.10, 1.80], "ignore_categories": ["hazard", "floor", "roof"]},
        "instances": inst_info,
        "solver": {"stats": sol.stats, "warnings": [i.model_dump() for i in sol.issues if i.severity != "error"]},
        "generation_seconds": round(time.time() - t0, 2),
        "objects": recs,
    }
    (out_dir / "scene_graph.json").write_text(json.dumps(graph, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return usd, graph


# --------------------------------------------------------------------------------------------------- resolved spec
def resolved_spec(sol: Solution) -> dict:
    """Fully grounded spec: every instance explicit (pose + sampled params), seed pinned.  Compiling it with the same seed
    reproduces the scene."""
    d = sol.spec.model_dump(mode="json", by_alias=True, exclude_none=True)
    d["seed"] = sol.seed
    by_obj: dict[str, list] = {}
    for inst in sol.instances:
        by_obj.setdefault(inst.obj_id, []).append(inst)
    objs = []
    for o in d["objects"]:
        insts = by_obj.get(o["id"], [])
        if not insts:
            continue
        o = dict(o)
        o["instance_params"] = [i.params.model_dump(mode="json") for i in insts]
        o["params"] = {}
        o.pop("jitter", None)
        if "layout" not in o["placement"]:
            poses = []
            for i in insts:
                ps = {"xy": [float(i.pose.pos[0]), float(i.pose.pos[1])], "yaw": math.degrees(i.pose.yaw)}  # full precision: exact replay
                if i.parent and i.anchor:
                    ps["support"] = f"{i.parent}.{i.anchor}" if not i.parent.startswith("door_") else None
                elif i.relation == "attach_to":
                    ps["z"] = float(i.pose.pos[2])
                if i.meta.get("zone") and not o.get("zone_label"):
                    ps["zone"] = i.meta["zone"]
                poses.append({k: v for k, v in ps.items() if v is not None})
            o["placement"] = {"pose": poses}
            o.pop("yaw", None)
        o["count"] = len(insts)
        objs.append(o)
    for o in objs:  # catalog queries are pinned to the exact path + revision they resolved to
        if isinstance(o["asset"], dict) and "catalog" in o["asset"]:
            o["asset"] = pin_catalog(o["asset"])
    d["objects"] = objs
    d.setdefault("metadata", {})["resolved_from"] = spec_hash(sol.spec)
    return d


# --------------------------------------------------------------------------------------------------- entry point
def compile_scene(src, seed: int | None = None, out_dir: str | Path = "out", *, base_dir: Path | None = None, run_check: bool = True,
                  write_resolved: bool = True) -> CompileResult:
    t0 = time.time()
    spec, issues = validate_check(src, base_dir)
    if spec is None or has_errors(issues):
        return CompileResult(False, issues=issues, seconds=time.time() - t0)
    if base_dir is None and isinstance(src, (str, Path)) and len(str(src)) < 4096 and Path(str(src)).exists():
        base_dir = Path(str(src)).resolve().parent
    seed = seed if seed is not None else (spec.seed or 0)
    sol = solve(spec, seed, base_dir)
    issues = issues + sol.issues
    if has_errors(sol.issues):
        return CompileResult(False, issues=issues, stats=dict(sol.stats), seconds=time.time() - t0)
    out_dir = Path(out_dir)
    try:
        usd, graph = build(sol, out_dir)
    except Exception as e:  # surface build failures as structured issues
        import traceback

        issues.append(error("build.failed", f"{type(e).__name__}: {e}", "", "This is a generator bug or an asset param combination the "
                            "asset cannot build; try other params.", traceback=traceback.format_exc()[-1500:]))
        return CompileResult(False, out_dir, issues=issues, seconds=time.time() - t0)
    if write_resolved:
        (out_dir / "resolved_spec.json").write_text(json.dumps(resolved_spec(sol), indent=1))
    res = CompileResult(True, out_dir, usd, out_dir / "scene_graph.json", issues, None, dict(sol.stats, counts=graph["counts"]), 0.0)
    if run_check:
        from .check import check_scene

        res.check = check_scene(out_dir)
        res.issues = issues + [Issue(**i) for i in res.check["issues"]]
        res.ok = bool(res.check["passed"])
    res.seconds = time.time() - t0
    return res
