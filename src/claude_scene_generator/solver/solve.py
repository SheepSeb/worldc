"""Layout solver: SceneSpec + seed -> Solution (every instance with params and a world pose), deterministic.

Order of work (each stage only sees what earlier stages placed):
  1. environment: bounds, slope zones (keep-outs), keep_clear zones, wall openings + their doors
  2. objects in dependency order; priority layout/pose/against/attach < fill/on < next_to < inside/near
     (agents' corridors, the robot spawn and the nav goals are reserved before the free-floor objects)
  3. clearance constraints, as soon as all their `adjust` objects are placed (1-D search along the gap axis)
  4. deferred nav goals, reachability (dropping optional instances that block a goal), count_at_least

Failures are reported as Issues (codes ``solve.*``) whose messages say what failed, by how much and what blocked it.
"""
from __future__ import annotations

import math
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..assets import registry
from ..assets.base import Asset, Footprint, Pose
from ..assets.procedural.building import BuildingShell, Opening, SlopeZone
from ..assets.resolve import AssetResolutionError, resolve
from ..geom import Q_IDENT, qyaw, qrpy
from ..spec.issues import Issue, error, pointer, warning
from ..spec.models import (AgainstPlacement, AttachPlacement, FillPlacement, FillProbability, InsidePlacement, LayoutPlacement,
                           NearPlacement, NextToPlacement, OnPlacement, PosePlacement, SceneSpec, YawFace, YawRange)
from ..spec.refs import match, parse_ref
from . import nav as NV
from .geometry import Entry, Index, bbox, contains, overlap, principal_angle, ray_exit, rect_poly, scan_x, transform

K_SAMPLES = 64
FLOOR_PAD = 0.1  # floor objects keep this far from the interior wall line
GOAL_HALF = 0.32  # extra half-size of a goal's reserved square beyond the robot radius


def rng_for(seed: int, *keys) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed & 0xFFFFFFFF] + [zlib.crc32(str(k).encode()) for k in keys]))


# ---------------------------------------------------------------------------------------------------------- results
@dataclass
class Placed:
    key: str  # "<obj_id>.<index>"
    obj_id: str
    index: int
    asset: Asset
    params: object
    pose: Pose
    relation: str
    parent: str | None = None
    anchor: str | None = None
    optional: bool = False
    zone_label: str | None = None
    pieces: list = field(default_factory=list)  # world [(poly, z0, z1)]
    meta: dict = field(default_factory=dict)
    floor: bool = True  # rests on the floor (subject to keep-outs)


@dataclass
class EnvInfo:
    kind: str
    x_range: tuple
    y_range: tuple
    height: float | None
    shell: Asset | None
    shell_params: object | None
    slope_zones: list  # (SlopeZoneSpec, params)
    ports: dict  # name -> world Anchor (env.opening_<id>)

    def floor_z(self, x, y) -> float:
        z = 0.0
        for zs, p in self.slope_zones:
            cx, cy = (zs.rect[0] + zs.rect[2]) / 2, (zs.rect[1] + zs.rect[3]) / 2
            z = max(z, SlopeZone.height_fn(cx, cy, p)(x, y))
        return z

    @property
    def interior(self):
        return (self.x_range[0], self.y_range[0], self.x_range[1], self.y_range[1])


@dataclass
class Solution:
    spec: SceneSpec
    seed: int
    env: EnvInfo
    zones: dict  # id -> rect
    instances: list  # [Placed] in build order
    agents: list  # [{id, asset, params, waypoints, mode, speed, phase, width}]
    robot: dict | None  # {asset, params, pose, radius}
    goals: list
    clearance: list
    issues: list
    stats: dict

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def by_key(self) -> dict:
        return {p.key: p for p in self.instances}


class SolveError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------- solver
PHASE = {"pose": 0, "layout": 0, "against": 0, "attach_to": 1, "fill": 1, "on": 1, "next_to": 2, "near": 4, "inside": 4}


def _relation(o) -> str:
    for k, cls in (("pose", PosePlacement), ("layout", LayoutPlacement), ("against", AgainstPlacement), ("attach_to", AttachPlacement),
                   ("fill", FillPlacement), ("on", OnPlacement), ("next_to", NextToPlacement), ("near", NearPlacement), ("inside", InsidePlacement)):
        if isinstance(o.placement, cls):
            return k
    raise SolveError(f"unknown placement {o.placement!r}")


class Solver:
    def __init__(self, spec: SceneSpec, seed: int, base_dir: Path | None = None):
        self.spec, self.seed, self.base_dir = spec, seed, base_dir
        self.issues: list[Issue] = []
        self.index = Index()
        self.placed: dict[str, Placed] = {}
        self.order: list[str] = []
        self.by_obj: dict[str, list[str]] = {}
        self.zones: dict[str, tuple] = {z.id: tuple(z.rect) for z in spec.zones}
        self.obj_idx = {o.id: i for i, o in enumerate(spec.objects)}
        self.margin = max([c.margin for c in spec.constraints if c.type == "no_overlap" and c.between is None] + [0.0])
        self.clearance_done: dict[str, dict] = {}
        self.goals: list[dict] = []
        self.robot: dict | None = None
        self.agents: list[dict] = []
        self.stats = Counter()

    # ------------------------------------------------------------------ helpers
    def err(self, code, msg, path, hint="", **data):
        self.issues.append(error(code, msg, path, hint, **data))

    def warn(self, code, msg, path, hint="", **data):
        self.issues.append(warning(code, msg, path, hint, **data))

    def _asset(self, ref, path) -> Asset | None:
        try:
            return resolve(ref, self.base_dir)
        except (KeyError, AssetResolutionError, ValueError) as e:
            self.err(getattr(e, "code", "asset.unknown"), str(e).strip("'\""), path, getattr(e, "hint", ""))
            return None

    def _pieces(self, asset, p, pose: Pose):
        fp = asset.footprint(p)
        yaw = pose.yaw
        return [(transform(f.polygon, pose.pos[0], pose.pos[1], yaw), pose.pos[2] + f.z0, pose.pos[2] + f.z1) for f in fp.pieces()]

    def _register(self, inst: Placed):
        inst.pieces = self._pieces(inst.asset, inst.params, inst.pose)
        for poly, z0, z1 in inst.pieces:
            self.index.add(Entry(poly, z0, z1, inst.key, "object", inst.obj_id))
        for k in inst.asset.keepouts(inst.params):
            self.index.add(Entry(transform(k.polygon, inst.pose.pos[0], inst.pose.pos[1], inst.pose.yaw), k.z[0], k.z[1], inst.key, "keepout", k.tag))
        self.placed[inst.key] = inst
        self.order.append(inst.key)
        self.by_obj.setdefault(inst.obj_id, []).append(inst.key)

    def _unregister(self, key: str):
        for child in [k for k, p in self.placed.items() if p.parent == key]:
            self._unregister(child)
        self.index.remove_owner(key)
        inst = self.placed.pop(key)
        self.order.remove(key)
        self.by_obj[inst.obj_id].remove(key)

    def _ancestors(self, key):
        out = set()
        while key:
            out.add(key)
            key = self.placed[key].parent if key in self.placed else None
        return out

    def _free(self, asset, p, pose, parent=None, floor=True, interior=True, owner_ok=()):
        """(ok, blockers, reason) for a candidate pose."""
        pieces = self._pieces(asset, p, pose)
        if interior and self.env.kind == "indoor" and floor:
            if not all(contains(self.env.interior, poly, FLOOR_PAD) for poly, _, _ in pieces):
                return False, [], "outside_building"
        ignore = self._ancestors(parent) | set(owner_ok) if parent else set(owner_ok)
        hits = self.index.collisions(pieces, ignore, self.margin, keepouts=floor)
        if hits:
            return False, hits, "collision"
        return True, [], ""

    def _yaw(self, o, rng, default=0.0, ctx=None) -> float:
        y = o.yaw
        if y is None or y == "aligned":
            return default
        if y == "random":
            return float(rng.uniform(0, 2 * math.pi))
        if isinstance(y, YawRange):
            return math.radians(float(rng.uniform(*y.range)))
        if isinstance(y, YawFace):
            tgt = self._point_of(y.face)
            if tgt is None or ctx is None:
                return default
            return math.atan2(tgt[1] - ctx[1], tgt[0] - ctx[0]) - math.pi / 2
        return math.radians(float(y))

    def _jitter(self, o, rng, x, y, yaw):
        if o.jitter is None:
            return x, y, yaw
        r, a = o.jitter.xy * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        return x + r * math.cos(a), y + r * math.sin(a), yaw + math.radians(rng.uniform(-o.jitter.yaw, o.jitter.yaw))

    # ------------------------------------------------------------------ references
    def _instances(self, pr) -> list[Placed]:
        keys = self.by_obj.get(pr.target, [])
        if isinstance(pr.instance, int):
            keys = [k for k in keys if self.placed[k].index == pr.instance]
        return [self.placed[k] for k in keys]

    def _anchors(self, ref: str, kind=None):
        """[(instance or None, world Anchor)] matching a reference."""
        pr = parse_ref(ref)
        if pr.kind == "env":
            a = self.env.ports.get(pr.target)
            return [(None, a)] if a is not None else []
        out = []
        for inst in self._instances(pr):
            for a in inst.asset.anchors(inst.params):
                if pr.anchor and match(pr.anchor, a.name) and (kind is None or a.kind == kind):
                    out.append((inst, a.to_world(inst.pose)))
        return out

    def _point_of(self, ref):
        """World xy (+z) of a reference: zone centre, instance origin, anchor position."""
        if not isinstance(ref, str):
            return (float(ref[0]), float(ref[1]), 0.0)
        if ref in self.zones:
            x0, y0, x1, y1 = self.zones[ref]
            return ((x0 + x1) / 2, (y0 + y1) / 2, 0.0)
        pr = parse_ref(ref)
        if pr.kind == "clearance":
            c = self.clearance_done.get(pr.target)
            return None if c is None else (*c["gap_center"], 0.0)
        if pr.kind == "robot":
            return None if self.robot is None else tuple(self.robot["pose"].pos)
        if pr.anchor is not None or pr.kind == "env":
            hits = self._anchors(ref)
            return tuple(hits[0][1].pos) if hits else None
        insts = self._instances(pr)
        return tuple(insts[0].pose.pos) if insts else None

    # ------------------------------------------------------------------ environment
    def _environment(self):
        env = self.spec.environment
        W, D = env.footprint.size
        shell, sp, ports = None, None, {}
        if env.kind == "indoor":
            shell = registry.get("building_shell")
            sp = shell.params({"size": (W, D), "height": env.height or 8.0, "wall_thickness": env.walls.thickness,
                               "openings": [Opening(id=o.id, wall=o.wall, center=o.center, width=o.width, height=o.height) for o in env.walls.openings],
                               "roof": env.roof, "apron_depth": env.apron_depth, "wall_material": env.walls.material})
            for a in shell.anchors(sp):
                if a.kind == "port":
                    ports[a.name] = a
        slopes = []
        sz_asset = registry.get("slope_zone")
        for zs in env.floor.slope_zones:
            p = sz_asset.params({"size": (zs.rect[2] - zs.rect[0], zs.rect[3] - zs.rect[1]), "slope": zs.slope, "max_height": zs.max_height})
            slopes.append((zs, p))
            x0, y0, x1, y1 = zs.rect
            self.index.add(Entry(rect_poly(x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + 0.3), 0, zs.max_height, "", "keepout", f"slope:{zs.id}"))
        self.env = EnvInfo(env.kind, (-W / 2, W / 2), (-D / 2, D / 2), env.height, shell, sp, slopes, ports)
        for z in self.spec.zones:
            if z.keep_clear:
                self.index.add(Entry(rect_poly(*z.rect), 0, 3.0, "", "keepout", f"zone:{z.id}"))
        for k in self.spec.constraints:
            if k.type == "keep_clear":
                if k.zone and k.zone in self.zones:
                    self.index.add(Entry(rect_poly(*self.zones[k.zone]), 0, 3.0, "", "keepout", f"keep_clear:{k.id or k.zone}"))
                if k.path:
                    self._corridor(k.path, k.width, f"keep_clear:{k.id or 'path'}", closed=False)
        # doors in openings
        for j, o in enumerate(env.walls.openings):
            if o.door is None:
                continue
            a = self._asset(o.door, pointer("environment", "walls", "openings", j, "door"))
            if a is None:
                continue
            port = ports[f"opening_{o.id}"]
            raw = {**({"width": o.width, "height": o.height} if "width" in a.Params.model_fields else {}), **o.door_params}
            if "wall_thickness" in a.Params.model_fields:
                raw.setdefault("wall_thickness", env.walls.thickness)
            if a.id == "swing_door":
                raw = {k: v for k, v in raw.items() if k not in ("width", "height")}
                raw.setdefault("leaf_width", max(0.6, o.width - 0.25))
                raw.setdefault("leaf_height", max(1.8, o.height - 0.14))
            try:
                p = a.params(raw, rng=rng_for(self.seed, "door", o.id))
            except Exception as e:
                self.err("param.invalid", str(e), pointer("environment", "walls", "openings", j, "door_params"), "See describe_asset for valid params.")
                continue
            inst = Placed(f"door_{o.id}.0", f"door_{o.id}", 0, a, p, Pose(port.pos, port.quat), "attach_to", zone_label="door", floor=False)
            self._register(inst)

    def _corridor(self, pts, width, tag, closed=False):
        pts = [tuple(map(float, q)) for q in pts]
        n = len(pts)
        polys = []
        for i in range(n if closed else n - 1):
            (ax, ay), (bx, by) = pts[i], pts[(i + 1) % n]
            L = math.hypot(bx - ax, by - ay)
            if L < 1e-6:
                continue
            poly = transform(rect_poly(-L / 2 - width / 2, -width / 2, L / 2 + width / 2, width / 2), (ax + bx) / 2, (ay + by) / 2, math.atan2(by - ay, bx - ax))
            polys.append(poly)
            self.index.add(Entry(poly, 0, 3.0, "", "keepout", tag))
        return polys

    # ------------------------------------------------------------------ objects
    def _order(self):
        objs = self.spec.objects
        deps = {}
        for o in objs:
            d = set()
            refs = []
            pl = o.placement
            if isinstance(pl, OnPlacement):
                refs.append(pl.on)
            elif isinstance(pl, NextToPlacement):
                refs.append(pl.next_to.target)
            elif isinstance(pl, AttachPlacement):
                refs.append(pl.attach_to)
            elif isinstance(pl, FillPlacement):
                refs.append(pl.fill.anchors)
            elif isinstance(pl, NearPlacement) and isinstance(pl.near.target, str):
                refs.append(pl.near.target)
            elif isinstance(pl, InsidePlacement):
                refs += pl.inside if isinstance(pl.inside, list) else [pl.inside]
            elif isinstance(pl, PosePlacement):
                refs += [ps.support for ps in (pl.pose if isinstance(pl.pose, list) else [pl.pose]) if ps.support]
            if isinstance(o.yaw, YawFace):
                refs.append(o.yaw.face)
            for r in refs:
                t = parse_ref(r).target
                if t in self.obj_idx and t != o.id:
                    d.add(t)
                if "." in r and r.split(".")[0] in self.obj_idx and "aisle_" in r:
                    d.add(r.split(".")[0])
            deps[o.id] = d
        done, out = set(), []
        pending = sorted(objs, key=lambda o: (PHASE[_relation(o)], self.obj_idx[o.id]))
        while pending:
            for o in pending:
                if deps[o.id] <= done:
                    out.append(o)
                    done.add(o.id)
                    pending.remove(o)
                    break
            else:  # cycle (already reported by validation): place the rest in priority order
                out += pending
                break
        return out

    def _count(self, o, rng):
        if isinstance(o.count, tuple):
            lo, hi = o.count
            return int(rng.integers(lo, hi + 1)), lo
        return o.count, o.count

    def _params(self, asset, o, i, path):
        raw = dict(o.params)
        if o.instance_params and i < len(o.instance_params):
            raw.update(o.instance_params[i])
        try:
            return asset.params(raw, rng=rng_for(self.seed, o.id, i, "params"))
        except Exception as e:
            self.err("param.invalid", str(e), path, "See describe_asset for valid params and ranges.")
            return None

    def _place_object(self, o):
        i_obj = self.obj_idx[o.id]
        path = pointer("objects", i_obj, "placement")
        asset = self._asset(o.asset, pointer("objects", i_obj, "asset"))
        if asset is None:
            return
        rel = _relation(o)
        rng = rng_for(self.seed, o.id, "place")
        if rel == "fill":
            return self._fill(o, asset, rng, path)
        if rel == "layout":
            return self._layout(o, asset, rng, path)
        n, n_req = self._count(o, rng)
        if rel == "pose":
            poses = o.placement.pose if isinstance(o.placement.pose, list) else [o.placement.pose]
            n = n_req = len(poses) if isinstance(o.placement.pose, list) else n
        reasons = Counter()
        placed = 0
        for i in range(n):
            p = self._params(asset, o, i, pointer("objects", i_obj, "params"))
            if p is None:
                return
            irng = rng_for(self.seed, o.id, i, "place")
            ok = False
            for attempt in range(K_SAMPLES if rel in ("inside", "near", "on", "against", "next_to") else 1):
                cand = self._sample(o, rel, asset, p, irng, i, attempt)
                if cand is None:
                    reasons["no_candidate"] += 1
                    break
                pose, parent, anchor, floor, meta = cand
                good, hits, why = self._free(asset, p, pose, parent, floor)
                if good:
                    inst = Placed(f"{o.id}.{placed}", o.id, placed, asset, p, pose, rel, parent, anchor, i >= n_req, o.zone_label or meta.get("zone"),
                                  meta=meta, floor=floor)
                    self._register(inst)
                    placed += 1
                    ok = True
                    break
                if why == "collision":
                    for h in hits[:3]:
                        reasons[h.tag if h.kind != "object" else h.tag] += 1
                else:
                    reasons[why] += 1
            if not ok:
                if i >= n_req:
                    self.stats["dropped_optional"] += 1
                    continue
                self._fail(o, rel, path, placed, n_req, reasons)
                return
        self.stats["placed"] += placed

    def _fail(self, o, rel, path, placed, n_req, reasons):
        top = ", ".join(f"{k} ({v}x)" for k, v in reasons.most_common(4)) or "no candidate positions"
        hints = {
            "inside": "Lower the count, add or enlarge zones, use smaller params, or reduce the no_overlap margin.",
            "near": "Increase near.radius, lower the count or move the target point to free floor.",
            "on": "Use a bigger support or smaller objects, lower the count, or arrange: grid.",
            "next_to": "Try side: any, reduce gap, or free the space beside the target.",
            "against": "Change the wall / along range, or free the wall section.",
            "pose": "Move the pose to free space (blocked by the objects listed) or remove the conflicting object.",
            "attach_to": "Check the port reference and that nothing else occupies the opening.",
        }
        self.err(f"solve.unsatisfiable.{rel}", f"Placed {placed} of {n_req} required '{o.id}' instances ({rel}); rejected candidates "
                 f"were blocked by: {top}.", path, hints.get(rel, ""), placed=placed, required=n_req, blockers=dict(reasons))

    def _sample(self, o, rel, asset, p, rng, i, attempt):
        """-> (pose, parent_key, anchor_name, floor, meta) or None."""
        pl = o.placement
        fz = self.env.floor_z
        if rel == "pose":
            ps = pl.pose[i] if isinstance(pl.pose, list) else pl.pose
            x, y = ps.xy
            if ps.support:
                hits = self._anchors(ps.support, "support")
                if not hits:
                    return None
                inst, a = hits[0]
                yaw = math.radians(ps.yaw) if ps.yaw is not None else 0.0
                return (Pose.xy_yaw(x, y, yaw, ps.z if ps.z is not None else a.pos[2]), inst.key if inst else None, a.name, False,
                        {"support": ps.support, **({"zone": ps.zone} if ps.zone else {})})
            if ps.rpy is not None:
                q = qrpy(*[math.radians(v) for v in ps.rpy])
                return Pose((x, y, ps.z if ps.z is not None else fz(x, y)), q), None, None, ps.z is None, {}
            yaw = math.radians(ps.yaw) if ps.yaw is not None else self._yaw(o, rng, 0.0, (x, y))
            z = ps.z if ps.z is not None else fz(x, y)
            return Pose.xy_yaw(x, y, yaw, z), None, None, ps.z is None or ps.z <= fz(x, y) + 1e-6, ({"zone": ps.zone} if ps.zone else {})
        if rel == "inside":
            zones = pl.inside if isinstance(pl.inside, list) else [pl.inside]
            rects = [self.zones[z] for z in zones if z in self.zones]
            if not rects:
                return None
            areas = np.array([(r[2] - r[0]) * (r[3] - r[1]) for r in rects])
            zi = int(rng.choice(len(rects), p=areas / areas.sum()))
            r = rects[zi]
            zone_id = [z for z in zones if z in self.zones][zi]
            yaw = self._yaw(o, rng)
            fp = transform(asset.footprint(p).polygon, 0, 0, yaw)
            lo, hi = fp.min(0), fp.max(0)
            x0, x1 = r[0] + pl.margin - lo[0], r[2] - pl.margin - hi[0]
            y0, y1 = r[1] + pl.margin - lo[1], r[3] - pl.margin - hi[1]
            if x0 > x1 or y0 > y1:
                return None
            x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
            return Pose.xy_yaw(x, y, yaw, fz(x, y)), None, None, True, {"zone": zone_id}
        if rel == "near":
            c = self._point_of(pl.near.target)
            if c is None:
                return None
            rr, a = pl.near.radius * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
            x, y = c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)
            yaw = self._yaw(o, rng, 0.0, (x, y))
            return Pose.xy_yaw(x, y, yaw, fz(x, y)), None, None, True, {}
        if rel == "on":
            hits = self._anchors(pl.on, "support")
            if not hits:
                return None
            inst, a = hits[int(rng.integers(len(hits)))]
            poly = np.asarray(a.polygon, float)
            a_yaw = Pose(a.pos, a.quat).yaw
            fp_poly = asset.footprint(p).polygon
            if o.yaw is None or o.yaw == "aligned":
                yaw = a_yaw + principal_angle(poly) - principal_angle(fp_poly)
            else:
                yaw = self._yaw(o, rng)
            local_fp = transform(fp_poly, 0, 0, yaw - a_yaw)
            lo, hi = local_fp.min(0), local_fp.max(0)
            plo, phi = poly.min(0) + pl.margin, poly.max(0) - pl.margin
            if pl.arrange == "center" or attempt == 0 and pl.arrange == "center":
                lx, ly = (plo + phi) / 2 - (lo + hi) / 2
            else:
                if plo[0] - lo[0] > phi[0] - hi[0] or plo[1] - lo[1] > phi[1] - hi[1]:
                    return None
                lx, ly = rng.uniform(plo[0] - lo[0], phi[0] - hi[0]), rng.uniform(plo[1] - lo[1], phi[1] - hi[1])
            wx, wy = transform([(lx, ly)], a.pos[0], a.pos[1], a_yaw)[0]
            return Pose.xy_yaw(wx, wy, yaw, a.pos[2]), inst.key if inst else None, a.name, False, {}
        if rel == "next_to":
            spec = pl.next_to
            tinsts = self._instances(parse_ref(spec.target))
            if not tinsts:
                return None
            t = tinsts[int(rng.integers(len(tinsts)))] if parse_ref(spec.target).instance is None else tinsts[0]
            sides = [spec.side] if spec.side != "any" else [["front", "back", "left", "right"][k] for k in rng.permutation(4)]
            side = sides[attempt % len(sides)]
            ty = t.pose.yaw
            yaw = ty if spec.align else self._yaw(o, rng)
            tfp = np.asarray(t.asset.footprint(t.params).polygon)
            ofp = transform(asset.footprint(p).polygon, 0, 0, yaw - ty)
            d = {"front": (0, 1), "back": (0, -1), "right": (1, 0), "left": (-1, 0)}[side]
            ax = 1 if d[1] else 0
            if d[ax] > 0:
                off = tfp[:, ax].max() + spec.gap - ofp[:, ax].min()
            else:
                off = tfp[:, ax].min() - spec.gap - ofp[:, ax].max()
            lc = [0.0, 0.0]
            lc[ax] = off
            lc[1 - ax] = (tfp[:, 1 - ax].min() + tfp[:, 1 - ax].max()) / 2 - (ofp[:, 1 - ax].min() + ofp[:, 1 - ax].max()) / 2
            lc[1 - ax] += float(rng.uniform(-0.3, 0.3)) * (attempt > 3)
            wx, wy = transform([lc], t.pose.pos[0], t.pose.pos[1], ty)[0]
            return Pose.xy_yaw(wx, wy, yaw, fz(wx, wy)), None, None, True, {"next_to": t.key, "side": side}
        if rel == "against":
            ref = pl.against
            x0, y0, x1, y1 = self.env.interior
            if ref.startswith("wall:"):
                side, line = ref[5:], None
            else:
                zid, side = ref.rsplit(".", 1)
                x0, y0, x1, y1 = self.zones[zid]
                side = {"north": "south_of_north", "south": "north_of_south", "east": "west_of_east", "west": "east_of_west"}[side]
            spec = {"south": ((0, 1), y0, "x", (x0, x1)), "north": ((0, -1), y1, "x", (x0, x1)), "west": ((1, 0), x0, "y", (y0, y1)),
                    "east": ((-1, 0), x1, "y", (y0, y1)), "north_of_south": ((0, 1), y0, "x", (x0, x1)), "south_of_north": ((0, -1), y1, "x", (x0, x1)),
                    "east_of_west": ((1, 0), x0, "y", (y0, y1)), "west_of_east": ((-1, 0), x1, "y", (y0, y1))}[side]
            n, line, axis, (a0, a1) = spec
            yaw = math.atan2(n[1], n[0]) - math.pi / 2
            fp = transform(asset.footprint(p).polygon, 0, 0, yaw)
            m = (fp @ np.asarray(n, float)).min()
            k = 0 if axis == "x" else 1
            lo, hi = fp[:, k].min(), fp[:, k].max()
            if pl.along is None:
                rng_a = (a0 + FLOOR_PAD - lo, a1 - FLOOR_PAD - hi)
            elif isinstance(pl.along, (int, float)):
                rng_a = (pl.along, pl.along)
            else:
                rng_a = tuple(pl.along)
            if rng_a[0] > rng_a[1]:
                return None
            t = float(rng.uniform(*rng_a)) if rng_a[0] < rng_a[1] else rng_a[0]
            # place the footprint's wall-side extreme (min along the inward normal, m) at line + pad + gap
            off = (FLOOR_PAD if ref.startswith("wall:") else 0.0) + pl.gap - m
            if axis == "x":
                x, y = t, line + n[1] * off
            else:
                x, y = line + n[0] * off, t
            return Pose.xy_yaw(x, y, yaw, fz(x, y)), None, None, True, {"against": ref}
        if rel == "attach_to":
            hits = self._anchors(pl.attach_to, "port")
            if not hits:
                return None
            inst, a = hits[min(i, len(hits) - 1)]
            return Pose(a.pos, a.quat), inst.key if inst else None, a.name, False, {"attach_to": pl.attach_to}
        return None

    # ------------------------------------------------------------------ fill / layout
    def _fill(self, o, asset, rng, path):
        f = o.placement.fill
        pr = parse_ref(f.anchors)
        forced = set()
        for fr in f.force:
            fpr = parse_ref(fr)
            for inst in self._instances(fpr):
                for a in inst.asset.anchors(inst.params):
                    if fpr.anchor and match(fpr.anchor, a.name):
                        forced.add((inst.key, a.name))
        cands = []
        for inst in self._instances(pr):
            for a in inst.asset.anchors(inst.params):
                if a.kind != "support" or not (pr.anchor and match(pr.anchor, a.name)):
                    continue
                if all(a.meta.get(k) in (v if isinstance(v, list) else [v]) for k, v in f.where.items()):
                    cands.append((inst, a))
        n = 0
        dropped = 0
        for inst, a in cands:
            forced_here = (inst.key, a.name) in forced
            prob = f.probability
            if isinstance(prob, FillProbability):
                prob = prob.values.get(str(a.meta.get(prob.by)), prob.default)
            roll = rng.random()
            if not forced_here and roll >= prob:
                continue
            if f.max is not None and n >= f.max and not forced_here:
                continue
            p = self._params(asset, o, n, pointer("objects", self.obj_idx[o.id], "params"))
            if p is None:
                return
            aw = a.to_world(inst.pose)
            a_yaw = Pose(aw.pos, aw.quat).yaw
            yaw = a_yaw + principal_angle(a.polygon) - principal_angle(asset.footprint(p).polygon) if (o.yaw in (None, "aligned")) else self._yaw(o, rng)
            irng = rng_for(self.seed, o.id, n, "place")
            x, y, yaw = self._jitter(o, irng, aw.pos[0], aw.pos[1], yaw)
            pose = Pose.xy_yaw(x, y, yaw, aw.pos[2])
            good, hits, _ = self._free(asset, p, pose, inst.key, floor=False)
            if not good:
                dropped += 1
                continue
            self._register(Placed(f"{o.id}.{n}", o.id, n, asset, p, pose, "fill", inst.key, a.name, not forced_here, o.zone_label,
                                  meta={**a.meta, "forced": forced_here, "parent": inst.key, "anchor": a.name}, floor=False))
            n += 1
        if forced and not all(any(self.placed[k].parent == ik and self.placed[k].anchor == an for k in self.by_obj.get(o.id, [])) for ik, an in forced):
            self.err("solve.unsatisfiable.fill", f"Some forced anchors of '{o.id}' could not be filled (blocked).", path + "/fill/force",
                     "Free those anchors or remove them from force.")
        self.stats["placed"] += n
        self.stats["fill_blocked"] += dropped

    def _layout(self, o, asset, rng, path):
        lay = o.placement.layout
        poses, extra = [], []
        if lay.type == "rack_aisles":
            p0 = self._params(asset, o, 0, pointer("objects", self.obj_idx[o.id], "params"))
            if p0 is None:
                return
            depth = getattr(p0, "depth", lay.row_depth)
            segs = getattr(p0, "segments", [(-5.0, 5.0)])
            ylo, yhi = min(s[0] for s in segs) - 0.35, max(s[1] for s in segs) + 0.35
            x = lay.start_x
            poses.append((x + depth / 2, 1))
            x += depth
            for k in range(lay.aisles):
                self.zones[f"{o.id}.aisle_{k}"] = (x, max(self.env.y_range[0], lay.y + ylo), x + lay.aisle_width, min(self.env.y_range[1], lay.y + yhi))
                x += lay.aisle_width
                if k < lay.aisles - 1:
                    poses.append((x + depth / 2, -1))
                    x += depth + lay.back_gap
                    poses.append((x + depth / 2, 1))
                    x += depth
                else:
                    poses.append((x + depth / 2, -1))
                    x += depth
            items = [(Pose.xy_yaw(px, lay.y, 0.0), {"face": face, "aisle": (len(extra) + 1) // 2 if True else 0}) for px, face in poses]
            for k, (pose, d) in enumerate(items):
                d["aisle"] = (k + 1) // 2 if k < len(items) - 1 else lay.aisles - 1
                d["aisle"] = min(lay.aisles - 1, k // 2 if k % 2 == 0 and k > 0 else (k + 1) // 2 if k > 0 else 0)
            # row k faces aisle: row 0 -> aisle 0; rows 2a-1 (face -1) -> aisle a-1 ... recompute plainly:
            faces_aisle = [0] + [((k - 1) // 2) + ((k - 1) % 2) for k in range(1, len(items))]
            items = [(pose, {"face": d["face"], "aisle": faces_aisle[k]}) for k, (pose, d) in enumerate(items)]
            overrides = [{"face": d["face"]} for _, d in items]
        elif lay.type == "row":
            n, _ = self._count(o, rng)
            d = math.radians(lay.direction)
            items = [(Pose.xy_yaw(lay.start[0] + k * lay.spacing * math.cos(d), lay.start[1] + k * lay.spacing * math.sin(d), 0.0), {}) for k in range(n)]
            overrides = [{} for _ in items]
        elif lay.type == "grid":
            items = [(Pose.xy_yaw(lay.origin[0] + c * lay.spacing[0], lay.origin[1] + r * lay.spacing[1], 0.0), {"row": r, "col": c})
                     for r in range(lay.rows) for c in range(lay.cols)]
            overrides = [{} for _ in items]
        else:  # along_path
            pts = np.asarray(lay.path, float)
            seg = np.diff(pts, axis=0)
            L = np.linalg.norm(seg, axis=1)
            cum = np.concatenate([[0], np.cumsum(L)])
            items = []
            for s in np.arange(0, cum[-1] + 1e-9, lay.spacing):
                k = min(int(np.searchsorted(cum, s, side="right") - 1), len(L) - 1)
                u = (s - cum[k]) / L[k]
                dvec = seg[k] / L[k]
                q = pts[k] + u * seg[k] + lay.offset * np.array([-dvec[1], dvec[0]])
                items.append((Pose.xy_yaw(q[0], q[1], math.atan2(dvec[1], dvec[0]) - math.pi / 2), {}))
            overrides = [{} for _ in items]
        i_obj = self.obj_idx[o.id]
        for k, ((pose, meta), ov) in enumerate(zip(items, overrides)):
            raw = {**o.params, **(o.instance_params[k] if o.instance_params and k < len(o.instance_params) else {}), **ov}
            try:
                p = asset.params(raw, rng=rng_for(self.seed, o.id, k, "params"))
            except Exception as e:
                self.err("param.invalid", str(e), pointer("objects", i_obj, "params"), "See describe_asset.")
                return
            yaw = pose.yaw if o.yaw is None else self._yaw(o, rng, pose.yaw, pose.pos)
            x, y = pose.pos[0], pose.pos[1]
            pose = Pose.xy_yaw(x, y, yaw, self.env.floor_z(x, y))
            good, hits, why = self._free(asset, p, pose)
            if not good:
                who = ", ".join(sorted({h.tag for h in hits})) or why
                self.err("solve.unsatisfiable.layout", f"Layout instance {k} of '{o.id}' at ({x:.2f}, {y:.2f}) is blocked by: {who}.", path,
                         "Shift the layout (start / origin), reduce its size, or move the blocking objects / zones.")
                return
            self._register(Placed(f"{o.id}.{k}", o.id, k, asset, p, pose, "layout", zone_label=o.zone_label, meta=meta))
        self.stats["placed"] += len(items)

    # ------------------------------------------------------------------ agents / robot / goals
    def _agents(self):
        for i, ag in enumerate(self.spec.agents):
            a = self._asset(ag.asset, pointer("agents", i, "asset"))
            if a is None:
                continue
            wps = []
            for k, w in enumerate(ag.path.waypoints):
                pt = self._point_of(w)
                if pt is None:
                    self.err("ref.unknown", f"Cannot resolve waypoint {w!r}.", pointer("agents", i, "path", "waypoints", k), "Use [x, y] or a placed zone/object reference.")
                    break
                wps.append((float(pt[0]), float(pt[1])))
            else:
                raw = {**ag.params, "waypoints": wps, "speed_mps": ag.path.speed, "path_mode": ag.path.mode, "path_phase_m": ag.path.phase}
                try:
                    p = a.params(raw, rng=rng_for(self.seed, ag.id, "params"))
                except Exception as e:
                    self.err("param.invalid", str(e), pointer("agents", i, "params"), "See describe_asset.")
                    continue
                width = ag.keep_clear_width or (a.footprint(p).size[1] + 0.9)
                polys = self._corridor(wps, width, f"agent:{ag.id}", closed=ag.path.mode == "loop")
                hits = {h.tag for P in polys for h in self.index.collisions([(P, NV.Z_BAND[0], NV.Z_BAND[1])], keepouts=False) if h.kind == "object"}
                if hits:
                    self.warn("solve.agent_corridor_blocked", f"Agent '{ag.id}' corridor ({width:.2f} m wide) overlaps already-placed: "
                              f"{', '.join(sorted(hits))}.", pointer("agents", i, "path"),
                              "Move the waypoints into free aisles, narrow keep_clear_width, or move those objects.", objects=sorted(hits))
                self.agents.append({"id": ag.id, "asset": a, "params": p, "waypoints": wps, "width": width})

    def _robot(self):
        r = self.spec.robot
        if r is None:
            return
        a = self._asset(r.asset, "/robot/asset")
        if a is None:
            return
        p = a.params({**r.params, **({"footprint_radius": r.footprint_radius} if "footprint_radius" in a.Params.model_fields else {})})
        rng = rng_for(self.seed, "robot", "spawn")
        pl = r.spawn
        fp = Footprint.circle(r.footprint_radius + 0.05, 1.5)
        pose = None
        for attempt in range(K_SAMPLES):
            if isinstance(pl, PosePlacement):
                ps = pl.pose[0] if isinstance(pl.pose, list) else pl.pose
                cand = Pose.xy_yaw(ps.xy[0], ps.xy[1], math.radians(ps.yaw or 0.0))
            elif isinstance(pl, InsidePlacement):
                zs = pl.inside if isinstance(pl.inside, list) else [pl.inside]
                x0, y0, x1, y1 = self.zones[zs[int(rng.integers(len(zs)))]]
                cand = Pose.xy_yaw(rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5), rng.uniform(0, 2 * math.pi))
            else:
                c = self._point_of(pl.near.target)
                rr, ang = pl.near.radius * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
                cand = Pose.xy_yaw(c[0] + rr * math.cos(ang), c[1] + rr * math.sin(ang), rng.uniform(0, 2 * math.pi))
            poly = transform(fp.polygon, cand.pos[0], cand.pos[1], 0)
            hits = [h for h in self.index.collisions([(poly, 0.1, 1.5)], margin=0.0, keepouts=False) if h.kind == "object"]
            if not hits:
                pose = cand
                break
            if isinstance(pl, PosePlacement):
                self.err("solve.robot_spawn_blocked", f"Robot spawn ({cand.pos[0]:.2f}, {cand.pos[1]:.2f}) overlaps " +
                         ", ".join(sorted({h.tag for h in hits})) + ".", "/robot/spawn", "Move the spawn to free floor.")
                return
        if pose is None:
            self.err("solve.robot_spawn_blocked", "No free robot spawn found.", "/robot/spawn", "Use a larger zone or a free pose.")
            return
        self.index.add(Entry(transform(fp.polygon, pose.pos[0], pose.pos[1], 0), 0, 2.0, "", "spawn", "robot_spawn"))
        self.robot = {"asset": a, "params": p, "pose": pose, "radius": r.footprint_radius}

    def _goal_pose(self, g, gi):
        """Stand pose for a nav goal -> dict or None (deferred)."""
        path = pointer("robot", "nav_goals", gi, "target")
        tgt, front, extra = g.target, None, {}
        if isinstance(tgt, str):
            pr = parse_ref(tgt)
            if pr.kind == "clearance":
                c = self.clearance_done.get(pr.target)
                if c is None:
                    return None
                pt = (*c["gap_center"], 0.0)
                front = None
            elif pr.kind == "env" or pr.anchor is not None:
                hits = self._anchors(tgt)
                if not hits:
                    return None if pr.kind == "object" and pr.target not in self.placed_ids() else self._goal_fail(g, path, "target not found")
                inst, a = hits[0]
                pt = a.pos
                apose = Pose(a.pos, a.quat)
                if g.approach == "port" or (a.kind == "port" and a.port_kind in ("parking", "approach")):
                    return {"stand": (a.pos[0], a.pos[1]), "yaw": apose.yaw if g.yaw is None else math.radians(g.yaw), "target_point": list(map(float, a.pos)),
                            "target_meta": a.meta, "on_port": True}
                if "face" in a.meta and inst is not None:
                    front = transform([(a.meta["face"], 0.0)], 0, 0, inst.pose.yaw)[0]
                    # stand-off measured from the parent's footprint boundary along `front`
                    for poly, z0, z1 in inst.pieces:
                        pass
                    hull = np.concatenate([q for q, _, _ in inst.pieces])
                    from scipy.spatial import ConvexHull

                    hull = hull[ConvexHull(hull).vertices]
                    base = ray_exit(hull, a.pos[:2], front)
                    extra["target_meta"] = a.meta
                    stand = np.asarray(a.pos[:2]) + front * (base + g.distance)
                    yaw = math.atan2(-front[1], -front[0])
                    return {"stand": tuple(stand), "yaw": yaw if g.yaw is None else math.radians(g.yaw), "target_point": list(map(float, a.pos)), **extra}
                front = transform([(0.0, 1.0)], 0, 0, apose.yaw)[0]
            else:
                insts = self._instances(pr)
                if not insts:
                    return None
                inst = insts[0]
                pt = inst.pose.pos
                front = transform([(0.0, 1.0)], 0, 0, inst.pose.yaw)[0]
        else:
            pt = (float(tgt[0]), float(tgt[1]), 0.0)
        pt = np.array(pt[:2], float) + (np.asarray(g.offset, float) if g.offset else 0.0)
        if front is not None and g.approach in ("front", "any") and g.distance > 0:
            stand = pt + np.asarray(front) * g.distance
            yaw = math.atan2(pt[1] - stand[1], pt[0] - stand[0])
        else:
            stand, yaw = pt, (math.radians(g.yaw) if g.yaw is not None else math.pi / 2)
            if g.distance > 0 and front is None and g.approach == "any":
                stand, yaw = pt, (math.radians(g.yaw) if g.yaw is not None else 0.0)
        if g.yaw is not None:
            yaw = math.radians(g.yaw)
        return {"stand": tuple(stand), "yaw": yaw, "target_point": [float(pt[0]), float(pt[1]), 0.0], **extra}

    def placed_ids(self):
        return set(self.by_obj)

    def _goal_fail(self, g, path, why):
        self.err("solve.goal_target", f"Nav goal '{g.id}': {why}.", path, "Point the goal at a placed object, anchor, port or [x, y].")
        return False

    def _goals(self, final=False):
        if self.spec.robot is None:
            return
        have = {g["id"] for g in self.goals}
        for gi, g in enumerate(self.spec.robot.nav_goals):
            if g.id in have:
                continue
            res = self._goal_pose(g, gi)
            if res is False:
                have.add(g.id)
                continue
            if res is None:
                if final:
                    self._goal_fail(g, pointer("robot", "nav_goals", gi, "target"), "target could not be resolved after placement")
                continue
            sx, sy = res.pop("stand")
            yaw = res.pop("yaw")
            res.pop("on_port", None)
            gd = {"id": g.id, "name": g.id, "type": g.type, "base_pose": [round(float(sx), 3), round(float(sy), 3), round(float(yaw), 4)],
                  "target": g.target if isinstance(g.target, str) else list(g.target), **res, **g.extra, "_gi": gi}
            self.goals.append(gd)
            h = (self.spec.robot.footprint_radius + GOAL_HALF)
            self.index.add(Entry(rect_poly(sx - h, sy - h, sx + h, sy + h), 0, 2.0, "", "goal", f"goal:{g.id}"))

    # ------------------------------------------------------------------ clearance
    def _clearances(self):
        for ci, k in enumerate(self.spec.constraints):
            if k.type != "clearance" or (k.id or f"clearance_{ci}") in self.clearance_done:
                continue
            adjust_ids = set()
            for ref in k.adjust:
                adjust_ids.add(parse_ref(ref).target)
            if not all(a in self.by_obj or a not in self.obj_idx for a in adjust_ids):
                continue
            if any(a in self.obj_idx and a not in self.by_obj and self._pending(a) for a in adjust_ids):
                continue
            self._solve_clearance(ci, k, adjust_ids)

    def _pending(self, oid):
        return oid not in self._done_objects

    def _solve_clearance(self, ci, k, adjust_ids):
        cid = k.id or f"clearance_{ci}"
        path = pointer("constraints", ci)
        if k.zone not in self.zones:
            self.err("ref.unknown", f"clearance zone {k.zone!r} not found", path + "/zone", "Use a zone id or a layout aisle zone.")
            self.clearance_done[cid] = {"id": cid, "ok": False}
            return
        r = self.zones[k.zone]
        swap = k.axis == "y"

        def tf(P):  # work in a frame where the gap axis is x
            P = np.asarray(P, float)
            return P[:, ::-1] * np.array([1, 1]) if swap else P

        zx0, zy0, zx1, zy1 = (r[1], r[0], r[3], r[2]) if swap else r
        span = k.span or (zy0, zy1)
        movers = set()
        for key, inst in self.placed.items():
            if inst.obj_id in adjust_ids:
                movers |= {key}
        changed = True
        while changed:
            changed = False
            for key, inst in self.placed.items():
                if inst.parent in movers and key not in movers:
                    movers.add(key)
                    changed = True
        fixed, moving = [], []
        for e in self.index.entries:
            if e.kind != "object" or e.z1 < NV.Z_BAND[0] or e.z0 > NV.Z_BAND[1]:
                continue
            P = tf(e.poly)
            lo, hi = P.min(0), P.max(0)
            if hi[1] < span[0] or lo[1] > span[1] or hi[0] < zx0 - 0.6 or lo[0] > zx1 + 0.6:
                continue
            (moving if e.owner in movers else fixed).append(P)
        if not moving:
            self.err("solve.clearance", f"Clearance '{cid}': none of the adjust objects {sorted(adjust_ids)} lie in zone {k.zone!r}.", path,
                     "Place the adjust objects inside the zone first (e.g. pose/inside the zone).")
            self.clearance_done[cid] = {"id": cid, "ok": False}
            return
        ys = np.arange(span[0], span[1] + 1e-9, 0.01)
        fixed_iv = [[iv for iv in (scan_x(P, y) for P in fixed) if iv] for y in ys]
        move_iv = [[iv for iv in (scan_x(P, y) for P in moving) if iv] for y in ys]

        def narrowest(dx):
            worst, at = 1e9, None
            for j, y in enumerate(ys):
                iv = sorted([(-1e9, zx0), (zx1, 1e9)] + fixed_iv[j] + [(a + dx, b + dx) for a, b in move_iv[j]])
                best, cur, gap_at = 0.0, None, None
                for a, b in iv:
                    if cur is not None and a > cur and a - cur > best:
                        best, gap_at = a - cur, (cur + a) / 2
                    cur = b if cur is None else max(cur, b)
                if best < worst:
                    worst, at = best, (gap_at, y)
            return worst, at

        all_m = np.concatenate(moving)
        dx_lo, dx_hi = zx0 - all_m[:, 0].min(), zx1 - all_m[:, 0].max()
        best = None
        for dx in np.arange(dx_lo, dx_hi + 1e-9, 0.005):
            g, at = narrowest(dx)
            err = abs(g - k.target)
            if best is None or err < best[0] - 1e-9 or (abs(err - best[0]) < 1e-9 and abs(dx) < abs(best[1])):
                if self._movers_free(movers, dx, swap):
                    best = (err, float(dx), g, at)
        if best is None:
            self.err("solve.clearance", f"Clearance '{cid}': no collision-free position of {sorted(adjust_ids)} inside {k.zone!r}.", path,
                     "Use smaller adjust objects or a wider zone.")
            self.clearance_done[cid] = {"id": cid, "ok": False}
            return
        err, dx, g, at = best
        self._shift(movers, dx, swap)
        gx, gy = at
        _, at_mid = narrowest(0.0)
        center = (gy, gx) if swap else (gx, (span[0] + span[1]) / 2)
        res = {"id": cid, "zone": k.zone, "rect": list(r), "axis": k.axis, "span": list(span), "target": k.target, "tolerance": k.tolerance,
               "planned_m": round(g, 4), "shift_m": round(dx, 4), "gap_center": [round(center[0], 4), round(center[1], 4)], "ok": err <= k.tolerance}
        self.clearance_done[cid] = res
        if err > k.tolerance:
            self.err("solve.clearance", f"Clearance '{cid}': best achievable narrowest gap is {g:.3f} m, target {k.target:.3f} ± {k.tolerance:.3f} m.",
                     path, "Change the target, the adjust objects' sizes, or the zone width.", achievable=round(g, 4))

    def _movers_free(self, movers, dx, swap):
        for key in movers:
            inst = self.placed[key]
            for poly, z0, z1 in inst.pieces:
                P = poly + (np.array([0, dx]) if swap else np.array([dx, 0]))
                hits = [h for h in self.index.collisions([(P, z0, z1)], ignore_owners=movers, margin=self.margin, keepouts=False)]
                if hits:
                    return False
        return True

    def _shift(self, movers, dx, swap):
        d = np.array([0, dx]) if swap else np.array([dx, 0])
        for key in movers:
            inst = self.placed[key]
            x, y, z = inst.pose.pos
            inst.pose = Pose((x + d[0], y + d[1], z), inst.pose.quat)
            for e in self.index.entries:
                if e.owner == key:
                    e.poly = e.poly + d
            self.index._bb = [bbox(e.poly) for e in self.index.entries]
            inst.pieces = [(P + d, z0, z1) for P, z0, z1 in inst.pieces]

    # ------------------------------------------------------------------ reachability
    def _reachability(self):
        if self.robot is None or not self.goals:
            return {}
        reach_cs = [k for k in self.spec.constraints if k.type == "reachable"]
        targets = set()
        for k in reach_cs:
            for t in k.to:
                targets |= {g["id"] for g in self.goals if t == "*" or match(t, g["id"])}
        radius = self.robot["radius"]
        grid = NV.Grid(self.env.x_range, self.env.y_range)
        result = {}
        for round_ in range(4):
            ents = [e for e in self.index.entries if e.kind == "object" and e.z1 > NV.Z_BAND[0] and e.z0 < NV.Z_BAND[1]]
            occ, lab = grid.raster([e.poly for e in ents])
            free = grid.free(occ, radius)
            start = NV.nearest_free(free, grid.w2g(*self.robot["pose"].pos[:2]), 4)
            if start is None:
                self.err("solve.robot_spawn_blocked", "Robot spawn is not navigable (too close to obstacles for the footprint radius).",
                         "/robot/spawn", "Move the spawn at least footprint_radius + 2 cm away from obstacles.")
                return {}
            cells = {}
            for g in self.goals:
                c = NV.nearest_free(free, grid.w2g(*g["base_pose"][:2]), 6)
                cells[g["id"]] = c
            cost = np.where(free, 1.0, np.inf)
            dist, prev = NV.dijkstra(cost, start, [c for c in cells.values() if c])
            unreachable = [g for g in self.goals if g["id"] in targets and (cells[g["id"]] is None or not np.isfinite(dist[cells[g["id"]]]))]
            for g in self.goals:
                c = cells[g["id"]]
                ok = c is not None and np.isfinite(dist[c])
                result[g["id"]] = {"reachable": bool(ok), "path_length_m": round(float(dist[c] * grid.res), 2) if ok else None}
                if ok and c != grid.w2g(*g["base_pose"][:2]):
                    x, y = grid.g2w(*c)
                    g["base_pose"][:2] = [round(x, 3), round(y, 3)]
            if not unreachable:
                return result
            # which instances block? soft-cost path where blocked cells are expensive
            soft = np.where(free, 1.0, 40.0)
            soft[occ & ~self._border(occ)] = 400.0
            d2, p2 = NV.dijkstra(soft, start, [cells[g["id"]] or grid.w2g(*g["base_pose"][:2]) for g in unreachable])
            blockers = Counter()
            from scipy.ndimage import binary_dilation

            rad_cells = int(math.ceil((radius + 0.02) / grid.res))
            for g in unreachable:
                c = cells[g["id"]] or grid.w2g(*g["base_pose"][:2])
                pth = NV.path_to(p2, start, c)
                mask = np.zeros_like(occ)
                for rc in pth:
                    if not free[rc]:
                        mask[rc] = True
                mask = binary_dilation(mask, iterations=rad_cells)
                for li in np.unique(lab[mask & (lab >= 0)]):
                    blockers[ents[li].owner or ents[li].tag] += 1
            optional = [b for b in blockers if b in self.placed and self.placed[b].optional]
            if optional and round_ < 3:
                for b in optional:
                    if b in self.placed:
                        self._unregister(b)
                        self.stats["dropped_for_reachability"] += 1
                continue
            for g in unreachable:
                names = sorted({self.placed[b].obj_id if b in self.placed else b for b in blockers})
                self.err("solve.unreachable", f"Nav goal '{g['id']}' at ({g['base_pose'][0]:.2f}, {g['base_pose'][1]:.2f}) is not reachable from the "
                         f"robot spawn with footprint radius {radius} m; blocked by: {', '.join(names[:6]) or 'walls / unknown'}.",
                         pointer("robot", "nav_goals", g["_gi"]), "Move or shrink those objects, widen aisles (>= 2 x radius + 0.1 m), or move the goal.",
                         blockers=names[:10])
            return result
        return result

    @staticmethod
    def _border(occ):
        b = np.zeros_like(occ)
        b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
        return b

    # ------------------------------------------------------------------ main
    def run(self) -> Solution:
        self._done_objects = set()
        self._environment()
        objs = self._order()
        reserved = False
        for o in objs:
            if not reserved and PHASE[_relation(o)] >= 4:
                self._agents()
                self._robot()
                self._goals()
                reserved = True
            self._place_object(o)
            self._done_objects.add(o.id)
            self._clearances()
            if reserved:
                self._goals()
        if not reserved:
            self._agents()
            self._robot()
        self._clearances()
        self._goals(final=True)
        reach = self._reachability()
        for g in self.goals:
            g.update(reach.get(g["id"], {}))
        for ci, k in enumerate(self.spec.constraints):
            if k.type == "count_at_least" and len(self.by_obj.get(parse_ref(k.object).target, [])) < k.n:
                self.err("solve.count", f"Only {len(self.by_obj.get(parse_ref(k.object).target, []))} instances of {k.object!r} placed, need >= {k.n}.",
                         pointer("constraints", ci), "Enlarge the zones or reduce other clutter.")
        insts = [self.placed[k] for k in self.order]
        goals = [{k: v for k, v in g.items() if k != "_gi"} for g in self.goals]
        stats = dict(self.stats, n_instances=len(insts))
        return Solution(self.spec, self.seed, self.env, self.zones, insts, self.agents, self.robot, goals,
                        list(self.clearance_done.values()), self.issues, stats)


def solve(spec: SceneSpec, seed: int, base_dir: Path | None = None) -> Solution:
    return Solver(spec, seed, base_dir).run()
