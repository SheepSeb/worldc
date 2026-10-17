"""Semantic validation (layer 2): everything that can be checked without geometry solving.

    issues = validate(spec_or_dict)            # schema + semantic
    spec, issues = check(src)                  # same, also returns the parsed spec

Codes (all ``error`` unless noted):
  asset.unknown              registry id not found (hint: closest ids)
  asset.file_missing         FileAssetRef.file does not exist
  asset.catalog_unavailable  a catalog query needs the Omniverse index, and none is installed
  asset.catalog_no_match     no catalog asset matches the query well enough (the hint lists the closest paths)
  asset.generated_but_match_exists (warning)  a generated asset's name closely matches a catalog asset
  asset.catalog_limited_license (warning)  the catalog asset is Limited Use Content (Isaac/Environments)
  asset.generated_invalid    see the nested gen.* issues (re-pathed under /objects/i/asset/generate)
  param.invalid              asset param failed validation (unknown name, out of range, wrong type)
  id.duplicate               two things share an id (objects, zones, agents, goals, cameras share one namespace)
  zone.degenerate            x0 >= x1 or y0 >= y1
  zone.outside               zone not inside the environment footprint
  ref.unknown                reference to an unknown zone / object / opening / wall
  ref.anchor_unknown         anchor / port pattern matches nothing on the target asset
  ref.instance_range         instance index >= max count
  ref.kind                   reference of the wrong kind (e.g. 'on' needs a support anchor, 'attach_to' a port)
  relation.cycle             objects depend on each other in a cycle
  count.pose_mismatch        number of explicit poses != count
  fill.capacity  (warning)   count asks for more instances than matching anchors
  env.outdoor_walls (warning) walls / roof / openings are ignored outdoors
  env.opening_outside        opening does not fit on its wall
  env.opening_overlap        two openings on one wall overlap
  agent.waypoint_outside     waypoint outside the footprint
"""
from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np

from ..assets import registry
from ..assets.base import ParamError
from .issues import Issue, error, has_errors, pointer, warning
from .io import parse
from .models import AttachPlacement, FillPlacement, InsidePlacement, LayoutPlacement, NearPlacement, NextToPlacement, OnPlacement, PosePlacement, SceneSpec
from .refs import match, parse_ref


def _close(name, options):
    c = difflib.get_close_matches(name, list(options), n=3)
    return f" Did you mean {', '.join(repr(x) for x in c)}?" if c else ""


class _Ctx:
    def __init__(self, spec: SceneSpec, base_dir: Path | None):
        self.spec, self.base_dir = spec, base_dir
        self.issues: list[Issue] = []
        self.objects = {o.id: (i, o) for i, o in enumerate(spec.objects)}
        self.zones = {z.id: z for z in spec.zones}
        self.openings = {o.id: o for o in spec.environment.walls.openings}
        self.assets = {}  # object id -> Asset | None
        self.params = {}  # object id -> params at seed 0 (for anchor checks)
        self.layout_zones = set()
        for o in spec.objects:
            if isinstance(o.placement, LayoutPlacement) and o.placement.layout.type == "rack_aisles":
                self.layout_zones |= {f"{o.id}.aisle_{k}" for k in range(o.placement.layout.aisles)}

    def add(self, issue):
        self.issues.append(issue)


def _max_count(o) -> int:
    """Upper bound on the instances an object can produce (layouts and fills define their own counts)."""
    pl = o.placement
    if isinstance(pl, LayoutPlacement):
        lay = pl.layout
        return {"rack_aisles": lambda: 2 * lay.aisles, "grid": lambda: lay.rows * lay.cols}.get(lay.type, lambda: 10_000)()
    if isinstance(pl, FillPlacement):
        return pl.fill.max if pl.fill.max is not None else 10_000
    if isinstance(pl, PosePlacement) and isinstance(pl.pose, list):
        return len(pl.pose)
    return o.count[1] if isinstance(o.count, tuple) else o.count


def _resolve_asset(c: _Ctx, ref, path) -> object | None:
    if isinstance(ref, str):
        if not registry.exists(ref):
            c.add(error("asset.unknown", f"Unknown asset id {ref!r}.", path,
                        "Use list_assets / search_assets to find an id." + _close(ref, registry.ids())))
            return None
        return registry.get(ref)
    if hasattr(ref, "file"):
        f = Path(ref.file)
        if not f.is_absolute() and c.base_dir:
            f = c.base_dir / f
        if not f.exists():
            c.add(error("asset.file_missing", f"Asset file not found: {ref.file}", path + "/file", "Check the path (relative paths resolve against the spec file's folder)."))
        return None
    if hasattr(ref, "catalog"):
        from ..assets.catalog_index import license_of
        from ..assets.resolve import AssetResolutionError, pin_catalog

        try:  # no download here: the query is matched against the offline index only
            pinned = pin_catalog(ref)
            if license_of(pinned["path"]) == "limited-use":
                c.add(warning("asset.catalog_limited_license", f"{pinned['path']} is NVIDIA Limited Use Content (use only, no modification, "
                              "no redistribution grant).", path, "Fine for local generation; see docs/CATALOG.md before shipping it in a dataset."))
        except AssetResolutionError as e:
            c.add(error(e.code, str(e), path, e.hint))
        return None
    from ..assets.generated import GeneratedAsset, validate_generated

    sub = validate_generated(ref.generate)
    for i in sub:
        c.add(i.model_copy(update={"path": path + "/generate" + i.path}))
    if has_errors(sub):
        return None
    from ..assets.catalog_index import resolve_query

    gid = str(ref.generate["id"])
    near = resolve_query(gid.rsplit("/", 1)[-1].replace("_", " "))
    if near:
        c.add(warning("asset.generated_but_match_exists", f"The catalog already has {near['path']} for {gid!r}.", path,
                      "Prefer {catalog: omniverse, path: ...} over generating; generate only when nothing existing fits.", match=near["path"]))
    return GeneratedAsset(ref.generate)


def _check_params(c: _Ctx, oid, asset, raw, path):
    for s in range(3):  # a few samples so distribution literals are exercised
        try:
            p = asset.params(raw, rng=np.random.default_rng(s))
        except ParamError as e:
            fields = list(asset.Params.model_fields)
            for err in e.errors:
                loc = [x for x in err["loc"]]
                hint = f"Valid params for {asset.id}: {', '.join(fields)}." if err["type"] == "extra_forbidden" else "See describe_asset for ranges."
                if err["type"] == "extra_forbidden" and loc:
                    hint = f"'{loc[0]}' is not a param of {asset.id}." + _close(str(loc[0]), fields) + f" Valid: {', '.join(fields)}."
                c.add(error("param.invalid", err["msg"], pointer(*path, *loc), hint, asset=asset.id))
            return None
        except (TypeError, ValueError) as e:
            c.add(error("param.invalid", f"Bad distribution literal: {e}", pointer(*path), "Use {uniform:[a,b]}, {randint:[a,b]}, {choice:[...]}, {normal:[mu,sd]}."))
            return None
        if s == 0:
            c.params[oid] = p
    return c.params.get(oid)


def _anchors_of(c: _Ctx, oid):
    a, p = c.assets.get(oid), c.params.get(oid)
    if a is None or p is None:
        return None
    return a.anchors(p)


def _check_ref(c: _Ctx, ref: str, path: str, want: str):
    """want: zone | object | support | port | any"""
    pr = parse_ref(ref)
    if pr.kind == "wall":
        if pr.target not in ("south", "north", "east", "west"):
            c.add(error("ref.unknown", f"Unknown wall {pr.target!r}.", path, "Use wall:south | wall:north | wall:east | wall:west."))
        return pr
    if pr.kind == "env":
        if pr.target.startswith("opening_"):
            oid = pr.target[len("opening_"):]
            if oid not in c.openings:
                c.add(error("ref.unknown", f"No opening {oid!r} in environment.walls.openings.", path, "Known openings: " + (", ".join(c.openings) or "none") + "."))
        elif pr.target not in ("floor",):
            c.add(error("ref.unknown", f"Unknown environment reference {ref!r}.", path, "Use env.floor or env.opening_<id>."))
        return pr
    if pr.kind == "robot":
        return pr
    if pr.kind == "clearance":
        ids = {k.id for k in c.spec.constraints if getattr(k, "type", "") == "clearance" and k.id}
        if pr.target not in ids:
            c.add(error("ref.unknown", f"No clearance constraint with id {pr.target!r}.", path,
                        "Known clearance constraints: " + (", ".join(sorted(ids)) or "none") + "."))
        return pr
    if want == "zone" or (pr.anchor is None and pr.instance is None and pr.target in c.zones):
        if ref in c.zones or ref in c.layout_zones:
            return pr
        if want == "zone":
            c.add(error("ref.unknown", f"Unknown zone {ref!r}.", path, "Known zones: " + ", ".join(sorted(c.zones) + sorted(c.layout_zones))[:400] + "." + _close(ref, list(c.zones) + list(c.layout_zones))))
            return pr
    if pr.target not in c.objects:
        c.add(error("ref.unknown", f"Unknown object {pr.target!r} in reference {ref!r}.", path,
                    "Known objects: " + ", ".join(c.objects) + "." + _close(pr.target, list(c.objects) + list(c.zones))))
        return pr
    o = c.objects[pr.target][1]
    if isinstance(pr.instance, int) and pr.instance >= _max_count(o):
        c.add(error("ref.instance_range", f"{ref!r}: object {o.id!r} has at most {_max_count(o)} instances.", path, f"Use an index < {_max_count(o)}."))
    if pr.anchor is not None:
        anchors = _anchors_of(c, pr.target)
        if anchors is not None:
            hits = [a for a in anchors if match(pr.anchor, a.name)]
            if not hits:
                names = sorted({a.name for a in anchors})
                c.add(error("ref.anchor_unknown", f"{ref!r}: no anchor/port matches {pr.anchor!r} on {c.assets[pr.target].id}.", path,
                            ("Available: " + ", ".join(names[:12]) + (" ..." if len(names) > 12 else "") + ".") if names else "This asset exposes no anchors."))
            elif want in ("support", "port") and not any(a.kind == want for a in hits):
                c.add(error("ref.kind", f"{ref!r} is a {hits[0].kind}, but this relation needs a {want}.", path,
                            "Use a support anchor for 'on'/'fill' and a port for 'attach_to'."))
    elif want in ("support", "port"):
        c.add(error("ref.kind", f"{ref!r} names an object, but this relation needs a {want} anchor ('<object>.<anchor>').", path,
                    "Append an anchor name, e.g. '" + ref + ".top'."))
    return pr


def _deps(o) -> list[str]:
    pl = o.placement
    refs = []
    if isinstance(pl, OnPlacement):
        refs = [pl.on]
    elif isinstance(pl, NextToPlacement):
        refs = [pl.next_to.target]
    elif isinstance(pl, AttachPlacement):
        refs = [pl.attach_to]
    elif isinstance(pl, FillPlacement):
        refs = [pl.fill.anchors]
    elif isinstance(pl, NearPlacement) and isinstance(pl.near.target, str):
        refs = [pl.near.target]
    elif isinstance(pl, PosePlacement):
        refs = [ps.support for ps in (pl.pose if isinstance(pl.pose, list) else [pl.pose]) if ps.support]
    if o.yaw is not None and hasattr(o.yaw, "face"):
        refs.append(o.yaw.face)
    out = []
    for r in refs:
        pr = parse_ref(r)
        if pr.kind == "object":
            out.append(pr.target)
    return out


def _cycles(c: _Ctx):
    graph = {oid: [d for d in _deps(o) if d in c.objects] for oid, (_, o) in c.objects.items()}
    state = {}
    stack = []

    def dfs(n):
        state[n] = 1
        stack.append(n)
        for m in graph[n]:
            if state.get(m) == 1:
                cyc = stack[stack.index(m):] + [m]
                c.add(error("relation.cycle", "Placement relations form a cycle: " + " -> ".join(cyc) + ".",
                            pointer("objects", c.objects[n][0], "placement"), "Place one of these objects independently (inside / pose)."))
            elif state.get(m) is None:
                dfs(m)
        stack.pop()
        state[n] = 2

    for n in graph:
        if state.get(n) is None:
            dfs(n)


def semantic(spec: SceneSpec, base_dir: Path | None = None) -> list[Issue]:
    c = _Ctx(spec, base_dir)
    env = spec.environment
    W, D = env.footprint.size
    X0, X1, Y0, Y1 = -W / 2, W / 2, -D / 2, D / 2

    # ids
    seen = {}
    for kind, items in (("zones", spec.zones), ("objects", spec.objects), ("agents", spec.agents), ("cameras", spec.cameras),
                        ("robot/nav_goals", spec.robot.nav_goals if spec.robot else [])):
        for i, it in enumerate(items):
            if it.id in seen:
                c.add(error("id.duplicate", f"id {it.id!r} is used by both {seen[it.id]} and {kind}[{i}].", pointer(*kind.split("/"), i, "id"),
                            "Ids share one namespace; rename one."))
            seen[it.id] = f"{kind}[{i}]"

    # environment
    if env.kind == "outdoor" and (env.walls.openings or env.apron_depth):
        c.add(warning("env.outdoor_walls", "Walls, openings and apron are ignored for outdoor environments.", "/environment/walls", "Remove them or set kind: indoor."))
    for wall in ("south", "north", "east", "west"):
        ops = sorted((o.center - o.width / 2, o.center + o.width / 2, j, o) for j, o in enumerate(env.walls.openings) if o.wall == wall)
        L = W if wall in ("south", "north") else D
        lo_lim = -L / 2
        for a0, a1, j, o in ops:
            if a0 < lo_lim + 0.05 or a1 > L / 2 - 0.05:
                c.add(error("env.opening_outside", f"Opening {o.id!r} [{a0:.2f}, {a1:.2f}] does not fit on the {wall} wall ({-L / 2:.2f} .. {L / 2:.2f}).",
                            pointer("environment", "walls", "openings", j), "Move its center or reduce its width."))
            if env.height and o.height > env.height - 0.2:
                c.add(error("env.opening_outside", f"Opening {o.id!r} is taller than the walls ({env.height} m).", pointer("environment", "walls", "openings", j, "height"), "Reduce height."))
        for (a0, a1, j, o), (b0, b1, k, p) in zip(ops, ops[1:]):
            if b0 < a1 + 0.2:
                c.add(error("env.opening_overlap", f"Openings {o.id!r} and {p.id!r} on the {wall} wall overlap or are < 0.2 m apart.",
                            pointer("environment", "walls", "openings", k), "Move them apart."))
        for j, o in enumerate(env.walls.openings):
            if o.door is not None:
                _resolve_asset(c, o.door, pointer("environment", "walls", "openings", j, "door"))

    # zones
    for i, z in enumerate(spec.zones):
        x0, y0, x1, y1 = z.rect
        if x0 >= x1 or y0 >= y1:
            c.add(error("zone.degenerate", f"Zone {z.id!r} rect {list(z.rect)} is empty.", pointer("zones", i, "rect"), "Use [x0, y0, x1, y1] with x0 < x1 and y0 < y1."))
        elif x0 < X0 - 1e-6 or x1 > X1 + 1e-6 or y0 < Y0 - 1e-6 or y1 > Y1 + 1e-6:
            c.add(error("zone.outside", f"Zone {z.id!r} {list(z.rect)} extends outside the footprint [{X0}, {Y0}, {X1}, {Y1}].",
                        pointer("zones", i, "rect"), "Clip the rect to the footprint (origin = floor centre)."))

    # assets + params
    for i, o in enumerate(spec.objects):
        a = _resolve_asset(c, o.asset, pointer("objects", i, "asset"))
        c.assets[o.id] = a
        if a is not None:
            _check_params(c, o.id, a, o.params, ("objects", i, "params"))
            for k, ip in enumerate(o.instance_params or []):
                _check_params(c, f"{o.id}#{k}", a, {**o.params, **ip}, ("objects", i, "instance_params", k))
    for i, ag in enumerate(spec.agents):
        a = _resolve_asset(c, ag.asset, pointer("agents", i, "asset"))
        if a is not None:
            raw = {**ag.params, "waypoints": [w for w in ag.path.waypoints if not isinstance(w, str)] or [(0, 0), (1, 0)]}
            _check_params(c, ag.id, a, raw, ("agents", i, "params"))
        for k, w in enumerate(ag.path.waypoints):
            p = pointer("agents", i, "path", "waypoints", k)
            if isinstance(w, str):
                _check_ref(c, w, p, "any")
            elif not (X0 <= w[0] <= X1 and Y0 <= w[1] <= Y1):
                c.add(error("agent.waypoint_outside", f"Waypoint {list(w)} of agent {ag.id!r} is outside the footprint.", p, f"Keep x in [{X0}, {X1}], y in [{Y0}, {Y1}]."))

    # relations
    for i, o in enumerate(spec.objects):
        pl, base = o.placement, pointer("objects", i, "placement")
        if isinstance(pl, InsidePlacement):
            for k, z in enumerate(pl.inside if isinstance(pl.inside, list) else [pl.inside]):
                _check_ref(c, z, base + "/inside" + (f"/{k}" if isinstance(pl.inside, list) else ""), "zone")
        elif isinstance(pl, OnPlacement):
            _check_ref(c, pl.on, base + "/on", "support")
        elif isinstance(pl, NextToPlacement):
            _check_ref(c, pl.next_to.target, base + "/next_to/target", "object")
        elif isinstance(pl, AttachPlacement):
            _check_ref(c, pl.attach_to, base + "/attach_to", "port")
        elif isinstance(pl, FillPlacement):
            pr = _check_ref(c, pl.fill.anchors, base + "/fill/anchors", "support")
            anchors = _anchors_of(c, pr.target) if pr.kind == "object" else None
            if anchors is not None and pr.anchor:
                n_parent = _max_count(c.objects[pr.target][1]) if pr.target in c.objects else 1
                hits = [a for a in anchors if match(pr.anchor, a.name) and all(
                    a.meta.get(k) in (v if isinstance(v, list) else [v]) for k, v in pl.fill.where.items())]
                cap = len(hits) * (1 if isinstance(pr.instance, int) else n_parent)
                if isinstance(o.count, tuple) or o.count > 1:
                    lo = o.count[0] if isinstance(o.count, tuple) else o.count
                    if lo > cap:
                        c.add(warning("fill.capacity", f"{o.id!r} needs at least {lo} instances but only {cap} anchors match.",
                                      base + "/fill", "Lower count or widen the anchor pattern / where filter.", capacity=cap))
            for k, f in enumerate(pl.fill.force):
                _check_ref(c, f, base + f"/fill/force/{k}", "support")
        elif isinstance(pl, NearPlacement) and isinstance(pl.near.target, str):
            _check_ref(c, pl.near.target, base + "/near/target", "any")
        if isinstance(pl, PosePlacement):
            for k, ps in enumerate(pl.pose if isinstance(pl.pose, list) else [pl.pose]):
                if ps.support:
                    _check_ref(c, ps.support, base + (f"/pose/{k}" if isinstance(pl.pose, list) else "/pose") + "/support", "support")
        if isinstance(pl, PosePlacement) and isinstance(pl.pose, list):
            if isinstance(o.count, int) and len(pl.pose) != o.count and o.count != 1:
                c.add(error("count.pose_mismatch", f"{o.id!r}: {len(pl.pose)} poses for count {o.count}.", base + "/pose",
                            f"Give exactly {o.count} poses or set count to {len(pl.pose)}."))
        if o.yaw is not None and hasattr(o.yaw, "face"):
            _check_ref(c, o.yaw.face, pointer("objects", i, "yaw", "face"), "any")
    _cycles(c)

    # constraints, goals, cameras
    for i, k in enumerate(spec.constraints):
        base = pointer("constraints", i)
        for fld in ("zone", "object"):
            v = getattr(k, fld, None)
            if isinstance(v, str):
                _check_ref(c, v, f"{base}/{fld}", "zone" if fld == "zone" else "any")
        for fld in ("adjust", "objects", "between"):
            for j, v in enumerate(getattr(k, fld, None) or []):
                if v != "*" and not v.startswith("*"):
                    _check_ref(c, v, f"{base}/{fld}/{j}", "any")
    if spec.robot:
        goal_ids = {g.id for g in spec.robot.nav_goals}
        for i, k in enumerate(spec.constraints):
            if getattr(k, "type", "") == "reachable":
                for j, t in enumerate(k.to):
                    if t != "*" and not any(match(t, g) for g in goal_ids):
                        c.add(error("ref.unknown", f"reachable target {t!r} matches no nav goal.", pointer("constraints", i, "to", j),
                                    "Known goals: " + ", ".join(sorted(goal_ids)) + "."))
        for i, g in enumerate(spec.robot.nav_goals):
            if isinstance(g.target, str):
                _check_ref(c, g.target, pointer("robot", "nav_goals", i, "target"), "any")
    for i, cam in enumerate(spec.cameras):
        if cam.mount and cam.mount != "robot":
            _check_ref(c, cam.mount, pointer("cameras", i, "mount"), "object")
        if cam.mode == "perspective" and cam.position is None:
            c.add(error("schema.missing_field", f"Camera {cam.id!r} needs a position.", pointer("cameras", i, "position"), "Add position [x, y, z] (and look_at)."))
    return c.issues


def check(src, base_dir: Path | None = None) -> tuple[SceneSpec | None, list[Issue]]:
    """Schema + semantic validation.  ``src``: dict, SceneSpec, JSON/YAML text or a path."""
    if isinstance(src, SceneSpec):
        spec, issues = src, []
    else:
        if base_dir is None and isinstance(src, (str, Path)) and len(str(src)) < 4096 and Path(str(src)).exists():
            base_dir = Path(str(src)).resolve().parent
        spec, issues = parse(src)
    if spec is None:
        return None, issues
    return spec, issues + semantic(spec, base_dir)


def validate(src, base_dir: Path | None = None) -> list[Issue]:
    return check(src, base_dir)[1]
