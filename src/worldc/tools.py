"""The agent tool surface (PLAN.md §6) as plain functions, usable without the ``mcp`` package.

Every tool returns ``{"ok": bool, "result": {...}, "issues": [Issue dicts]}``; nothing raises.  Unexpected exceptions become
one ``internal.<ExceptionType>`` issue.  ``mcp_server.py`` registers these functions as MCP tools and turns a returned
``result["image"]`` path into image content.

    spec arguments accept a dict, a JSON/YAML string, or a path to a spec file.
    output directories live under the workspace ($WORLDC_WORKSPACE, default ./mcp_out); scene directories from earlier
    ``build_scene`` calls can be passed back from anywhere.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

from .jsonpatch import PatchError, apply_patch, diff_summary
from .spec.io import load_yaml
from .spec.issues import Issue, error, has_errors

DOCS = "docs/SPEC.md"
MAX_SPEC_CHARS = 2_000_000


# ------------------------------------------------------------------------------------------------------ plumbing
def workspace() -> Path:
    return Path(os.environ.get("WORLDC_WORKSPACE", "mcp_out")).expanduser().resolve()


def _issues(items) -> list[dict]:
    return [i.model_dump() if isinstance(i, Issue) else dict(i) for i in items]


def _out(ok: bool | None = None, result: dict | None = None, issues=()) -> dict:
    issues = _issues(issues)
    if ok is None:
        ok = not any(i.get("severity") == "error" for i in issues)
    return {"ok": bool(ok), "result": result or {}, "issues": issues}


def tool(fn: Callable) -> Callable:
    """Never raise: turn an unexpected exception into an ``internal.*`` issue."""

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            return _out(False, issues=[error(f"internal.{type(e).__name__}", str(e)[:800], hint="This is a bug in the tool, not in your input; "
                                             "report it with the arguments you passed.")])

    return wrapper


class SpecInputError(ValueError):
    pass


def _spec_dict(spec: Any) -> tuple[dict, Path | None]:
    """(spec as dict, folder for relative file assets)."""
    if isinstance(spec, dict):
        return spec, None
    if isinstance(spec, str):
        text, base = spec, None
        if len(spec) < 4096 and "\n" not in spec and Path(spec).expanduser().is_file():
            p = Path(spec).expanduser().resolve()
            text, base = p.read_text(), p.parent
        if len(text) > MAX_SPEC_CHARS:
            raise SpecInputError(f"spec is larger than {MAX_SPEC_CHARS} characters")
        try:
            d = load_yaml(text)  # JSON is valid YAML; the spec loader keeps `on:` a string (YAML 1.1 reads it as true)
        except yaml.YAMLError as e:
            raise SpecInputError(f"spec is neither valid JSON nor YAML: {str(e)[:300]}") from e
        if not isinstance(d, dict):
            raise SpecInputError("spec must be an object (mapping) at the top level")
        return d, base
    raise SpecInputError(f"spec must be an object, a JSON/YAML string or a file path, got {type(spec).__name__}")


def _bad_spec(e: Exception) -> dict:
    return _out(False, issues=[error("spec.unreadable", str(e), hint="Pass the spec as a JSON object (or a JSON/YAML string / file path).")])


def _scene_dir(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_file() and p.name.endswith((".usd", ".usda", ".usdc")):
        p = p.parent
    if not (p / "scene_graph.json").is_file():
        raise FileNotFoundError(f"{path!r} is not a scene directory (no scene_graph.json). Build one with build_scene first.")
    return p.resolve()


def _under_workspace(out_dir: str) -> Path:
    ws = workspace()
    p = (ws / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir).resolve()
    if ws != p and ws not in p.parents:
        raise PermissionError(f"out_dir {out_dir!r} is outside the workspace {ws}; use a relative name.")
    return p


def _isaac():
    from .isaac_env import require_isaac_sim

    return require_isaac_sim()


def _isaac_env() -> dict:
    return dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))


# ------------------------------------------------------------------------------------------------------ assets
@tool
def list_assets(tags: list[str] | None = None, query: str | None = None) -> dict:
    """Registered assets (procedural, imported, generated) having all ``tags`` and matching the substring ``query``."""
    from .assets import list_assets as _list

    items = [{k: a[k] for k in ("id", "description", "tags", "source") if k in a} for a in _list(tags or None, query)]
    return _out(True, {"count": len(items), "assets": items})


@tool
def search_assets(query: str, tags: list[str] | None = None, sources: list[str] | None = None, limit: int = 10) -> dict:
    """Ranked matches over the registry and the Omniverse catalog.  Prefer a registry hit, then a catalog hit, and only
    generate a new asset when nothing fits.  ``sources``: any of "registry", "omniverse"."""
    from .assets.resolve import search

    hits = search(query, max(1, min(int(limit), 50)), tags or None)
    if sources:
        hits = [h for h in hits if h["source"] in sources]
    out = []
    for h in hits:
        if h["source"] == "registry":
            out.append({"source": "registry", "id": h["id"], "score": h["score"], "description": h.get("description", ""), "tags": h.get("tags", []),
                        "use": f'"asset": "{h["id"]}"'})
        else:
            phys = h.get("physics") or {}
            out.append({"source": "omniverse", "path": h["path"], "score": h["score"], "size_m": h.get("size_m"),
                        "has_physics": bool(phys.get("rigid_body") or phys.get("collision")), "license": h.get("license"),
                        "use": f'"asset": {{"catalog": "omniverse", "path": "{h["path"]}"}}'})
    hint = []
    if not out:
        hint.append(error("search.no_match", f"Nothing matches {query!r}.", hint="Try other words, or describe the object as a "
                          "`generate` asset.").model_copy(update={"severity": "info"}))
    return _out(True, {"count": len(out), "matches": out}, hint)


@tool
def describe_asset(id: str) -> dict:
    """Full description of a registered asset: params with ranges and defaults, footprint, anchors, ports, physics, joints."""
    from .assets import registry

    if not registry.exists(id):
        near = registry.suggest(id)
        return _out(False, issues=[error("asset.unknown", f"Unknown asset id {id!r}.", hint=("Did you mean: " + ", ".join(near) + "? " if near else "")
                                         + "Use list_assets or search_assets.")])
    return _out(True, registry.describe(id))


@tool
def import_asset(file: str, overrides: dict | None = None) -> dict:
    """Import a user mesh/USD file (obj, stl, glb, usd...) as a registry asset for this session.  ``overrides``: id, mass_kg,
    tags, body (static|dynamic|kinematic), material, description, up, units, max_hulls.  Reference it afterwards by id, or
    keep using ``{"file": path}`` in a spec to make the spec self-contained."""
    from .assets import registry

    try:
        from .assets.mesh.file_asset import file_asset
        from .assets.mesh.import_mesh import ImportError_
    except ImportError as e:
        return _out(False, issues=[error("env.mesh_extra_missing", f"Mesh import needs the 'mesh' extra: {e}", hint="Install with `uv sync --extra mesh`.")])
    try:
        asset = file_asset(file, overrides)
    except FileNotFoundError as e:
        return _out(False, issues=[error("asset.file_missing", str(e), hint="Pass an absolute path.")])
    except (ImportError_, ValueError, NotImplementedError) as e:
        return _out(False, issues=[error("asset.import_failed", str(e), hint="Check the file format and the override names.")])
    if not registry.exists(asset.id):
        registry.register(asset)
    return _out(True, {"id": asset.id, "summary": asset.summary(), "use": f'"asset": "{asset.id}"'})


@tool
def validate_generated_asset(asset: Any, render: bool = True) -> dict:
    """Check a declarative ``generate`` asset on its own (parts, joints, mass, stability) and, if it is valid, return a proxy
    render of it so you can see what you described."""
    from .assets.generated import validate_generated

    try:
        spec, _ = _spec_dict(asset)
    except SpecInputError as e:
        return _bad_spec(e)
    spec = spec.get("generate", spec)  # accept {"generate": {...}} or the bare asset spec
    issues = validate_generated(spec)
    res: dict = {"id": spec.get("id")}
    if has_errors(issues) or not render:
        return _out(None, res, issues)
    try:
        res.update(_preview_generated(spec))
    except Exception as e:  # noqa: BLE001 - the verdict stands even if the preview fails
        issues = list(issues) + [error("render.preview_failed", f"{type(e).__name__}: {e}", hint="The asset is valid; only its preview failed.").model_copy(update={"severity": "warning"})]
    return _out(None, res, issues)


def _preview_generated(spec: dict) -> dict:
    import hashlib

    from .assets.generated import GeneratedAsset
    from .compile import compile_scene
    from .render import render_proxy

    a = GeneratedAsset(spec)
    pieces = a.footprint(a.params({})).pieces()
    xs = [x for f in pieces for x, _ in f.polygon]
    ys = [y for f in pieces for _, y in f.polygon]
    h = max(f.z1 for f in pieces)
    dim = max(max(xs) - min(xs), max(ys) - min(ys), h, 0.3)
    room = max(6.0, 2.0 * dim + 2.0)
    scene = {"environment": {"footprint": {"size": [room, room]}}, "zones": [],
             "objects": [{"id": "preview", "asset": {"generate": spec}, "placement": {"pose": {"xy": [0, 0]}}}]}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    out = workspace() / "_asset_previews" / key
    res = compile_scene(scene, 0, out, run_check=False)
    if not res.ok:
        raise RuntimeError("; ".join(str(i) for i in res.issues if i.severity == "error")[:400])
    d = 1.6 * dim + 0.6
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    cam = {"position": [cx + d, cy - d, h * 0.6 + d * 0.5], "look_at": [cx, cy, h / 2], "focal_mm": 18.0, "name": "preview"}
    png = render_proxy(out, cam, out / "preview.png", 640, 480, hide=["/World/Building"], annotate=False)
    return {"image": str(png), "size_m": [round(max(xs) - min(xs), 3), round(max(ys) - min(ys), 3), round(h, 3)], "preview_dir": str(out)}


# ------------------------------------------------------------------------------------------------------ spec
@tool
def get_spec_schema(section: str | None = None) -> dict:
    """JSON Schema of the scene spec: the whole schema, or one ``$defs`` entry (plus what it references) to save tokens.
    Examples and every relation / constraint are explained in docs/SPEC.md."""
    from .spec.schema import build_schema, section as _section

    whole = build_schema()
    names = sorted(whole.get("$defs", {}))
    if not section:
        return _out(True, {"schema": whole, "sections": names, "docs": DOCS})
    try:
        return _out(True, {"schema": _section(section), "sections": names, "docs": DOCS})
    except KeyError as e:
        return _out(False, issues=[error("schema.unknown_section", str(e).strip("\"'"), hint="Pass one of the listed sections or none for the whole schema.")],
                    result={"sections": names})


def _solve_summary(sol) -> dict:
    per_obj = Counter(p.obj_id for p in sol.instances)
    return {"placed_instances": len(sol.instances), "instances_by_object": dict(sorted(per_obj.items())),
            "solver_stats": {k: int(v) for k, v in sorted(sol.stats.items())},
            "clearance": [dict(c) if isinstance(c, dict) else c for c in sol.clearance] if sol.clearance else []}


@tool
def validate_spec(spec: Any, seed: int = 0) -> dict:
    """Schema + semantic validation, then a dry-run solve (no USD written) to find placement failures early.  Returns every
    problem as an issue with a JSON-pointer path and a hint, plus how many instances of each object were placed."""
    from .solver.solve import solve
    from .spec.validate import check

    try:
        d, base = _spec_dict(spec)
    except SpecInputError as e:
        return _bad_spec(e)
    parsed, issues = check(d, base)
    if parsed is None or has_errors(issues):
        return _out(None, {"stage": "validate"}, issues)
    sol = solve(parsed, int(seed), base)
    return _out(None, {"stage": "solved", "seed": int(seed), **_solve_summary(sol)}, list(issues) + list(sol.issues))


@tool
def edit_scene(spec: Any, json_patch: list[dict]) -> dict:
    """Apply an RFC 6902 JSON Patch (add / remove / replace / move / copy / test) to a spec, then validate the result.
    Returns the new spec, a diff, and the validation issues.  A bad operation is reported with its index and nothing is
    applied."""
    from .spec.validate import check

    try:
        d, base = _spec_dict(spec)
    except SpecInputError as e:
        return _bad_spec(e)
    try:
        new = apply_patch(d, json_patch)
    except PatchError as e:
        return _out(False, {"failed_op": e.index}, [error("patch.failed", e.message, f"/json_patch/{e.index}" if e.index >= 0 else "", e.hint, op=e.op)])
    _, issues = check(new, base)
    return _out(None, {"spec": new, "diff": diff_summary(d, new), "valid": not has_errors(issues)}, issues)


# ------------------------------------------------------------------------------------------------------ build / check / render
def _scene_files(out: Path) -> dict:
    names = ("scene.usd", "scene_graph.json", "resolved_spec.json", "check.json", "nav_occupancy.png")
    return {n: str(out / n) for n in names if (out / n).is_file()}


def _check_summary(chk: dict) -> dict:
    nav = chk.get("navigation", {})
    return {"passed": chk.get("passed"), "spawn_interpenetrations": len(chk.get("spawn_interpenetrations", [])),
            "floating_bodies": len(chk.get("floating_bodies", [])), "hazard_path_conflicts": len(chk.get("hazard_path_conflicts", [])),
            "goals_reachable": nav.get("all_reachable"), "n_goals": nav.get("n_goals"),
            "clearance": [{"id": c.get("id"), "measured_m": c.get("measured_m"), "target_m": c.get("target_m")} for c in chk.get("clearance", [])]}


@tool
def build_scene(spec: Any, seed: int = 0, out_dir: str | None = None, run_check: bool = True) -> dict:
    """Compile a spec into USD + scene_graph.json + resolved_spec.json (and run the pure-Python checks).  The output goes to
    a folder under the workspace; pass it to check_scene / render."""
    from .compile import compile_scene, spec_hash
    from .spec.validate import check

    try:
        d, base = _spec_dict(spec)
    except SpecInputError as e:
        return _bad_spec(e)
    parsed, issues = check(d, base)
    if parsed is None or has_errors(issues):
        return _out(None, {"stage": "validate"}, issues)
    out = _under_workspace(out_dir) if out_dir else workspace() / f"{spec_hash(parsed)}_{int(seed)}"
    res = compile_scene(parsed, int(seed), out, base_dir=base, run_check=run_check)
    result: dict = {"seconds": round(res.seconds, 2)}
    if res.ok:
        g = json.loads((out / "scene_graph.json").read_text())
        result.update(out_dir=str(out), files=_scene_files(out), counts=g.get("counts", {}), n_dynamic_bodies=g.get("n_dynamic_bodies"),
                      cameras=[c.get("name") for c in g.get("cameras", [])], goals=[x.get("name") for x in g.get("goals", [])],
                      seed=int(seed), stats=res.stats)
        if res.check is not None:
            result["check"] = _check_summary(res.check)
    return _out(res.ok, result, res.issues)


@tool
def check_scene(path: str, physics: bool = True, sim_seconds: float = 6.0, timeout_s: float = 900.0) -> dict:
    """Post-build checks of a scene directory: overlaps, floating bodies, hazard paths, goal reachability and clearances (pure
    Python), plus a physics settle in Isaac Sim (bodies must come to rest; subprocess with a timeout)."""
    from .check import check_scene as _check

    d = _scene_dir(path)
    chk = _check(d)
    issues = [Issue(**i) for i in chk.get("issues", [])]
    result = {"scene": str(d), "static": _check_summary(chk)}
    ok = bool(chk.get("passed"))
    if physics:
        phys, extra = _physics(d, float(sim_seconds), float(timeout_s))
        result["physics"] = phys
        issues += extra
        ok = ok and bool(phys.get("ok"))
    return _out(ok, result, issues)


def _physics(d: Path, sim_seconds: float, timeout_s: float) -> tuple[dict, list[Issue]]:
    from .isaac_env import IsaacSimUnavailable

    try:
        sim = _isaac()
    except IsaacSimUnavailable as e:
        return {"ok": False, "ran": False}, [Issue(**e.issue)]
    f = d / "isaac_validation.json"
    f.unlink(missing_ok=True)
    try:
        p = subprocess.run([str(sim.python), "-m", "worldc.isaac_validate", str(d), "--sim-seconds", str(sim_seconds),
                            "--skip-render", "--skip-art"], env=_isaac_env(), capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "ran": False}, [error("physics.timeout", f"Physics settle did not finish within {timeout_s:.0f} s.", hint="Use a smaller scene or a larger timeout_s.")]
    (d / "isaac_run.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if not f.is_file():
        return {"ok": False, "ran": False}, [error("physics.failed", "Isaac Sim produced no result.", hint=f"See {d / 'isaac_run.log'}.", returncode=p.returncode)]
    st = json.loads(f.read_text()).get("stages", {})
    ph, load, rob = st.get("physics", {}), st.get("load", {}), st.get("robot", {})
    issues: list[Issue] = []
    if not load.get("ok"):
        issues.append(error("physics.load_failed", "Isaac Sim could not load the stage.", hint=str(load.get("message", ""))[:300]))
    for p_ in ph.get("fell_through_floor", [])[:10]:
        issues.append(error("physics.fell_through_floor", f"{p_} fell through the floor.", p_, "Check its collider and spawn height."))
    for m in ph.get("faster_than_1mps", [])[:10]:
        issues.append(error("physics.unstable", f"{m['body']} moved faster than 1 m/s ({m['max_speed_mps']}).", m["body"], "Usually an interpenetrating spawn."))
    if ph and ph.get("fraction_displacement_below_2cm", 1.0) < ph.get("target", 0.95):
        issues.append(error("physics.not_settled", f"Only {100 * ph['fraction_displacement_below_2cm']:.1f}% of bodies stayed within 2 cm.", hint="Look at worst_movers; stacks may be unstable or overlapping."))
    if ph.get("conveyor_payload_speeds") and not ph.get("conveyor_ok", True):
        issues.append(error("physics.conveyor_speed", "Cartons on a conveyor did not reach the belt speed.", hint="Run longer (sim_seconds >= 10) or check the "
                            "belt end; speeds: " + json.dumps(ph["conveyor_payload_speeds"])[:300]).model_copy(update={"severity": "warning"}))
    if rob.get("present") and not rob.get("ok", True):
        issues.append(error("physics.robot_unstable", rob.get("error") or f"The robot did not come to rest: drifted {rob.get('drift_xy_m')} m, rose {rob.get('max_rise_m')} m, final speed {rob.get('final_speed_mps')} m/s.",
                            "/robot", "Spawn it on clear, flat floor away from objects; check the robot loaded (needs the Isaac asset root)."))
    settled = bool(ph) and ph.get("fraction_displacement_below_2cm", 0.0) >= ph.get("target", 0.95) and not ph.get("fell_through_floor") and not ph.get("faster_than_1mps")
    summary = {k: ph.get(k) for k in ("n_bodies_evaluated", "fraction_displacement_below_2cm", "max_displacement_m", "worst_movers", "fell_through_floor") if k in ph}
    if rob.get("present"):
        summary["robot"] = {k: rob.get(k) for k in ("asset", "kind", "ok", "settled", "drift_xy_m", "dz_m", "max_rise_m", "final_speed_mps", "n_dofs", "needs_controller", "error") if k in rob}
    summary.update(ran=True, ok=bool(load.get("ok") and settled and rob.get("ok", True)), result_file=str(f))
    return summary, issues


def auto_cameras(graph: dict) -> dict[str, dict]:
    """Cameras framed from the scene extents (works for any scene): ``auto_topdown`` and a 3/4 ``auto_overview``.  Both hide roofs."""
    import math

    env = graph.get("environment") or graph.get("building") or {}
    xr, yr = env.get("x_range") or [-10.0, 10.0], env.get("y_range") or [-10.0, 10.0]
    cx, cy = (xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2
    wx, wy = xr[1] - xr[0], yr[1] - yr[0]
    half_fov = math.atan(20.955 / 2 / 18.0)  # 18 mm lens on the default aperture
    width = max(wx, wy * 16 / 9) * 1.08
    top = {"name": "auto_topdown", "position": [cx, cy, width / 2 / math.tan(half_fov)], "look_at": [cx, cy, 0.0], "up": [0, 1, 0],
           "focal_mm": 18.0, "hide_roof": True}
    d = max(wx, wy) * 0.95 + 4.0
    over = {"name": "auto_overview", "position": [cx + 0.3 * d, cy - 0.55 * d, 0.95 * d], "look_at": [cx, cy, 0.0], "focal_mm": 18.0, "hide_roof": True}
    return {"auto_topdown": top, "auto_overview": over}


@tool
def render(path: str, camera: Any = "overview", width: int = 960, height: int = 540, engine: str = "rtx") -> dict:
    """Render a built scene from a scene-graph camera name, the string "topdown_map", or an ad-hoc camera
    ``{"position": [x,y,z], "look_at": [x,y,z], "focal_mm": 18}`` / ``{"mode": "top_down"}``.  engine "rtx" = Isaac Sim ray
    tracing (slow, realistic); "proxy" = fast flat-shaded preview that shows layout, not appearance."""
    d = _scene_dir(path)
    width, height = max(64, min(int(width), 4096)), max(64, min(int(height), 4096))
    if isinstance(camera, str) and camera.lstrip().startswith("{"):
        camera = json.loads(camera)
    if engine == "proxy":
        from .render import render_proxy, render_topdown_map

        png = render_topdown_map(d) if camera == "topdown_map" else render_proxy(d, camera, width=width, height=height)
        return _out(True, {"image": str(png), "engine": "proxy", "camera": camera if isinstance(camera, str) else camera.get("name", "custom")})
    if engine != "rtx":
        return _out(False, issues=[error("render.engine", f"Unknown engine {engine!r}.", hint='Use "rtx" or "proxy".')])
    graph = json.loads((d / "scene_graph.json").read_text())
    if camera == "topdown_map" or (isinstance(camera, dict) and camera.get("mode") == "top_down"):
        camera = auto_cameras(graph)["auto_topdown"]
    elif camera in ("auto_overview", "auto_topdown"):
        camera = auto_cameras(graph)[camera]
    from .isaac_env import IsaacSimUnavailable

    try:
        sim = _isaac()
    except IsaacSimUnavailable as e:
        return _out(False, issues=[Issue(**e.issue)])
    name = camera if isinstance(camera, str) else camera.get("name", "custom")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name)).strip("_") or "camera"
    png = d / "renders" / f"rtx_{safe}.png"
    cmd = [str(sim.python), "-m", "worldc.render.isaac_render", str(d), "--out", str(png), "--width", str(width), "--height", str(height)]
    if isinstance(camera, str):
        cmd += ["--camera", camera]
    elif "position" in camera and "look_at" in camera:
        cmd += ["--camera-json", json.dumps(camera)]
    else:
        return _out(False, issues=[error("render.camera", "Ad-hoc RTX cameras need position and look_at.", hint='Use {"position": [x,y,z], "look_at": [x,y,z]} or a named camera.')])
    try:
        p = subprocess.run(cmd, env=_isaac_env(), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return _out(False, issues=[error("render.timeout", "RTX render did not finish within 900 s.", hint='Try engine="proxy" or a smaller image.')])
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("RENDER_RESULT:")), None)
    res = json.loads(line[len("RENDER_RESULT:"):]) if line else {"ok": False, "error": (p.stderr or p.stdout)[-600:]}
    if not res.get("ok") and "file" not in res:
        return _out(False, issues=[error("render.failed", str(res.get("error", "render failed"))[:800], hint="Check camera names with build_scene's `cameras` list.")])
    issues = [] if res.get("ok") else [error("render.suspicious", "The render looks black, saturated or has missing textures.", hint=json.dumps({k: res[k] for k in res if k.startswith(("mean", "frac"))})).model_copy(update={"severity": "warning"})]
    return _out(True, {"image": res["file"], "engine": "rtx", "camera": res["camera"], "stats": {k: v for k, v in res.items() if k.startswith(("mean", "std", "frac"))}}, issues)


TOOLS: dict[str, Callable] = {f.__name__: f for f in (
    list_assets, search_assets, import_asset, validate_generated_asset, describe_asset, get_spec_schema, validate_spec, build_scene,
    check_scene, render, edit_scene)}
