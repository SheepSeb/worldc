"""E3 tier variants of a built scene (spec transforms, rebuilt and statically checked; no Isaac physics).

    from worldc.e3.tiers import make_variants; make_variants("dataset/.../scene", "out/e3/scenes/P18")

* ``base`` (easy, medium): the scene's resolved spec with its LLM-authored agents removed (E3's seeded walkers replace them).
* ``hard``: ``base`` plus OCCLUDERS_RANGE tall storage units (``OCCLUDER_ASSET``) standing beside human walkways, axis-aligned
  with the long side the one closer to the walkway's direction, leaving at least MIN_AISLE_WIDTH_M of aisle.  Candidates are
  taken in an order drawn with the transform seed and added greedily: a unit is kept when the set still passes a 2D pre-check
  (every task point reachable from the door through MIN_AISLE_WIDTH_M aisles, every robot goal reachable from the spawn) and
  the solver places every unit (door swing arcs, spec keep-out zones, the building interior).  The rebuilt scene must then
  pass ``check_scene`` (all mandatory static checks).

Both variants are rebuilt with the same compiler, so they differ only by the occluders.  Each variant directory holds an
``e3_variant.json`` (base scene, transform, check result, and the base scene's robot frame for ``world.robot_frame``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..check.safety import MIN_AISLE_WIDTH_M, poly_distance, ROBOT_CELL_CLEARANCE_M
from .world import EIGHT, World, load_world

OCCLUDER_ASSET = "parts_shelf"  # SimReady heavy-duty shelving, 0.92 x 0.61 x 2.44 m (tall: blocks the line of sight)
OCCLUDER_SIZE = (0.92, 0.61)  # local x (long side) x local y
OCCLUDERS_RANGE = (2, 4)
GAP_M = 0.05  # between the aisle edge and the unit
KEEP_FROM_POI_M = 1.0  # a unit's centre stays this far from task points ...
KEEP_FROM_GOAL_M = 1.2  # ... robot goals and the spawn
MIN_SPACING_M = 2.5  # between units
ENDS_M = 1.0  # no unit within this of a walking path's ends
ATTEMPTS = 6


def _rect(cx, cy, yaw, sx, sy) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)])
    return loc @ np.array([[c, s], [-s, c]]) + (cx, cy)


def candidates(world: World) -> list[dict]:
    """Unit poses beside the walking paths, clear of everything the scene needs."""
    occ = world.hgrid.occ
    g = world.hgrid
    off = MIN_AISLE_WIDTH_M / 2 + OCCLUDER_SIZE[1] / 2 + GAP_M
    keep = [p for v in world.pois.values() for p in v]
    goals = [gl["pose"][:2] for gl in world.goals] + [world.spawn[:2]]
    out, seen = [], set()
    for path in world.walkway_paths():
        seg = np.diff(path, axis=0)
        L = np.hypot(*seg.T)
        cum = np.concatenate([[0.0], np.cumsum(L)])
        for s in np.arange(ENDS_M, cum[-1] - ENDS_M, 0.5):
            i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(L) - 1)
            t = seg[i] / max(L[i], 1e-9)
            p = path[i] + t * (s - cum[i])
            n = np.array([-t[1], t[0]])
            for side in (-1, 1):
                cx, cy = p + side * off * n
                key = (round(cx * 4), round(cy * 4))
                if key in seen:
                    continue
                seen.add(key)
                if min((math.hypot(cx - x, cy - y) for x, y in keep), default=9) < KEEP_FROM_POI_M:
                    continue
                if min(math.hypot(cx - x, cy - y) for x, y in goals) < KEEP_FROM_GOAL_M + world.robot_r:
                    continue
                yaw = round(math.atan2(t[1], t[0]) / (math.pi / 2)) * (math.pi / 2)  # axis-aligned, long side along the walkway
                R = _rect(cx, cy, yaw, OCCLUDER_SIZE[0] + 2 * GAP_M, OCCLUDER_SIZE[1] + 2 * GAP_M)
                lo, hi = R.min(0), R.max(0)
                if lo[0] < g.x0 or lo[1] < g.y0 or hi[0] > g.x1 or hi[1] > g.y1:
                    continue
                xs = np.linspace(lo[0], hi[0], 8)
                ys = np.linspace(lo[1], hi[1], 8)
                pts = [(x, y) for x in xs for y in ys if _inside(R, x, y)]
                if any(occ[g.clamp(g.cell(x, y))] for x, y in pts):
                    continue
                if any(poly_distance(R, P) < ROBOT_CELL_CLEARANCE_M + GAP_M for P in world.cells):
                    continue
                out.append({"xy": [round(float(cx), 3), round(float(cy), 3)], "yaw_deg": round(math.degrees(yaw), 2),
                            "rect": _rect(cx, cy, yaw, *OCCLUDER_SIZE)})
    return out


def _inside(R, x, y) -> bool:
    e = np.roll(R, -1, 0) - R
    cr = e[:, 0] * (y - R[:, 1]) - e[:, 1] * (x - R[:, 0])
    return bool((cr >= 0).all() or (cr <= 0).all())


def precheck(world: World, units: list[dict]) -> tuple[bool, str]:
    """2D stand-in for the static checks the units could break: aisle-width reachability of every task point from the door,
    and robot reachability of every goal."""
    w = world.with_obstacles({f"occ{i}": u["rect"] for i, u in enumerate(units)})
    g = w.hgrid
    clear = ndimage.distance_transform_edt(~g.occ) * g.res
    lab, _ = ndimage.label(clear >= MIN_AISLE_WIDTH_M / 2, structure=EIGHT)
    door = (world.pois.get("door") or [None])[0]
    if door is not None:
        near = lambda x, y: lab[max(0, g.cell(x, y)[0] - 6):g.cell(x, y)[0] + 7, max(0, g.cell(x, y)[1] - 6):g.cell(x, y)[1] + 7]  # noqa: E731
        src = set(np.unique(near(*door))) - {0}
        for kind, pts in world.pois.items():
            for p in pts:
                if not (set(np.unique(near(*p))) & src):
                    return False, f"{kind} {p} loses its {MIN_AISLE_WIDTH_M} m aisle"
    rl, _ = ndimage.label(w.grid.free, structure=EIGHT)
    s = rl[w.grid.clamp(w.grid.cell(*world.spawn[:2]))]
    for gl in world.goals:
        if rl[w.grid.clamp(w.grid.cell(*gl["pose"][:2]))] != s:
            return False, f"robot goal {gl['id']} unreachable"
    return True, ""


def _solves(spec: dict, seed: int) -> tuple[bool, str]:
    """Does the solver place everything (no build)?"""
    from ..solver.solve import solve
    from ..spec.validate import check

    parsed, issues = check(spec, None)
    errs = [i for i in issues if i.severity == "error"]
    if parsed is not None and not errs:
        errs = [i for i in solve(parsed, seed, None).issues if i.severity == "error"]
    return not errs, (f"{errs[0].code}: {errs[0].message}" if errs else "")


def _with_units(spec: dict, units: list[dict]) -> dict:
    return {**spec, "objects": spec["objects"] + [{
        "id": "e3_occluders", "asset": OCCLUDER_ASSET, "count": len(units), "tags": ["e3_occluder"],
        "placement": {"pose": [{"xy": u["xy"], "yaw": u["yaw_deg"]} for u in units]}}]}


def _build(spec: dict, out: Path, seed: int) -> dict:
    from ..compile import compile_scene

    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    res = compile_scene(out / "spec.json", seed, out / "scene")
    return {"ok": bool(res.ok), "seconds": round(res.seconds, 1),
            "errors": [f"{i.code}: {i.message}" for i in res.issues if getattr(i, "severity", "") == "error"][:10]}


def make_variants(scene_dir, out_dir, seed: int = 0, build_seed: int = 0, log=print) -> dict:
    scene_dir, out_dir = Path(scene_dir), Path(out_dir)
    spec = json.loads((scene_dir / "resolved_spec.json").read_text())
    world = load_world(scene_dir)
    n_agents = len(spec.get("agents", []))
    base_spec = {**spec, "agents": []}
    meta = {"base_scene": str(scene_dir), "robot_frame": {"travel_heading_rad": world.phi0, "source": world.robot["source"]},
            "agents_removed": n_agents}
    res = {}
    b = _build(base_spec, out_dir / "base", build_seed)
    (out_dir / "base" / "scene" / "e3_variant.json").write_text(json.dumps({**meta, "tier": "base", "check": b}, indent=1))
    res["base"] = b
    log(f"base: {'ok' if b['ok'] else 'FAILED ' + str(b['errors'])} ({b['seconds']} s)")

    rng = np.random.default_rng(seed)
    n = int(rng.integers(OCCLUDERS_RANGE[0], OCCLUDERS_RANGE[1] + 1))
    cands = candidates(world)
    log(f"hard: {n} occluders from {len(cands)} candidate poses")
    tried = []
    for attempt in range(ATTEMPTS):
        order = rng.permutation(len(cands))
        units: list[dict] = []
        for i in order:
            c = cands[i]
            if any(math.hypot(c["xy"][0] - u["xy"][0], c["xy"][1] - u["xy"][1]) < MIN_SPACING_M for u in units):
                continue
            if precheck(world, units + [c])[0] and _solves(_with_units(base_spec, units + [c]), build_seed)[0]:
                units.append(c)
            if len(units) == n:
                break
        if len(units) < n:
            tried.append({"attempt": attempt, "placed": len(units), "reason": "not enough candidates pass the 2D pre-check and the solver"})
            continue
        h = _build(_with_units(base_spec, units), out_dir / "hard", build_seed)
        tried.append({"attempt": attempt, "units": [{k: u[k] for k in ("xy", "yaw_deg")} for u in units], "check": h})
        log(f"hard attempt {attempt}: {[u['xy'] for u in units]} -> {'ok' if h['ok'] else 'FAILED ' + str(h['errors'])}")
        if h["ok"]:
            break
    else:
        res["hard"] = {"ok": False, "attempts": tried}
        return res
    res["hard"] = {"ok": True, "occluders": tried[-1]["units"], "attempts": len(tried), "check": tried[-1]["check"]}
    (out_dir / "hard" / "scene" / "e3_variant.json").write_text(json.dumps(
        {**meta, "tier": "hard", "transform_seed": seed, "occluder_asset": OCCLUDER_ASSET, "occluders": res["hard"]["occluders"],
         "attempts": tried, "check": tried[-1]["check"]}, indent=1))
    return res
