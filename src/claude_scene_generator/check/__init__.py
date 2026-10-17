"""Generic post-build scene checker: ``check_scene(path)``.

Everything (bounds, robot radius, goals, hazards, clearance checks) comes from ``scene_graph.json``; see
docs/PLAN.md section 6.  All problems are reported both in the legacy-style result lists and as ``Issue`` records.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..colliders import collect
from ..spec.issues import Issue, error, warning
from . import clearance as CL
from . import collide as CO
from . import nav as NV
from . import scene as SC

__all__ = ["check_scene"]

MAX_PAIR_ISSUES = 100


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _who(sd: SC.SceneData, path):
    sid = sd.spec_id(path)
    return f"{path} (spec id '{sid}')" if sid else str(path)


def _empty(issues, seconds):
    return {"passed": False, "issues": [i.model_dump() for i in issues], "n_colliders": 0, "spawn_interpenetrations": [],
            "floating_bodies": [], "hazard_path_conflicts": [],
            "navigation": {"resolution_m": None, "robot_radius_m": None, "spawn_free": False, "goals": [],
                           "all_reachable": False, "n_goals": 0},
            "clearance": [], "seconds": round(seconds, 2)}


def check_scene(path, *, write: bool = True, nav_png: bool = True) -> dict:
    t0 = time.time()
    issues: list[Issue] = []
    try:
        d, usd, graph = SC.locate(path)
        from pxr import Usd

        stage = Usd.Stage.Open(str(usd))
        if stage is None:
            raise SC.LoadError(f"cannot open stage {usd}")
        cols = collect(stage)
    except Exception as e:  # noqa: BLE001
        issues.append(error("check.load_failed", f"{type(e).__name__}: {e}", hint="Check that the scene directory holds "
                            "scene_graph.json and the USD file named by its 'usd' key, and that both open cleanly."))
        res = _empty(issues, time.time() - t0)
        return res
    sd = SC.SceneData(d, usd, graph, stage, cols, SC.build_spec_id_map(graph))
    res = {"passed": True, "issues": [], "n_colliders": len(cols)}

    lo, hi = CO.aabbs(cols) if cols else (np.zeros((0, 3)), np.zeros((0, 3)))
    adj = CO.joint_adjacency(stage)

    # ---- interpenetration
    pens = CO.spawn_penetrations(cols, adj, lo, hi)
    res["spawn_interpenetrations"] = []
    for a, b, pen in pens:
        A, B = cols[a], cols[b]
        res["spawn_interpenetrations"].append({"a": A.path, "b": B.path, "depth_mm": round(float(pen) * 1000, 1),
                                               "a_spec_id": sd.spec_id(A.path), "b_spec_id": sd.spec_id(B.path)})
    for r in res["spawn_interpenetrations"][:MAX_PAIR_ISSUES]:
        issues.append(error(
            "check.interpenetration", f"{_who(sd, r['a'])} and {_who(sd, r['b'])} overlap by {r['depth_mm']} mm at spawn.",
            hint=f"Move one of them apart by at least {r['depth_mm'] / 1000 + 0.005:.3f} m, or shrink one (check the spec "
                 f"placements of {r['a_spec_id'] or r['a']} and {r['b_spec_id'] or r['b']}).",
            a=r["a"], b=r["b"], a_spec_id=r["a_spec_id"], b_spec_id=r["b_spec_id"], depth_mm=r["depth_mm"]))
    if len(pens) > MAX_PAIR_ISSUES:
        issues.append(error("check.interpenetration", f"{len(pens) - MAX_PAIR_ISSUES} further interpenetrating pairs not listed.",
                            hint="Fix the listed pairs first and re-run check_scene; see 'spawn_interpenetrations' for the full list."))

    # ---- floating
    res["floating_bodies"] = []
    for i, body, zb in CO.floating_bodies(cols, adj, lo, hi):
        sid = sd.spec_id(body)
        res["floating_bodies"].append({"body": body, "bottom_z": round(zb, 3), "spec_id": sid})
        issues.append(error("check.floating", f"{_who(sd, body)} hovers with its lowest point at z={zb:.3f} m and nothing under it.",
                            hint=f"Lower it by {zb:.3f} m onto the floor, or place it on a support (pallet, shelf, table) whose top is at "
                                 f"z={zb:.3f} m.", body=body, spec_id=sid, bottom_z=round(zb, 3)))

    # ---- hazards
    res["hazard_path_conflicts"] = []
    for hz_path, hit, pen in CO.hazard_conflicts(cols, graph.get("hazards") or [], lo, hi):
        obst = cols[hit].path
        r = {"hazard": hz_path, "obstacle": obst, "depth_mm": round(float(pen) * 1000, 1)}
        res["hazard_path_conflicts"].append(r)
        issues.append(error("check.hazard_path_blocked",
                            f"Path of hazard {_who(sd, hz_path)} runs into {_who(sd, obst)} (penetration {r['depth_mm']} mm).",
                            hint=f"Move {sd.spec_id(obst) or obst} out of the corridor, or re-route the waypoints of "
                                 f"{sd.spec_id(hz_path) or hz_path} around it.", hazard=hz_path, obstacle=obst,
                            hazard_spec_id=sd.spec_id(hz_path), obstacle_spec_id=sd.spec_id(obst), depth_mm=r["depth_mm"]))

    # ---- navigation + clearance
    cfg = SC.nav_config(graph)
    zb = cfg["z_band"]
    xr, yr = SC.environment_bounds(graph, cols)
    rr = SC.robot_radius(graph)
    idx = NV.nav_colliders(cols, zb, set(cfg["ignore_categories"]))
    polys = NV.polygons(cols, idx)
    grid = NV.Grid(xr, yr, float(cfg["resolution_m"]))
    NV.build_grid(grid, polys, rr)
    blockers = NV.Blockers(cols, polys)
    goals = graph.get("goals") or []
    navres = {"resolution_m": grid.res, "robot_radius_m": rr, "spawn_free": False, "goals": [], "all_reachable": False,
              "n_goals": len(goals), "grid_shape": [grid.ny, grid.nx]}
    robot = graph.get("robot")
    paths = {}
    start = None
    if not robot or not robot.get("spawn"):
        issues.append(warning("check.no_robot", "scene_graph.json has no robot spawn; navigation was not checked.",
                              hint="Add robot.spawn=[x, y, yaw] (and robot.radius_m) to the scene."))
        navres.update(spawn_free=True, all_reachable=True)
    else:
        sx, sy = robot["spawn"][:2]
        start = grid.clamp(grid.cell(sx, sy))
        navres["spawn_free"] = grid.is_free(sx, sy)
        if not navres["spawn_free"]:
            near = blockers.nearest(sx, sy, 3)
            names = ", ".join(f"{_who(sd, cols[i].path)} ({dd:.2f} m)" for i, dd in near) or "the environment boundary"
            issues.append(error("check.spawn_blocked", f"Robot spawn ({sx:.2f}, {sy:.2f}) is not free for radius {rr:.2f} m "
                                f"(+{NV.INFLATE_MARGIN} m margin).",
                                hint=f"Move the spawn or the nearest obstacles: {names}. Needs {rr + NV.INFLATE_MARGIN:.2f} m free radius.",
                                spawn=[sx, sy], blockers=[cols[i].path for i, _ in near]))
        gcell = [grid.clamp(grid.cell(*g["base_pose"][:2])) for g in goals]
        gfree = [grid.is_free(*g["base_pose"][:2]) for g in goals]
        if navres["spawn_free"]:
            dist, prev = NV.dijkstra(grid.free, start, [c for c, f in zip(gcell, gfree) if f])
        for g, cell, gf in zip(goals, gcell, gfree):
            ok = bool(navres["spawn_free"] and gf and dist[cell] < 1e17)
            gx, gy = g["base_pose"][:2]
            nm = g.get("name", g.get("id"))
            navres["goals"].append({"id": g.get("id"), "name": g.get("name"), "type": g.get("type"), "reachable": ok,
                                    "goal_cell_free": bool(gf),
                                    "path_length_m": round(float(dist[cell] * grid.res), 2) if ok else None})
            if ok:
                paths[nm] = NV.trace_path(prev, start, cell)
            elif navres["spawn_free"]:
                issues.append(_goal_issue(sd, g, grid, blockers, cols, (sx, sy), gf, rr))
        navres["all_reachable"] = bool(navres["spawn_free"] and all(g["reachable"] for g in navres["goals"]))
    if not goals:
        issues.append(warning("check.no_goals", "The scene defines no goals, so reachability was not tested.",
                              hint="Add at least one goal (manipulation target with a base_pose) to the spec."))
    res["navigation"] = navres

    # ---- clearance
    res["clearance"] = []
    cpolys = [polys[i] for i in polys]
    cboxes = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in cpolys]
    for ck in SC.clearance_checks(graph):
        meas, at = CL.measure(cpolys, cboxes, ck)
        tol = float(ck.get("tolerance", 0.03))
        tgt = float(ck["target"])
        ok = abs(meas - tgt) <= tol
        res["clearance"].append({"id": ck["id"], "target_m": tgt, "measured_m": round(meas, 3), "tolerance_m": tol, "ok": bool(ok),
                                 "narrowest_at": round(at, 2)})
        if not ok:
            diff = meas - tgt
            ax = ck.get("axis", "x")
            other = "y" if ax == "x" else "x"
            issues.append(error("check.clearance",
                                f"Clearance '{ck['id']}' measures {meas:.3f} m against target {tgt:.3f} m +/-{tol:.3f} "
                                f"({diff:+.3f} m), narrowest at {other}={at:.2f}.",
                                hint=f"{'Widen' if diff < 0 else 'Narrow'} the gap along {ax} by {abs(diff):.3f} m near {other}={at:.2f} "
                                     f"(move the objects bounding the gap inside rect {ck['rect']}).",
                                id=ck["id"], measured_m=round(meas, 3), target_m=tgt, difference_m=round(diff, 3), narrowest_at=round(at, 2)))

    res["issues"] = [i.model_dump() for i in issues]
    res["passed"] = not any(i.severity == "error" for i in issues)
    res["seconds"] = round(time.time() - t0, 2)
    if write:
        if nav_png and start is not None:
            NV.render_png(d / "nav_occupancy.png", grid, start, [g["base_pose"][:2] for g in goals], paths, navres["spawn_free"])
        (d / "check.json").write_text(json.dumps(res, indent=1, default=_jsonable))
    return res


def _goal_issue(sd, g, grid, blockers, cols, spawn, goal_free, rr) -> Issue:
    gx, gy = g["base_pose"][:2]
    label = f"'{g.get('name') or g.get('id')}' ({g.get('type')}) at ({gx:.2f}, {gy:.2f})"
    need = rr + NV.INFLATE_MARGIN
    if not goal_free:
        near = [(i, d) for i, d in blockers.nearest(gx, gy, 3) if d < need]
        if not grid.inside(gx, gy):
            why = "the base pose lies outside the environment ranges"
        elif near:
            why = "blocked by " + ", ".join(f"{_who(sd, cols[i].path)} ({d:.2f} m away)" for i, d in near)
        else:
            why = "too close to the environment boundary"
        return error("check.goal_unreachable", f"Goal {label} is unreachable: its base pose is not free for the robot.",
                     hint=f"The goal cell itself is blocked: {why}; the robot needs {need:.2f} m clear radius. Move the "
                          f"blocking object(s) or the goal base pose.", goal=g.get("id"), goal_cell_free=False,
                     blockers=[cols[i].path for i, _ in near])
    # disconnected: walk the straight line spawn -> goal, collect blockers where it is not free
    sx, sy = spawn
    L = math.hypot(gx - sx, gy - sy)
    n = max(2, int(L / 0.1))
    tally: dict[int, int] = {}
    order: list[int] = []
    for k in range(n + 1):
        x, y = sx + (gx - sx) * k / n, sy + (gy - sy) * k / n
        if not grid.is_free(x, y):
            near = blockers.nearest(x, y, 1)
            if near:
                i = near[0][0]
                tally[i] = tally.get(i, 0) + 1
                if i not in order:
                    order.append(i)
    top = sorted(order, key=lambda i: -tally[i])[:3]
    if not top:  # straight line is clear: the enclosure is elsewhere, report what hugs the goal
        top = [i for i, _ in blockers.nearest(gx, gy, 3)]
    names = ", ".join(_who(sd, cols[i].path) for i in top) or "unknown obstacles"
    return error("check.goal_unreachable", f"Goal {label} is unreachable: no free path exists from the spawn.",
                 hint=f"The goal cell is free but disconnected from the spawn; nearest blockers along the straight line "
                      f"spawn->goal: {names}. Open a corridor at least {2 * need:.2f} m wide through them (move or remove one).",
                 goal=g.get("id"), goal_cell_free=True, blockers=[cols[i].path for i in top])
