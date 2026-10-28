"""Mandatory safety checks, run by every ``check_scene`` against fixed thresholds (not spec constraints), so an accepted
scene means the same thing whichever model wrote the spec:

* human reachability: from the main door, every workbench front, storage zone and other exit is reachable on the
  occupancy grid inflated by ``HUMAN_RADIUS_M`` (robot-cell safety footprints count as obstacles);
* minimum aisle width: the widest route to each of those targets is at least ``MIN_AISLE_WIDTH_M`` wide at its narrowest;
* robot-cell clearance: nothing outside a robot cell comes within ``ROBOT_CELL_CLEARANCE_M`` of its safety footprint.

Spec ``min_clearance`` constraints are checked here too (they are declared, so they are not mandatory).  Inputs come from
``scene_graph.json["safety"]`` (written by the compiler) plus the stage colliders; every failure is an error-level issue
with a hint, so it feeds the repair loop.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..geom import footprint
from ..spec.issues import error, warning
from ..spec.refs import match, parse_ref
from . import nav as NV

HUMAN_RADIUS_M = 0.26
MIN_AISLE_WIDTH_M = 1.20  # E1 protocol (the constrained prompts ask for 1.2 m walking aisles); grid precision ~ one 5 cm cell
ROBOT_CELL_CLEARANCE_M = 0.50
REACH_TOL_M = 0.30  # a target counts as reached when a free cell lies within this distance of it
MIN_CLEARANCE_TOL_M = 0.01
THRESHOLDS = {"human_radius_m": HUMAN_RADIUS_M, "min_aisle_width_m": MIN_AISLE_WIDTH_M, "robot_cell_clearance_m": ROBOT_CELL_CLEARANCE_M}
EIGHT = np.ones((3, 3), bool)


# ------------------------------------------------------------------------------------------------------- geometry
def _seg_dist(pts: np.ndarray, poly: np.ndarray) -> float:
    """Smallest distance from any of ``pts`` to the edges of ``poly``."""
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    L2 = np.maximum((e * e).sum(1), 1e-12)
    t = np.clip(((pts[:, None, :] - a[None]) * e[None]).sum(2) / L2[None], 0.0, 1.0)
    q = a[None] + t[..., None] * e[None]
    return float(np.sqrt(((pts[:, None, :] - q) ** 2).sum(2)).min())


def _overlap(P: np.ndarray, Q: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons."""
    for poly in (P, Q):
        e = np.roll(poly, -1, axis=0) - poly
        for n in np.stack([-e[:, 1], e[:, 0]], 1):
            p, q = P @ n, Q @ n
            if p.max() < q.min() or q.max() < p.min():
                return False
    return True


def poly_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Distance between two convex polygons (0 if they overlap)."""
    if _overlap(P, Q):
        return 0.0
    return min(_seg_dist(P, Q), _seg_dist(Q, P))


def _bbox_gap(P: np.ndarray, Q: np.ndarray) -> float:
    g = np.maximum(0.0, np.maximum(P.min(0) - Q.max(0), Q.min(0) - P.max(0)))
    return float(np.hypot(*g))


def _owned(path: str, roots: list[str]) -> bool:
    return any(path == r or path.startswith(r + "/") for r in roots)


def _disk(grid: NV.Grid, x: float, y: float, r: float) -> np.ndarray:
    rows = (grid.y1 - (np.arange(grid.ny) + 0.5) * grid.res)[:, None]
    cols = (grid.x0 + (np.arange(grid.nx) + 0.5) * grid.res)[None, :]
    return (cols - x) ** 2 + (rows - y) ** 2 <= r * r


def _rect(grid: NV.Grid, rect, pad: float) -> np.ndarray:
    x0, y0, x1, y1 = rect
    rows = (grid.y1 - (np.arange(grid.ny) + 0.5) * grid.res)[:, None]
    cols = (grid.x0 + (np.arange(grid.nx) + 0.5) * grid.res)[None, :]
    return (cols >= x0 - pad) & (cols <= x1 + pad) & (rows >= y0 - pad) & (rows <= y1 + pad)


def _xy(grid: NV.Grid, rc) -> tuple[float, float]:
    return grid.x0 + (rc[1] + 0.5) * grid.res, grid.y1 - (rc[0] + 0.5) * grid.res


# ------------------------------------------------------------------------------------------------------- checks
class SafetyCheck:
    def __init__(self, sd, cols, polys: dict, graph: dict, xr, yr, res: float, who):
        self.sd, self.cols, self.polys, self.graph, self.who = sd, cols, polys, graph, who
        self.xr, self.yr, self.res = xr, yr, res
        self.inputs = graph.get("safety") or {}
        self.issues: list = []
        self.out: dict = {"thresholds": THRESHOLDS}

    def run(self) -> tuple[dict, list]:
        if "safety" not in self.graph:  # not written by the compiler (hand-made or older scene graph): exits etc. are unknown, not absent
            self.issues.append(warning("check.safety_not_run", "scene_graph.json has no 'safety' block, so the mandatory safety checks "
                                       "(human reachability, aisle width, robot-cell clearance) did not run.",
                                       hint="Rebuild the scene from its spec with the current compiler."))
            self.out["ran"] = False
            return self.out, self.issues
        self.out["ran"] = True
        self.robot_cells()
        self.human()
        self.min_clearance()
        return self.out, self.issues

    def _instance(self, i: int) -> str:
        roots = [v["root"] for v in (self.graph.get("instances") or {}).values()]
        return max((r for r in roots if _owned(self.cols[i].path, [r])), key=len, default=self.cols[i].path)

    def _name(self, i: int) -> str:
        return self.who(self.sd, self.cols[i].path)

    # ---- robot cells ----------------------------------------------------------------------------------------
    def robot_cells(self):
        res = []
        for cell in self.inputs.get("robot_cells", []):
            P = np.asarray(cell["polygon"], float)
            hits = []
            for i, Q in self.polys.items():
                if _owned(self.cols[i].path, cell["members"]) or _bbox_gap(P, Q) >= ROBOT_CELL_CLEARANCE_M:
                    continue
                d = poly_distance(P, Q)
                if d < ROBOT_CELL_CLEARANCE_M - 1e-6:
                    hits.append((d, i))
            hits.sort()
            seen, worst = set(), []
            for d, i in hits:  # one entry per offending object, nearest collider first
                owner = self.sd.spec_id(self.cols[i].path) or self.cols[i].path
                if owner not in seen:
                    seen.add(owner)
                    worst.append((d, i, owner))
            label = cell.get("spec_id") or cell["id"]
            res.append({"id": cell["id"], "spec_id": cell.get("spec_id"), "ok": not worst,
                        "min_distance_m": round(worst[0][0], 3) if worst else None, "offenders": [o for _, _, o in worst[:10]]})
            for d, i, owner in worst[:5]:
                inside = d <= 1e-6
                need = ROBOT_CELL_CLEARANCE_M - d
                self.issues.append(error(
                    "check.robot_cell_clearance",
                    f"{self._name(i)} {'intrudes into' if inside else f'is {d:.2f} m from'} the safety footprint of robot cell '{label}' "
                    f"(required clearance {ROBOT_CELL_CLEARANCE_M:.2f} m).",
                    hint=f"Move {owner} at least {need + 0.02:.2f} m away from the cell, or move the cell '{label}'. The safeguarded "
                         f"area (fence or keep-out) needs {ROBOT_CELL_CLEARANCE_M:.2f} m free all around for access and escape.",
                    cell=cell["id"], cell_spec_id=cell.get("spec_id"), obstacle=self.cols[i].path, obstacle_spec_id=self.sd.spec_id(self.cols[i].path),
                    distance_m=round(d, 3), required_m=ROBOT_CELL_CLEARANCE_M))
        self.out["robot_cells"] = res

    # ---- human reachability + aisle width -------------------------------------------------------------------
    def _targets(self, grid) -> list[dict]:
        t = []
        for e in self.inputs.get("exits", [])[1:]:
            t.append({"kind": "exit", "id": e["id"], "label": f"exit '{e['id']}' ({e['wall']} wall)", "at": e["xy"],
                      "mask": _disk(grid, *e["xy"], REACH_TOL_M)})
        for b in self.inputs.get("workbenches", []):
            mask = np.zeros((grid.ny, grid.nx), bool)
            for x, y in b["stand_points"]:
                mask |= _disk(grid, x, y, REACH_TOL_M)
            side = "front" if b["has_front"] else "any side"
            t.append({"kind": "workbench", "id": b["id"], "spec_id": b["spec_id"], "label": f"workbench '{b['id']}' ({side})",
                      "at": b["stand_points"][0], "mask": mask})
        for z in self.inputs.get("storage_zones", []):
            x0, y0, x1, y1 = z["rect"]
            t.append({"kind": "storage_zone", "id": z["id"], "label": f"storage zone '{z['id']}'", "at": [(x0 + x1) / 2, (y0 + y1) / 2],
                      "mask": _rect(grid, z["rect"], REACH_TOL_M)})
        return t

    def human(self):
        env = self.graph.get("environment") or {}
        exits = self.inputs.get("exits", [])
        polys = dict(self.polys)
        for j, cell in enumerate(self.inputs.get("robot_cells", [])):  # people never route through a safeguarded area
            polys[-1 - j] = np.asarray(cell["polygon"], float)
        grid = NV.Grid(self.xr, self.yr, self.res)
        NV.build_grid(grid, polys, HUMAN_RADIUS_M)
        clear = ndimage.distance_transform_edt(~grid.occ) * grid.res
        free = grid.free
        out = {"radius_m": HUMAN_RADIUS_M, "min_aisle_width_m": MIN_AISLE_WIDTH_M, "source": None, "targets": []}
        self.out["human"] = out
        if env.get("kind", "indoor") == "indoor":
            if not exits:
                self.issues.append(error("check.safety_no_exit", "The building has no door, so people cannot enter or leave and human "
                                         "reachability cannot be tested.",
                                         hint="Add an opening with sill 0 on a wall (environment.walls.openings, e.g. {id: entrance, wall: "
                                              "south, center: 0, width: 1.2, height: 2.2, door: swing_door})."))
                return
            sx, sy = exits[0]["xy"]
            src = _disk(grid, sx, sy, REACH_TOL_M)
            out["source"] = {"exit": exits[0]["id"], "xy": [sx, sy]}
            src_label = f"door '{exits[0]['id']}'"
        else:  # outdoors people arrive from anywhere along the footprint boundary
            band = int(math.ceil((HUMAN_RADIUS_M + NV.INFLATE_MARGIN + REACH_TOL_M) / grid.res)) + 1  # the grid's own border is occupied
            src = np.zeros_like(free)
            src[:band, :] = src[-band:, :] = src[:, :band] = src[:, -band:] = True
            out["source"] = {"boundary": True}
            sx, sy = (self.xr[0] + self.xr[1]) / 2, (self.yr[0] + self.yr[1]) / 2
            src_label = "the site boundary"
        targets = self._targets(grid)

        def reach(free_mask):
            lab, _ = ndimage.label(free_mask, structure=EIGHT)
            ids = np.unique(lab[src & free_mask])
            return np.isin(lab, ids[ids > 0])

        if not (src & free).any():
            if out["source"].get("boundary"):
                self.issues.append(error("check.safety_exit_blocked", "Nothing along the site boundary is free for a person, so nobody can "
                                         f"walk in (radius {HUMAN_RADIUS_M:.2f} m).", hint="Leave a gap of at least "
                                         f"{MIN_AISLE_WIDTH_M:.2f} m between objects along the footprint edge."))
                return
            near = NV.Blockers(self.cols, self.polys).nearest(sx, sy, 3)
            self.issues.append(error("check.safety_exit_blocked", f"The area just inside {src_label} at ({sx:.2f}, {sy:.2f}) is not free for a "
                                     f"person (radius {HUMAN_RADIUS_M:.2f} m).",
                                     hint="Move what blocks the door: " + (", ".join(self._name(i) for i, _ in near) or "the walls around it") + "."))
            return
        if not targets:
            return
        reach_h = reach(free)
        w_req = MIN_AISLE_WIDTH_M / 2
        reach_w = reach(clear >= w_req)
        for t in targets:
            ok_h = bool((reach_h & t["mask"]).any())
            ok_w = bool((reach_w & t["mask"]).any())
            row = {"kind": t["kind"], "id": t["id"], "reachable": ok_h, "aisle_ok": ok_w, "narrowest_m": None}
            out["targets"].append(row)
            if not ok_h:
                self.issues.append(self._unreachable(t, grid, src_label, (sx, sy)))
            elif not ok_w:
                lo, hi = HUMAN_RADIUS_M + NV.INFLATE_MARGIN, w_req
                for _ in range(8):  # widest route: largest half-width at which the target is still connected to the source
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if (reach(clear >= mid) & t["mask"]).any() else (lo, mid)
                width = 2 * lo
                row["narrowest_m"] = round(width, 2)
                self.issues.append(self._narrow(t, grid, clear, src, lo, hi, width, src_label))

    def _unreachable(self, t, grid, src_label, src_xy):
        gx, gy = t["at"]
        blockers = NV.Blockers(self.cols, self.polys)
        tally: dict[int, int] = {}
        n = max(2, int(math.hypot(gx - src_xy[0], gy - src_xy[1]) / 0.1))
        for k in range(n + 1):  # what blocks the straight line source -> target
            x, y = src_xy[0] + (gx - src_xy[0]) * k / n, src_xy[1] + (gy - src_xy[1]) * k / n
            if not grid.is_free(x, y):
                for i, _ in blockers.nearest(x, y, 1):
                    tally[i] = tally.get(i, 0) + 1
        top = sorted(tally, key=lambda i: -tally[i])[:3] or [i for i, _ in blockers.nearest(gx, gy, 3)]
        names = ", ".join(self._name(i) for i in top) or "unknown obstacles"
        return error("check.human_unreachable", f"A person (radius {HUMAN_RADIUS_M:.2f} m) cannot reach {t['label']} at ({gx:.2f}, {gy:.2f}) "
                     f"from {src_label}.",
                     hint=f"Open a walkway at least {MIN_AISLE_WIDTH_M:.2f} m wide from {src_label} to it; blockers along the way: {names}.",
                     target=t["id"], target_kind=t["kind"], spec_id=t.get("spec_id"), blockers=[self.cols[i].path for i in top])

    def _narrow(self, t, grid, clear, src, half, fail_half, width, src_label):
        free = clear >= half - 1e-9
        starts = [tuple(rc) for rc in np.argwhere(src & free)[:1]]
        goals = [tuple(rc) for rc in np.argwhere(t["mask"] & free)]
        where, names, paths = t["at"], "", []
        if starts and goals:
            dist, prev = NV.dijkstra(free, starts[0], goals)
            g = min(goals, key=lambda rc: dist[rc])
            if dist[g] < 1e17:
                path = NV.trace_path(prev, starts[0], g)[::-1]  # source -> target
                # the bottleneck is where the widest route first gets narrower than the half-width that failed
                rc = next((c for c in path if clear[c] < fail_half), min(path, key=lambda c: clear[c]))
                where = _xy(grid, rc)
                near = NV.Blockers(self.cols, self.polys).nearest(*where, 2)
                names = ", ".join(self._name(i) for i, _ in near)
                paths = [self.cols[i].path for i, _ in near]
        return error("check.aisle_too_narrow", f"The widest walkway from {src_label} to {t['label']} is {width:.2f} m at its narrowest, near "
                     f"({where[0]:.2f}, {where[1]:.2f}); the minimum aisle width is {MIN_AISLE_WIDTH_M:.2f} m.",
                     hint=f"Widen it by {MIN_AISLE_WIDTH_M - width + 0.02:.2f} m near ({where[0]:.2f}, {where[1]:.2f})"
                          + (f" by moving {names}" if names else "") + ", or open another route.",
                     target=t["id"], target_kind=t["kind"], spec_id=t.get("spec_id"), narrowest_m=round(width, 3),
                     at=[round(float(where[0]), 2), round(float(where[1]), 2)], blockers=paths)

    # ---- declared min_clearance --------------------------------------------------------------------------------
    def _group(self, ref: str):
        """(label, polygons as [(collider index | None, polygon)], owner roots) for a constraint reference."""
        pr = parse_ref(ref)
        x0, x1 = self.xr
        y0, y1 = self.yr
        if pr.kind == "wall":
            t = 0.01
            line = {"south": [(x0, y0 - t), (x1, y0 - t), (x1, y0), (x0, y0)], "north": [(x0, y1), (x1, y1), (x1, y1 + t), (x0, y1 + t)],
                    "west": [(x0 - t, y0), (x0, y0), (x0, y1), (x0 - t, y1)], "east": [(x1, y0), (x1 + t, y0), (x1 + t, y1), (x1, y1)]}.get(pr.target)
            return ([(None, np.asarray(line, float))] if line else []), []
        if pr.kind == "robot":
            rb = self.graph.get("robot") or {}
            if not rb.get("spawn"):
                return [], []
            (x, y), r = rb["spawn"][:2], float(rb.get("radius_m", 0.3))
            return [(None, np.array([(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]))], ["/World/Robot"]
        if pr.kind != "object":
            return None, []
        roots = [v["root"] for v in (self.graph.get("instances") or {}).values()
                 if match(pr.target, v["spec_id"]) and (pr.instance in (None, "*") or v["index"] == pr.instance)]
        out = []
        for i, c in enumerate(self.cols):
            if c.category in ("floor", "roof") or not _owned(c.path, roots):
                continue
            P = self.polys.get(i)
            if P is None:
                try:
                    P = footprint(c.obb)
                except Exception:  # noqa: BLE001  flat hull
                    continue
            out.append((i, np.asarray(P, float)))
        return out, roots

    def min_clearance(self):
        res = []
        for k in self.inputs.get("min_clearance", []):
            a, b = k["between"]
            ga, ra = self._group(a)
            gb, rb = self._group(b)
            bad = [r for r, g in ((a, ga), (b, gb)) if not g]
            if bad:
                self.issues.append(error("check.min_clearance_unresolved", f"min_clearance '{k['id']}': {', '.join(repr(r) for r in bad)} matches "
                                         "no built geometry, so the clearance cannot be measured.", path=k["path"] + "/between",
                                         hint="Reference built object ids (optionally '<id>.<n>' or '<id>.*'), 'robot', or 'wall:<side>'."))
                res.append({"id": k["id"], "value_m": k["value"], "measured_m": None, "ok": False})
                continue
            best, pair = math.inf, None
            for ia, P in ga:
                for ib, Q in gb:
                    if ia is not None and ib is not None and self._instance(ia) == self._instance(ib):
                        continue  # the same instance on both sides (e.g. between: [shelf, shelf])
                    if _bbox_gap(P, Q) >= best:
                        continue
                    d = poly_distance(P, Q)
                    if d < best:
                        best, pair = d, (ia, ib)
            ok = best >= k["value"] - MIN_CLEARANCE_TOL_M
            res.append({"id": k["id"], "between": [a, b], "value_m": k["value"], "measured_m": None if math.isinf(best) else round(best, 3), "ok": ok})
            if not ok:
                na = self._name(pair[0]) if pair[0] is not None else a
                nb = self._name(pair[1]) if pair[1] is not None else b
                self.issues.append(error("check.min_clearance", f"min_clearance '{k['id']}': {na} and {nb} are {best:.2f} m apart, "
                                         f"required {k['value']:.2f} m.", path=k["path"],
                                         hint=f"Move {a} and {b} at least {k['value'] - best + 0.02:.2f} m further apart (or lower the "
                                              f"constraint value if it is wrong).",
                                         measured_m=round(best, 3), required_m=k["value"], a=a, b=b))
        self.out["min_clearance"] = res


def check_safety(sd, cols, polys, graph, xr, yr, res, who) -> tuple[dict, list]:
    return SafetyCheck(sd, cols, polys, graph, xr, yr, res, who).run()
