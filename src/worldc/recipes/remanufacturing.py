"""E2 recipe baseline: a remanufacturing / disassembly cell from one prompt row (``prompts/prompts.csv``), no LLM.

    spec = remanufacturing_cell(seed, width_m, depth_m, cell_type, difficulty, n_people, n_benches, n_robot_cells, conveyor,
                                n_racks, n_doors)
    worldc recipe remanufacturing --prompts prompts/prompts.csv --out out/e2/recipe      # one scene per prompt

Same frozen library and spec format as the LLM runs (curated SimReady assets, procedural conveyor, robot_cell, human
agents, one mobile robot).  The row fixes the footprint, people and the large-item counts (-1 = drawn from a range).  The
seed draws the structural knobs, all written to ``metadata.recipe``: door walls and positions, bench arrangement
(along_wall | single_row | facing_rows | along_conveyor), robot-cell positions (corner | wall_centred) and arms, conveyor
orientation and which end feeds the cell, shelving walls (door walls avoided), pallet staging spot next to a door, charging
station spot, lighting.  Cell type picks the props (same vocabulary as the prompts); difficulty sets the slack above the
mandatory 1.2 m aisle (AISLE) and adds occluding storage when hard.

Layout: large items get explicit poses from a rectangle planner that keeps AISLE between items (robot cells also 0.6 m
from walls) and keeps doorways clear; props are left to the solver (on bench tops / pallet decks, inside zones).  Nothing is
adjusted to pass the checks: ``build_prompt`` resamples the whole recipe with seed + 1000 k (k = 1..10) when a build fails
``check_scene`` and records the attempts.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from ..spec.models import SceneSpec

PAD = 0.12                                        # wall-mounted items stand this far from the wall line (solver floor pad 0.1)
AISLE = {"easy": 2.0, "medium": 1.6, "hard": 1.3}  # free floor kept between large items (mandatory aisle is 1.2 m)
CELL_WALL_GAP = 0.6                                # robot cell to wall (mandatory clearance 0.5 m)
BENCH = ("workbench_steel", 1.8, 0.7)              # asset, length (local x), depth (local y)
SHELF = ("parts_shelf", 0.92, 0.61)
BENCH_GAP = 0.6                                    # between benches in a row
CONVEYOR_W = 0.8
CONVEYOR_CELL_GAP = 0.7                            # conveyor end to the robot cell it feeds
ARRANGEMENTS = ("along_wall", "single_row", "facing_rows", "along_conveyor")
SPAWN_PLACEMENTS = ("left_of_lane", "right_of_lane", "near_charger")  # where the robot parks; near_charger only with a charger
LIGHTING = ("indoor_industrial", "bright_day", "mixed_warm")
WALLS = ("south", "north", "west", "east")
ARMS = ("ur10", "fanuc_crx10ia", "fanuc_crx20ia_l", "franka")
# props per cell type: (bench props [(asset, count range)], floor props [(asset, count range)], on-pallet unit, n pallets range)
PROPS = {
    "ev_battery_dismantling": ([("toolbox", (1, 2)), ("crate_gray", (1, 2))], [("crate_blue", (2, 4)), ("drum_steel", (1, 3)), ("ibc_tank", (1, 1)),
                               ("flammables_cabinet", (1, 1))], "crate_blue", (1, 2)),
    "electronics_weee": ([("laptop", (1, 3)), ("monitor", (1, 2)), ("pc_tower", (0, 2))], [("scrap_bin_wheeled", (1, 2)), ("scrap_bin", (1, 2))],
                         "pc_tower", (1, 2)),
    "motor_gearbox_reman": ([("electric_motor", (1, 2)), ("transfer_case", (0, 2)), ("planetary_gearbox", (1, 2)), ("toolbox", (0, 1))],
                            [("parts_washer", (1, 1)), ("drum_steel", (1, 2))], "electric_motor", (1, 2)),
    "appliance_repair": ([("microwave", (1, 2)), ("toolbox", (1, 2))], [("scrap_bin", (1, 1))], "washing_machine", (1, 3)),
    "sorting_grading": ([("laptop", (0, 1)), ("planetary_gearbox", (0, 2)), ("toolbox", (0, 1))], [("scrap_bin_wheeled", (2, 3)), ("scrap_bin", (1, 3))],
                        "crate_blue", (1, 2)),
    "returned_parts_inspection": ([("planetary_gearbox", (1, 2)), ("laptop", (1, 1)), ("differential", (0, 1))], [("parts_washer", (1, 1))],
                                  "transfer_case", (1, 2)),
}
PROPS["engine_gearbox_reman"] = PROPS["motor_gearbox_reman"]
OCCLUDERS = ("crate_plywood", "ibc_tank", "flammables_cabinet")  # extra storage for hard cells (not counted as shelving)


# ---------------------------------------------------------------------------------------------------- planner
class Plan:
    """Axis-aligned rectangles in the room (x in [-W/2, W/2], y in [-D/2, D/2])."""

    def __init__(self, W, D):
        self.W, self.D = W, D
        self.rects: list[tuple[tuple, str]] = []  # ((x0, y0, x1, y1), kind)

    def inside(self, r, margin=PAD):
        return r[0] >= -self.W / 2 + margin - 1e-9 and r[2] <= self.W / 2 - margin + 1e-9 and r[1] >= -self.D / 2 + margin - 1e-9 and r[3] <= self.D / 2 - margin + 1e-9

    def free(self, r, gap, ignore=()):
        for q, kind in self.rects:
            if kind in ignore:
                continue
            g = (0.0 if kind == "reserved" else gap) - 1e-6  # exactly `gap` apart is fine (rows are placed at exactly A)
            if r[0] < q[2] + g and r[2] > q[0] - g and r[1] < q[3] + g and r[3] > q[1] - g:
                return False
        return True

    def add(self, r, kind):
        self.rects.append((tuple(float(v) for v in r), kind))

    def segment_clear(self, a, b, inflate=0.35):
        a, b = np.asarray(a, float), np.asarray(b, float)
        for (x0, y0, x1, y1), kind in self.rects:
            if kind == "reserved":
                continue
            lo, hi = np.array([x0 - inflate, y0 - inflate]), np.array([x1 + inflate, y1 + inflate])
            d = b - a
            t0, t1 = 0.0, 1.0
            hit = True
            for k in range(2):  # slab test
                if abs(d[k]) < 1e-12:
                    if a[k] < lo[k] or a[k] > hi[k]:
                        hit = False
                        break
                else:
                    u, v = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
                    t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
                    if t0 > t1:
                        hit = False
                        break
            if hit:
                return False
        return True


def _rect(cx, cy, sx, sy):
    return (cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2)


def _wall_frame(wall, W, D):
    """(point on the wall line at t=0, along unit, inward unit, wall length)."""
    return {"south": ((0, -D / 2), (1, 0), (0, 1), W), "north": ((0, D / 2), (1, 0), (0, -1), W),
            "west": ((-W / 2, 0), (0, 1), (1, 0), D), "east": ((W / 2, 0), (0, 1), (-1, 0), D)}[wall]


def _against(wall, t, along, depth, W, D, off=PAD):
    """Centre and rect of an item of size (along, depth) against ``wall`` at position t along it; yaw makes local -y face the wall."""
    (px, py), (ax, ay), (nx, ny), _ = _wall_frame(wall, W, D)
    cx, cy = px + ax * t + nx * (off + depth / 2), py + ay * t + ny * (off + depth / 2)
    sx, sy = (along, depth) if ax else (depth, along)
    yaw = {"south": 0, "north": 180, "west": -90, "east": 90}[wall]  # local +y = into the room
    return (cx, cy), _rect(cx, cy, sx, sy), yaw


def _face_yaw(cx, cy):
    """Yaw for a robot cell at (cx, cy) whose front (local -y) points to the room centre (snapped to an axis)."""
    if abs(cx) >= abs(cy):
        return 90 if cx < 0 else -90
    return 180 if cy < 0 else 0


# ---------------------------------------------------------------------------------------------------- recipe
def remanufacturing_cell(seed: int, width_m, depth_m, cell_type, difficulty, n_people, n_benches, n_robot_cells, conveyor, n_racks,
                         n_doors) -> SceneSpec:
    from ..assets.procedural.robot_cell import RobotCell

    rng = np.random.default_rng(seed)
    W, D = float(width_m), float(depth_m)
    diff = difficulty if difficulty in AISLE else "medium"
    A = AISLE[diff]
    n_doors = int(n_doors) if int(n_doors) >= 0 else int(rng.integers(1, 3))
    n_racks = int(n_racks) if int(n_racks) >= 0 else int(rng.integers(1, 4))
    n_benches = int(n_benches) if int(n_benches) >= 0 else int(rng.integers(2, 7))
    n_cells = int(n_robot_cells) if int(n_robot_cells) >= 0 else int(rng.integers(0, 2))
    has_conveyor = str(conveyor).lower() == "true" if not isinstance(conveyor, bool) else conveyor
    props = PROPS.get(cell_type, PROPS["electronics_weee"])
    plan, knobs, objects, zones = Plan(W, D), {"aisle_m": A}, [], []

    # ---- doors
    first = WALLS[(int(seed) + int(seed) // 1000) % 4]  # the main door's wall cycles over the four walls across seeds
    walls = [first] + [w for w in rng.permutation(WALLS).tolist() if w != first]
    door_walls = [walls[i % 4] for i in range(max(1, n_doors))]
    openings, door_pts, spawn_pts = [], [], []
    for i, wall in enumerate(door_walls):
        L = _wall_frame(wall, W, D)[3]
        rollup = i == 0 and min(W, D) >= 9 and rng.random() < 0.6
        w = 2.4 if rollup else 1.15
        for _ in range(20):
            t = float(rng.uniform(-L / 2 + w / 2 + 1.0, L / 2 - w / 2 - 1.0))
            depth = 3.4 if rollup else 2.2
            (cx, cy), r, _ = _against(wall, t, w + 1.2, depth, W, D, off=0.0)
            if plan.free(r, 0.0):
                break
        plan.add(r, "reserved")
        if i == 0:
            main_lane = [r]
        if rollup:  # the dock leveler plate is a real obstacle for walkers (worker paths must go round it)
            plan.add(_against(wall, t, w, 2.4, W, D, off=0.0)[1], "leveler")
        (px, py), (ax, ay), (nx, ny), _ = _wall_frame(wall, W, D)
        inset = 2.8 if rollup else 0.8  # a walking point beyond the roll-up door's dock leveler / in the doorway
        door_pts.append((px + ax * t + nx * inset, py + ay * t + ny * inset))
        spawn_pts.append((px + ax * t + nx * (depth + 0.4), py + ay * t + ny * (depth + 0.4)))  # past the door swing / dock lane
        openings.append({"id": f"door_{i}", "wall": wall, "center": round(t, 3), "width": w, "height": 3.0 if rollup else 2.2,
                         "door": "rollup_door" if rollup else "swing_door"})
    knobs["doors"] = [{"wall": o["wall"], "center": o["center"], "kind": o["door"]} for o in openings]
    lane_end = _rect(spawn_pts[0][0], spawn_pts[0][1], 1.2, 1.2)
    plan.add(lane_end, "reserved")  # the main door's lane continues past the door swing / dock: kept free of furniture
    main_lane.append(lane_end)

    # ---- robot cells
    cells, cell_rects = [], []
    for c in range(n_cells):
        fits = [a for a in ARMS if 2 * RobotCell().half(RobotCell().params({"arm": a})) + 2 * CELL_WALL_GAP + A <= min(W, D)]
        # cycles over the arms whose cell (+ wall gaps + an aisle) fits, across seeds and cells (balanced over the prompt set)
        arm = fits[(int(seed) + c + int(seed) // 1000) % len(fits)] if fits else "franka"
        h = RobotCell().half(RobotCell().params({"arm": arm}))
        cands = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                cands.append(("corner", sx * (W / 2 - h - CELL_WALL_GAP), sy * (D / 2 - h - CELL_WALL_GAP)))
        for wall in WALLS:
            (px, py), (ax, ay), (nx, ny), L = _wall_frame(wall, W, D)
            t = float(rng.uniform(-max(0.0, L / 2 - h - CELL_WALL_GAP - 1.0), max(0.0, L / 2 - h - CELL_WALL_GAP - 1.0)))
            cands.append(("wall_centred", px + ax * t + nx * (h + CELL_WALL_GAP), py + ay * t + ny * (h + CELL_WALL_GAP)))
        order = rng.permutation(len(cands))
        placed = None
        for k in order:
            kind, cx, cy = cands[k]
            r = _rect(cx, cy, 2 * h, 2 * h)
            if plan.inside(r, CELL_WALL_GAP - 0.01) and plan.free(r, A):
                placed = (kind, cx, cy, r)
                break
        if placed is None:  # no room left: put it anyway at the first candidate (the build check decides)
            kind, cx, cy = cands[order[0]]
            placed = (kind, cx, cy, _rect(cx, cy, 2 * h, 2 * h))
        kind, cx, cy, r = placed
        plan.add(r, "cell")
        cell_rects.append(r)
        cells.append({"xy": [round(cx, 3), round(cy, 3)], "yaw": _face_yaw(cx, cy), "arm": arm, "position": kind, "half": h})
    if cells:
        objects.append({"id": "cells", "asset": "robot_cell", "count": len(cells),
                        "instance_params": [{"arm": c["arm"], "arm_yaw": float(rng.choice([0, 90, 180, -90]))} for c in cells],
                        "placement": {"pose": [{"xy": c["xy"], "yaw": c["yaw"]} for c in cells]}})
    knobs["robot_cells"] = [{k: c[k] for k in ("xy", "arm", "position")} for c in cells]

    # ---- conveyor
    belt = None
    if has_conveyor:
        if cells:
            c0, r0 = cells[0], cell_rects[0]
            cx, cy = c0["xy"]
            axes = [ax for ax in ("x", "y") if (ax == "x" and abs(cx) > 0.5) or (ax == "y" and abs(cy) > 0.5)] or ["x"]
            axis = str(rng.choice(axes))
            sign = -np.sign(cx) if axis == "x" else -np.sign(cy)  # from the cell towards the room centre
            sign = sign or 1.0
            room_len = W if axis == "x" else D
            start = (r0[2] if sign > 0 else r0[0]) if axis == "x" else (r0[3] if sign > 0 else r0[1])
            free_len = (room_len / 2 - PAD - A) - sign * start if sign > 0 else (room_len / 2 - PAD - A) + sign * start
            L = float(min(rng.uniform(3.0, 6.0), max(2.0, abs(free_len) - CONVEYOR_CELL_GAP)))
            mid = start + sign * (CONVEYOR_CELL_GAP + L / 2)
            bx, by = (mid, cy) if axis == "x" else (cx, mid)
            feeds = str(rng.choice(["outfeed", "infeed"]))
            # conveyor local +y is the outfeed end: point it at the cell when the outfeed feeds the cell
            toward = -sign
            if axis == "x":
                yaw = (-90 if toward > 0 else 90) if feeds == "outfeed" else (90 if toward > 0 else -90)
            else:
                yaw = (0 if toward > 0 else 180) if feeds == "outfeed" else (180 if toward > 0 else 0)
            knobs["conveyor"] = {"axis": axis, "length_m": round(L, 2), "feeds_cell_with": feeds, "cell": 0}
        else:
            axis = "x" if W >= D else "y"
            L = float(rng.uniform(3.0, min(6.0, (W if axis == "x" else D) - 2 * A - 1.0)))
            off = float(rng.uniform(-0.25, 0.25)) * (D if axis == "x" else W)
            bx, by = (0.0, off) if axis == "x" else (off, 0.0)
            yaw = float(rng.choice([90, -90] if axis == "x" else [0, 180]))
            knobs["conveyor"] = {"axis": axis, "length_m": round(L, 2), "feeds_cell_with": None, "cell": None}
        r = _rect(bx, by, L, CONVEYOR_W) if axis == "x" else _rect(bx, by, CONVEYOR_W, L)
        plan.add(r, "conveyor")
        belt = {"xy": (bx, by), "axis": axis, "rect": r}
        objects.append({"id": "belt", "asset": "conveyor_belt", "params": {"length": round(L, 2), "speed": 0.4},
                        "placement": {"pose": {"xy": [round(bx, 3), round(by, 3)], "yaw": yaw}}})

    def wall_slots(wall, along, depth, n, gap, kind, min_free):
        """Place n items side by side against ``wall`` (first free run found); returns [(xy, yaw)]."""
        L = _wall_frame(wall, W, D)[3]
        run = n * along + (n - 1) * gap
        starts = rng.permutation(np.arange(-L / 2 + PAD, L / 2 - PAD - run + 1e-6, 0.25)) if run < L - 2 * PAD else []
        for s in starts:
            items = []
            for j in range(n):
                t = s + j * (along + gap) + along / 2
                xy, r, yaw = _against(wall, t, along, depth, W, D)
                if not (plan.inside(r) and plan.free(r, min_free)):
                    break
                items.append((xy, r, yaw))
            if len(items) == n:
                for _, r, _ in items:
                    plan.add(r, kind)
                return [(xy, yaw) for xy, _, yaw in items]
        return []

    # ---- pallet staging next to a door (any door, sliding outward along its wall, three sizes)
    staging = None
    for di in [0] + [i for i in rng.permutation(len(openings)).tolist() if i != 0]:
        d0 = openings[di]
        L0 = _wall_frame(d0["wall"], W, D)[3]
        for along, depth in ((2.6, 2.0), (2.0, 1.6), (1.6, 1.3)):
            for shift in (0.6, 1.2, 2.0, 3.0):
                for side in rng.permutation([-1, 1]):
                    t = d0["center"] + side * (d0["width"] / 2 + shift + along / 2)
                    if abs(t) > L0 / 2 - along / 2 - PAD:
                        continue
                    xy, r, yaw = _against(d0["wall"], t, along, depth, W, D)
                    if plan.inside(r) and plan.free(r, A):
                        plan.add(r, "staging")
                        staging = r
                        break
                if staging is not None:
                    break
            if staging is not None:
                break
        if staging is not None:
            break
    knobs["staging"] = None if staging is None else [round(v, 2) for v in staging]
    if staging is not None:
        zones.append({"id": "staging", "rect": [round(v, 3) for v in staging], "tags": ["staging"], "marking": "outline"})
        lo, hi = props[3]
        n_pal = int(rng.integers(lo, hi + 1))
        objects.append({"id": "pallets", "asset": "pallet", "count": [1, n_pal], "placement": {"inside": "staging"}, "yaw": "aligned"})
        unit = props[2]
        objects.append({"id": "returned_units", "asset": unit, "count": [1, n_pal] if unit == "washing_machine" else [1, 2 * n_pal],
                        "placement": {"on": "pallets.*.deck"}, "yaw": "random" if unit != "washing_machine" else {"range": [-10, 10]}})


    # ---- benches
    arrangements = [a for a in ARRANGEMENTS if a != "along_conveyor" or belt]
    # cycles over the options across consecutive seeds (balanced over the prompt set); a resample (seed + 1000 k) shifts it
    arrangement = arrangements[(int(seed) + int(seed) // 1000) % len(arrangements)]
    knobs["bench_arrangement_drawn"] = arrangement
    benches = []
    L_b, D_b = BENCH[1], BENCH[2]

    def greedy_row(axis, band, n, tag):
        """Up to n benches along the line ``band`` (perpendicular coordinate) parallel to ``axis``, each slid into the next free
        gap (BENCH_GAP to its row neighbours, the aisle A to everything else); returns [(xy, yaw)]."""
        span = W if axis == "x" else D
        yaw = 0 if axis == "x" else 90
        placed, s_pos = [], -span / 2 + A + L_b / 2
        while len(placed) < n and s_pos <= span / 2 - A - L_b / 2 + 1e-9:
            cx, cy = (s_pos, band) if axis == "x" else (band, s_pos)
            r = _rect(cx, cy, L_b, D_b) if axis == "x" else _rect(cx, cy, D_b, L_b)
            if plan.inside(r, A) and plan.free(r, A, ignore=(tag,)):
                plan.add(r, tag)
                placed.append(((cx, cy), yaw))
                s_pos += L_b + BENCH_GAP
            else:
                s_pos += 0.1
        return placed

    def undo(tag):
        plan.rects = [q for q in plan.rects if q[1] != tag]

    def best_rows(axes, band_sets, per_row):
        """Try every axis / band combination; keep the one placing the most benches (ties: first in the shuffled order)."""
        best = None
        for axis in axes:
            span_perp = D if axis == "x" else W
            for bands in band_sets(axis, span_perp):
                got = []
                for k, (b, n) in enumerate(zip(bands, per_row)):
                    got += greedy_row(axis, b, n, f"row{k}")
                for k in range(len(bands)):
                    undo(f"row{k}")
                if best is None or len(got) > len(best[1]):
                    best = ((axis, bands), got)
                if len(got) >= sum(per_row):
                    break
            if best and len(best[1]) >= sum(per_row):
                break
        if best:
            (axis, bands), _ = best
            out = []
            for k, (b, n) in enumerate(zip(bands, per_row)):
                out += greedy_row(axis, b, n, f"row{k}")
            plan.rects = [(q, "bench" if kind.startswith("row") else kind) for q, kind in plan.rects]
            return out
        return []

    def interior_bands(axis, span_perp, offsets=(0.0,)):
        lo, hi = -span_perp / 2 + A + D_b / 2, span_perp / 2 - A - D_b / 2 - max(offsets)
        return [[b + o for o in offsets] for b in rng.permutation(np.arange(lo, hi + 1e-9, 0.25)).tolist()]

    if arrangement == "along_wall":
        for wall in rng.permutation([w for w in walls if w not in door_walls]).tolist() + door_walls:  # benches come before shelving now
            if len(benches) == n_benches:
                break
            for n in range(n_benches - len(benches), 0, -1):
                got = wall_slots(wall, L_b, D_b, n, BENCH_GAP, "bench", A)
                if got:
                    benches += got
                    break
    elif arrangement == "along_conveyor":
        bx, by = belt["xy"]
        axis = belt["axis"]
        c = by if axis == "x" else bx
        off = CONVEYOR_W / 2 + A + D_b / 2 + 0.01
        sides = [[c + s * off] for s in rng.permutation([-1, 1]).tolist()]
        both = [[c - off, c + off]]
        half = (n_benches + 1) // 2
        benches += best_rows([axis], lambda ax, sp: sides, [n_benches]) or []
        if len(benches) < n_benches:  # one side is not enough: both sides of the belt
            for (x, y), _ in benches:
                plan.rects = [q for q in plan.rects if not (q[1] == "bench" and abs((q[0][0] + q[0][2]) / 2 - x) < 1e-6 and abs((q[0][1] + q[0][3]) / 2 - y) < 1e-6)]
            benches = best_rows([axis], lambda ax, sp: both, [half, n_benches - half])
    else:
        axes = (["x", "y"] if W >= D else ["y", "x"])
        if arrangement == "facing_rows" and n_benches >= 2:
            per = [n_benches - n_benches // 2, n_benches // 2]
            benches += best_rows(axes, lambda ax, sp: interior_bands(ax, sp, (0.0, D_b + A + 0.01)), per)
        else:
            benches += best_rows(axes, lambda ax, sp: interior_bands(ax, sp), [n_benches])
    knobs["benches_by_arrangement"] = len(benches)  # the rest (if any) come from the fallbacks below
    # what the layout actually shows: the drawn arrangement only if it placed every bench, else mixed (some by the fallbacks
    # below) or fallback (none); records and analyses use this, not the drawn value
    knobs["bench_arrangement"] = ("n/a" if not n_benches else arrangement if len(benches) == n_benches else "mixed" if benches else "fallback")
    for wall in rng.permutation(WALLS).tolist():  # fallback 1: single free wall slots
        while len(benches) < n_benches:
            got = wall_slots(wall, L_b, D_b, 1, 0.0, "bench", A)
            if not got:
                break
            benches += got
    while len(benches) < n_benches:  # fallback 2: anywhere free, so the count is respected (the check decides)
        grid = [(x, y) for x in np.arange(-W / 2 + A, W / 2 - A, 0.4) for y in np.arange(-D / 2 + A, D / 2 - A, 0.4)]
        spot = next(((x, y) for k in rng.permutation(len(grid)) for x, y in [grid[k]]
                     if plan.inside(_rect(x, y, L_b, D_b), A) and plan.free(_rect(x, y, L_b, D_b), A)), None)
        if spot is None:
            spot = (float(rng.uniform(-W / 4, W / 4)), float(rng.uniform(-D / 4, D / 4)))
        plan.add(_rect(*spot, L_b, D_b), "bench")
        benches.append((spot, 0))
    objects.append({"id": "benches", "asset": BENCH[0], "count": len(benches),
                    "placement": {"pose": [{"xy": [round(x, 3), round(y, 3)], "yaw": yaw} for (x, y), yaw in benches]}})
    for asset, (lo, hi) in props[0]:
        objects.append({"id": f"bench_{asset}", "asset": asset, "count": [lo, max(lo, hi * max(1, len(benches) // 2))],
                        "placement": {"on": "benches.*.top"}, "yaw": "random"})

    # ---- shelving
    free_walls = [w for w in walls if w not in door_walls] or list(walls)
    shelves, shelf_walls, shelf_runs = [], [], []
    split = n_racks >= 2 and bool(rng.random() < 0.5)  # knob: one run, or split over two walls
    remaining = n_racks
    want = int(rng.integers(1, n_racks)) if split else n_racks
    for wall in rng.permutation(free_walls).tolist() + [w for w in walls if w in door_walls]:
        while remaining > 0:
            n_try = min(want, remaining)
            got = wall_slots(wall, SHELF[1], SHELF[2], n_try, 0.05, "shelf", A)
            if got:
                shelves += got
                shelf_runs.append((wall, got))
                shelf_walls.append(wall)
                remaining -= n_try
                want = remaining
                break  # the rest (if any) goes on another wall
            if n_try == 1:
                break
            want = max(1, n_try // 2)  # does not fit here: a shorter run
        if remaining == 0:
            break
    for wall in rng.permutation(WALLS).tolist():  # fallback: single units on any free wall slot, so the count is respected
        while remaining > 0:
            got = wall_slots(wall, SHELF[1], SHELF[2], 1, 0.05, "shelf", A)
            if not got:
                break
            shelves += got
            shelf_runs.append((wall, got))
            remaining -= 1
    while remaining > 0:  # last resort: anywhere (the build check decides)
        x, y = float(rng.uniform(-W / 4, W / 4)), float(rng.uniform(-D / 4, D / 4))
        plan.add(_rect(x, y, SHELF[1], SHELF[2]), "shelf")
        shelves.append(((x, y), 0))
        remaining -= 1
    knobs["shelving_walls"] = shelf_walls
    knobs["shelving_split"] = split
    if shelves:
        objects.append({"id": "shelving", "asset": SHELF[0], "count": len(shelves),
                        "placement": {"pose": [{"xy": [round(x, 3), round(y, 3)], "yaw": yaw} for (x, y), yaw in shelves]}})
        xs = [x for (x, _), _ in shelves]
        ys = [y for (_, y), _ in shelves]
        for zi, (wall, run) in enumerate(shelf_runs):  # per wall run: the shelves + 0.8 m in front (a person stands there), nothing beside
            (nx, ny) = _wall_frame(wall, W, D)[2]
            rs = [_rect(x, y, SHELF[1], SHELF[2]) if nx == 0 else _rect(x, y, SHELF[2], SHELF[1]) for (x, y), _ in run]
            x0, y0, x1, y1 = min(q[0] for q in rs), min(q[1] for q in rs), max(q[2] for q in rs), max(q[3] for q in rs)
            x0, x1 = (x0, x1 + 0.8) if nx > 0 else (x0 - 0.8, x1) if nx < 0 else (x0, x1)
            y0, y1 = (y0, y1 + 0.8) if ny > 0 else (y0 - 0.8, y1) if ny < 0 else (y0, y1)
            zones.append({"id": f"parts_storage_{zi}", "rect": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)], "tags": ["storage"]})
        objects.append({"id": "klt_bins", "asset": "klt_bin", "count": [len(shelves), 3 * len(shelves)],
                        "placement": {"on": "shelving.*.shelf_*"}, "yaw": {"range": [-5, 5]}})

    # ---- charging station: at a wall, its parking area (2.9 m deep) reserved in front
    charger = None
    for wall in rng.permutation(WALLS).tolist():
        got = wall_slots(wall, 1.8, 2.9, 1, 0.0, "charger", A)
        if got:
            (x, y), yaw = got[0]
            (nx, ny) = _wall_frame(wall, W, D)[2]
            charger = (x - nx * (2.9 / 2 - 0.15 - PAD), y - ny * (2.9 / 2 - 0.15 - PAD))
            objects.append({"id": "charger", "asset": "charging_station", "placement": {"pose": {"xy": [round(charger[0], 3), round(charger[1], 3)], "yaw": yaw}}})
            break
    knobs["charging_station"] = None if charger is None else [round(v, 2) for v in charger]

    # ---- floor props, charging station, occluders (against free walls)
    def place_on_walls(asset, along, depth, ident, kind="prop"):
        for wall in rng.permutation(WALLS).tolist():
            got = wall_slots(wall, along, depth, 1, 0.0, kind, A)
            if got:
                (x, y), yaw = got[0]
                objects.append({"id": ident, "asset": asset, "placement": {"pose": {"xy": [round(x, 3), round(y, 3)], "yaw": yaw}}})
                return (x, y)
        return None

    from ..assets import get

    for asset, (lo, hi) in props[1]:
        a = get(asset)
        sx, sy, _ = a.footprint(a.params({})).size  # local x runs along the wall, local y is the depth (yaw from _against)
        for j in range(int(rng.integers(lo, hi + 1))):
            place_on_walls(asset, sx + 0.05, sy + 0.05, f"{asset}_{j}")
    if diff == "hard":
        for j in range(int(rng.integers(1, 3))):
            asset = str(rng.choice(OCCLUDERS))
            a = get(asset)
            sx, sy, _ = a.footprint(a.params({})).size
            place_on_walls(asset, sx + 0.05, sy + 0.05, f"occluder_{j}")
    knobs["lighting"] = str(rng.choice(LIGHTING))

    # ---- agents: workers walk between points that can see each other
    points = [("door", p) for p in door_pts]
    for (x, y), yaw in benches:
        n = (-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)))
        for s in (1, -1):
            points.append(("bench", (x + s * n[0] * (D_b / 2 + 0.65), y + s * n[1] * (D_b / 2 + 0.65))))
    for (x, y), yaw in shelves:
        n = (-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)))
        points.append(("shelving", (x + n[0] * (SHELF[2] / 2 + 0.65), y + n[1] * (SHELF[2] / 2 + 0.65))))
    for c in cells:
        yaw = math.radians(c["yaw"])
        points.append(("cell", (c["xy"][0] + math.sin(yaw) * (c["half"] + 0.7), c["xy"][1] - math.cos(yaw) * (c["half"] + 0.7))))
    points = [(k, p) for k, p in points if plan.inside(_rect(p[0], p[1], 0.6, 0.6)) and plan.free(_rect(p[0], p[1], 0.6, 0.6), 0.05, ignore=("reserved",))]
    pairs = [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))
             if points[i][0] != points[j][0] and math.dist(points[i][1], points[j][1]) > 1.5 and plan.segment_clear(points[i][1], points[j][1])]
    agents = []
    for w in range(int(n_people)):
        if pairs:
            i, j = pairs[int(rng.integers(len(pairs)))]
            wp = [list(map(lambda v: round(float(v), 3), points[i][1])), list(map(lambda v: round(float(v), 3), points[j][1]))]
        elif points:
            p = points[int(rng.integers(len(points)))][1]
            wp = [[round(p[0], 3), round(p[1], 3)], [round(p[0] + 0.8, 3), round(p[1], 3)]]
        else:
            wp = [[0.0, 0.0], [0.8, 0.0]]
        agents.append({"id": f"worker_{w}", "asset": "human", "path": {"waypoints": wp, "mode": "pingpong", "speed": round(float(rng.uniform(0.7, 1.0)), 2),
                                                                       "phase": round(float(rng.uniform(0, 3)), 2)}, "keep_clear_width": 0.8})

    # ---- robot: parked beside the main door lane (seeded knob), clear of the lane and of every worker path by the
    # check.human_path_robot_spawn distance, on free floor
    from ..assets.robots import table as robot_table
    from ..check.safety import HUMAN_RADIUS_M, HUMAN_SPAWN_GAP_M, path_clearance, robot_body

    rhx, rhy = (v / 2 for v in robot_table()["dingo"].size_m[:2])
    r_nav = float(get("dingo").info.footprint_radius_m)
    clear = HUMAN_RADIUS_M + HUMAN_SPAWN_GAP_M + 0.05
    lx0, ly0 = min(q[0] for q in main_lane), min(q[1] for q in main_lane)
    lx1, ly1 = max(q[2] for q in main_lane), max(q[3] for q in main_lane)
    paths = [a["path"]["waypoints"] for a in agents]

    def spawn_ok(x, y):
        body = _rect(x, y, 2 * rhx, 2 * rhy)
        nav = _rect(x, y, 2 * (r_nav + 0.1), 2 * (r_nav + 0.1))  # the solver wants footprint_radius + 2 cm of free floor
        gap = max(lx0 - body[2], body[0] - lx1, ly0 - body[3], body[1] - ly1)
        return (plan.inside(nav, 0.1) and plan.free(nav, 0.0, ignore=("reserved",)) and gap >= clear
                and all(path_clearance(wp, False, robot_body({"spawn": [x, y, 0.0], "asset": "dingo"}))[0] >= clear for wp in paths))

    (px, py), (ax, ay), (nx, ny), _ = _wall_frame(openings[0]["wall"], W, D)
    t0, half_lane = openings[0]["center"], max(abs(ax) * (lx1 - lx0), abs(ay) * (ly1 - ly0)) / 2
    lane_depth = max(abs(nx) * (lx1 - lx0), abs(ny) * (ly1 - ly0))
    body_along, body_in = (rhx, rhy) if ax else (rhy, rhx)

    def beside(side):
        for extra in np.arange(0.0, 3.01, 0.25):
            off = half_lane + clear + body_along + extra
            for d in np.arange(0.5, lane_depth + 1.01, 0.25):
                yield px + ax * (t0 + side * off) + nx * (body_in + d), py + ay * (t0 + side * off) + ny * (body_in + d)

    def by_charger():
        if charger is None:
            return
        for rr in np.arange(1.8, 4.01, 0.25):
            for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                yield charger[0] + rr * math.cos(ang), charger[1] + rr * math.sin(ang)

    options = [k for k in SPAWN_PLACEMENTS if k != "near_charger" or charger is not None]
    drawn = options[(int(seed) + int(seed) // 1000 + 1) % len(options)]  # cycles across seeds, like the other knobs
    gens = {"left_of_lane": lambda: beside(-1), "right_of_lane": lambda: beside(1), "near_charger": by_charger}
    spawn, placement = None, None
    for k in [drawn] + [o for o in options if o != drawn]:  # the drawn option first, then the others
        spawn = next(((x, y) for x, y in gens[k]() if spawn_ok(x, y)), None)
        if spawn:
            placement = k
            break
    if spawn is None:  # anywhere valid, nearest the main door first (the build's checks decide)
        door = (px + ax * t0, py + ay * t0)
        grid = sorted(((x, y) for x in np.arange(-W / 2 + 0.6, W / 2 - 0.6, 0.25) for y in np.arange(-D / 2 + 0.6, D / 2 - 0.6, 0.25)),
                      key=lambda q: math.dist(q, door))
        spawn, placement = next(((x, y) for x, y in grid if spawn_ok(x, y)), spawn_pts[0]), "fallback"
    knobs["robot_spawn_drawn"], knobs["robot_spawn"] = drawn, placement
    goals = [{"id": "at_bench", "type": "place", "target": "benches.0", "distance": 0.9}]
    if cells:
        goals.append({"id": "at_cell", "type": "pick", "target": "cells.0.gate", "approach": "port"})
    if shelves:
        goals.append({"id": "at_shelving", "type": "place", "target": "shelving.0", "distance": 0.9})

    spec = {
        "spec_version": "1.0", "name": f"e2_recipe_{cell_type}_seed{seed}",
        "description": f"E2 recipe baseline: {W:g} x {D:g} m {cell_type.replace('_', ' ')} ({diff}).", "seed": int(seed),
        "metadata": {"baseline": "e2_recipe", "recipe": knobs},
        "environment": {"kind": "indoor", "footprint": {"type": "rect", "size": [W, D]}, "height": 5.0,
                        "floor": {"material": "concrete", "wear": round(float(rng.uniform(0.2, 0.7)), 2)}, "walls": {"openings": openings}},
        "zones": zones, "objects": objects, "agents": agents,
        "robot": {"asset": "dingo", "spawn": {"pose": {"xy": [round(spawn[0], 3), round(spawn[1], 3)], "yaw": 0}}, "nav_goals": goals},
        "constraints": [{"type": "reachable", "from": "robot.spawn", "to": ["*"]}],
        "cameras": [{"id": "topdown", "mode": "top_down", "position": [0, 0, round(1.1 * max(W, D), 1)], "focal_mm": 10.5, "hide": ["roof"]},
                    {"id": "overview", "position": [round(-W / 2 + 0.8, 2), round(-D / 2 + 0.8, 2), 3.4], "look_at": [0, 0, 0.6], "focal_mm": 14}],
        "lighting": {"preset": knobs["lighting"]},
    }
    spec["metadata"]["recipe"] = json.loads(json.dumps(knobs, default=lambda o: o.item()))  # numpy scalars -> plain JSON
    return SceneSpec.model_validate(spec)


# ---------------------------------------------------------------------------------------------------- batch build
ROW_ARGS = ("width_m", "depth_m", "cell_type", "difficulty", "n_people", "n_benches", "n_robot_cells", "conveyor", "n_racks", "n_doors")
MAX_RESAMPLES = 10


def spec_for_row(row: dict, seed: int) -> SceneSpec:
    return remanufacturing_cell(seed, float(row["width_m"]), float(row["depth_m"]), row["cell_type"], row["difficulty"], int(row["n_people"]),
                                int(row["n_benches"]), int(row["n_robot_cells"]), row["conveyor"], int(row["n_racks"]), int(row["n_doors"]))


def provenance() -> dict:
    """What a baseline was built against.  The recipe lives outside the E1 freeze, so the checkout is not the freeze commit:
    record the freeze tag / commit, the curated-table hash and every path changed since the freeze (expected: recipes,
    tests, docs; never the asset library, compiler or checks)."""
    from ..datagen.runner import FREEZE_TAG, _git, freeze_info

    info = freeze_info()
    changed = _git("diff", "--name-only", FREEZE_TAG, "--").split() if info["freeze_commit"] else []
    return {"e1_freeze_tag": info["freeze_tag"], "e1_freeze_commit": info["freeze_commit"], "baseline_commit": info["head_commit"],
            "tree_dirty": info["tree_dirty"], "curated_table_sha256": info["curated_table_sha256"], "changed_since_freeze": changed}


def _attempt(spec, seed: int, out_dir: Path) -> tuple[dict, Path]:
    from ..compile import compile_scene

    t0 = time.time()
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        res = compile_scene(spec, seed, tmp)  # static checks only (no Isaac Sim)
        ok, codes = res.ok, sorted({i.code for i in res.issues if i.severity == "error"})
    except Exception as e:  # noqa: BLE001 - a spec that cannot even be expressed is a failed attempt
        ok, codes = False, [f"recipe.error:{type(e).__name__}"]
    return {"seed": seed, "ok": ok, "errors": codes, "seconds": round(time.time() - t0, 1)}, tmp


def _finish(out_dir: Path, tmp: Path, ok: bool):
    shutil.rmtree(out_dir, ignore_errors=True)
    if tmp.exists():
        tmp.rename(out_dir)  # a failed last attempt is kept for inspection
    out_dir.mkdir(parents=True, exist_ok=True)


def _record(out_dir: Path, status: dict, prov: dict) -> dict:
    spec = json.loads((out_dir / "resolved_spec.json").read_text()) if (out_dir / "resolved_spec.json").is_file() else {}
    status = {**status, "knobs": (spec.get("metadata") or {}).get("recipe"), "provenance": prov}
    (out_dir / "e2_attempts.json").write_text(json.dumps(status, indent=1))
    return status


def build_prompt(row: dict, out_dir: Path, log=print, prov: dict | None = None) -> dict:
    """Build the prompt's scene with seed int(id[1:]); on a failed build / check resample with seed + 1000 k (k = 1..10)."""
    base = int(row["id"][1:])
    out_dir = Path(out_dir)
    attempts = []
    for k in range(MAX_RESAMPLES + 1):
        seed = base + 1000 * k
        try:
            spec = spec_for_row(row, seed)
        except Exception as e:  # noqa: BLE001
            a, tmp = {"seed": seed, "ok": False, "errors": [f"recipe.error:{type(e).__name__}"], "seconds": 0.0}, out_dir.with_name(out_dir.name + ".tmp")
        else:
            a, tmp = _attempt(spec, seed, out_dir)
        attempts.append(a)
        log(f"  {row['id']} seed {seed}: {'OK' if a['ok'] else 'FAILED ' + ','.join(a['errors'])}")
        if a["ok"]:
            break
    _finish(out_dir, tmp, attempts[-1]["ok"])
    status = {"method": "recipe", "prompt_id": row["id"], "accepted": attempts[-1]["ok"], "attempts": len(attempts), "first_try": attempts[0]["ok"],
              "final_seed": attempts[-1]["seed"], "history": attempts}
    return _record(out_dir, status, prov or provenance())


def build_all(prompts_csv, out_root, ids=None, log=print) -> list[dict]:
    rows = [r for r in csv.DictReader(open(prompts_csv)) if not ids or r["id"] in ids]
    prov = provenance()
    out = [build_prompt(r, Path(out_root) / r["id"], log, prov) for r in rows]
    (Path(out_root) / "e2_recipe_summary.json").write_text(json.dumps(out, indent=1))
    return out


TEMPLATE = Path(__file__).resolve().parents[3] / "specs" / "e2_template_reman_cell.json"


def build_template(out_root, n_seeds: int = 30, spec_path=TEMPLATE, log=print) -> list[dict]:
    """The template baseline: one fixed floor plan (based on its prompt, P08), seeds 0..n-1 vary only the props.  No
    resampling (a failed seed stays failed); folders <prompt>_seed<NN> so worldc eval reads the prompt id from the name."""
    spec = json.loads(Path(spec_path).read_text())
    pid = spec["metadata"]["based_on_prompt"]
    prov = provenance()
    out = []
    for seed in range(n_seeds):
        out_dir = Path(out_root) / f"{pid}_seed{seed:02d}"
        a, tmp = _attempt(spec, seed, out_dir)
        log(f"  {pid} seed {seed}: {'OK' if a['ok'] else 'FAILED ' + ','.join(a['errors'])}")
        _finish(out_dir, tmp, a["ok"])
        status = {"method": "template", "prompt_id": pid, "accepted": a["ok"], "attempts": 1, "first_try": a["ok"], "final_seed": seed, "history": [a]}
        out.append(_record(out_dir, status, prov))
    (Path(out_root) / "e2_template_summary.json").write_text(json.dumps(out, indent=1))
    return out
