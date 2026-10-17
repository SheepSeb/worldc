"""Warehouse recipe: seed -> SceneSpec reproducing the original procedural warehouse (``generate.py`` before the spec
pipeline) declaratively.

40 x 25 m, 8 m ceiling; 6 rack aisles (12 rows, 4 levels); dock with 3 roll-up bays + levelers; 2 swing doors; packing
zone (2 belts, roller conveyor, work table with drawers, tote pallet); charging with 2 stations; wire-mesh cage with a
sliding gate; 2 slope zones; a forklift loop and 2 walking people; a 0.90 m chokepoint in aisle 2 made of fallen
pallets + spilled cartons; floor clutter; 3 lighting setups.

Structural choices that the old generator drew from its RNG (lighting, rack palette, box size classes, slope gradient,
forced pick slots, staging stacks, ...) are made here with a recipe RNG and written into the spec; per-instance
randomness (rack fill, carton jitter, clutter positions) is left to the compiler.
"""
from __future__ import annotations

import math

import numpy as np

from ..geom import rect_overlap_2d
from ..spec.models import SceneSpec

W, D, H = 40.0, 25.0, 8.0
X0, X1, Y0, Y1 = -W / 2, W / 2, -D / 2, D / 2
AISLES, AISLE_W, RACK_D, BACK_GAP, START_X = 6, 3.2, 0.9, 0.1, -19.4
SEGS = [(-4.3, 1.1), (4.3, 9.7)]
LEVELS = [0.2, 1.8, 3.4, 5.0]
CROSS = [(-7.5, -4.3), (1.1, 4.3)]
DOCK_X = (-15.0, -9.0, -3.0)
PARKING_X = (15.2, 18.0)
CAGE = (14.3, -7.6, 19.5, -3.6)
PACK_ZONE = (13.8, 1.2, 19.7, 12.4)
CHARGE_ZONE = (12.6, -12.4, 19.7, -8.3)
SLOPE_ZONES = [(2.0, -12.1, 10.5, -9.5), (13.4, -3.0, 19.4, -0.8)]
LIGHTING = ("bright_day", "mixed_warm", "dim_night")
PALETTES = [  # (uprights, beams)
    ((0.10, 0.25, 0.65), (0.95, 0.45, 0.08)),
    ((0.35, 0.37, 0.40), (0.80, 0.12, 0.10)),
    ((0.08, 0.40, 0.25), (0.95, 0.55, 0.05)),
    ((0.55, 0.10, 0.10), (0.20, 0.22, 0.25)),
]
PICK_SPECS = [(1, 2, 0), (4, 2, 1), (3, 3, 0), (5, 3, 1)]  # (aisle, level, segment)
CHOKE_AISLE, CHOKE_Y, CHOKE_CLEAR = 2, 7.0, 0.9
PAL_L, PAL_W = 1.2, 0.8


def aisle_x(k: int) -> tuple[float, float, float]:
    x0 = START_X + RACK_D + k * (AISLE_W + 2 * RACK_D + BACK_GAP)
    return x0, x0 + AISLE_W, x0 + AISLE_W / 2


def _r(v, n=4):
    return round(float(v), n)


def warehouse(seed: int = 0, lighting: int | None = None, light_scale: float = 1.0, **overrides) -> SceneSpec:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x57A3]))
    light = LIGHTING[seed % 3 if lighting is None else lighting]
    slope = _r(rng.uniform(0.02, 0.03))
    up_c, beam_c = PALETTES[int(rng.integers(4))]
    tints = [(np.clip(np.array(up_c) * rng.uniform(0.9, 1.1, 3), 0, 1), np.clip(np.array(beam_c) * rng.uniform(0.9, 1.1, 3), 0, 1)) for _ in range(3)]

    def rr(a, b):
        return float(rng.uniform(a, b))

    classes = [
        (rr(0.15, 0.25), rr(0.15, 0.25), rr(0.15, 0.25)), (rr(0.2, 0.32), rr(0.2, 0.32), rr(0.18, 0.3)),
        (rr(0.3, 0.42), rr(0.25, 0.38), rr(0.2, 0.35)), (rr(0.4, 0.5), rr(0.3, 0.45), rr(0.25, 0.4)),
        (rr(0.5, 0.6), rr(0.4, 0.5), rr(0.28, 0.42)), (rr(0.45, 0.6), rr(0.3, 0.45), rr(0.15, 0.22)),
    ]
    classes = [[_r(v) for v in c] for c in classes]
    # rack slots are 1.273 m apart: plan with the bare pallet outline like the original generator (overhang is checked post-build)
    jitter = {"box_size_scale": [0.94, 1.06], "min_box_dim": 0.15, "footprint_pad": 0.0}

    def cls(lo, hi):  # choice literal over classes [lo, hi)
        return {"choice": classes[lo:hi]}

    # ---- rack rows (layout: 12 rows, 6 aisles) + per-row colour tints
    racks = {"id": "racks", "asset": "rack_row", "zone_label": "racking",
             "params": {"segments": [list(s) for s in SEGS], "bays_per_segment": 2, "levels": LEVELS},
             "instance_params": [{"upright_color": [_r(c) for c in tints[i % 3][0]], "beam_color": [_r(c) for c in tints[i % 3][1]]} for i in range(12)],
             "placement": {"layout": {"type": "rack_aisles", "aisles": AISLES, "aisle_width": AISLE_W, "start_x": START_X,
                                      "back_gap": BACK_GAP, "row_depth": RACK_D}}}
    # forced pick slots: in aisle a, row 2a faces it from the west (face +1), row 2a+1 from the east (face -1)
    picks = []
    for k, (a, lvl, seg) in enumerate(PICK_SPECS):
        row = 2 * a if k % 2 == 0 else 2 * a + 1
        picks.append((row, lvl, seg, f"racks.{row}.slot_s{seg}_b0_p0_L{lvl}"))
    pick_pallets = {"id": "pick_pallets", "asset": "pallet_load", "zone_label": "racking",
                    "params": {"variant": "boxes", "box_size": cls(1, 6), "max_load_height": 1.12, "max_boxes": 12, **jitter},
                    "placement": {"fill": {"anchors": "racks.*.slot_*", "probability": 0.0, "force": [p[3] for p in picks]}},
                    "jitter": {"xy": 0.01, "yaw": 1.15}}
    rack_pallets = {"id": "rack_pallets", "asset": "pallet_load", "zone_label": "racking",
                    "params": {"variant": {"choice": ["boxes", "wrapped", "totes", "empty"], "p": [0.55, 0.28, 0.08, 0.09]},
                               "box_size": cls(1, 6), "max_load_height": 1.12, "max_boxes": 12, **jitter},
                    "placement": {"fill": {"anchors": "racks.*.slot_*", "probability": {"by": "level", "values": {"0": 0.55, "1": 0.45, "2": 0.4, "3": 0.35}}}},
                    "jitter": {"xy": 0.01, "yaw": 1.15}}

    # ---- dock staging row: 7 positions, 3 of them 2-high stacks
    xs = [-17.0 + i * 2.2 for i in range(7)]
    stacks = sorted(int(v) for v in rng.choice(7, 3, replace=False))
    lower_poses, lower_params, upper_poses, upper_params = [], [], [], []
    for i, x in enumerate(xs):
        yaw = math.degrees(rr(-0.05, 0.05))
        lower_poses.append({"xy": [_r(x), -8.55], "yaw": _r(yaw, 3)})
        if i in stacks:
            lower_params.append({"variant": "boxes", "box_size": classes[int(rng.integers(2, 5))], "max_load_height": 0.9, "full": True})
            upper_poses.append({"xy": [_r(x + rr(-0.015, 0.015)), _r(-8.55 + rr(-0.015, 0.015))], "yaw": _r(yaw + math.degrees(rr(-0.03, 0.03)), 3),
                                "support": f"staging.{i}.top"})
            upper_params.append({"variant": "boxes", "box_size": classes[int(rng.integers(2, 5))], "max_load_height": 0.8, "max_boxes": 10})
        elif rng.random() < 0.5:
            lower_params.append({"variant": "wrapped", "box_size": classes[int(rng.integers(2, 6))], "max_load_height": 1.1})
        else:
            lower_params.append({"variant": "boxes", "box_size": classes[int(rng.integers(1, 6))], "max_load_height": 1.1})
    staging = {"id": "staging", "asset": "pallet_load", "zone_label": "dock_staging", "count": 7, "params": dict(jitter),
               "instance_params": lower_params, "placement": {"pose": lower_poses}}
    staging_upper = {"id": "staging_upper", "asset": "pallet_load", "zone_label": "dock_staging", "count": len(upper_poses), "params": dict(jitter),
                     "instance_params": upper_params, "placement": {"pose": upper_poses}}

    # ---- chokepoint in aisle 2 (north segment): two fallen pallets leave CHOKE_CLEAR to the opposite rack
    ax0, ax1, _ = aisle_x(CHOKE_AISLE)
    left_edge = ax0 + 0.01
    y1_ = CHOKE_Y - 0.05
    p1_yaw = math.radians(6.0)
    span1 = PAL_L * abs(math.cos(p1_yaw)) + PAL_W * abs(math.sin(p1_yaw)) + 0.06
    p1x = left_edge + span1 / 2 + 0.01
    p2_yaw = math.pi / 2 + math.radians(12.0)
    span2 = PAL_L * abs(math.cos(p2_yaw)) + PAL_W * abs(math.sin(p2_yaw))
    p2x, p2y = ax1 - CHOKE_CLEAR - span2 / 2, CHOKE_Y + 0.05
    for _ in range(80):  # keep both on overlapping y rows so the squeeze is a true cross-section
        if not rect_overlap_2d((p1x, y1_), (PAL_L / 2 + 0.03, PAL_W / 2 + 0.03), p1_yaw, (p2x, p2y), (PAL_L / 2, PAL_W / 2), p2_yaw, 0.03):
            break
        p2y += 0.02
    choke_load = {"id": "choke_load", "asset": "pallet_load", "zone_label": "chokepoint",
                  "params": {"variant": "boxes", "box_size": classes[2], "max_load_height": 0.4, "max_boxes": 4, **jitter},
                  "placement": {"pose": {"xy": [_r(p1x), _r(y1_)], "yaw": 6.0}}}
    choke_pallet = {"id": "choke_pallet", "asset": "pallet", "zone_label": "chokepoint",
                    "placement": {"pose": {"xy": [_r(p2x), _r(p2y)], "yaw": _r(math.degrees(p2_yaw), 3)}}}
    choke_cones = {"id": "choke_cones", "asset": "traffic_cone", "zone_label": "chokepoint", "count": 2,
                   "placement": {"pose": [{"xy": [_r(ax0 + 1.0), CHOKE_Y - 2.1]}, {"xy": [_r(ax0 + 1.0), CHOKE_Y + 2.1]}]}, "yaw": "random"}
    choke_spill = {"id": "choke_spill", "asset": "cardboard_box", "zone_label": "chokepoint", "count": [4, 5],
                   "params": {"size": [{"uniform": [0.15, 0.4]}] * 3}, "placement": {"inside": "choke_spill_area", "margin": 0.0}, "yaw": "random"}

    # ---- packing zone
    belts = [
        {"id": "belt_a", "asset": "conveyor_belt", "zone_label": "packing", "params": {"length": 4.0, "speed": 0.5}, "placement": {"pose": {"xy": [15.2, 4.0], "yaw": 0}}},
        {"id": "belt_b", "asset": "conveyor_belt", "zone_label": "packing", "params": {"length": 4.0, "speed": 0.5}, "placement": {"pose": {"xy": [17.2, 4.0], "yaw": 0}}},
        {"id": "roller", "asset": "conveyor_roller", "zone_label": "packing", "params": {"length": 3.0, "speed": 0.5}, "placement": {"pose": {"xy": [15.2, 7.8], "yaw": 0}}},
        {"id": "table", "asset": "work_table", "zone_label": "packing", "placement": {"pose": {"xy": [17.2, 8.3], "yaw": 0}}},
    ]
    belt_boxes = []
    for bid, bx, ys in (("belt_a", 15.2, (2.6, 3.6)), ("belt_b", 17.2, (2.6, 3.8))):
        poses, params = [], []
        for y in ys:
            poses.append({"xy": [bx, y], "yaw": _r(math.degrees(rr(-0.1, 0.1)), 3), "support": f"{bid}.0.surface"})
            params.append({"size": [_r(rr(0.18, 0.45)), _r(rr(0.18, 0.45)), _r(rr(0.15, 0.35))]})
        belt_boxes.append({"id": f"{bid}_boxes", "asset": "cardboard_box", "zone_label": "packing", "count": 2, "instance_params": params, "placement": {"pose": poses}})
    belt_boxes.append({"id": "roller_box", "asset": "cardboard_box", "zone_label": "packing", "params": {"size": [0.4, 0.3, 0.2], "variant": 1},
                       "placement": {"pose": {"xy": [15.2, 6.6], "yaw": 0, "support": "roller.0.surface"}}})
    packing = [
        {"id": "table_totes", "asset": "tote", "zone_label": "packing", "count": 2,
         "placement": {"pose": [{"xy": [16.8, 8.3], "yaw": 0, "support": "table.0.top"}, {"xy": [17.6, 8.3], "yaw": 0, "support": "table.0.top"}]}},
        {"id": "tote_pallet", "asset": "pallet_load", "zone_label": "packing", "params": {"variant": "totes"}, "placement": {"pose": {"xy": [18.9, 10.4], "yaw": 90}}},
        {"id": "packing_bin", "asset": "trash_bin", "zone_label": "packing", "placement": {"pose": {"xy": [19.2, 1.9]}}},
    ]

    # ---- cage + contents, charging stations
    cx0, cy0, cx1, cy1 = CAGE
    cage = {"id": "cage", "asset": "cage", "zone_label": "cage", "params": {"size": [_r(cx1 - cx0), _r(cy1 - cy0)], "gate_offset": 0.0},
            "placement": {"pose": {"xy": [_r((cx0 + cx1) / 2), _r((cy0 + cy1) / 2)], "yaw": 0}}}
    cage_loads = {"id": "cage_loads", "asset": "pallet_load", "zone_label": "cage", "count": 3, "params": dict(jitter),
                  "instance_params": [{"variant": "wrapped", "box_size": classes[3], "max_load_height": 1.0},
                                      {"variant": "boxes", "box_size": classes[4], "max_load_height": 1.0, "max_boxes": 12},
                                      {"variant": "boxes", "box_size": classes[2], "max_load_height": 0.8, "max_boxes": 12}],
                  "placement": {"pose": [{"xy": [_r(cx0 + 1.1), _r(cy1 - 1.0)], "yaw": _r(math.degrees(0.1), 3)},
                                         {"xy": [_r(cx1 - 1.2), _r(cy1 - 1.2)], "yaw": _r(math.degrees(-0.2), 3)},
                                         {"xy": [_r(cx0 + 1.3), _r(cy0 + 1.1)], "yaw": _r(math.degrees(1.4), 3)}]}}
    chargers = {"id": "chargers", "asset": "charging_station", "zone_label": "charging", "count": 2,
                "placement": {"pose": [{"xy": [x, Y0 + 0.3], "yaw": 0} for x in PARKING_X]}}

    # ---- floor clutter
    box_dims = [{"uniform": [0.15, 0.6]}] * 3
    clutter = [
        {"id": "loose_boxes", "asset": "cardboard_box", "count": [14, 16], "params": {"size": box_dims},
         "placement": {"inside": ["dock_front", "east_lane", "north_lane", "charging_area", "packing", "cross_aisle_1", "cross_aisle_2"], "margin": 0.0},
         "yaw": "random"},
        {"id": "stacked_boxes", "asset": "cardboard_box", "count": [2, 3], "params": {"size": [{"uniform": [0.15, 0.3]}, {"uniform": [0.15, 0.3]}, {"uniform": [0.15, 0.3]}]},
         "placement": {"on": "loose_boxes.*.top", "arrange": "center"}, "jitter": {"xy": 0.02, "yaw": 8}},
        {"id": "pallet_jack", "asset": "pallet_jack_tipped", "zone_label": "dock_front", "placement": {"inside": "jack_area", "margin": 0.0}, "yaw": "random"},
        {"id": "cones", "asset": "traffic_cone", "zone_label": "dock_front", "count": [2, 3], "placement": {"inside": "dock_apron"}, "yaw": "random"},
    ]

    # ---- zones
    zones = [
        {"id": "dock_front", "rect": [-17.0, -12.0, 1.0, -9.9], "tags": ["dock"]},
        {"id": "dock_apron", "rect": [-17.0, -12.0, 12.0, -9.9]},
        {"id": "jack_area", "rect": [-17.5, -12.4, 9.5, -9.1]},
        {"id": "east_lane", "rect": [11.4, -7.2, 13.4, 2.0]},
        {"id": "north_lane", "rect": [-18.5, 9.9, 10.0, 12.2]},
        {"id": "charging_area", "rect": [12.8, -12.0, 19.4, -8.6], "tags": ["charging"]},
        {"id": "charging", "rect": list(CHARGE_ZONE), "tags": ["charging"], "marking": "outline"},
        {"id": "packing", "rect": [13.9, 1.2, 19.4, 12.2], "tags": ["packing"]},
        {"id": "packing_zone", "rect": list(PACK_ZONE), "tags": ["packing"], "marking": "outline"},
        {"id": "cross_aisle_1", "rect": [-17.0, -7.4, 9.0, -4.4]},
        {"id": "cross_aisle_2", "rect": [-17.0, 1.2, 9.0, 4.2]},
        {"id": "choke_spill_area", "rect": [_r(ax0 + 0.2), CHOKE_Y - 1.9, _r(ax0 + 2.0), CHOKE_Y + 1.9]},
    ]
    a1 = aisle_x(1)[2]
    a3 = aisle_x(3)[2]
    xa, xb = aisle_x(0)[2], aisle_x(5)[2]
    ya, yb = sum(CROSS[0]) / 2, sum(CROSS[1]) / 2
    goals = [{"id": f"parking_{i}", "type": "parking", "target": f"chargers.{i}.parking", "approach": "port"} for i in range(2)]
    goals += [{"id": f"dock_bay_{i}", "type": "dock", "target": f"env.opening_dock_{i}", "distance": 2.0} for i in range(3)]
    goals += [{"id": "belt_a_infeed", "type": "packing", "target": "belt_a.infeed", "approach": "port"},
              {"id": "belt_b_infeed", "type": "packing", "target": "belt_b.infeed", "approach": "port"},
              {"id": "roller_end", "type": "packing", "target": "roller.outfeed", "approach": "port"},
              {"id": "work_table", "type": "packing", "target": "table.front", "approach": "port"},
              {"id": "cage_gate", "type": "cage", "target": "cage.gate", "approach": "port"}]
    for row, lvl, seg, ref in picks:
        goals.append({"id": f"pick_{str(LEVELS[lvl]).replace('.', '_')}m_row{row}", "type": "pick_high", "target": ref, "approach": "front",
                      "distance": 0.95, "extra": {"pick": {"shelf_level_m": LEVELS[lvl], "rack_row": row, "segment": seg, "bay": 0, "slot": 0, "needs_lift": True}}})
    goals += [{"id": "choke_before", "type": "chokepoint", "target": "clearance:chokepoint", "offset": [0, -2.6], "yaw": 90},
              {"id": "choke_beyond", "type": "chokepoint", "target": "clearance:chokepoint", "offset": [0, 2.6], "yaw": 90},
              {"id": "staging_pickup", "type": "staging", "target": [-9.9, -10.2], "yaw": 0}]

    spec = {
        "spec_version": "1.0",
        "name": f"warehouse_seed{seed}",
        "description": "40 x 25 m warehouse: 6 rack aisles, dock with 3 roll-up bays, packing zone with conveyors and a work table, "
                       "charging area, wire-mesh cage, forklift loop and walking people, a 0.9 m chokepoint and floor clutter.",
        "seed": seed,
        "environment": {
            "kind": "indoor", "footprint": {"type": "rect", "size": [W, D]}, "height": H, "apron_depth": 6.5,
            "floor": {"material": "concrete_worn", "wear": {"uniform": [0.3, 1.0]},
                      "slope_zones": [{"id": f"crown_{i}", "rect": list(z), "slope": slope, "max_height": 0.04} for i, z in enumerate(SLOPE_ZONES)],
                      "extra_markings": [
                          {"type": "line", "points": [[-18.8, 9.95], [10.0, 9.95]], "color": [30, 150, 70], "width": 0.12, "kind": "pedestrian_walkway"},
                          {"type": "rect", "rect": [cx0 - 0.3, cy0 - 1.6, cx1 + 0.1, cy1 + 0.3], "color": [255, 200, 20], "width": 0.1, "kind": "cage_keepout"},
                          {"type": "dashed", "points": [[-17.5, -9.45], [-3.6, -9.45]], "color": [235, 235, 230], "width": 0.08, "kind": "staging_line"}]},
            "walls": {"thickness": 0.3, "openings":
                      [{"id": f"dock_{i}", "wall": "south", "center": x, "width": 3.0, "height": 3.2, "door": "rollup_door"} for i, x in enumerate(DOCK_X)] +
                      [{"id": "west_door", "wall": "west", "center": 11.0, "width": 1.15, "height": 2.24, "door": "swing_door"},
                       {"id": "north_door", "wall": "north", "center": 16.4, "width": 1.15, "height": 2.24, "door": "swing_door"}]},
        },
        "zones": zones,
        "objects": [racks, pick_pallets, rack_pallets, staging, staging_upper, choke_load, choke_pallet, choke_cones, cage, cage_loads, chargers,
                    *belts, *belt_boxes, *packing, choke_spill, *clutter],
        "constraints": [
            {"type": "clearance", "id": "chokepoint", "zone": f"racks.aisle_{CHOKE_AISLE}", "axis": "x", "span": [CHOKE_Y - 1.9, CHOKE_Y + 1.9],
             "target": CHOKE_CLEAR, "tolerance": 0.03, "side": "max", "adjust": ["choke_pallet"]},
            {"type": "reachable", "from": "robot.spawn", "to": ["*"]},
        ],
        "agents": [
            {"id": "forklift", "asset": "forklift_proxy", "params": {"mass_kg": 3500.0},
             "path": {"waypoints": [[xa, ya], [xb, ya], [xb, yb], [xa, yb]], "mode": "loop", "speed": 0.8}, "keep_clear_width": 2.1},
            {"id": "human_0", "asset": "human_proxy", "params": {"mass_kg": 80.0, "outfit": "vest"},
             "path": {"waypoints": [[-17.0, 11.2], [8.5, 11.2]], "mode": "pingpong", "speed": 0.9}, "keep_clear_width": 1.4},
            {"id": "human_1", "asset": "human_proxy", "params": {"mass_kg": 75.0, "outfit": "blue"},
             "path": {"waypoints": [[a3, -9.3], [a3, 10.9]], "mode": "pingpong", "speed": 0.8, "phase": 6.0}, "keep_clear_width": 1.4},
        ],
        "robot": {"spawn": {"pose": {"xy": [12.4, -9.0], "yaw": 180}}, "footprint_radius": 0.38, "nav_goals": goals},
        "cameras": [
            {"id": "overhead", "mode": "top_down", "position": [0, 0, 23.0], "focal_mm": 10.5, "clip": [1.0, 200.0], "hide": ["roof"]},
            {"id": "aisle_level", "position": [a1, -8.0, 1.7], "look_at": [a1, 8.0, 2.6], "focal_mm": 14.0},
        ],
        "lighting": {"preset": light, "scale": light_scale},
        "physics": {"hz": 120, "solver": "TGS"},
        "output": {"usd_name": "warehouse.usd", "namespace": "warehouse"},
        "metadata": {"recipe": "warehouse", "recipe_seed": seed, "box_classes_m": classes},
    }
    for k, v in overrides.items():  # shallow top-level overrides (e.g. lighting={...})
        spec[k] = v
    return SceneSpec.model_validate(spec)
