"""Post-hoc evaluation (eval/): adherence counts against a prompt row, plausibility rules and their aggregation.
Hand-built scene graphs where the rule needs no occupancy grid; tiny compiled rooms where it does."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from worldc.compile import compile_scene
from worldc.eval import adherence, plausibility
from worldc.eval.scene import Inst, Scene, load

DOOR = {"id": "door", "wall": "south", "center": 0.0, "width": 1.2, "height": 2.2}


def inst(key, asset, xy, yaw_deg=0.0, params=None, parent=None):
    return Inst(key, asset, key.split(".")[0], tuple(xy), math.radians(yaw_deg), params or {}, parent, f"/World/{key}")


def hand_scene(instances, hazards=(), openings=(), objects=()):
    graph = {"environment": {"kind": "indoor", "x_range": [-5, 5], "y_range": [-5, 5]}, "hazards": list(hazards), "objects": list(objects),
             "instances": {i.key: {"asset": i.asset} for i in instances}}
    spec = {"environment": {"walls": {"openings": list(openings)}}}
    return Scene(Path("."), graph, spec, list(instances))


# ---------------------------------------------------------------------------------------------------- adherence
PROMPT = {"n_benches": "2", "n_robot_cells": "1", "conveyor": "True", "n_racks": "-1", "n_doors": "1", "n_people": "2"}


def test_adherence_all_fields_match():
    sc = hand_scene([inst("bench.0", "work_table", (0, 0)), inst("bench.1", "workbench_steel", (3, 0)), inst("cell.0", "robot_cell", (0, 3)),
                     inst("belt.0", "conveyor_belt", (-3, 0)), inst("door_door.0", "rollup_door", (0, -5)), inst("shelf.0", "parts_shelf", (4, 4))],
                    hazards=[{"kind": "human"}, {"kind": "human"}, {"kind": "forklift"}])
    r = adherence.score(sc, PROMPT)
    assert r["counts"]["n_humans"] == 2  # the forklift agent is not counted
    assert "n_racks" not in r["fields"]  # -1 = unspecified, not scored
    assert r["adherence"] == 1.0 and all(f["match"] for f in r["fields"].values())


def test_adherence_counts_mismatch():
    sc = hand_scene([inst("bench.0", "work_table", (0, 0)), inst("door_door.0", "swing_door", (0, -5))], hazards=[{"kind": "human"}])
    r = adherence.score(sc, PROMPT)
    f = r["fields"]
    assert f["n_benches"] == {"requested": 2, "got": 1, "match": False, "abs_err": 1}
    assert f["conveyor"] == {"requested": True, "got": False, "match": False} and f["n_doors"]["match"]
    assert r["adherence"] == pytest.approx(1 / 5)  # 5 specified fields, only n_doors matches


def test_adherence_legacy_graph_uses_categories():
    sc = Scene(Path("."), {"objects": [{"category": "work_table"}, {"category": "rack_row"}, {"category": "rack_row"}, {"category": "door_swing"}],
                           "hazards": [{"kind": "human_capsule"}]}, None, [])
    assert adherence.counts(sc) == {"n_benches": 1, "n_robot_cells": 0, "n_conveyors": 0, "n_racks": 2, "n_doors": 1, "n_humans": 1}


# ---------------------------------------------------------------------------------------------------- conveyor_connected
BELT = inst("belt.0", "conveyor_belt", (0, 0), params={"length": 4.0})  # runs along y, ends at y = -2 / +2


def _conveyor(*others):
    rate, details = plausibility.conveyor_connected(hand_scene([BELT, *others]))
    return rate, details[0]


def test_conveyor_with_station_and_linked_end_passes():
    rate, d = _conveyor(inst("pallet.0", "pallet", (0, 2.75)), inst("bench.0", "work_table", (1.6, 0)))
    assert d["end_linked"] and d["station_near"] and d["pass"] and rate == {"n": 1, "passed": 1, "rate": 1.0}


def test_conveyor_without_station_fails():
    rate, d = _conveyor(inst("pallet.0", "pallet", (0, 2.75)))  # an end feeds a pallet, but no bench / cell works at it
    assert d["end_linked"] and not d["station_near"] and not d["pass"] and rate["rate"] == 0.0


def test_conveyor_with_ends_in_the_open_fails():
    rate, d = _conveyor(inst("bench.0", "work_table", (1.6, 0)))  # bench along the side (b), but both ends lead nowhere (a)
    assert d["station_near"] and not d["end_linked"] and not d["pass"]
    assert all(e["distance_m"] > plausibility.CONVEYOR_LINK_M for e in d["ends"])


@pytest.mark.parametrize("gap, ok", [(1.3, True), (1.6, False)])
def test_conveyor_station_distance(gap, ok):
    """(b) allows a bench up to 1.5 m from the belt (the 1.2 m mandatory aisle + 0.3 m): 1.3 m passes, 1.6 m fails."""
    x = 0.35 + gap + 0.8  # belt footprint half-width 0.35 m, work_table half-length 0.8 m along x
    rate, d = _conveyor(inst("pallet.0", "pallet", (0, 2.75)), inst("bench.0", "work_table", (x, 0)))
    assert d["end_linked"] and d["station_distance_m"] == pytest.approx(gap, abs=0.01)
    assert d["station_near"] is ok and d["pass"] is ok


def test_conveyor_end_at_a_door_counts():
    door = {"id": "dock", "wall": "south", "center": 0.0, "width": 2.0, "height": 2.5}
    sc = hand_scene([inst("belt.0", "conveyor_belt", (0, -2.6), params={"length": 4.0}), inst("bench.0", "work_table", (1.6, -2.6))], openings=[door])
    _, d = plausibility.conveyor_connected(sc)
    assert d[0]["pass"] and any(e["nearest"] == "door:dock" for e in d[0]["ends"])


# ---------------------------------------------------------------------------------------------------- aggregation
def test_non_applicable_rules_are_excluded_from_the_mean():
    rules = {"bench_access": {"rate": 1.0}, "conveyor_connected": {"rate": None}, "storage_reachable": {"rate": 0.5}, "cell_not_blocking": {"rate": None}}
    assert plausibility.aggregate(rules) == 0.75  # not (1 + 1 + 0.5 + 1) / 4
    assert plausibility.aggregate({k: {"rate": None} for k in rules}) is None


# ---------------------------------------------------------------------------------------------------- grid rules on built rooms
def room(objects):
    return {"environment": {"footprint": {"size": [6, 6]}, "height": 3.0, "walls": {"openings": [DOOR]}}, "objects": objects}


BENCH = {"id": "bench", "asset": "work_table", "params": {"drawers": 0}, "placement": {"pose": {"xy": [0.0, 1.3], "yaw": 0}}}


def _built(d, tmp_path, name):
    compile_scene(d, 0, tmp_path / name, run_check=False)
    return load(tmp_path / name)


def test_bench_access_pass_and_fail(tmp_path):
    ok = plausibility.score(_built(room([BENCH]), tmp_path, "ok"))
    b = next(x for x in ok["details"] if x["rule"] == "bench_access")
    assert b["pass"] and b["method"] == "port" and b["side"] == "+y"  # work_table's front port points to local +y
    assert ok["rules"]["conveyor_connected"]["rate"] is None and ok["plausibility"] == 1.0  # n/a rules do not count
    block = {"id": "crate", "asset": "enclosure", "params": {"size": [1.4, 0.4, 1.0]}, "placement": {"pose": {"xy": [0.0, 2.35], "yaw": 0}}}
    bad = plausibility.score(_built(room([BENCH, block]), tmp_path, "bad"))
    b = next(x for x in bad["details"] if x["rule"] == "bench_access")
    assert not b["pass"] and b["blocked_cells"] > 0 and bad["rules"]["bench_access"]["rate"] == 0.0


def test_storage_reachable_from_the_door(tmp_path):
    shelf = {"id": "shelf", "asset": "rack_row", "params": {"segments": [[-1.0, 1.0]], "bays_per_segment": 1, "levels": [0.2, 1.4]},
             "placement": {"pose": {"xy": [-2.0, 0.0], "yaw": 0}}}
    r = plausibility.score(_built(room([shelf]), tmp_path, "s"))
    d = next(x for x in r["details"] if x["rule"] == "storage_reachable")
    assert d["pass"] and d["from"] == "doors"


# ---------------------------------------------------------------------------------------------------- worldc eval (CSV)
def _prompts_csv(path, text):
    import csv as _csv

    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "text_plain", "text_constrained", "n_benches", "n_robot_cells", "conveyor", "n_racks", "n_doors", "n_people"])
        w.writeheader()
        w.writerow({"id": "P07", "text_plain": text, "text_constrained": text + " Keep aisles wide.", "n_benches": 0, "n_robot_cells": 0,
                    "conveyor": False, "n_racks": -1, "n_doors": -1, "n_people": 0})


def test_eval_csv_from_records(tmp_path, monkeypatch):
    import csv as _csv
    import sys

    from worldc.datagen.runner import generate
    from worldc.eval.run import columns, main

    fake = Path(__file__).parent / "fixtures" / "fake_claude.py"
    root = tmp_path / "ds"
    for mode, prompt in (("ok", "two pallets in a small room"), ("invalid", "a broken one")):
        monkeypatch.setenv("FAKE_MODE", mode)
        generate(prompt, root=root, final_checks=False, command_override=[sys.executable, str(fake)], log=lambda s: None)
    _prompts_csv(tmp_path / "p.csv", "two pallets in a small room")
    main(["--runs", str(root), "--prompts", str(tmp_path / "p.csv"), "--out", str(tmp_path / "e.csv")])
    rows = list(_csv.DictReader(open(tmp_path / "e.csv")))
    assert list(rows[0]) == columns() and len(rows) == 2
    ok, bad = rows
    assert ok["prompt_id"] == "P07" and ok["variant"] == "plain" and ok["accepted"] == "True" and ok["repairs"] == "1"
    assert ok["match_n_benches"] == "True" and ok["adherence"] and ok["storage_reachable_n"] == "2"  # two floor pallets = staging areas
    assert bad["accepted"] == "False" and bad["prompt_id"] == "" and bad["adherence"] == "" and bad["plausibility"] == ""


def test_eval_csv_baselines(tmp_path):
    import csv as _csv

    from worldc.eval.run import main

    compile_scene(room([BENCH]), 0, tmp_path / "runs" / "P07_seed0", run_check=False)
    _prompts_csv(tmp_path / "p.csv", "x")
    main(["--runs", str(tmp_path / "runs"), "--prompts", str(tmp_path / "p.csv"), "--out", str(tmp_path / "e.csv")])
    (row,) = list(_csv.DictReader(open(tmp_path / "e.csv")))
    assert row["prompt_id"] == "P07" and row["model"] == "baseline" and row["accepted"] == "True"
    assert row["got_n_benches"] == "1" and row["match_n_benches"] == "False" and row["bench_access_rate"] == "1.0"
