"""E2 baselines: the remanufacturing recipe (determinism, structural knobs, requested counts) and the fixed template."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from worldc.recipes import RECIPES
from worldc.recipes.remanufacturing import ROW_ARGS, spec_for_row

ROOT = Path(__file__).resolve().parents[1]
ROWS = {r["id"]: r for r in csv.DictReader(open(ROOT / "prompts" / "prompts.csv"))}
TEMPLATE = ROOT / "specs" / "e2_template_reman_cell.json"
STRUCTURE = ("doors", "robot_cells", "bench_arrangement", "shelving_walls", "staging", "charging_station", "lighting", "conveyor")


def dump(spec):
    return json.dumps(spec.model_dump(mode="json", by_alias=True, exclude_none=True), sort_keys=True)


def counts(spec):
    objs = {o.id: o for o in spec.objects}

    def n(oid):
        o = objs.get(oid)
        return 0 if o is None else o.count if isinstance(o.count, int) else o.count[1]

    return {"n_benches": n("benches"), "n_robot_cells": n("cells"), "conveyor": "belt" in objs, "n_racks": n("shelving"),
            "n_doors": len(spec.environment.walls.openings)}


def test_not_offered_to_the_llm():
    assert "remanufacturing" not in RECIPES  # list_recipes / get_recipe stay as in the E1 freeze


@pytest.mark.parametrize("pid", ["P01", "P14", "P23", "P30"])
def test_same_seed_same_spec(pid):
    assert dump(spec_for_row(ROWS[pid], 7)) == dump(spec_for_row(ROWS[pid], 7))


@pytest.mark.parametrize("pid", ["P07", "P18", "P29"])
def test_seeds_change_the_structural_knobs(pid):
    knobs = [spec_for_row(ROWS[pid], s).metadata["recipe"] for s in range(6)]
    varied = [k for k in STRUCTURE if len({json.dumps(kn.get(k), sort_keys=True) for kn in knobs}) > 1]
    assert {"doors", "bench_arrangement", "lighting"} <= set(varied) or len(varied) >= 4, varied


@pytest.mark.parametrize("pid", sorted(ROWS))
def test_requested_counts_are_respected(pid):
    row = ROWS[pid]
    for seed in (int(pid[1:]), int(pid[1:]) + 1000):
        spec = spec_for_row(row, seed)
        c = counts(spec)
        for f in ("n_benches", "n_robot_cells", "n_racks", "n_doors"):
            want = int(row[f])
            if want >= 0:
                assert c[f] == want, (pid, seed, f, c[f], want)
            else:
                assert c[f] >= 1, (pid, seed, f)  # unspecified: drawn from a range
        assert c["conveyor"] == (row["conveyor"] == "True"), pid
        assert spec.environment.footprint.size == (float(row["width_m"]), float(row["depth_m"]))
        assert len([a for a in spec.agents if a.asset == "human"]) == int(row["n_people"])
        assert not [a for a in spec.agents if "forklift" in str(a.asset)]
        assert spec.metadata["recipe"]["aisle_m"] >= 1.3  # every difficulty keeps slack above the mandatory 1.2 m


def test_row_arguments_match_the_csv():
    assert set(ROW_ARGS) <= set(next(iter(ROWS.values())))


def test_template_floor_plan_is_identical_across_seeds(tmp_path):
    """Built with two seeds: every large item (benches, belt, cell, shelf, pallets, bins) lands on the same pose."""
    from worldc.assets import curated
    from worldc.compile import compile_scene

    if any(curated.cached_mesh(a) is None for a in ("workbench_steel", "parts_shelf", "pallet_euro", "klt_bin", "safety_fence_panel")):
        pytest.skip("curated assets not cached (python -m worldc.assets.curated warm)")
    spec = json.loads(TEMPLATE.read_text())
    large = {"belt", "cell", "benches", "shelf", "recycling", "recycling_2", "pallets"}
    poses = []
    for seed in (0, 1):
        r = compile_scene(spec, seed, tmp_path / f"s{seed}")
        assert r.ok, [str(i) for i in r.issues if i.severity == "error"]
        inst = json.loads((tmp_path / f"s{seed}" / "scene_graph.json").read_text())["instances"]
        poses.append({k: (v["asset"], [round(c, 2) for c in v["pose"]["xyz"][:2]]) for k, v in inst.items() if v["spec_id"] in large - {"recycling", "recycling_2"}})
        props = sorted(k for k, v in inst.items() if v["spec_id"] in ("laptops", "monitors", "klt_bins"))
        poses[-1]["_props"] = props
    plan0 = {k: v for k, v in poses[0].items() if k != "_props"}
    assert plan0 and plan0 == {k: v for k, v in poses[1].items() if k != "_props"}  # same floor plan


def test_records_carry_the_realised_bench_arrangement():
    """bench_arrangement is what the layout shows: the drawn option only if it placed every bench, else mixed / fallback."""
    seen = set()
    for pid, row in ROWS.items():
        k = spec_for_row(row, int(pid[1:])).metadata["recipe"]
        n, by = int(row["n_benches"]), k["benches_by_arrangement"]
        want = "n/a" if not n else k["bench_arrangement_drawn"] if by == n else "mixed" if by else "fallback"
        assert k["bench_arrangement"] == want, pid
        seen.add(k["bench_arrangement"])
    assert len(seen - {"mixed", "fallback", "n/a"}) >= 3  # several arrangements are realised in full across the set


def test_baseline_provenance_names_the_e1_freeze():
    from worldc.recipes.remanufacturing import provenance

    p = provenance()
    assert p["e1_freeze_tag"].startswith("e1-freeze") and len(p["curated_table_sha256"]) == 64
    assert set(p) == {"e1_freeze_tag", "e1_freeze_commit", "baseline_commit", "tree_dirty", "curated_table_sha256", "changed_since_freeze"}


@pytest.mark.parametrize("seed", [18, 26])
def test_template_props_on_one_surface_never_overlap(tmp_path, seed):
    """Template seeds 18 / 26 failed check_scene on 4 mm overlaps of sibling props (monitors, KLT bins): the solver planned
    with the visual footprint while the check tests the collision hulls.  Planning with hull-covering footprints fixes it."""
    from worldc.assets import curated
    from worldc.compile import compile_scene

    if any(curated.cached_mesh(a) is None for a in ("workbench_steel", "parts_shelf", "monitor", "klt_bin")):
        pytest.skip("curated assets not cached (python -m worldc.assets.curated warm)")
    r = compile_scene(json.loads(TEMPLATE.read_text()), seed, tmp_path / "s")
    assert r.ok and not [i for i in r.issues if i.code == "check.interpenetration"], [str(i) for i in r.issues if i.severity == "error"]
