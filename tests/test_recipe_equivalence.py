"""The warehouse recipe compiled through the spec pipeline reproduces the original generator (baseline recorded at
f4daa5f in fixtures/warehouse_baseline.json).

Equivalence (docs/PLAN.md §5.2):
* exact: structural categories and counts, articulations (kind + limits), conveyors, hazards, cameras, goal names/types,
  building size, lighting preset per seed;
* statistical: stochastic counts per seed within [0.5 x min, 1.5 x max] of the baseline's seeds 0-4 (and, in the slow
  test, the 5-seed means within 25%).  The RNG stream differs from the old generator, so per-seed values differ; totes
  come in blocks of 4 per tote pallet (sd ~14 per seed), hence the wide per-seed band;
* physics validity without Isaac Sim: post-build checks pass (no interpenetration, nothing floating, hazard paths clear),
  every goal reachable, chokepoint 0.90 +/- 0.03 m, every dynamic body with positive mass and inertia.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdPhysics

from claude_scene_generator.compile import compile_scene
from claude_scene_generator.recipes import warehouse

BASE = json.loads((Path(__file__).parent / "fixtures" / "warehouse_baseline.json").read_text())
STOCHASTIC = ("pallet", "box", "tote", "tote_item", "clutter")


def norm(name: str) -> str:
    return name.lower().replace(".", "_")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cache = {}

    def get(seed):
        if seed not in cache:
            out = tmp_path_factory.mktemp(f"wh{seed}")
            res = compile_scene(warehouse(seed), seed, out)
            cache[seed] = (res, json.loads((out / "scene_graph.json").read_text()))
        return cache[seed]

    return get


def _check_seed(res, g, seed):
    base = BASE["per_seed"][str(seed)]
    assert res.ok, [str(i) for i in res.issues if i.severity == "error"]
    c = g["counts"]
    for cat in BASE["structural_categories"]:
        assert c.get(cat, 0) == base["counts"].get(cat, 0), (cat, c.get(cat), base["counts"].get(cat))
    lo_hi = BASE["count_ranges"]
    for cat in STOCHASTIC:
        lo, hi = lo_hi[cat]
        assert 0.5 * lo <= c.get(cat, 0) <= 1.5 * hi, (cat, c.get(cat), (lo, hi))
    lo, hi = lo_hi["n_dynamic_bodies"]
    assert 0.5 * lo <= g["n_dynamic_bodies"] <= 1.5 * hi
    assert g["lighting"]["setup_name"] == base["lighting"]
    assert g["building"]["size_m"] == base["building_size_m"]
    arts = Counter((a["kind"], round(a["lower"], 4), round(a["upper"], 4), a["unit"]) for a in g["articulations"])
    assert arts == Counter((k, round(lo_, 4), round(hi_, 4), u) for _, k, lo_, hi_, u in base["articulations"])
    assert Counter((cv["kind"], cv["speed_mps"]) for cv in g["conveyors"]) == Counter((k, s) for _, k, s in base["conveyors"])
    assert Counter((h["kind"], h["path_mode"], h["speed_mps"]) for h in g["hazards"]) == Counter(tuple(h) for h in base["hazards"])
    assert sorted(cam["name"] for cam in g["cameras"]) == base["cameras"]
    assert sorted((norm(x["name"]), x["type"]) for x in g["goals"]) == sorted((norm(n), t) for n, t in base["goals"])
    ck = res.check
    assert ck["passed"] and not ck["spawn_interpenetrations"] and not ck["floating_bodies"] and not ck["hazard_path_conflicts"]
    assert ck["navigation"]["all_reachable"]
    choke = next(x for x in ck["clearance"] if x["id"] == "chokepoint")
    assert abs(choke["measured_m"] - 0.90) <= 0.03
    stage = Usd.Stage.Open(str(res.usd))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(prim)
            assert m.GetMassAttr().Get() > 0 and all(v > 0 for v in m.GetDiagonalInertiaAttr().Get())


@pytest.mark.parametrize("seed", [0, 1])
def test_recipe_matches_baseline(built, seed):
    res, g = built(seed)
    _check_seed(res, g, seed)


@pytest.mark.slow
def test_recipe_matches_baseline_all_seeds(built):
    rows = []
    for seed in BASE["seeds"]:
        res, g = built(seed)
        _check_seed(res, g, seed)
        rows.append(g["counts"])
    for cat in ("pallet", "box", "tote", "tote_item"):
        new = np.mean([r.get(cat, 0) for r in rows])
        old = np.mean([BASE["per_seed"][str(s)]["counts"].get(cat, 0) for s in BASE["seeds"]])
        assert abs(new - old) <= 0.25 * old, (cat, new, old)


def test_recipe_spec_is_compact_and_valid():
    from claude_scene_generator.spec.validate import semantic

    spec = warehouse(3)
    assert not [i for i in semantic(spec) if i.severity == "error"]
    assert len(json.dumps(spec.model_dump(mode="json", exclude_defaults=True))) < 20_000  # LLM-sized
    assert warehouse(3) == spec and warehouse(4) != spec


def test_legacy_cli_through_pipeline(tmp_path):
    from claude_scene_generator.generate import build

    d = build(0, tmp_path)
    assert (d / "warehouse.usd").exists() and (d / "scene_graph.json").exists() and (d / "resolved_spec.json").exists()
