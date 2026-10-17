"""Compiler: examples build + pass checks, determinism, exact replay of the resolved spec, failure paths."""
import json
from pathlib import Path

import pytest
from pxr import Usd

from claude_scene_generator.compile import compile_scene, resolved_spec
from claude_scene_generator.spec.io import parse

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples" / "specs").glob("*.yaml"))


@pytest.mark.parametrize("spec", EXAMPLES, ids=lambda p: p.stem)
def test_examples_build_and_pass_checks(tmp_path, spec):
    r = compile_scene(spec, 0, tmp_path / "s")
    assert r.ok, [str(i) for i in r.issues]
    g = json.loads((tmp_path / "s" / "scene_graph.json").read_text())
    for k in ("environment", "objects", "instances", "goals", "counts", "robot", "cameras", "lighting", "spec_hash"):
        assert k in g
    assert Usd.Stage.Open(str(r.usd)).GetDefaultPrim().GetPath().pathString == "/World"
    assert (tmp_path / "s" / "resolved_spec.json").exists() and (tmp_path / "s" / "check.json").exists()
    assert all(goal["reachable"] for goal in g["goals"])


def test_deterministic_bytes(tmp_path):
    spec = ROOT / "examples" / "specs" / "warehouse_lite.yaml"
    a = compile_scene(spec, 7, tmp_path / "a", run_check=False)
    b = compile_scene(spec, 7, tmp_path / "b", run_check=False)
    assert a.usd.read_bytes() == b.usd.read_bytes()
    assert (tmp_path / "a" / "resolved_spec.json").read_text() == (tmp_path / "b" / "resolved_spec.json").read_text()
    c = compile_scene(spec, 8, tmp_path / "c", run_check=False)
    assert c.usd.read_bytes() != a.usd.read_bytes()


@pytest.mark.parametrize("seed", [0, 3])
def test_resolved_spec_replays_exactly(tmp_path, seed):
    spec = ROOT / "examples" / "specs" / "warehouse_lite.yaml"
    compile_scene(spec, seed, tmp_path / "a", run_check=False)
    rs = tmp_path / "a" / "resolved_spec.json"
    parsed, issues = parse(rs)
    assert parsed is not None, issues
    r = compile_scene(rs, None, tmp_path / "b", run_check=False)
    assert r.ok, [str(i) for i in r.issues]
    A = {o["path"]: o for o in json.loads((tmp_path / "a" / "scene_graph.json").read_text())["objects"]}
    B = {o["path"]: o for o in json.loads((tmp_path / "b" / "scene_graph.json").read_text())["objects"]}
    assert A == B


def test_invalid_spec_returns_issues_and_writes_nothing(tmp_path):
    r = compile_scene({"environment": {"footprint": {"size": [5, 5]}}, "objects": [{"id": "x", "asset": "nope", "placement": {"inside": "z"}}]},
                      0, tmp_path / "out")
    assert not r.ok and {"asset.unknown", "ref.unknown"} <= {i.code for i in r.issues}
    assert not (tmp_path / "out").exists()


def test_solver_failure_surfaces(tmp_path):
    d = {"environment": {"footprint": {"size": [5, 5]}}, "zones": [{"id": "z", "rect": [-1, -1, 1, 1]}],
         "objects": [{"id": "p", "asset": "pallet_load", "count": 10, "placement": {"inside": "z"}}]}
    r = compile_scene(d, 0, tmp_path / "out")
    assert not r.ok and any(i.code == "solve.unsatisfiable.inside" for i in r.issues)


def test_generated_and_mesh_assets_in_spec(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    trimesh.creation.box((0.5, 0.4, 0.3)).export(tmp_path / "crate.obj")
    d = {"environment": {"footprint": {"size": [6, 6]}},
         "objects": [{"id": "crates", "asset": {"file": "crate.obj", "id": "user_crate", "mass_kg": 4.0}, "count": 2, "placement": {"inside": "z"}},
                     {"id": "lib", "asset": "example_crate", "placement": {"pose": {"xy": [2, 2]}}},
                     {"id": "block", "asset": {"generate": {"id": "gen/block", "description": "block", "body": "dynamic", "mass_kg": 5,
                                                            "parts": [{"name": "b", "shape": "box", "size": [0.4, 0.4, 0.4], "pos": [0, 0, 0.2]}]}},
                      "placement": {"pose": {"xy": [-2, -2]}}}],
         "zones": [{"id": "z", "rect": [-2.5, 0, 0, 2.5]}]}
    r = compile_scene(d, 0, tmp_path / "out", base_dir=tmp_path)
    assert r.ok, [str(i) for i in r.issues]
    g = json.loads((tmp_path / "out" / "scene_graph.json").read_text())
    srcs = {v["spec_id"]: v["source"] for v in g["instances"].values()}
    assert srcs == {"crates": "mesh", "lib": "mesh", "block": "generated"}
