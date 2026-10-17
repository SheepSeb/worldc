"""Spec schema + semantic validation."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from sceneforge.spec.io import load_yaml

from sceneforge.spec.io import parse
from sceneforge.spec.models import SceneSpec
from sceneforge.spec.schema import SCHEMA_PATH, build_schema, section
from sceneforge.spec.validate import check, validate

ROOT = Path(__file__).resolve().parents[1]
LITE = ROOT / "examples" / "specs" / "warehouse_lite.yaml"


def lite():
    return load_yaml(LITE.read_text())


def codes(issues, severity="error"):
    return [i.code for i in issues if i.severity == severity]


def test_schema_in_sync():
    assert SCHEMA_PATH.read_text() == json.dumps(build_schema(), indent=1, sort_keys=True) + "\n", \
        "run: python -m sceneforge.spec.schema"


def test_schema_sections():
    s = section("ObjectSpec")
    assert "ObjectSpec" in s["$defs"] and "FillSpec" in s["$defs"]
    assert "GeneratedAssetSpec" in build_schema()["$defs"]
    with pytest.raises(KeyError):
        section("Nope")


def test_example_specs_valid():
    for p in sorted((ROOT / "examples" / "specs").glob("*.yaml")):
        spec, issues = check(p)
        assert spec is not None and not codes(issues), (p.name, [str(i) for i in issues])


def test_roundtrip():
    spec, _ = parse(lite())
    d = spec.model_dump(mode="json", by_alias=True)
    assert SceneSpec.model_validate(d) == spec


def test_minimal_spec():
    spec, issues = check({"environment": {"footprint": {"size": [10, 8]}}})
    assert spec is not None and not issues


@pytest.mark.parametrize("mutate, code, path", [
    (lambda d: d["objects"][0].update(lenght=3), "schema.extra_field", "/objects/0/lenght"),
    (lambda d: d["objects"][3]["placement"].update(inside="x"), "schema.placement", "/objects/3/placement"),
    (lambda d: d["objects"][0].update(asset="rack_rows"), "asset.unknown", "/objects/0/asset"),
    (lambda d: d["objects"][0]["params"].update(levles=[1]), "param.invalid", "/objects/0/params/levles"),
    (lambda d: d["objects"][0]["params"].update(depth=5.0), "param.invalid", "/objects/0/params/depth"),
    (lambda d: d["objects"][2]["placement"].update(inside="stagng"), "ref.unknown", "/objects/2/placement/inside"),
    (lambda d: d["objects"][4]["placement"].update(on="belt.top"), "ref.anchor_unknown", "/objects/4/placement/on"),
    (lambda d: d["objects"][4]["placement"].update(on="belt"), "ref.kind", "/objects/4/placement/on"),
    (lambda d: d["objects"][4]["placement"].update(on="belt.infeed"), "ref.kind", "/objects/4/placement/on"),
    (lambda d: d["zones"][0].update(rect=[-12, -6, -1, -4]), "zone.outside", "/zones/0/rect"),
    (lambda d: d["zones"][0].update(id="belt"), "id.duplicate", "/objects/3/id"),
    (lambda d: d["environment"]["walls"]["openings"][0].update(center=-9.5), "env.opening_outside", "/environment/walls/openings/0"),
    (lambda d: d["agents"][0]["path"]["waypoints"].append([30, 0]), "agent.waypoint_outside", "/agents/0/path/waypoints/2"),
    (lambda d: d["robot"]["nav_goals"][0].update(target="racks.9.slot_s0_b0_p0_L1"), "ref.instance_range", "/robot/nav_goals/0/target"),
    (lambda d: d["constraints"][1].update(to=["pick_9"]), "ref.unknown", "/constraints/1/to/0"),
    (lambda d: d.update(spec_version="0.3"), "schema.version", "/spec_version"),
    (lambda d: d["objects"].append({"id": "a", "asset": "cardboard_box", "placement": {"on": "b.top"}}) or
               d["objects"].append({"id": "b", "asset": "cardboard_box", "placement": {"on": "a.top"}}), "relation.cycle", None),
    (lambda d: d["objects"].append({"id": "gen", "asset": {"generate": {"id": "gen/x", "description": "x", "body": "dynamic",
                                    "parts": [{"name": "p", "shape": "box", "size": [0.5, 0.5, 0.5], "pos": [0, 0, 0.25], "material": "stell"}],
                                    "mass_kg": 3}}, "placement": {"inside": "packing"}}), "gen.material_unknown", "/objects/8/asset/generate/parts/0/material"),
])
def test_error_codes(mutate, code, path):
    d = copy.deepcopy(lite())
    mutate(d)
    issues = validate(d)
    hit = [i for i in issues if i.code == code]
    assert hit, (code, [str(i) for i in issues])
    if path is not None:
        assert any(i.path == path for i in hit), (path, [i.path for i in hit])
    assert all(i.hint for i in hit), [str(i) for i in hit]


def test_extra_field_hint_suggests():
    d = lite()
    d["objects"][0]["placment"] = d["objects"][0].pop("placement")
    issues = validate(d)
    assert any("placement" in i.hint for i in issues if i.code == "schema.extra_field")


def test_param_hint_lists_valid_params():
    d = lite()
    d["objects"][0]["params"]["levles"] = [1]
    i = next(i for i in validate(d) if i.code == "param.invalid")
    assert "levels" in i.hint
