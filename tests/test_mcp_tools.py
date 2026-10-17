"""Agent tool surface (tools.py), JSON Patch, and the MCP server wrapper.  None of this needs Isaac Sim to run."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from worldc import tools as T
from worldc.jsonpatch import PatchError, apply_patch, diff_summary

ROOT = Path(__file__).resolve().parents[1]
LITE = str(ROOT / "examples" / "specs" / "warehouse_lite.yaml")
MINI = {"environment": {"footprint": {"size": [6, 6]}}, "zones": [{"id": "z", "rect": [-2.5, -2.5, 2.5, 2.5]}],
        "objects": [{"id": "pallets", "asset": "pallet", "count": 2, "placement": {"inside": "z"}}]}
BLOCK = {"id": "gen/test_block", "description": "a steel block", "body": "dynamic", "mass_kg": 5,
         "parts": [{"name": "b", "shape": "box", "size": [0.4, 0.4, 0.4], "pos": [0, 0, 0.2]}]}


@pytest.fixture(autouse=True)
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDC_WORKSPACE", str(tmp_path / "ws"))
    return tmp_path / "ws"


def codes(res):
    return [i["code"] for i in res["issues"]]


# ------------------------------------------------------------------------------------------------ json patch
def test_patch_ops_and_no_mutation():
    doc = {"a": [1, 2, 3], "b": {"c": 1}, "k/s": 5}
    out = apply_patch(doc, [{"op": "add", "path": "/a/-", "value": 4}, {"op": "add", "path": "/a/0", "value": 0},
                            {"op": "replace", "path": "/b/c", "value": 2}, {"op": "remove", "path": "/a/1"},
                            {"op": "copy", "from": "/b", "path": "/d"}, {"op": "move", "from": "/k~1s", "path": "/e"},
                            {"op": "test", "path": "/b/c", "value": 2}])
    assert out == {"a": [0, 2, 3, 4], "b": {"c": 2}, "d": {"c": 2}, "e": 5}
    assert doc == {"a": [1, 2, 3], "b": {"c": 1}, "k/s": 5}


@pytest.mark.parametrize("patch,idx", [
    ([{"op": "replace", "path": "/nope", "value": 1}], 0),
    ([{"op": "add", "path": "/a/-", "value": 4}, {"op": "remove", "path": "/a/9"}], 1),
    ([{"op": "test", "path": "/b/c", "value": 99}], 0),
    ([{"op": "frobnicate", "path": "/a"}], 0),
    ([{"op": "add", "path": "/a"}], 0),
    ([{"op": "move", "from": "/b", "path": "/b/x"}], 0),
    ([{"op": "add", "path": "a/b", "value": 1}], 0),
])
def test_patch_errors_cite_the_op(patch, idx):
    with pytest.raises(PatchError) as e:
        apply_patch({"a": [1, 2], "b": {"c": 1}}, patch)
    assert e.value.index == idx


def test_diff_summary():
    assert diff_summary({"a": 1, "b": [1, 2], "c": 1}, {"a": 2, "b": [1, 2, 3], "d": 1}) == [
        "~ /a: 1 -> 2", "~ /b: list length 2 -> 3", "- /c", "+ /d"]


# ------------------------------------------------------------------------------------------------ asset tools
def test_list_search_describe():
    r = T.list_assets(tags=["dynamic"], query="pallet")
    assert r["ok"] and any(a["id"] == "pallet" for a in r["result"]["assets"])
    s = T.search_assets("pallet")
    assert s["ok"] and {m["source"] for m in s["result"]["matches"]} == {"registry", "omniverse"}
    assert s["result"]["matches"][0]["source"] == "registry" and "use" in s["result"]["matches"][0]
    assert all(m["source"] == "omniverse" for m in T.search_assets("pallet", sources=["omniverse"])["result"]["matches"])
    assert T.search_assets("zzqx")["result"]["count"] == 0
    d = T.describe_asset("pallet")
    assert d["ok"] and {"params_schema", "defaults", "anchors", "example"} <= set(d["result"])
    bad = T.describe_asset("palet")
    assert not bad["ok"] and codes(bad) == ["asset.unknown"] and "pallet" in bad["issues"][0]["hint"]


def test_schema_tool():
    whole = T.get_spec_schema()
    assert whole["ok"] and "ObjectSpec" in whole["result"]["sections"]
    sec = T.get_spec_schema("ObjectSpec")
    assert sec["ok"] and len(json.dumps(sec["result"]["schema"])) < len(json.dumps(whole["result"]["schema"]))
    assert codes(T.get_spec_schema("Nope")) == ["schema.unknown_section"]


def test_import_asset_errors(tmp_path):
    pytest.importorskip("trimesh")
    assert codes(T.import_asset(str(tmp_path / "missing.obj"))) == ["asset.file_missing"]


def test_import_asset_ok(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    trimesh.creation.box((0.5, 0.4, 0.3)).export(tmp_path / "crate.obj")
    r = T.import_asset(str(tmp_path / "crate.obj"), {"id": "mcp_test_crate", "mass_kg": 3.0})
    assert r["ok"] and r["result"]["id"] == "mcp_test_crate"
    assert T.describe_asset("mcp_test_crate")["ok"]


def test_validate_generated_asset_with_preview():
    r = T.validate_generated_asset(BLOCK)
    assert r["ok"], r["issues"]
    assert Path(r["result"]["image"]).is_file() and r["result"]["size_m"] == pytest.approx([0.4, 0.4, 0.4], abs=0.01)
    assert T.validate_generated_asset({"generate": BLOCK}, render=False)["ok"]  # wrapped form accepted
    bad = T.validate_generated_asset({**BLOCK, "mass_kg": -1}, render=False)
    assert not bad["ok"] and "image" not in bad["result"]


# ------------------------------------------------------------------------------------------------ spec tools
def test_validate_spec_levels():
    ok = T.validate_spec(MINI)
    assert ok["ok"] and ok["result"]["stage"] == "solved" and ok["result"]["instances_by_object"] == {"pallets": 2}
    assert T.validate_spec(LITE)["ok"]  # YAML text with `on:` placements, given as a path
    yml = Path(LITE).read_text()
    assert T.validate_spec(yml)["ok"]  # ... and as YAML text
    bad = T.validate_spec({**MINI, "objects": [{"id": "p", "asset": "palet", "placement": {"inside": "z"}}]})
    assert not bad["ok"] and bad["result"]["stage"] == "validate" and codes(bad) == ["asset.unknown"]
    assert bad["issues"][0]["path"] == "/objects/0/asset" and "pallet" in bad["issues"][0]["hint"]
    unsolvable = T.validate_spec({**MINI, "objects": [{"id": "p", "asset": "pallet", "count": 40, "placement": {"inside": "z"}}]})
    assert not unsolvable["ok"] and unsolvable["result"]["stage"] == "solved" and any(c.startswith("solve.") for c in codes(unsolvable))
    assert codes(T.validate_spec("{ not: [valid")) == ["spec.unreadable"]
    assert codes(T.validate_spec(42)) == ["spec.unreadable"]


def test_edit_scene():
    r = T.edit_scene(MINI, [{"op": "replace", "path": "/objects/0/count", "value": 3}])
    assert r["ok"] and r["result"]["valid"] and r["result"]["spec"]["objects"][0]["count"] == 3
    assert r["result"]["diff"] == ["~ /objects/0/count: 2 -> 3"] and MINI["objects"][0]["count"] == 2
    bad_op = T.edit_scene(MINI, [{"op": "replace", "path": "/objects/0/count", "value": 3}, {"op": "remove", "path": "/zones/7"}])
    assert not bad_op["ok"] and bad_op["result"]["failed_op"] == 1 and bad_op["issues"][0]["path"] == "/json_patch/1"
    invalid = T.edit_scene(MINI, [{"op": "replace", "path": "/objects/0/asset", "value": "palet"}])
    assert not invalid["ok"] and invalid["result"]["spec"]["objects"][0]["asset"] == "palet"  # new spec returned so the agent can keep editing
    assert codes(invalid) == ["asset.unknown"]


# ------------------------------------------------------------------------------------------------ build / check / render
def test_build_check_render_flow(ws):
    b = T.build_scene(MINI, seed=2)
    assert b["ok"], b["issues"]
    out = Path(b["result"]["out_dir"])
    assert ws in out.parents and (out / "scene.usd").is_file() and (out / "resolved_spec.json").is_file()
    assert b["result"]["counts"]["pallet"] == 2 and b["result"]["check"]["passed"]
    assert b["result"]["files"]["scene_graph.json"] == str(out / "scene_graph.json")
    again = T.build_scene(MINI, seed=2)
    assert again["result"]["out_dir"] == b["result"]["out_dir"]  # same spec + seed -> same folder

    c = T.check_scene(str(out), physics=False)
    assert c["ok"] and c["result"]["static"]["passed"] and "physics" not in c["result"]

    r = T.render(str(out), {"position": [4, -4, 3], "look_at": [0, 0, 0.3], "name": "iso"}, 320, 200, engine="proxy")
    assert r["ok"] and Path(r["result"]["image"]).is_file()
    assert T.render(str(out), "topdown_map", engine="proxy")["ok"]
    assert codes(T.render(str(out), "nope", engine="proxy")) == ["internal.ValueError"]
    assert codes(T.render(str(out), "iso", engine="vulkan")) == ["render.engine"]
    assert codes(T.render(str(ws), "overview")) == ["internal.FileNotFoundError"]


def test_build_scene_failures_and_confinement(ws, tmp_path):
    bad = T.build_scene({**MINI, "objects": [{"id": "p", "asset": "palet", "placement": {"inside": "z"}}]})
    assert not bad["ok"] and codes(bad) == ["asset.unknown"]
    assert codes(T.build_scene(MINI, out_dir="../escape")) == ["internal.PermissionError"]
    assert codes(T.build_scene(MINI, out_dir=str(tmp_path / "elsewhere"))) == ["internal.PermissionError"]
    ok = T.build_scene(MINI, out_dir="named/run1")
    assert ok["ok"] and Path(ok["result"]["out_dir"]) == (ws / "named" / "run1").resolve()


def test_physics_check_reports_missing_isaac(ws, monkeypatch, tmp_path):
    b = T.build_scene(MINI)
    monkeypatch.setenv("ISAACSIM_PATH", str(tmp_path / "no_isaac"))
    c = T.check_scene(b["result"]["out_dir"], physics=True)
    assert not c["ok"] and "env.isaac_sim_missing" in codes(c) and c["result"]["physics"]["ran"] is False
    r = T.render(b["result"]["out_dir"], "overview", engine="rtx")
    assert not r["ok"] and codes(r) == ["env.isaac_sim_missing"]


def test_unexpected_exceptions_become_issues(monkeypatch):
    import worldc.assets as assets

    monkeypatch.setattr(assets, "list_assets", lambda *a, **k: 1 / 0)
    r = T.list_assets()
    assert not r["ok"] and codes(r) == ["internal.ZeroDivisionError"]


# ------------------------------------------------------------------------------------------------ MCP wrapper
def test_mcp_server_registers_and_serves():
    pytest.importorskip("mcp")
    from mcp.client import Client

    from worldc.mcp_server import build_server

    async def run():
        async with Client(build_server()) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names == set(T.TOOLS) and len(names) == 14
            schema = {t.name: t.input_schema for t in (await c.list_tools()).tools}
            assert "spec" in schema["validate_spec"]["properties"] and "json_patch" in schema["edit_scene"]["properties"]
            res = await c.call_tool("describe_asset", {"id": "pallet"})
            assert json.loads(res.content[0].text)["ok"]
            res = await c.call_tool("build_scene", {"spec": MINI, "seed": 1})
            built = json.loads(res.content[0].text)
            assert built["ok"]
            res = await c.call_tool("render", {"path": built["result"]["out_dir"], "camera": "topdown_map", "engine": "proxy"})
            kinds = [x.type for x in res.content]
            assert kinds == ["text", "image"], kinds
            res = await c.call_tool("validate_spec", {"spec": {"objects": 3}})
            assert not json.loads(res.content[0].text)["ok"]

    asyncio.run(run())


def test_recipe_tools():
    names = [r["name"] for r in T.list_recipes()["result"]["recipes"]]
    assert "warehouse" in names
    r = T.get_recipe("warehouse", 3)
    assert r["ok"] and r["result"]["spec"]["environment"]["footprint"]["size"] == [40.0, 25.0] and r["result"]["spec"]["robot"]["asset"] == "dingo"
    assert T.get_recipe("warehouse", 3)["result"]["spec"] != T.get_recipe("warehouse", 4)["result"]["spec"]
    assert codes(T.get_recipe("nope")) == ["recipe.unknown"]
    e = T.edit_scene(r["result"]["spec"], [{"op": "replace", "path": "/robot/asset", "value": "jackal"}])  # a recipe is a normal spec
    assert e["result"]["valid"]


def test_recipe_spec_round_trips_and_builds():
    spec = T.get_recipe("warehouse", 1)["result"]["spec"]
    assert T.validate_spec(spec)["ok"]  # the dumped spec parses back (discriminator fields kept)
