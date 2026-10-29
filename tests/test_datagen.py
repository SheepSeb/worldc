"""Dataset collection through the Claude Code runner (with a fake `claude`), and the exports.  No Claude Code, no Isaac Sim."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from worldc.datagen import export as ex
from worldc.datagen import prompts
from worldc.datagen.records import RECORD_VERSION, load_records, split_of
from worldc.datagen.runner import claude_command, env_signature, generate

FAKE = Path(__file__).parent / "fixtures" / "fake_claude.py"


def run(root, prompt="two pallets in a small room", mode="ok", monkeypatch=None, **kw):
    monkeypatch.setenv("FAKE_MODE", mode)
    return generate(prompt, root=root, final_checks=False, command_override=[sys.executable, str(FAKE)], log=lambda s: None, **kw)


@pytest.fixture
def root(tmp_path):
    return tmp_path / "dataset"


def test_accepted_run_is_fully_collected(root, monkeypatch):
    rec = run(root, monkeypatch=monkeypatch)
    assert rec.accepted and rec.stop_reason == "end_turn" and rec.error is None
    d = root / "runs" / rec.run_id
    for f in ("prompt.txt", "spec.json", "record.json", "trace.jsonl", "mcp.json"):
        assert (d / f).is_file(), f
    assert (d / "scene" / "scene.usd").is_file() and (d / "scene" / "resolved_spec.json").is_file() and not (d / "work").exists()
    assert rec.scene_dir == f"runs/{rec.run_id}/scene" and (root / "README.md").is_file()
    assert rec.spec["objects"][0]["asset"] == "pallet" and rec.asset_sources == {"registry": 1} and rec.env_signature == "indoor|pallet"
    assert rec.usage.est_cost_usd == 0.123 and rec.usage.tool_calls == 5 and rec.usage.cache_read_input_tokens == 1000
    assert rec.usage.repairs == 1 and rec.params["max_repairs"] == 10  # the first validate_spec came back with asset.unknown
    g = rec.generator
    assert len(g["curated_table_sha256"]) == 64 and g["freeze_tag"] == "e1-freeze-v6" and g["at_freeze"] in ("true", "false")
    assert rec.summary == "Built a 6x6 room with two pallets." and rec.record_version == RECORD_VERSION and rec.split in ("train", "val", "test")
    assert rec.validation["last_validate"]["ok"] and rec.validation["last_build"]["ok"] and rec.validation["last_build"]["check"]["passed"]
    assert any(p.startswith(f"runs/{rec.run_id}/images/") for p in rec.render_paths) and (d / "images" / "001.png").is_file()
    assert [r.id for r in load_records(root)] == [rec.id]


def test_trace_uses_the_data_traces_vocabulary(root, monkeypatch):
    rec = run(root, monkeypatch=monkeypatch)
    d = root / "runs" / rec.run_id
    ev = [json.loads(ln) for ln in (d / "trace.jsonl").read_text().splitlines()]
    assert [e["seq"] for e in ev] == list(range(len(ev))) and ev[0]["type"] == "user" and ev[-1]["type"] == "result"
    assert {e["type"] for e in ev} <= {"user", "init", "assistant", "thinking", "write", "tool_call", "tool_result", "result"}
    assert ev[0]["prompt_id"] == prompts.SPEC_WRITER_ID and ev[1]["type"] == "init" and ev[1]["mcp_servers"] == {"worldc": "connected"}
    calls = [e for e in ev if e["type"] == "tool_call"]
    results = [e for e in ev if e["type"] == "tool_result"]
    assert [c["tool"] for c in calls] == ["search_assets", "validate_spec", "validate_spec", "build_scene", "render"] == [r["tool"] for r in results]
    assert [c["tool_use_id"] for c in calls] == [r["tool_use_id"] for r in results]
    assert results[1]["ok"] is False and results[1]["exit_code"] == 1 and json.loads(results[1]["stdout"])["issues"][0]["code"] == "asset.unknown"
    assert results[-1]["images"] == ["images/001.png"]
    # spec drafts are kept as numbered versions, including the failed one (repair data)
    writes = [e for e in ev if e["type"] == "write"]
    assert [w["path"] for w in writes] == ["specs/spec_v1.json", "specs/spec_v2.json"]
    assert json.loads((d / "specs" / "spec_v1.json").read_text())["objects"][0]["asset"] == "palet" and json.loads(writes[1]["content"]) == rec.spec
    assert "base64" not in (d / "trace.jsonl").read_text() and not any("signature" in json.dumps(e) for e in ev)
    assert ev[-2]["type"] == "assistant" and ev[-2]["content"] == "Built a 6x6 room with two pallets."


def test_trace_markdown_is_readable(root, monkeypatch):
    rec = run(root, monkeypatch=monkeypatch)
    md = (root / "runs" / rec.run_id / "trace.md").read_text()
    for needle in ("# Run ", "ACCEPTED", "## Prompt", "two pallets in a small room", "### Tool call: `validate_spec`", "`asset.unknown`",
                   "hint:", "![images/001.png](images/001.png)", "specs/spec_v2.json", "Built a 6x6 room with two pallets."):
        assert needle in md, needle


def test_rejected_runs_are_recorded_too(root, monkeypatch):
    bad = run(root, "q", "invalid", monkeypatch)
    assert not bad.accepted and bad.scene_dir is None and bad.validation["last_validate"]["issues"][0]["code"] == "asset.unknown"
    assert bad.spec["objects"][0]["asset"] == "palet" and bad.stop_reason == "end_turn"
    nospec = run(root, "r", "nospec", monkeypatch)
    assert not nospec.accepted and nospec.spec is None
    crash = run(root, "s", "crash", monkeypatch)
    assert not crash.accepted and "exited without a result" in crash.error
    assert len(load_records(root)) == 3


def test_missing_claude_binary(root):
    rec = generate("x", root=root, claude_bin="definitely-not-claude", final_checks=False, log=lambda s: None)
    assert not rec.accepted and "not found" in rec.error and rec.stop_reason == "error"


def test_max_turns_kills_the_run(root, monkeypatch):
    rec = run(root, monkeypatch=monkeypatch, max_turns=2)
    assert rec.stop_reason == "max_turns" and not rec.accepted


def test_max_repairs_stops_the_run(root, monkeypatch):
    rec = run(root, monkeypatch=monkeypatch, max_repairs=0)
    assert rec.stop_reason == "max_repairs" and not rec.accepted and rec.usage.repairs == 1
    assert run(root, "p2", monkeypatch=monkeypatch, max_repairs=None).accepted  # no cap


def test_seed_is_pinned_for_the_mcp_server(root, monkeypatch):
    rec = run(root, monkeypatch=monkeypatch, seed=7)
    cfg = json.loads((root / "runs" / rec.run_id / "mcp.json").read_text())
    assert cfg["mcpServers"]["worldc"]["env"]["WORLDC_SEED"] == "7" and rec.params["seed"] == 7
    assert rec.params["loop_seconds"] >= 0 and rec.params["final_checks_seconds"] >= 0


def test_exports(root, monkeypatch):
    good = run(root, "two pallets", monkeypatch=monkeypatch)
    run(root, "bad one", "invalid", monkeypatch)
    out = ex.export(root)
    assert set(out) == {"prompt_spec", "sft", "records", "stats"}
    rows = [json.loads(ln) for ln in out["prompt_spec"].read_text().splitlines()]
    assert [r["prompt"] for r in rows] == ["two pallets"] and rows[0]["spec"] == good.spec and rows[0]["split"] == good.split
    rows_all = [json.loads(ln) for ln in ex.export(root, ["prompt_spec"], include_rejected=True)["prompt_spec"].read_text().splitlines()]
    assert len(rows_all) == 2 and rows_all[1]["accepted"] is False and rows_all[1]["issues"][0]["code"] == "asset.unknown"

    sft = [json.loads(ln) for ln in out["sft"].read_text().splitlines()]
    assert len(sft) == 1
    m = sft[0]["messages"]
    assert m[0]["role"] == "user" and m[0]["content"].startswith("Build this scene")
    roles = [x["role"] for x in m]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles  # strictly alternating
    assert roles[-1] == "assistant" and m[-1]["content"][0]["text"] == "Built a 6x6 room with two pallets."
    uses = [b for x in m if x["role"] == "assistant" for b in x["content"] if b["type"] == "tool_use"]
    results = [b for x in m if x["role"] == "user" and isinstance(x["content"], list) for b in x["content"]]
    assert [u["id"] for u in uses] == [r["tool_use_id"] for r in results]  # every call answered, in order
    assert any(c.get("type") == "image_ref" and c["path"].startswith("runs/") for r in results for c in r["content"])
    assert sft[0]["final_spec"] == good.spec and "thinking" not in json.dumps(m)

    st = json.loads(out["stats"].read_text())
    assert st["n_runs"] == 2 and st["n_accepted"] == 1 and st["acceptance_rate"] == 0.5 and st["total_cost_usd"] > 0
    assert len(out["records"].read_text().splitlines()) == 2
    with pytest.raises(ValueError):
        ex.export(root, ["parquet"])


def test_split_is_by_family_and_deterministic():
    assert split_of("indoor|pallet") == split_of("indoor|pallet")
    seen = {split_of(f"indoor|asset_{i}") for i in range(200)}
    assert seen == {"train", "val", "test"}
    assert env_signature({"environment": {"kind": "outdoor"}, "objects": [{"asset": "b"}, {"asset": "a"}, {"asset": {"catalog": "omniverse", "path": "x"}}]}) == "outdoor|a,b"
    assert env_signature(None) == "none"


def test_claude_command_is_isolated_and_capped(tmp_path):
    cmd = claude_command("claude", "hi", tmp_path / "m.json", model="opus", effort="high", max_budget_usd=2.5)
    j = lambda flag: cmd[cmd.index(flag) + 1]  # noqa: E731
    assert j("--tools") == "" and j("--allowedTools") == "mcp__worldc__*" and "--strict-mcp-config" in cmd
    assert j("--max-budget-usd") == "2.5" and j("--model") == "opus" and j("--effort") == "high" and j("--permission-mode") == "dontAsk"
    assert "mcp__worldc" not in j("--append-system-prompt") and "SPEC REFERENCE" in j("--append-system-prompt")


def test_auto_cameras_frame_the_scene():
    from worldc.tools import auto_cameras

    c = auto_cameras({"environment": {"x_range": [-10, 10], "y_range": [-7, 7]}})
    t, o = c["auto_topdown"], c["auto_overview"]
    assert t["position"][:2] == [0.0, 0.0] and t["position"][2] > 15 and t["hide_roof"] and o["position"][2] > 0 and o["look_at"] == [0.0, 0.0, 0.0]
    small = auto_cameras({"environment": {"x_range": [-3, 3], "y_range": [-3, 3]}})
    assert small["auto_topdown"]["position"][2] < t["position"][2]  # framing scales with the scene


def test_final_renders_are_collected_and_shown(root, monkeypatch):
    """With final checks on, every camera + the auto cameras are rendered by Isaac RTX; here the Isaac calls are stubbed."""
    from worldc import tools as T
    from worldc.datagen import runner

    calls = []

    def fake_render(path, camera, width, height, engine):
        name = camera if isinstance(camera, str) else camera["name"]
        calls.append((name, engine))
        png = Path(path) / "renders" / f"rtx_{name}.png"
        png.parent.mkdir(exist_ok=True)
        png.write_bytes(b"\x89PNG fake")
        return {"ok": True, "result": {"image": str(png), "stats": {"mean_luminance": 80}}, "issues": []}

    monkeypatch.setattr(T, "render", fake_render)
    monkeypatch.setattr(T, "check_scene", lambda path, physics, sim_seconds, timeout_s=900.0: {"ok": True, "result": {"static": {"passed": True}, "physics": {"ran": True, "ok": True}}, "issues": []})
    monkeypatch.setenv("FAKE_MODE", "ok")
    rec = runner.generate("two pallets", root=root, final_checks=True, command_override=[sys.executable, str(FAKE)], log=lambda s: None)
    assert rec.accepted and [c for c, _ in calls] == ["auto_topdown", "auto_overview"] and {e for _, e in calls} == {"rtx"}
    assert rec.final_render_paths == [f"runs/{rec.run_id}/final_renders/auto_overview.png", f"runs/{rec.run_id}/final_renders/auto_topdown.png"]
    assert all((root / p).is_file() for p in rec.final_render_paths)
    assert [r["camera"] for r in rec.final_checks["renders"]] == ["auto_topdown", "auto_overview"] and rec.final_checks["check"]["ok"]
    md = (root / "runs" / rec.run_id / "trace.md").read_text()
    assert "## Final Isaac Sim renders" in md and "![auto_overview](final_renders/auto_overview.png)" in md and "Final checks" in md
    assert md.index("Final Isaac Sim renders") > md.index("Built a 6x6 room")  # at the end


@pytest.mark.parametrize("second, accepted, status", [("timeout", False, "physics.timeout"), ("ok", True, "accepted")])
def test_physics_timeout_is_retried_once_then_not_accepted(root, monkeypatch, second, accepted, status):
    """A physics timeout is not a pass: one rerun with twice the timeout; a second timeout makes the run not accepted."""
    from worldc import tools as T

    calls = []

    def fake_check(path, physics, sim_seconds, timeout_s=900.0):
        calls.append(timeout_s)
        if len(calls) == 1 or second == "timeout":
            return {"ok": False, "result": {"static": {"passed": True}, "physics": {"ran": False, "ok": False}},
                    "issues": [{"severity": "error", "code": "physics.timeout", "message": "", "hint": "", "path": ""}]}
        return {"ok": True, "result": {"static": {"passed": True}, "physics": {"ran": True, "ok": True}}, "issues": []}

    monkeypatch.setattr(T, "check_scene", fake_check)
    monkeypatch.setattr(T, "render", lambda *a, **k: {"ok": False, "result": {}, "issues": []})
    monkeypatch.setenv("FAKE_MODE", "ok")
    rec = generate("two pallets", root=root, final_checks=True, command_override=[sys.executable, str(FAKE)], log=lambda s: None)
    assert calls == [900.0, 1800.0]
    assert rec.accepted is accepted and rec.final_status == status and rec.final_checks["physics_status"] == second
    assert rec.final_checks["physics_first_attempt"]["status"] == "timeout"


def test_isaac_timeout_kills_the_whole_process_tree(tmp_path):
    """python.sh is a shell wrapper around Kit's python: a timeout must kill the child too, not leave it on the GPU."""
    import os
    import subprocess
    import time

    from worldc.isaac_env import run_isaac

    pidfile = tmp_path / "child.pid"
    wrapper = f"sleep 60 & echo $! > {pidfile}; wait"  # like python.sh: a shell that runs the real process as a child
    with pytest.raises(subprocess.TimeoutExpired):
        run_isaac(["bash", "-c", wrapper], timeout=1.0)
    pid = int(pidfile.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
