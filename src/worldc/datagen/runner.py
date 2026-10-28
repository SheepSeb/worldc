"""Run one prompt through Claude Code (headless) with the worldc MCP server attached, and collect everything as a dataset record.

Uses the user's own Claude Code login (``claude -p``), so no API key is needed.  Claude Code runs with its built-in tools off
and only the ``mcp__worldc__*`` tools allowed; its stream-json output is folded into ``trace.jsonl`` (images saved to files and
referenced by path), and the final scene is re-checked independently of what the model claimed.

    record = generate("a small machine shop with two lathes", root=dataset_root())
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import tools
from . import prompts
from .records import Record, Trace, Usage, append_record, canonical, new_run, now, sha, split_of

MCP_PREFIX = "mcp__worldc__"
MAX_TEXT = 60_000  # tool-result text kept per event in the trace
REPAIR_TOOLS = ("validate_spec", "build_scene")


@dataclass
class _State:
    session_id: str = ""
    model: str = ""
    msg_ids: set = field(default_factory=set)
    pending: dict = field(default_factory=dict)       # tool_use_id -> (name, input)
    candidate_spec: dict | None = None                # last spec validated / produced by edit_scene
    built_spec: dict | None = None                    # spec of the last successful build_scene
    built_dir: str | None = None
    last_validate: dict = field(default_factory=dict)
    last_build: dict = field(default_factory=dict)
    last_check: dict = field(default_factory=dict)
    n_tool_calls: int = 0
    n_repairs: int = 0                                # REPAIR_TOOLS calls that came back with error issues
    n_images: int = 0
    n_specs: int = 0
    final_text: str = ""
    result: dict = field(default_factory=dict)
    mcp_status: dict = field(default_factory=dict)


def mcp_config(path: Path, workspace: Path, seed: int | None = None) -> Path:
    env = {"WORLDC_WORKSPACE": str(workspace), **({"WORLDC_SEED": str(int(seed))} if seed is not None else {})}
    cfg = {"mcpServers": {"worldc": {"command": "uv", "args": ["run", "--directory", str(prompts.REPO), "worldc-mcp"], "env": env}}}
    path.write_text(json.dumps(cfg, indent=1))
    return path


def claude_command(claude_bin: str, user_msg: str, cfg: Path, *, model: str | None, effort: str | None, max_budget_usd: float,
                   extra: tuple = ()) -> list[str]:
    cmd = [claude_bin, "-p", user_msg, "--output-format", "stream-json", "--verbose", "--mcp-config", str(cfg), "--strict-mcp-config",
           "--tools", "", "--allowedTools", f"{MCP_PREFIX}*", "--permission-mode", "dontAsk", "--setting-sources", "project",
           "--disable-slash-commands", "--no-session-persistence", "--max-budget-usd", str(max_budget_usd),
           "--append-system-prompt", prompts.spec_writer_prompt()]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    return cmd + list(extra)


def _short(name: str, inp: dict) -> str:
    if name == "search_assets":
        return repr(inp.get("query"))
    if name in ("render",):
        return f"{inp.get('camera')} ({inp.get('engine', 'rtx')})"
    if name in ("describe_asset",):
        return str(inp.get("id"))
    if name == "edit_scene":
        return f"{len(inp.get('json_patch') or [])} ops"
    return ""


def _jsonable_result(text: str) -> dict | None:
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) and "ok" in d else None
    except ValueError:
        return None


def _spec_from(inp: dict) -> dict | None:
    try:
        return tools._spec_dict(inp.get("spec"))[0]
    except Exception:  # noqa: BLE001
        return None


def _track(st: _State, name: str, inp: dict, res: dict | None):
    """Update run state from one finished tool call."""
    if res is None:
        return
    r, issues = res.get("result", {}), res.get("issues", [])
    if name == "validate_spec":
        st.candidate_spec = _spec_from(inp) or st.candidate_spec
        st.last_validate = {"ok": res["ok"], "stage": r.get("stage"), "placed_instances": r.get("placed_instances"),
                            "instances_by_object": r.get("instances_by_object"), "issues": issues[:30]}
    elif name == "edit_scene" and isinstance(r.get("spec"), dict):
        st.candidate_spec = r["spec"]
    elif name == "build_scene":
        spec = _spec_from(inp)
        st.last_build = {"ok": res["ok"], "out_dir": r.get("out_dir"), "counts": r.get("counts"), "check": r.get("check"),
                         "issues": issues[:30]}
        if spec:
            st.candidate_spec = spec
        if res["ok"] and r.get("out_dir"):
            st.built_spec, st.built_dir = spec or st.candidate_spec, r["out_dir"]
    elif name == "check_scene":
        st.last_check = {"ok": res["ok"], "static": r.get("static"), "physics": r.get("physics"), "issues": issues[:30]}


def _strip_signature(block: dict) -> dict:
    return {k: v for k, v in block.items() if k != "signature"}


def _write_spec(spec: dict, st: _State, trace: Trace, run_dir: Path, why: str):
    """Every distinct spec the model validates/builds/edits is kept as specs/spec_vN.json (like data/traces' spec_v1.yaml...)."""
    text = json.dumps(spec, indent=1)
    d = run_dir / "specs"
    if st.n_specs and (d / f"spec_v{st.n_specs}.json").read_text() == text:
        return
    st.n_specs += 1
    d.mkdir(exist_ok=True)
    p = d / f"spec_v{st.n_specs}.json"
    p.write_text(text)
    trace.log("write", path=f"specs/{p.name}", why=why, content=text)


def _save_image(block: dict, images: Path, st: _State) -> dict:
    src = block.get("source", {})
    if src.get("type") != "base64":
        return {"type": "image_ref", "path": None}
    st.n_images += 1
    ext = (src.get("media_type", "image/png").split("/")[-1]) or "png"
    p = images / f"{st.n_images:03d}.{ext}"
    images.mkdir(exist_ok=True)
    p.write_bytes(base64.b64decode(src["data"]))
    return {"type": "image_ref", "path": f"images/{p.name}"}


def _handle(ev: dict, st: _State, trace: Trace, run_dir: Path, log) -> None:
    t = ev.get("type")
    if t == "system" and ev.get("subtype") == "init":
        st.session_id, st.model = ev.get("session_id", ""), ev.get("model", "")
        st.mcp_status = {s.get("name"): s.get("status") for s in ev.get("mcp_servers", [])}
        trace.log("init", session_id=st.session_id, model=st.model, mcp_servers=st.mcp_status, tools=ev.get("tools"))
    elif t == "assistant":
        m = ev.get("message", {})
        mid = m.get("id")
        st.msg_ids.add(mid)
        first = True
        for b in (_strip_signature(x) for x in m.get("content", [])):
            usage = {"usage": m.get("usage")} if first else {}
            if b.get("type") == "text" and b.get("text", "").strip():
                trace.log("assistant", message_id=mid, content=b["text"], **usage)
                st.final_text, first = b["text"].strip(), False
            elif b.get("type") == "thinking" and b.get("thinking", "").strip():
                trace.log("thinking", message_id=mid, content=b["thinking"], **usage)
                first = False
            elif b.get("type") == "tool_use":
                name, inp = b["name"].removeprefix(MCP_PREFIX), b.get("input", {})
                st.pending[b["id"]] = (name, inp)
                st.n_tool_calls += 1
                trace.log("tool_call", message_id=mid, tool_use_id=b["id"], tool=name, input=inp, **usage)
                first = False
                spec = _spec_from(inp) if name in ("validate_spec", "build_scene", "edit_scene") else None
                if spec is not None:
                    _write_spec(spec, st, trace, run_dir, f"from {name}")
                log(f"  -> {name} {_short(name, inp)}")
    elif t == "user":
        for b in ev.get("message", {}).get("content", []):
            if not isinstance(b, dict) or b.get("type") != "tool_result":
                continue
            name, inp = st.pending.pop(b.get("tool_use_id"), ("?", {}))
            images, text = [], ""
            for c in b.get("content") if isinstance(b.get("content"), list) else [{"type": "text", "text": str(b.get("content"))}]:
                if c.get("type") == "image":
                    images.append(_save_image(c, run_dir / "images", st)["path"])
                else:
                    text += c.get("text", "")
            res = _jsonable_result(text)
            ok = (res["ok"] if res is not None else not b.get("is_error"))
            trace.log("tool_result", tool_use_id=b.get("tool_use_id"), tool=name, ok=bool(ok), exit_code=0 if ok else 1, is_error=bool(b.get("is_error")),
                      stdout=text[:MAX_TEXT], images=images)
            _track(st, name, inp, res)
            if name == "edit_scene" and res and isinstance(res.get("result", {}).get("spec"), dict):
                _write_spec(res["result"]["spec"], st, trace, run_dir, "result of edit_scene")
            if res is not None:
                n_err = sum(1 for i in res.get("issues", []) if i.get("severity") == "error")
                if name in REPAIR_TOOLS and n_err:
                    st.n_repairs += 1
                log(f"     {name}: {'ok' if res['ok'] else 'FAILED'}" + (f", {n_err} error(s)" if n_err else "")
                    + (f" [repair {st.n_repairs}]" if name in REPAIR_TOOLS and n_err else ""))
    elif t == "result":
        st.result = ev
        trace.log("result", **{k: ev.get(k) for k in ("subtype", "is_error", "stop_reason", "num_turns", "total_cost_usd", "usage", "duration_ms",
                                                      "permission_denials", "result")})


def _asset_sources(spec: dict | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for o in (spec or {}).get("objects", []) or []:
        a = o.get("asset")
        k = "registry" if isinstance(a, str) else "omniverse" if isinstance(a, dict) and "catalog" in a else \
            "generated" if isinstance(a, dict) and "generate" in a else "file" if isinstance(a, dict) and "file" in a else "unknown"
        out[k] = out.get(k, 0) + 1
    return out


def env_signature(spec: dict | None) -> str:
    """Family key for held-out splits: environment kind + the dominant registry assets."""
    if not spec:
        return "none"
    kind = (spec.get("environment") or {}).get("kind", "indoor")
    ids = sorted({o["asset"] for o in spec.get("objects", []) or [] if isinstance(o.get("asset"), str)})
    return f"{kind}|" + ",".join(ids[:4])


FREEZE_TAG = os.environ.get("WORLDC_FREEZE_TAG", "e1-freeze-v2")


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(prompts.REPO), *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def freeze_info() -> dict[str, str]:
    """What a run was generated against: the freeze tag's commit, whether the checkout is exactly that commit, and the
    sha256 of the curated asset table (the asset library a model could choose from)."""
    import hashlib

    from ..assets.curated import TABLE

    tagged = _git("rev-parse", "--verify", "--quiet", f"{FREEZE_TAG}^{{commit}}")
    head = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))  # tracked changes or untracked files (gitignored outputs do not count)
    return {"freeze_tag": FREEZE_TAG, "freeze_commit": tagged, "head_commit": head,
            "at_freeze": str(bool(tagged) and tagged == head and not dirty).lower(), "tree_dirty": str(dirty).lower(),
            "curated_table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest()}


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(prompts.REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


PHYSICS_TIMEOUT_S = 900.0


def physics_status(c: dict) -> str:
    """ok | failed | timeout | not_run, from a check_scene result."""
    if any(i.get("code") == "physics.timeout" for i in c.get("issues", [])):
        return "timeout"
    ph = (c.get("result") or {}).get("physics") or {}
    if not ph.get("ran"):
        return "not_run"
    return "ok" if ph.get("ok") else "failed"


def _final_checks(scene: Path, run_dir: Path, trace: Trace, physics: bool, rtx: bool, log) -> dict:
    """Independent verification after the loop: static + Isaac physics check, then Isaac RTX renders of every camera.
    A physics timeout is rerun once with twice the timeout (alone on the GPU: Isaac Sim runs are queued machine-wide); a
    second timeout leaves physics_status = "timeout", which is not accepted."""
    out: dict[str, Any] = {}
    log("  final checks: static" + (" + Isaac physics" if physics else ""))
    c = tools.check_scene(str(scene), physics=physics, sim_seconds=6.0, timeout_s=PHYSICS_TIMEOUT_S)
    status = physics_status(c) if physics else "not_run"
    if status == "timeout":
        log(f"  physics timed out after {PHYSICS_TIMEOUT_S:.0f} s: rerunning once with {2 * PHYSICS_TIMEOUT_S:.0f} s")
        trace.log("final_check", ok=c["ok"], static=c["result"].get("static"), physics=c["result"].get("physics"), issues=c["issues"][:30], attempt=1)
        out["physics_first_attempt"] = {"status": "timeout", "timeout_s": PHYSICS_TIMEOUT_S}
        c = tools.check_scene(str(scene), physics=True, sim_seconds=6.0, timeout_s=2 * PHYSICS_TIMEOUT_S)
        status = physics_status(c)
    out["physics_status"] = status
    out["check"] = {"ok": c["ok"], "static": c["result"].get("static"), "physics": c["result"].get("physics"), "issues": c["issues"][:30]}
    trace.log("final_check", ok=c["ok"], static=c["result"].get("static"), physics=c["result"].get("physics"), issues=c["issues"][:30],
              physics_status=status)
    if rtx:
        g = json.loads((scene / "scene_graph.json").read_text())
        cams: dict[str, Any] = {x["name"]: x["name"] for x in g.get("cameras", []) if x.get("name")}
        cams.update(tools.auto_cameras(g))
        out["renders"] = []
        for name, cam in cams.items():
            log(f"  final render (Isaac RTX): {name}")
            r = tools.render(str(scene), cam, 1280, 720, engine="rtx")
            entry: dict[str, Any] = {"camera": name, "ok": r["ok"], "stats": r["result"].get("stats"), "issues": r["issues"][:3]}
            if r["ok"] and r["result"].get("image"):
                dst = run_dir / "final_renders" / f"{name}.png"
                dst.parent.mkdir(exist_ok=True)
                shutil.copyfile(r["result"]["image"], dst)
                entry["path"] = f"final_renders/{name}.png"
            out["renders"].append(entry)
            trace.log("final_render", **entry)
    return out


def generate(prompt: str, *, root: Path, model: str | None = None, effort: str | None = None, max_budget_usd: float = 20.0,
             max_turns: int = 120, max_repairs: int | None = 10, seed: int | None = None, timeout_s: float = 3600.0, final_checks: bool = True, keep_work: bool = False,
             claude_bin: str = "claude", extra_args: tuple = (), log=lambda s: print(s, file=sys.stderr, flush=True),
             command_override: list[str] | None = None) -> Record:
    run_id, run_dir = new_run(root, prompt)
    (run_dir / "prompt.txt").write_text(prompt)
    work = run_dir / "work"
    work.mkdir()
    trace = Trace(run_dir / "trace.jsonl")
    trace.log("user", content=prompts.user_message(prompt), prompt_id=prompts.SPEC_WRITER_ID, description=prompt)
    log(f"[{run_id}] {prompt[:100]}")
    cfg = mcp_config(run_dir / "mcp.json", work, seed)
    cmd = command_override or claude_command(claude_bin, prompts.user_message(prompt), cfg, model=model, effort=effort,
                                             max_budget_usd=max_budget_usd, extra=extra_args)
    st, stop, err = _State(), "", None
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=run_dir, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=dict(os.environ, WORLDC_WORKSPACE=str(work)))
    except FileNotFoundError:
        err = f"'{claude_bin}' not found. Install Claude Code and log in (https://claude.com/claude-code), or pass --claude-bin."
        proc = None
    killed = {"why": ""}
    if proc is not None:
        timer = threading.Timer(timeout_s, lambda: (killed.update(why="timeout"), proc.kill()))
        timer.start()
        stderr_lines: list[str] = []
        threading.Thread(target=lambda: stderr_lines.extend(proc.stderr), daemon=True).start()
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                _handle(ev, st, trace, run_dir, log)
                if st.mcp_status and st.mcp_status.get("worldc") not in (None, "connected") and not st.msg_ids:
                    killed["why"] = "mcp_failed"
                    proc.kill()
                    break
                if len(st.msg_ids) > max_turns:
                    killed["why"] = "max_turns"
                    proc.kill()
                    break
                if max_repairs is not None and st.n_repairs > max_repairs:
                    killed["why"] = "max_repairs"
                    proc.kill()
                    break
        finally:
            timer.cancel()
            proc.wait()
        if killed["why"] == "mcp_failed":
            err = f"MCP server 'worldc' failed to start (status {st.mcp_status.get('worldc')}). Run `uv run worldc-mcp` to see why."
        elif killed["why"]:
            stop = killed["why"]
        elif not st.result:
            err = ("claude exited without a result: " + "".join(stderr_lines)[-600:]).strip()
    res = st.result
    if not stop:
        stop = (res.get("subtype") if res.get("is_error") else res.get("stop_reason")) or ("error" if err else "")
    if res.get("is_error") and not err:
        err = str(res.get("result", ""))[:600]

    # ---- usage from Claude Code's own accounting
    u = res.get("usage", {}) or {}
    usage = Usage(turns=res.get("num_turns") or len(st.msg_ids), tool_calls=st.n_tool_calls, repairs=st.n_repairs, input_tokens=u.get("input_tokens", 0),
                  output_tokens=u.get("output_tokens", 0), cache_read_input_tokens=u.get("cache_read_input_tokens", 0),
                  cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0), est_cost_usd=round(float(res.get("total_cost_usd") or 0), 5))

    # ---- keep the final scene, drop scratch builds
    scene = None
    if st.built_dir and Path(st.built_dir).is_dir():
        scene = run_dir / "scene"
        shutil.move(st.built_dir, scene)
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)

    loop_s = time.time() - t0
    # ---- independent verification of what was actually built
    final: dict[str, Any] = {}
    if scene is not None and final_checks:
        try:
            final = _final_checks(scene, run_dir, trace, physics=True, rtx=True, log=log)
        except Exception as e:  # noqa: BLE001
            final = {"error": f"{type(e).__name__}: {e}"}
    spec = st.built_spec or st.candidate_spec
    chk = final.get("check")
    static_ok = bool((st.last_build.get("check") or {}).get("passed")) if not chk else bool((chk.get("static") or {}).get("passed"))
    phys = final.get("physics_status", "not_run")  # not_run: final checks off, or Isaac Sim unavailable
    physics_ok = phys in ("ok", "not_run")
    accepted = bool(scene is not None and static_ok and physics_ok and not err)
    final_status = ("error" if err else "no_scene" if scene is None else "static_failed" if not static_ok else
                    "physics.timeout" if phys == "timeout" else "physics_failed" if phys == "failed" else "accepted")
    if spec is not None:
        (run_dir / "spec.json").write_text(json.dumps(spec, indent=1))
    root_ = root
    rec = Record(
        id=sha(canonical(spec) + prompt), run_id=run_id, created_at=now(), model=st.model or model or "", prompt_id=prompts.SPEC_WRITER_ID,
        params={"effort": effort, "max_turns": max_turns, "max_repairs": max_repairs, "seed": seed, "loop_seconds": round(loop_s, 1),
                "final_checks_seconds": round(time.time() - t0 - loop_s, 1), "max_budget_usd": max_budget_usd, "timeout_s": timeout_s, "final_checks": final_checks,
                "session_id": st.session_id},
        description=prompt, spec=spec, spec_hash=sha(canonical(spec)) if spec else None, accepted=accepted, final_status=final_status,
        stop_reason=stop or "unknown", error=err,
        validation={"last_validate": st.last_validate, "last_build": st.last_build, "last_check": st.last_check},
        final_checks=final, asset_sources=_asset_sources(spec), env_signature=env_signature(spec),
        scene_dir=str(scene.relative_to(root_)) if scene else None,
        final_render_paths=sorted(f"runs/{run_id}/{x['path']}" for x in final.get("renders", []) if x.get("path")),
        render_paths=sorted(str(p.relative_to(root_)) for p in ([*scene.glob("renders/*.png")] if scene else []) + list((run_dir / "images").glob("*"))),
        summary=st.final_text, usage=usage, n_events=trace.n,
        generator={"git": _git_rev(), "prompt": prompts.SPEC_WRITER_ID, "claude_code": "headless", **freeze_info()},
    )
    rec.split = split_of(rec.env_signature)
    append_record(root, rec, run_dir)
    from .view import write_markdown

    write_markdown(run_dir, rec)
    log(f"[{run_id}] {'ACCEPTED' if accepted else 'REJECTED'} stop={rec.stop_reason} turns={usage.turns} tools={usage.tool_calls} repairs={usage.repairs} "
        f"cost=${usage.est_cost_usd:.2f} ({time.time() - t0:.0f}s)" + (f"  error: {err}" if err else ""))
    return rec
