"""Readable transcript of a run (``trace.md``): prompt, the model's notes, every tool call and result, spec drafts, images inline.

    worldc dataset show <run_id>          print it
"""
from __future__ import annotations

import json
from pathlib import Path

MAX_RESULT_LINES = 14


def _fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def _result_summary(r: dict) -> str:
    """Compact view of a worldc tool result: verdict, issues, then the interesting result fields."""
    try:
        d = json.loads(r["stdout"])
    except (ValueError, KeyError):
        return _fence(r.get("stdout", "")[:1500])
    lines = [f"**{'ok' if d.get('ok') else 'FAILED'}**"]
    for i in d.get("issues", [])[:12]:
        lines.append(f"- [{i['severity']}] `{i['code']}` {('at `' + i['path'] + '`') if i.get('path') else ''}: {i['message']}" + (f"  \n  hint: {i['hint']}" if i.get("hint") else ""))
    res = {k: v for k, v in d.get("result", {}).items() if k not in ("schema", "assets", "matches", "spec", "files", "image")}
    if res:
        body = json.dumps(res, indent=1, default=str).splitlines()
        lines.append(_fence("\n".join(body[:MAX_RESULT_LINES] + (["..."] if len(body) > MAX_RESULT_LINES else [])), "json"))
    if d.get("result", {}).get("matches"):
        lines.append("matches: " + ", ".join(f"`{m.get('id') or m.get('path')}` ({m['score']})" for m in d["result"]["matches"][:6]))
    return "\n".join(lines)


def to_markdown(run_dir: Path, rec=None) -> str:
    ev = [json.loads(ln) for ln in (run_dir / "trace.jsonl").read_text().splitlines() if ln.strip()]
    out = [f"# Run {run_dir.name}", ""]
    seen_final = False
    if rec is not None:
        out += [f"- **{'ACCEPTED' if rec.accepted else 'REJECTED'}**, stop reason `{rec.stop_reason}`, model `{rec.model}`, "
                f"{rec.usage.turns} turns, {rec.usage.tool_calls} tool calls, est. cost ${rec.usage.est_cost_usd:.2f}",
                f"- environment family `{rec.env_signature}`, split `{rec.split}`, assets {rec.asset_sources}"]
        if rec.scene_dir:
            out.append(f"- scene: `{rec.scene_dir}/`")
        if rec.error:
            out.append(f"- error: {rec.error}")
        out.append("")
    for e in ev:
        t = e["type"]
        if t == "user":
            out += ["## Prompt", "", e.get("description") or e["content"], ""]
        elif t == "assistant":
            out += [f"### Assistant", "", e["content"].strip(), ""]
        elif t == "thinking":
            out += ["<details><summary>thinking</summary>", "", e["content"].strip(), "", "</details>", ""]
        elif t == "write":
            out += [f"### Spec draft `{e['path']}` ({e.get('why', '')})", "", f"[open]({e['path']})", ""]
        elif t == "tool_call":
            inp = {k: v for k, v in e["input"].items() if k != "spec"}
            has_spec = "spec" in e["input"]
            out += [f"### Tool call: `{e['tool']}`" + (" (spec inline, see draft above)" if has_spec else ""), ""]
            if inp:
                out += [_fence(json.dumps(inp, indent=1, default=str)[:3000], "json"), ""]
        elif t == "tool_result":
            out += [_result_summary(e), ""]
            out += [f"![{p}]({p})" for p in e.get("images", []) if p] + ([""] if e.get("images") else [])
        elif t == "final_check":
            out += ["## Final checks (run after the model finished, independent of what it claimed)", "",
                    f"**{'ok' if e['ok'] else 'FAILED'}**: static `{(e.get('static') or {}).get('passed')}`, physics `{(e.get('physics') or {}).get('ok', 'not run')}`", ""]
        elif t == "final_render":
            if not seen_final:
                out += ["## Final Isaac Sim renders", ""]
                seen_final = True
            out += [f"### {e['camera']}", ""] + ([f"![{e['camera']}]({e['path']})", ""] if e.get("path") else [f"render failed: {e.get('issues')}", ""])
        elif t == "result":
            out += ["---", f"Finished: `{e.get('stop_reason')}` after {e.get('num_turns')} turns, ${float(e.get('total_cost_usd') or 0):.2f}", ""]
    return "\n".join(out)


def write_markdown(run_dir: Path, rec=None) -> Path:
    p = run_dir / "trace.md"
    p.write_text(to_markdown(run_dir, rec))
    return p
