"""Turn the collected records into ready-to-train files (all derived from ``records.jsonl`` + each run's ``trace.jsonl``).

    prompt_spec.jsonl   {id, split, prompt, spec, accepted, env_signature, asset_sources[, issues]}   description -> spec pairs
    sft.jsonl           {id, split, messages, tools, final_spec}   full tool-use conversations (Anthropic message format)
    records.jsonl       the canonical records, one per line
    stats.json          counts, acceptance rate, cost, tokens, by split / environment family

Splits are assigned by environment family (``env_signature``), so a held-out family never appears in train.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .records import Record, load_records, split_of

FORMATS = ("prompt_spec", "sft", "records", "stats")


def _rows_prompt_spec(root: Path, recs: list[Record], include_rejected: bool):
    for r in recs:
        if r.spec is None or not (r.accepted or include_rejected):
            continue
        row = {"id": r.id, "split": r.split or split_of(r.env_signature), "prompt": r.description, "spec": r.spec, "accepted": r.accepted,
               "env_signature": r.env_signature, "asset_sources": r.asset_sources}
        if not r.accepted:
            row["issues"] = (r.validation.get("last_build") or {}).get("issues") or (r.validation.get("last_validate") or {}).get("issues") or []
            row["error"] = r.error
        yield row


def messages_from_trace(root: Path, r: Record) -> list[dict]:
    """The conversation as Anthropic-format messages (thinking dropped, images as path references)."""
    events = [json.loads(ln) for ln in (root / "runs" / r.run_id / "trace.jsonl").read_text().splitlines() if ln.strip()]
    msgs: list[dict] = []
    cur_id, cur, results = None, [], []

    def flush_assistant():
        nonlocal cur, cur_id
        if cur:
            msgs.append({"role": "assistant", "content": cur})
        cur, cur_id = [], None

    def flush_results():
        nonlocal results
        if results:
            msgs.append({"role": "user", "content": results})
        results = []

    for e in events:
        t = e["type"]
        if t == "user":
            msgs.append({"role": "user", "content": e["content"]})
        elif t in ("assistant", "tool_call"):
            if e.get("message_id") != cur_id:
                flush_assistant()
                flush_results()
                cur_id = e.get("message_id")
            cur.append({"type": "text", "text": e["content"]} if t == "assistant" else
                       {"type": "tool_use", "id": e["tool_use_id"], "name": e["tool"], "input": e["input"]})
        elif t == "tool_result":
            flush_assistant()
            content = [{"type": "text", "text": e["stdout"]}] + [{"type": "image_ref", "path": f"runs/{r.run_id}/{p}"} for p in e.get("images", []) if p]
            results.append({"type": "tool_result", "tool_use_id": e["tool_use_id"], "content": content, **({"is_error": True} if e.get("is_error") else {})})
    flush_assistant()
    flush_results()
    return msgs


def _rows_sft(root: Path, recs: list[Record], include_rejected: bool):
    for r in recs:
        if not (r.accepted or include_rejected) or not (root / "runs" / r.run_id / "trace.jsonl").is_file():
            continue
        yield {"id": r.id, "split": r.split or split_of(r.env_signature), "system_prompt_id": r.prompt_id, "accepted": r.accepted,
               "tools": sorted({b["name"] for m in messages_from_trace(root, r) if m["role"] == "assistant" for b in m["content"] if b.get("type") == "tool_use"}),
               "messages": messages_from_trace(root, r), "final_spec": r.spec}


def stats(recs: list[Record]) -> dict:
    acc = [r for r in recs if r.accepted]
    n = max(len(recs), 1)
    return {"n_runs": len(recs), "n_accepted": len(acc), "acceptance_rate": round(len(acc) / n, 3),
            "by_split": dict(Counter(r.split for r in recs)), "accepted_by_split": dict(Counter(r.split for r in acc)),
            "by_stop_reason": dict(Counter(r.stop_reason for r in recs)), "by_env_signature": dict(Counter(r.env_signature for r in recs).most_common(50)),
            "asset_sources": dict(sum((Counter(r.asset_sources) for r in recs), Counter())),
            "total_cost_usd": round(sum(r.usage.est_cost_usd for r in recs), 2), "mean_turns": round(sum(r.usage.turns for r in recs) / n, 1),
            "mean_tool_calls": round(sum(r.usage.tool_calls for r in recs) / n, 1), "mean_repairs": round(sum(r.usage.repairs for r in recs) / n, 1),
            "mean_repairs_accepted": round(sum(r.usage.repairs for r in acc) / max(len(acc), 1), 1),
            "tokens": {k: sum(getattr(r.usage, k) for r in recs) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}}


def export(root: Path, formats=FORMATS, include_rejected: bool = False, out_dir: Path | None = None) -> dict[str, Path]:
    recs = load_records(root)
    for r in recs:
        r.split = r.split or split_of(r.env_signature)
    out = Path(out_dir or root / "exports")
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def dump(name: str, rows):
        p = out / f"{name}.jsonl"
        n = 0
        with p.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
                n += 1
        written[name] = p
        return n

    for fmt in formats:
        if fmt == "prompt_spec":
            dump(fmt, _rows_prompt_spec(root, recs, include_rejected))
        elif fmt == "sft":
            dump(fmt, _rows_sft(root, recs, include_rejected))
        elif fmt == "records":
            dump(fmt, (json.loads(r.model_dump_json()) for r in recs))
        elif fmt == "stats":
            (out / "stats.json").write_text(json.dumps(stats(recs), indent=1))
            written[fmt] = out / "stats.json"
        else:
            raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(FORMATS)}")
    return written
