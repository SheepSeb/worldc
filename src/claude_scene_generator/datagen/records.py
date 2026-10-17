"""Dataset layout, record schema and the append-only index.

    dataset/
      README.md                  layout + schema (written once)
      records.jsonl              one ``Record`` per run (canonical index; append-only)
      runs/<run_id>/
        prompt.txt  spec.json  record.json  trace.jsonl
        scene/                   the final built scene (scene.usd, scene_graph.json, resolved_spec.json, check.json, renders/)
        images/                  renders the model looked at
      exports/                   files written by ``csg dataset export``

``records.jsonl`` is the source of truth; exports are derived and can always be regenerated.  Rejected runs are recorded too
(``accepted: false`` with the issues), which is what repair / negative training needs.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

RECORD_VERSION = 1

# USD per million tokens: (input, output, cache read, cache write).  Estimates for budgeting, not billing.
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0), "claude-opus-5": (5.0, 25.0, 0.25, 6.25),
          "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5), "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125)}

README = """# Scene dataset

Each run of `csg generate` adds one record to `records.jsonl` and one folder under `runs/`.

* `records.jsonl`: one JSON object per run (schema: `claude_scene_generator.datagen.records.Record`, `record_version` {v}).
  Key fields: `description` (the prompt), `spec` (the final scene spec as the model wrote it), `accepted` (built and all checks
  passed), `validation` (issues, check summary), `usage`, `env_signature`, `scene_dir`, `render_paths`.
* `runs/<run_id>/trace.jsonl`: every event of the conversation (`user`, `assistant` with text/tool calls, `tool_result` exactly as
  the model saw it, images by path), with token usage per turn.
* `runs/<run_id>/scene/`: the built scene. `spec.json` is the final spec; `scene/resolved_spec.json` replays it exactly.
* `exports/`: derived training files from `csg dataset export` (`prompt_spec.jsonl`, `sft.jsonl`, `stats.json`).

Paths in records are relative to this folder.
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def dataset_root(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("CSG_DATASET", "dataset")).expanduser().resolve()


class Usage(BaseModel):
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    est_cost_usd: float = 0.0

    def add(self, u: Any, model: str):
        g = lambda k: int(getattr(u, k, 0) or 0)  # noqa: E731
        self.turns += 1
        self.input_tokens += g("input_tokens")
        self.output_tokens += g("output_tokens")
        self.cache_read_input_tokens += g("cache_read_input_tokens")
        self.cache_creation_input_tokens += g("cache_creation_input_tokens")
        pi, po, pr, pw = PRICES.get(model, PRICES["claude-opus-5-5"])
        self.est_cost_usd = round(self.est_cost_usd + (g("input_tokens") * pi + g("output_tokens") * po + g("cache_read_input_tokens") * pr
                                                        + g("cache_creation_input_tokens") * pw) / 1e6, 5)


class Record(BaseModel):
    record_version: int = RECORD_VERSION
    id: str                                  # sha256 of canonical(spec) + description
    kind: str = "trajectory"
    run_id: str
    created_at: str
    model: str
    prompt_id: str                           # prompt template id@version
    params: dict[str, Any] = Field(default_factory=dict)   # effort, max_turns, seed, max_cost_usd ...
    description: str                         # the user's prompt
    spec: dict | None = None                 # final spec as written (last one that built; else last one validated)
    spec_hash: str | None = None
    accepted: bool = False                   # built, static checks pass, and (if run) the final physics check passed
    stop_reason: str = ""                    # end_turn | max_turns | max_cost | refusal | max_tokens | error
    error: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)  # last validate_spec / build / check summaries and issues
    final_checks: dict[str, Any] = Field(default_factory=dict)  # independent re-check after the loop (static, physics, rtx)
    asset_sources: dict[str, int] = Field(default_factory=dict)  # registry / omniverse / generated / file -> object count
    env_signature: str = ""                  # family key for held-out splits
    scene_dir: str | None = None
    render_paths: list[str] = Field(default_factory=list)            # every image of the run (scene renders + what the model looked at)
    final_render_paths: list[str] = Field(default_factory=list)      # the closing Isaac Sim RTX renders, one per camera
    summary: str = ""                        # the model's closing message
    usage: Usage = Field(default_factory=Usage)
    n_events: int = 0
    generator: dict[str, str] = Field(default_factory=dict)
    split: str | None = None


def new_run(root: Path, prompt: str) -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + sha(prompt, 8)
    d = root / "runs" / run_id
    d.mkdir(parents=True, exist_ok=False)
    if not (root / "README.md").exists():
        (root / "README.md").write_text(README.replace("{v}", str(RECORD_VERSION)))
    return run_id, d


class Trace:
    """Append-only JSONL event log for one run."""

    def __init__(self, path: Path):
        self.path, self.n = path, 0

    def log(self, type_: str, **data):
        ev = {"seq": self.n, "ts": now(), "type": type_, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        self.n += 1


def append_record(root: Path, rec: Record, run_dir: Path):
    (run_dir / "record.json").write_text(rec.model_dump_json(indent=1))
    with (root / "records.jsonl").open("a") as f:
        f.write(rec.model_dump_json() + "\n")


def load_records(root: Path) -> list[Record]:
    f = root / "records.jsonl"
    if not f.is_file():
        return []
    return [Record.model_validate_json(ln) for ln in f.read_text().splitlines() if ln.strip()]


def split_of(env_signature: str, ratios=(0.8, 0.1, 0.1)) -> str:
    """Deterministic train/val/test by environment family (all runs of a family land in the same split)."""
    x = int(sha(env_signature or "none", 8), 16) / 0xFFFFFFFF
    return "train" if x < ratios[0] else "val" if x < ratios[0] + ratios[1] else "test"
