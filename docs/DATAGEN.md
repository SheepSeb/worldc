# Prompt to scene, collected as a dataset

```
worldc generate "A 12x8 m workshop with two work tables on the north wall and a pallet near the door"
worldc generate --prompts-file prompts.txt          # one description per line (# comments allowed), run in sequence
worldc dataset list | stats | show <run_id>
worldc dataset export                               # training files, see below
```

`worldc generate` runs **your Claude Code** headless (`claude -p`, your own login, no API key). Claude Code gets only the `worldc` MCP
tools (built-in tools are off, project settings only), so every run is the same loop an interactive session would do: search
assets, write a spec, `validate_spec`, `build_scene`, `check_scene`, proxy renders, `edit_scene`. A run stops at the spend cap
(`--max-budget-usd`, default 20), `--max-turns` (60) or `--timeout`. Options: `--model`, `--effort`, `--dataset DIR`,
`--no-final-checks`, `--keep-work`. Requires Isaac Sim (the MCP server will not start without it) and `uv`.

After the model finishes, the runner re-checks what was actually built, independent of what the model claimed: the static
checks, an Isaac Sim physics settle, and **Isaac Sim RTX renders of every scene camera plus two auto-framed views**
(`auto_overview`, `auto_topdown`).

## What is collected

```
dataset/                         ($WORLDC_DATASET or --dataset)
  records.jsonl                  one record per run, accepted or not (canonical index, append-only)
  README.md                      layout + schema
  runs/<run_id>/
    trace.md                     readable transcript: prompt, notes, every tool call and result, spec drafts, images inline,
                                 and the final Isaac renders at the end
    trace.jsonl                  the same as events (user, assistant, thinking, tool_call, tool_result, write, final_check,
                                 final_render, result), the vocabulary of data/traces
    specs/spec_v1.json ...       every distinct spec the model validated, built or produced by edit_scene
    spec.json                    the final spec (last one that built)
    scene/                       the final built scene: scene.usd, scene_graph.json, resolved_spec.json (exact replay), check.json
    images/                      renders the model looked at (proxy and RTX)
    final_renders/*.png          the closing Isaac Sim RTX renders, one per camera
    prompt.txt  record.json  mcp.json
  exports/                       from `worldc dataset export`
```

A record (`datagen/records.py`, `record_version` 1) holds: `description` (prompt), `spec`, `accepted` (built, static checks
pass, final physics check passes), `stop_reason`, `error`, `validation` (last validate / build / check, with issues),
`final_checks` (independent re-check + per-camera render stats), `asset_sources` (registry / omniverse / generated / file
counts), `env_signature`, `split`, `scene_dir`, `render_paths`, `final_render_paths`, `summary` (the model's closing note),
`usage` (turns, tool calls, tokens, cost as reported by Claude Code), `generator` (git rev, prompt id), `params`.
Rejected runs are kept with their issues: they are the repair / negative data.

## Training files (`worldc dataset export [--include-rejected] [--format ...]`)

| file | rows |
|---|---|
| `prompt_spec.jsonl` | `{id, split, prompt, spec, accepted, env_signature, asset_sources}`: description to spec pairs |
| `sft.jsonl` | `{id, split, messages, tools, final_spec}`: the full tool-use conversation in Anthropic message format (`text`, `tool_use`, `tool_result` blocks; images as `{"type": "image_ref", "path": ...}`; thinking dropped) |
| `records.jsonl` | the canonical records |
| `stats.json` | counts, acceptance rate, cost, tokens, by split / environment family / stop reason |

Splits are by **environment family** (`env_signature` = environment kind + the dominant registry assets), assigned by a
deterministic hash (80/10/10), so a held-out family never appears in train.

## Notes

* The system prompt (`datagen/prompts.py`, id `spec_writer@1`) includes `docs/SPEC.md` and two example specs; its id is stored in
  every record, so changing the prompt means bumping the version.
* Cost figures come from Claude Code (`total_cost_usd`); on a subscription they are what the usage would cost at API rates.
* A run is not deterministic (the model samples). The *scene* is: `scene/resolved_spec.json` rebuilds it exactly.
* Not built yet: reverse captioning from recipes, preference pairs from multiple samples per prompt, dedup, Langfuse mirror
  (PLAN.md section 7). `--prompts-file` plus several runs per prompt is the way to get multiple samples today.
