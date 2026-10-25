# Running `worldc generate` on Ollama models

`worldc generate` drives the scene tools through Claude Code (`claude -p`). Ollama exposes an Anthropic-compatible API, so
Claude Code can talk to an Ollama model instead of Anthropic: the runner, the MCP server and the dataset format stay the same.

```
scripts/generate_ollama.sh "a room with one table and two chairs"
OLLAMA_MODEL=nemotron-3.5-bolt-32k scripts/generate_ollama.sh --prompts-file prompts.txt
```

Extra arguments go to `worldc generate` (`--timeout`, `--max-turns`, `--dataset`, ...).

## Requirements

* `ollama` (tested with 0.23.4) and Claude Code (`claude`) on the PATH, plus everything `worldc-mcp` needs ([MCP.md](MCP.md)).
* A model that supports tool calling: `ollama show <model>` must list `tools` under capabilities.
* `:cloud` models need `ollama signin`. They run on Ollama's servers, so prompts, specs and renders leave the machine.

## What the script sets up

| setting | value | why |
|---|---|---|
| `OLLAMA_MODEL` | `deepseek-v4.1-flash:cloud` (default) | model passed to `--model` |
| `OLLAMA_HOST` | `127.0.0.1:11434` | where Ollama listens |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | only applied if the script has to start `ollama serve` itself |
| `ANTHROPIC_BASE_URL` | `http://$OLLAMA_HOST` | sends Claude Code's API calls to Ollama |
| `ANTHROPIC_AUTH_TOKEN` | `ollama` | any non-empty value |
| `ANTHROPIC_API_KEY` | empty | so Claude Code does not use an Anthropic key |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `1` | no telemetry/update calls |
| `WORLDC_DATASET` | `dataset/<model>/` | one dataset folder per model (`:` and `/` become `_`); an explicit `--dataset` or a `WORLDC_DATASET` you set wins |

The script also starts `ollama serve` if nothing answers, and pulls the model if it is missing.

## Context length (the one that bites)

Ollama's default context is 4096 tokens. Claude Code's system prompt, the spec-writer prompt and the 14 tool definitions are
about 9k tokens, so the prompt is silently truncated and the model never sees the task. Symptom: the model says "the user has
not provided a request" and the run ends after 2 turns with `accepted: false`.

`ollama ps` shows the context actually in use. It must read 16k or more while a run is going.

If Ollama runs as a system service, `OLLAMA_CONTEXT_LENGTH` set by the script has no effect. Use one of:

**Derived model (no sudo, recommended).** Bake the context into a copy of the model; the weights are shared.

```
printf 'FROM nemotron-3.5-bolt:e4b\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create nemotron-3.5-bolt-32k -f Modelfile
OLLAMA_MODEL=nemotron-3.5-bolt-32k scripts/generate_ollama.sh "..."
```

**Service environment.**

```
sudo systemctl edit ollama        # add:  [Service]
                                  #       Environment="OLLAMA_CONTEXT_LENGTH=32768"
sudo systemctl restart ollama
```

A larger context uses more VRAM (the 1 GB e4b model took 1.2 GB at 32k). Isaac Sim needs the GPU too.

## Dataset layout

Runs are saved under `dataset/<model>/` (`records.jsonl`, `runs/`, `exports/`), and each record stores the model name. The
other commands read `./dataset` unless told otherwise:

```
worldc dataset stats --dataset dataset/nemotron-3.5-bolt-32k
WORLDC_DATASET=dataset/nemotron-3.5-bolt-32k worldc dataset list
```

A run that ends without a validated scene is stored as rejected and left out of exports unless `--include-rejected` is given.

## Caveats

* `--max-budget-usd` and the `est_cost_usd` in records assume Anthropic pricing; they are not meaningful for other models.
* `--effort` may be ignored by non-Anthropic models.
* `render` returns an image; a model without vision cannot use it.
* `spec_writer_prompt()` was written for Claude. Small models need many more `validate_spec` / `edit_scene` rounds and may hit
  `--max-turns`. Start with a one-room prompt.
* Quantisation lowers tool-call accuracy; prefer Q8/Q6 over Q4 where VRAM allows.
* The environment variables above follow Ollama's Anthropic-compatible endpoint; compare with `ollama launch claude` if your
  version expects others (for example model-alias variables).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| model asks what you want to build, 2 turns, rejected | context too small, see above |
| `claude: not found` / `--claude-bin` | install Claude Code or pass `--claude-bin` |
| `cannot find model ...` | `ollama pull <model>`, or `ollama signin` for `:cloud` |
| first tool call fails, no tool use at all | model has no `tools` capability |
| auth / 404 from the API | check `ANTHROPIC_BASE_URL` has no `/v1` suffix and Ollama is up (`curl $HOST/api/version`) |
| run dir has `trace.md` / `trace.jsonl` | read these first: they show every tool call and where the model got stuck |
