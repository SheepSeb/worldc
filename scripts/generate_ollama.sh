#!/usr/bin/env bash
# Run `worldc generate` with Claude Code talking to an Ollama model instead of Anthropic.
#
#   scripts/generate_ollama.sh "a small machine shop with two lathes"
#   OLLAMA_MODEL=qwen3-coder:30b scripts/generate_ollama.sh --prompts-file prompts.txt
#
# Any extra arguments go to `worldc generate`. Env: OLLAMA_MODEL (default below), OLLAMA_HOST (default 127.0.0.1:11434),
# OLLAMA_CONTEXT_LENGTH (default 32768; Ollama's small default context silently truncates the spec schema).
# Note: :cloud models run on Ollama's servers, so prompts, specs and renders leave this machine.
set -euo pipefail

MODEL="${OLLAMA_MODEL:-deepseek-v4.1-flash:cloud}"
HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
HOST="${HOST#http://}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v ollama >/dev/null || { echo "ollama not found" >&2; exit 1; }
command -v claude >/dev/null || { echo "claude (Claude Code) not found" >&2; exit 1; }

if ! curl -sf "http://$HOST/api/version" >/dev/null; then
  echo "starting ollama serve..." >&2
  OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}" nohup ollama serve >"${TMPDIR:-/tmp}/ollama-serve.log" 2>&1 &
  for _ in $(seq 30); do curl -sf "http://$HOST/api/version" >/dev/null && break; sleep 1; done
  curl -sf "http://$HOST/api/version" >/dev/null || { echo "ollama did not start (see ${TMPDIR:-/tmp}/ollama-serve.log)" >&2; exit 1; }
fi

# `ollama show` also fails for a :cloud model that isn't signed in / pulled yet.
ollama show "$MODEL" >/dev/null 2>&1 || ollama pull "$MODEL" || {
  echo "cannot find model '$MODEL' (for :cloud models run 'ollama signin')" >&2; exit 1; }

export ANTHROPIC_BASE_URL="http://$HOST"
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_API_KEY=""
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# One dataset folder per model: dataset/<model>/ (e.g. dataset/nemotron-3.5-bolt_e4b). An explicit --dataset still wins.
export WORLDC_DATASET="${WORLDC_DATASET:-$REPO/dataset/${MODEL//[:\/]/_}}"
echo "dataset: $WORLDC_DATASET" >&2

cd "$REPO"
exec uv run worldc generate --model "$MODEL" "$@"
