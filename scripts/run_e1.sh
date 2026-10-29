#!/usr/bin/env bash
# E1: LLM scene generation over the frozen prompt set, one dataset folder per model.
#
#   scripts/run_e1.sh claude-opus-5-5 ollama:qwen3:14b ollama:deepseek-v4.1-flash:cloud            # pass 1: constrained
#   scripts/run_e1.sh --plain claude-opus-5-5 ollama:qwen3:14b ollama:deepseek-v4.1-flash:cloud    # optional pass 2: plain
#   E1_PROMPTS=P01,P18,P24 scripts/run_e1.sh ollama:deepseek-v4.1-flash:cloud                     # a subset (pilot)
#
# Models are arguments (not edited into this file), so choosing them never dirties the frozen checkout:
#   <claude model id>        run through your Claude Code login (e.g. claude-opus-5-5)
#   ollama:<ollama model>    run through scripts/generate_ollama.sh (Claude Code against Ollama's Anthropic API)
# The models' generation loops run concurrently (E1_SEQUENTIAL=1 for one at a time); Isaac Sim work (physics checks, RTX
# renders) is queued machine-wide, one Isaac Sim on the GPU at a time, and a subprocess's timeout counts from its turn.
# Env: E1_PROMPTS (comma-separated ids, default all), E1_DATASET (default dataset/e1), E1_SEQUENTIAL.
set -euo pipefail

# ---- frozen protocol: identical for every model (the runner records all of it per run) ------------------------------
EFFORT=high          # Claude Code effort (ignored by non-Anthropic models)
MAX_TURNS=120        # generous: the repair cap below is the binding limit
MAX_BUDGET_USD=20    # Claude Code's spend cap (Anthropic pricing; not meaningful for Ollama models)
SEED=0               # every build_scene of a run uses this seed, whatever the model passes
MAX_REPAIRS=10       # stop after 10 validate_spec / build_scene calls that returned errors
TIMEOUT_S=3600
FREEZE_TAG=e1-freeze-v4
# ----------------------------------------------------------------------------------------------------------------------

VARIANT=constrained
if [ "${1:-}" = "--plain" ]; then VARIANT=plain; shift; fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 2; }

head_commit="$(git rev-parse HEAD)"
tag_commit="$(git rev-parse --verify --quiet "$FREEZE_TAG^{commit}" || true)"
dirty="$(git status --porcelain)"
if [ "$head_commit" != "$tag_commit" ] || [ -n "$dirty" ]; then
  echo "refusing: not at $FREEZE_TAG (HEAD ${head_commit:0:12}, tag ${tag_commit:0:12}${dirty:+, the tree is dirty})." >&2
  echo "use a clean worktree of the tag:  git worktree add --detach ../worldc-$FREEZE_TAG $FREEZE_TAG" >&2
  exit 1
fi

# The curated SimReady assets: download + collision import once into the cache (the frozen table is not rewritten).
echo "warming the curated asset cache..." >&2
uv run --extra mesh python -m worldc.assets.curated warm >/dev/null

DATASET_ROOT="${E1_DATASET:-$REPO/dataset/e1}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
uv run python - "$VARIANT" "${E1_PROMPTS:-}" >"$WORK/prompts.txt" <<'PY'
import csv, sys
variant, ids = sys.argv[1], [i for i in sys.argv[2].split(",") if i]
for r in csv.DictReader(open("prompts/prompts.csv")):
    if not ids or r["id"] in ids:
        print(r[f"text_{variant}"])
PY
[ -s "$WORK/prompts.txt" ] || { echo "no prompts selected (E1_PROMPTS=${E1_PROMPTS:-})" >&2; exit 1; }

common=(--prompts-file "$WORK/prompts.txt" --max-turns "$MAX_TURNS" --max-budget-usd "$MAX_BUDGET_USD" --seed "$SEED"
        --max-repairs "$MAX_REPAIRS" --timeout "$TIMEOUT_S" --effort "$EFFORT")
run_model() {
  local model="$1" name="${1#ollama:}"
  local ds="$DATASET_ROOT/${name//[:\/]/_}"
  echo "== $model, $VARIANT: $(wc -l <"$WORK/prompts.txt") prompts -> $ds" >&2
  if [[ "$model" == ollama:* ]]; then
    OLLAMA_MODEL="$name" WORLDC_DATASET="$ds" scripts/generate_ollama.sh "${common[@]}" >"$WORK/${name//[:\/]/_}.log" 2>&1 || true
  else
    uv run worldc generate --model "$model" --dataset "$ds" "${common[@]}" >"$WORK/${name//[:\/]/_}.log" 2>&1 || true
  fi
  tail -n 3 "$WORK/${name//[:\/]/_}.log" >&2
}
for model in "$@"; do
  if [ "${E1_SEQUENTIAL:-0}" = 1 ]; then run_model "$model"; else run_model "$model" & fi
done
wait
echo "done ($VARIANT): $DATASET_ROOT" >&2
