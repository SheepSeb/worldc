#!/usr/bin/env bash
# E1: LLM scene generation over the frozen prompt set, one dataset folder per model.
#
#   scripts/run_e1.sh claude-opus-5-5 ollama:qwen3:14b ollama:deepseek-v4.1-flash:cloud
#   E1_PROMPTS=P01,P18,P24 E1_VARIANTS=constrained scripts/run_e1.sh ollama:deepseek-v4.1-flash:cloud   # a subset (pilot)
#
# Models are arguments (not edited into this file), so choosing them never dirties the frozen checkout:
#   <claude model id>        run through your Claude Code login (e.g. claude-opus-5-5)
#   ollama:<ollama model>    run through scripts/generate_ollama.sh (Claude Code against Ollama's Anthropic API)
# Env: E1_PROMPTS (comma-separated ids, default all), E1_VARIANTS (default "plain constrained"), E1_DATASET (default
# dataset/e1), E1_ALLOW_UNFROZEN=1 to run off the e1-freeze tag (records then say at_freeze=false).
set -euo pipefail

# ---- frozen protocol: identical for every model (the runner records all of it per run) ------------------------------
EFFORT=high          # Claude Code effort (ignored by non-Anthropic models)
MAX_TURNS=60
MAX_BUDGET_USD=20    # Claude Code's spend cap (Anthropic pricing; not meaningful for Ollama models)
SEED=0               # every build_scene of a run uses this seed, whatever the model passes
MAX_REPAIRS=10       # stop after 10 validate_spec / build_scene calls that returned errors
TIMEOUT_S=3600
FREEZE_TAG=e1-freeze
# ----------------------------------------------------------------------------------------------------------------------

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 2; }

head_commit="$(git rev-parse HEAD)"
tag_commit="$(git rev-parse --verify --quiet "$FREEZE_TAG^{commit}" || true)"
dirty="$(git status --porcelain --untracked-files=no)"
if [ "${E1_ALLOW_UNFROZEN:-0}" != 1 ] && { [ "$head_commit" != "$tag_commit" ] || [ -n "$dirty" ]; }; then
  echo "not at $FREEZE_TAG (HEAD $head_commit, tag ${tag_commit:-missing}${dirty:+, tracked files modified})." >&2
  echo "check out the tag with a clean tree, or set E1_ALLOW_UNFROZEN=1 for a non-frozen trial." >&2
  exit 1
fi

# The curated SimReady assets: download + collision import once into the cache (the frozen table is not rewritten).
echo "warming the curated asset cache..." >&2
uv run --extra mesh python -m worldc.assets.curated warm >/dev/null

DATASET_ROOT="${E1_DATASET:-$REPO/dataset/e1}"
VARIANTS="${E1_VARIANTS:-plain constrained}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

prompts_file() {  # $1 = variant -> one prompt per line (selected ids only)
  uv run python - "$1" "${E1_PROMPTS:-}" <<'PY'
import csv, sys
variant, ids = sys.argv[1], [i for i in sys.argv[2].split(",") if i]
for r in csv.DictReader(open("prompts/prompts.csv")):
    if not ids or r["id"] in ids:
        print(r[f"text_{variant}"])
PY
}

common=(--max-turns "$MAX_TURNS" --max-budget-usd "$MAX_BUDGET_USD" --seed "$SEED" --max-repairs "$MAX_REPAIRS"
        --timeout "$TIMEOUT_S" --effort "$EFFORT")
for model in "$@"; do
  for variant in $VARIANTS; do
    pf="$WORK/${variant}.txt"
    prompts_file "$variant" >"$pf"
    [ -s "$pf" ] || { echo "no prompts selected (E1_PROMPTS=${E1_PROMPTS:-})" >&2; exit 1; }
    name="${model#ollama:}"
    ds="$DATASET_ROOT/${name//[:\/]/_}"
    echo "== $model, $variant: $(wc -l <"$pf") prompts -> $ds" >&2
    if [[ "$model" == ollama:* ]]; then
      OLLAMA_MODEL="$name" WORLDC_DATASET="$ds" scripts/generate_ollama.sh --prompts-file "$pf" "${common[@]}" || true
    else
      uv run worldc generate --model "$model" --prompts-file "$pf" --dataset "$ds" "${common[@]}" || true
    fi
  done
done
echo "done: $DATASET_ROOT" >&2
