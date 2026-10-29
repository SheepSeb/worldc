#!/usr/bin/env bash
# E2 within-prompt samples: extra constrained LLM runs with the frozen E1 protocol, run from the e1-freeze-v7 worktree into
# dataset/e2_samples/<model> (their own set, not E1 records).  Each pass runs every listed prompt once.
#
#   scripts/e2_samples.sh P05,P08,P15,P16 4               # 4 passes now
#   E2_WAIT_FOR=P23 scripts/e2_samples.sh P23 4            # wait until E1 has accepted P23, then 4 passes
#
# Env: E2_WORKTREE (default ../worldc-e1-freeze-v7), E2_MODEL (default ollama:deepseek-v4.1-flash:cloud),
# E2_E1_DATASET (default dataset/e1/deepseek-v4.1-flash_cloud), E2_WAIT_FOR, E2_POLL_S (default 300).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT="${E2_WORKTREE:-$REPO/../worldc-e1-freeze-v7}"
MODEL="${E2_MODEL:-ollama:deepseek-v4.1-flash:cloud}"
IDS="${1:?prompt ids, comma-separated}"
PASSES="${2:-4}"

if [ -n "${E2_WAIT_FOR:-}" ]; then
  E1DS="${E2_E1_DATASET:-$REPO/dataset/e1/deepseek-v4.1-flash_cloud}"
  echo "waiting for an accepted constrained E1 run of $E2_WAIT_FOR in $E1DS ($(date))"
  until python3 - "$E2_WAIT_FOR" "$E1DS/records.jsonl" "$REPO/prompts/prompts.csv" <<'PY'
import csv, json, sys
pid, records, prompts = sys.argv[1:]
text = " ".join(next(r for r in csv.DictReader(open(prompts)) if r["id"] == pid)["text_constrained"].split())
try:
    lines = open(records).read().splitlines()
except FileNotFoundError:
    sys.exit(1)
for ln in lines:
    try:
        r = json.loads(ln)
    except ValueError:
        continue  # a record being written
    if r.get("accepted") and " ".join((r.get("description") or "").split()) == text:
        print(f"E1 accepted {pid}: {r['run_id']}")
        sys.exit(0)
sys.exit(1)
PY
  do sleep "${E2_POLL_S:-300}"; done
fi

for k in $(seq 1 "$PASSES"); do
  echo "== pass $k/$PASSES: $IDS ($(date))"
  (cd "$WT" && E1_DATASET="$REPO/dataset/e2_samples" E1_PROMPTS="$IDS" scripts/run_e1.sh "$MODEL")
done
echo "done: $IDS x $PASSES -> $REPO/dataset/e2_samples ($(date))"
