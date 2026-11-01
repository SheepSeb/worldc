"""E2 within-prompt recipe sampling until enough samples pass physics.

    uv run python scripts/e2_recipe_sampling.py                       # recipe v3 and v4, prompts P05 P08 P15 P16 P23
    uv run python scripts/e2_recipe_sampling.py --methods v4 --ids P15

Per method and prompt: sample k = 0, 1, 2, ... (seed base + k, the +1000 retry scheme of ``e2_recipe_within.build_sample`` on a failed
rule / build / static check), and every statically accepted scene gets the Isaac Sim physics check of ``e2_physics_queue`` (same rule as
the E1 final check).  Stop when TARGET samples have passed physics or after MAX_ATTEMPTS samples, whichever comes first.  Every sample
counts as one attempt (a sample that never builds statically counts too), and the static build retries are counted separately.
Resumable: samples already built / physics-checked are reused.  Writes ``<root>/sampling_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_physics_queue as Q  # noqa: E402
import e2_recipe_within as W  # noqa: E402
from worldc.recipes.remanufacturing import provenance  # noqa: E402

ROOTS = {"v3": ("out/e2_within/recipe", False), "v4": ("out/e2_within/recipe_v4", True)}


def sample_prompt(pid: str, root: Path, rules: bool, target: int, max_attempts: int, prov, log) -> dict:
    passes, rows = [], []
    for k in range(max_attempts):
        d = root / f"{pid}_seed{k}"
        rec = json.loads((d / "record.json").read_text()) if (d / "record.json").is_file() else W.build_sample(pid, k, root, rules, prov, lambda m: None)
        status = "static_failed"
        if rec["accepted"]:
            f = d / "e2_physics.json"
            if not f.is_file():
                f.write_text(json.dumps(Q.check(d), indent=1))
            status = json.loads(f.read_text())["physics_status"]
        rows.append({"sample": k, "static_ok": rec["accepted"], "static_attempts": rec["attempts"], "final_seed": rec["final_seed"], "physics": status})
        log(f"{root.name}/{pid} sample {k}: static {'ok' if rec['accepted'] else 'FAILED'} ({rec['attempts']} builds), physics {status}")
        if status == "ok":
            passes.append(k)
        if len(passes) >= target:
            break
    return {"prompt": pid, "attempts": len(rows), "passes": passes, "reached_target": len(passes) >= target, "static_builds": sum(r["static_attempts"] for r in rows),
            "physics_checked": sum(r["static_ok"] for r in rows), "samples": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="v3,v4")
    ap.add_argument("--ids", default="P05,P08,P15,P16,P23")
    ap.add_argument("--target", type=int, default=5)
    ap.add_argument("--max-attempts", type=int, default=20)
    a = ap.parse_args()
    os.nice(19)
    prov = provenance()
    log = lambda m: print(time.strftime("%F %T"), m, flush=True)  # noqa: E731
    for m in a.methods.split(","):
        root, rules = ROOTS[m]
        out = [sample_prompt(pid, Path(root), rules, a.target, a.max_attempts, prov, log) for pid in a.ids.split(",")]
        (Path(root) / "sampling_summary.json").write_text(json.dumps(out, indent=1))
    log("done")


if __name__ == "__main__":
    main()
