"""E2 recipe baseline, within-prompt: seeds base..base+4 per prompt (base = prompt number), each resampled with seed + 1000 j on a
failed build, exactly like ``recipes.remanufacturing.build_prompt``.  Static checks only.  Folders ``<out>/<prompt>_seed<k>/``
with ``record.json`` (read by ``e2_within.py --recipe``).

    uv run python scripts/e2_recipe_within.py --out out/e2_within/recipe [--ids P05,P08,P15,P16,P23]
    uv run python scripts/e2_recipe_within.py --out out/e2_within/recipe_v4 --rules       # v4: the prompt's spatial rules are hard constraints

``--rules`` reads ``prompts/spatial_rules_e2_within.json`` (recipe v4): named door walls, cell / bench / shelving / conveyor / staging
placements as hard constraints; what the prompt leaves open stays seed-drawn.  A seed whose rules cannot be satisfied counts as a
failed attempt (resample with seed + 1000 j, as for a failed build); every attempt and the final seed are in ``record.json``.
"""
import argparse
import csv
import json
from pathlib import Path

from worldc.recipes.remanufacturing import MAX_RESAMPLES, _attempt, _finish, load_rules, provenance, spec_for_row

ROWS = {r["id"]: r for r in csv.DictReader(open("prompts/prompts.csv"))}


def build_sample(pid: str, k: int, out: Path, rules: bool, prov: dict | None = None, log=print) -> dict:
    """Sample k of a prompt: seed base + k, resampled with seed + 1000 j (up to MAX_RESAMPLES) on a failed rule / build / static check."""
    base, d, hist = int(pid[1:]) + k, Path(out) / f"{pid}_seed{k}", []
    for j in range(MAX_RESAMPLES + 1):
        seed = base + 1000 * j
        try:
            att, tmp = _attempt(spec_for_row(ROWS[pid], seed, load_rules(pid) if rules else None), seed, d)
        except Exception as e:  # noqa: BLE001
            att, tmp = {"seed": seed, "ok": False, "errors": [f"recipe.error:{type(e).__name__}"], "seconds": 0.0}, d.with_name(d.name + ".tmp")
        hist.append(att)
        log(f"  {pid} sample {k} seed {seed}: {'OK' if att['ok'] else 'FAILED ' + ','.join(att['errors'])}")
        if att["ok"]:
            break
    _finish(d, tmp, hist[-1]["ok"])
    spec = json.loads((d / "resolved_spec.json").read_text()) if (d / "resolved_spec.json").is_file() else {}
    rec = {"method": "recipe_v4" if rules else "recipe", "prompt_id": pid, "sample": k, "accepted": hist[-1]["ok"], "attempts": len(hist), "first_try": hist[0]["ok"],
           "final_seed": hist[-1]["seed"], "history": hist, "scene_dir": str(d.resolve()), "knobs": (spec.get("metadata") or {}).get("recipe"),
           "provenance": prov or provenance()}
    (d / "record.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/e2_within/recipe")
    ap.add_argument("--ids", default="P05,P08,P15,P16,P23")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--rules", action="store_true", help="recipe v4: spatial rules as hard constraints")
    a = ap.parse_args()
    prov = provenance()
    for pid in a.ids.split(","):
        for k in range(a.n):
            build_sample(pid, k, Path(a.out), a.rules, prov, lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
