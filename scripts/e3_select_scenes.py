"""Propose the E3 scene set: accepted E1 scenes covering every cell type and the S / M / L sizes.

    uv run python scripts/e3_select_scenes.py dataset/e1/<model> [more datasets] --n 10 --out results/e3_scene_selection

Light: it reads the E1 records and loads each candidate's 2D world (no Isaac Sim, no trials), so it is safe to run
while E1 is running.

* **Mapping:** a run maps to its prompt by matching ``prompt.txt`` against ``prompts/prompts.csv`` (plain or
  constrained text).
* **Candidates:** accepted runs whose final physics check passed. A candidate must also be usable for E3: its scene
  graph has a door, at least two kinds of task point, and at least two robot goals, every one statically reachable.
* **Families:** ``engine_gearbox_reman`` and ``motor_gearbox_reman`` count as one, ``gearbox_reman``, which gives 6
  cell types.
* **Greedy pick, one scene per prompt:**
  1. Cover every family and every size.
  2. Then add (family, size) combinations not yet chosen.
  3. Then balance the families.
  Ties go to the least-chosen difficulty and size, then to the higher prompt id.
* **Outputs:** ``selection.md`` (the proposal, coverage and what is missing) and ``selection.json`` (scene dirs, ready
  for ``scripts/e3_one_scene.py``).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAMILY = {"engine_gearbox_reman": "gearbox_reman", "motor_gearbox_reman": "gearbox_reman"}
SIZES = ("S", "M", "L")


def norm(s: str) -> str:
    return " ".join(s.split())


def candidates(datasets, prompts_csv, check: bool) -> tuple[list[dict], list[dict]]:
    P = list(csv.DictReader(open(prompts_csv)))
    by_text = {}
    for p in P:
        for variant in ("plain", "constrained"):
            by_text[norm(p[f"text_{variant}"])] = (p, variant)
    out, rejected = [], []
    for ds in map(Path, datasets):
        for line in open(ds / "records.jsonl"):
            r = json.loads(line)
            run = ds / "runs" / r["run_id"]
            hit = by_text.get(norm((run / "prompt.txt").read_text())) if (run / "prompt.txt").exists() else None
            row = {"dataset": str(ds), "run_id": r["run_id"], "prompt": hit[0]["id"] if hit else None}
            if hit is None:
                rejected.append({**row, "why": "prompt not in prompts.csv"})
                continue
            p, variant = hit
            row.update(variant=variant, cell_type=p["cell_type"], family=FAMILY.get(p["cell_type"], p["cell_type"]), size=p["size"],
                       difficulty=p["difficulty"], n_people=int(p["n_people"]), width_m=float(p["width_m"]), depth_m=float(p["depth_m"]),
                       scene=str(ds / r["scene_dir"]) if r.get("scene_dir") else None)
            physics = (r.get("final_checks") or {}).get("physics_status")
            if not r.get("accepted") or physics != "ok":
                rejected.append({**row, "why": f"not accepted ({r.get('final_status')}, physics {physics})"})
                continue
            if check:
                why = usable(row["scene"])
                if why:
                    rejected.append({**row, "why": why})
                    continue
            out.append(row)
    return out, rejected


def usable(scene) -> str | None:
    from worldc.e3.sim import static_legs
    from worldc.e3.world import load_world

    try:
        w = load_world(scene)
    except Exception as e:  # noqa: BLE001
        return f"world does not load: {e}"
    if "door" not in w.pois:
        return "no door task point"
    if len(w.pois) < 2:
        return f"only one kind of task point ({list(w.pois)})"
    if len(w.goals) < 2:
        return f"{len(w.goals)} robot goal(s)"
    bad = [g["id"] for g, p in zip(w.goals, static_legs(w)) if p is None]
    return f"unreachable robot goals {bad}" if bad else None


def select(cands: list[dict], n: int) -> list[dict]:
    pool = {}
    for c in sorted(cands, key=lambda c: (c["prompt"], c["dataset"])):  # one scene per prompt: the first dataset given wins
        pool.setdefault(c["prompt"], c)
    pool = list(pool.values())
    families = sorted({c["family"] for c in pool})
    chosen: list[dict] = []
    while pool and len(chosen) < n:
        fam = Counter(c["family"] for c in chosen)
        have_f, have_s = set(fam), {c["size"] for c in chosen}
        have_fs = {(c["family"], c["size"]) for c in chosen}
        dif, siz = Counter(c["difficulty"] for c in chosen), Counter(c["size"] for c in chosen)

        def score(c):
            return ((c["family"] not in have_f) + (c["size"] not in have_s),  # 1. coverage
                    (c["family"], c["size"]) not in have_fs,  # 2. new combination
                    -fam[c["family"]],  # 3. balance families ...
                    -dif[c["difficulty"]] - siz[c["size"]],  # ... then difficulty and size
                    -int(c["prompt"][1:]))
        best = max(pool, key=score)
        chosen.append(best)
        pool.remove(best)
    return chosen, families


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", help="E1 dataset folders (dataset/e1/<model>), in order of preference")
    ap.add_argument("--prompts", default="prompts/prompts.csv")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="results/e3_scene_selection")
    ap.add_argument("--no-check", action="store_true", help="skip the E3 usability check (no world loading)")
    a = ap.parse_args()
    cands, rejected = candidates(a.datasets, a.prompts, not a.no_check)
    chosen, _ = select(cands, a.n)
    all_fams = sorted({FAMILY.get(p["cell_type"], p["cell_type"]) for p in csv.DictReader(open(a.prompts))})
    miss_f = [f for f in all_fams if f not in {c["family"] for c in chosen}]
    miss_s = [s for s in SIZES if s not in {c["size"] for c in chosen}]
    lines = [f"# E3 scene selection ({len(chosen)} of {a.n})", "",
             f"From {', '.join(a.datasets)}: {len(cands)} usable candidates, {len(rejected)} rejected.", "",
             "| # | prompt | cell type | size | W x D (m) | difficulty | people | run |", "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(chosen, 1):
        lines.append(f"| {i} | {c['prompt']} | {c['cell_type']} | {c['size']} | {c['width_m']:g} x {c['depth_m']:g} | {c['difficulty']} | "
                     f"{c['n_people']} | `{c['run_id']}` |")
    lines += ["", f"Families: {dict(Counter(c['family'] for c in chosen))}. Sizes: {dict(Counter(c['size'] for c in chosen))}.",
              f"**Missing:** families {miss_f or 'none'}, sizes {miss_s or 'none'}"
              + ("" if len(chosen) == a.n else f", and {a.n - len(chosen)} scene(s) short of {a.n}") + "."]
    if rejected:
        lines += ["", "Rejected:", ""] + [f"- {r.get('prompt') or '?'} `{r['run_id']}`: {r['why']}" for r in rejected]
    md = "\n".join(lines) + "\n"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.md").write_text(md)
    (out / "selection.json").write_text(json.dumps({"datasets": a.datasets, "n": a.n, "chosen": chosen, "missing_families": miss_f,
                                                    "missing_sizes": miss_s, "rejected": rejected}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
