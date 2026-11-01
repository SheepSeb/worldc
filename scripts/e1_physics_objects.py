"""E1 physics failures: which objects failed (spec object and asset), per run.

    python3 scripts/e1_physics_objects.py --out results/e1

For every E1 record with ``final_status == physics_failed``: the bodies named by the error issues (``physics.fell_through_floor``,
``physics.unstable``) and, for ``physics.not_settled`` (no body in the issue), the bodies of ``physics.worst_movers`` that moved more than 2 cm.
A body ``/World/<spec id>/<spec id>_NNNN`` is an instance of spec object ``<spec id>``; its asset comes from the run's ``scene_graph.json``.
Writes ``physics_failure_objects.md`` / ``.csv``.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

PROPS = {"laptop": "laptop", "isaac_sm_laptop": "laptop", "isaac_sm_laptopclose": "laptop", "gen/returned_laptop": "laptop",
         "transfer_case": "transfer case", "simready_drivetrain_transfer_case_b01": "transfer case"}


def family(asset: str) -> str:
    return PROPS.get(asset, "other")


def failing_objects(ds: Path, r: dict) -> list[dict]:
    """The objects (spec id, asset, family, codes, largest displacement) behind the physics errors of a physics-failed record."""
    chk = (r.get("final_checks") or {}).get("check") or {}
    graph = json.loads((ds / r["scene_dir"] / "scene_graph.json").read_text())
    asset_of = {}
    for v in graph["instances"].values():
        asset_of.setdefault(v.get("spec_id"), v["asset"])
    bodies, disp = collections.defaultdict(set), {}
    for i in chk.get("issues", []):
        if i["severity"] == "error" and i["code"].startswith("physics."):
            if i["path"]:
                bodies[i["path"].split("/")[2]].add(i["code"].split(".")[1])
            else:
                for m in (chk.get("physics") or {}).get("worst_movers", []):
                    if m["displacement_m"] > 0.02:
                        o = m["body"].split("/")[2]
                        bodies[o].add(i["code"].split(".")[1])
                        disp[o] = max(disp.get(o, 0.0), m["displacement_m"])
    return [{"object": o, "asset": asset_of.get(o, "?"), "family": family(asset_of.get(o, "?")), "codes": sorted(c), "max_displacement_m": round(disp.get(o, 0.0), 3)} for o, c in sorted(bodies.items())]


def asset_induced(objs: list[dict]) -> bool:
    """Every failing object is a laptop or a transfer case (the two props behind the physics failures)."""
    return bool(objs) and all(o["family"] != "other" for o in objs)


def prompt_ids(ds: Path, prompts_csv="prompts/prompts.csv") -> dict:
    """run id -> prompt id, by matching the run's prompt.txt against the prompts (plain or constrained text)."""
    norm = lambda t: " ".join(t.split())  # noqa: E731
    by = {}
    for p in csv.DictReader(open(prompts_csv)):
        by[norm(p["text_plain"])] = by[norm(p["text_constrained"])] = p["id"]
    out = {}
    for run in (ds / "runs").iterdir():
        if (run / "prompt.txt").exists():
            out[run.name] = by.get(norm((run / "prompt.txt").read_text()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="dataset/e1")
    ap.add_argument("--out", default="results/e1")
    a = ap.parse_args()
    rows = []
    for ds in sorted(Path(a.e1).iterdir()):
        if not (ds / "records.jsonl").exists():
            continue
        pid = prompt_ids(ds)
        for ln in open(ds / "records.jsonl"):
            r = json.loads(ln)
            if r.get("final_status") != "physics_failed":
                continue
            objs = failing_objects(ds, r)
            rows.append({"model": ds.name, "prompt": pid.get(r["run_id"]), "run": r["run_id"], "codes": sorted({c for o in objs for c in o["codes"]}), "objects": objs,
                         "families": sorted({o["family"] for o in objs}), "all_props": asset_induced(objs)})
    md = ["# E1 physics failures: failing objects", "", f"{len(rows)} runs ended `physics_failed` (static check passed, final Isaac Sim physics check failed). Objects are spec object ids with their asset.", "",
          "| Model | Prompt | Run | Codes | Failing objects (asset; codes; largest displacement) | Props only? |", "|---|---|---|---|---|---|"]
    for r in rows:
        ob = "; ".join(f"{o['object']} ({o['asset']}; {','.join(o['codes'])}{'; ' + str(o['max_displacement_m']) + ' m' if o['max_displacement_m'] else ''})" for o in r["objects"])
        md.append(f"| {r['model']} | {r['prompt']} | `{r['run'][:15]}` | {', '.join(r['codes'])} | {ob} | {'yes' if r['all_props'] else 'no'} |")
    cnt = collections.Counter()
    for r in rows:
        for f in r["families"]:
            cnt[f] += 1
    only = collections.Counter(tuple(r["families"]) for r in rows)
    md += ["", f"Runs with a failing laptop: {sum('laptop' in r['families'] for r in rows)}; with a failing transfer case: {sum('transfer case' in r['families'] for r in rows)}; with another object: {sum('other' in r['families'] for r in rows)} (of {len(rows)}).",
           f"Runs whose failing objects are all laptops / transfer cases: {sum(r['all_props'] for r in rows)} of {len(rows)}.", "",
           "Failing-object families per run: " + ", ".join(f"{'+'.join(k) or 'none'}: {v}" for k, v in only.most_common()) + "."]
    (Path(a.out) / "physics_failure_objects.md").write_text("\n".join(md) + "\n")
    with open(Path(a.out) / "physics_failure_objects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "prompt", "run", "object", "asset", "family", "codes", "max_displacement_m"])
        for r in rows:
            for o in r["objects"] or [{"object": "", "asset": "", "family": "", "codes": [], "max_displacement_m": ""}]:
                w.writerow([r["model"], r["prompt"], r["run"], o["object"], o["asset"], o["family"], ";".join(o["codes"]), o["max_displacement_m"]])
    print("\n".join(md))


if __name__ == "__main__":
    main()
