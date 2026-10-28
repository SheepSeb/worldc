"""``worldc eval``: one CSV row per run with prompt adherence and layout plausibility (post-hoc, E1 / E2).

    worldc eval --runs dataset/e1/<model> --prompts prompts/prompts.csv --out eval.csv
    worldc eval --runs baselines/ --prompts prompts/prompts.csv --prompt-id ids.csv --out eval.csv

``--runs`` with a ``records.jsonl``: every record is a row; it is joined to its prompt by its description (= the row's
``text_plain`` or ``text_constrained``, which also gives the variant).  Without one (baseline scene folders): every folder
below ``--runs`` holding a ``scene_graph.json`` is a row; its prompt id comes from ``--prompt-id`` (CSV ``dir,prompt_id`` or
a JSON object; ``dir`` relative to ``--runs``) or else from a ``P<nn>`` in the folder name, and it counts as accepted when
``check_scene`` passes.  Non-accepted runs get a row with the metric columns left empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from . import adherence, plausibility
from .scene import load

BASE_COLUMNS = ["run_id", "prompt_id", "variant", "model", "accepted", "repairs", "at_freeze", "scene_dir"]


def columns() -> list[str]:
    cols = list(BASE_COLUMNS)
    for f, (_, kind) in adherence.FIELDS.items():
        cols += [f"req_{f}", f"got_{f}", f"match_{f}"] + ([f"abserr_{f}"] if kind == "count" else [])
    cols += ["adherence_fields", "adherence"]
    for r in plausibility.RULES:
        cols += [f"{r}_n", f"{r}_passed", f"{r}_rate"]
    return cols + ["plausibility"]


def _prompts(path) -> tuple[dict, dict]:
    rows = {r["id"]: r for r in csv.DictReader(open(path))} if path else {}
    by_text = {}
    for pid, r in rows.items():
        for v in ("plain", "constrained"):
            by_text[" ".join(r.get(f"text_{v}", "").split())] = (pid, v)
    return rows, by_text


def evaluate(scene_dir: Path, prompt_row: dict | None) -> dict:
    sc = load(scene_dir)
    out: dict = {}
    if prompt_row:
        a = adherence.score(sc, prompt_row)
        for f, v in a["fields"].items():
            out.update({f"req_{f}": v["requested"], f"got_{f}": v["got"], f"match_{f}": v["match"]})
            if "abs_err" in v:
                out[f"abserr_{f}"] = v["abs_err"]
        out.update(adherence_fields=len(a["fields"]), adherence=a["adherence"])
    p = plausibility.score(sc)
    for r, v in p["rules"].items():
        out.update({f"{r}_n": v["n"], f"{r}_passed": v["passed"], f"{r}_rate": v["rate"]})
    out["plausibility"] = p["plausibility"]
    return out


def _record_rows(runs: Path, prompts: dict, by_text: dict):
    from ..datagen.records import load_records

    for rec in load_records(runs):
        pid, variant = by_text.get(" ".join(rec.description.split()), ("", ""))
        row = {"run_id": rec.run_id, "prompt_id": pid, "variant": variant, "model": rec.model, "accepted": rec.accepted,
               "repairs": rec.usage.repairs, "at_freeze": rec.generator.get("at_freeze", ""), "scene_dir": rec.scene_dir or ""}
        if rec.accepted and rec.scene_dir and (runs / rec.scene_dir / "scene_graph.json").is_file():
            row.update(evaluate(runs / rec.scene_dir, prompts.get(pid)))
        yield row


def _baseline_rows(runs: Path, prompts: dict, mapping: dict):
    from ..check import check_scene

    for g in sorted(runs.rglob("scene_graph.json")):
        d = g.parent
        rel = d.relative_to(runs).as_posix()
        m = re.search(r"P\d{2}", rel)
        pid = mapping.get(rel) or mapping.get(d.name) or (m.group(0) if m else "")
        accepted = bool(check_scene(d, write=False)["passed"])
        row = {"run_id": rel, "prompt_id": pid, "variant": "", "model": "baseline", "accepted": accepted, "repairs": "", "at_freeze": "",
               "scene_dir": str(d)}
        if accepted:
            row.update(evaluate(d, prompts.get(pid)))
        yield row


def _mapping(path) -> dict:
    if not path:
        return {}
    p = Path(path)
    if p.suffix == ".json":
        return {str(k): str(v) for k, v in json.loads(p.read_text()).items()}
    return {r[0]: r[1] for r in csv.reader(p.open()) if len(r) >= 2 and r[0] and not r[0].startswith("#") and r[0] != "dir"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="worldc eval", description="Prompt adherence + layout plausibility per run (post-hoc)")
    ap.add_argument("--runs", required=True, help="dataset folder (records.jsonl) or a folder of baseline scene folders")
    ap.add_argument("--prompts", default="prompts/prompts.csv")
    ap.add_argument("--prompt-id", default=None, help="baselines: CSV dir,prompt_id or JSON {dir: prompt_id}")
    ap.add_argument("--out", default="eval.csv")
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    prompts, by_text = _prompts(a.prompts if Path(a.prompts).is_file() else None)
    rows = list(_record_rows(runs, prompts, by_text) if (runs / "records.jsonl").is_file() else _baseline_rows(runs, prompts, _mapping(a.prompt_id)))
    cols = columns()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"{len(rows)} run(s) -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
