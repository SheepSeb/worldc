"""E1 results: per model, how often the LLM loop ends in an accepted scene, at what effort, and how good the accepted
scenes are (Table 1, E1 rows).

    uv run python scripts/e1_analysis.py --out results/e1

Analysis only (no builds, no Isaac Sim, nice 19).  Reads every ``dataset/e1/<model>/records.jsonl`` (or ``--models``).

* **Runs:** the records of ``--variant`` (default constrained), the latest per prompt.  Every record must be at the
  freeze (``generator.at_freeze``); the script refuses otherwise.  Labelled PRELIMINARY until each model has all 30 prompts.
* **Acceptance:** ``accepted`` (static ``check_scene`` + final physics check), with a 95 % Wilson interval.
  ``final_status`` is broken down (``no_scene``, ``physics_failed``, ...).  First-try = accepted with 0 repairs.
* **Effort:** medians over all runs of turns, repairs (errored validate/build calls, capped at 10), wall time (loop +
  final checks) and output tokens.  Cost is Claude Code's estimate at Anthropic pricing: real for Claude models only.
* **Quality:** ``worldc eval`` adherence and plausibility, means over accepted scenes, 95 % percentile bootstrap over
  them; per-field match rates and per-rule pass rates in ``summary.json``.
* **Acceptance@k** (``acceptance_at_k.md/.csv``): the share of prompts accepted with at most k repairs, k = 0, 1, 2, 3, 4, 5, 6, 8, 10
  (each prompt is one run; a run that stopped after r repairs counts for every k >= r), with Wilson intervals.  ``static@k`` is the same
  for the static ``check_scene`` alone (accepted or physics-failed runs), which separates the final physics check from the loop.
* **Failure taxonomy** (``failure_taxonomy.md/.csv``): per model, runs by outcome class.  Physics failures (the loop produced a statically
  valid scene, the final Isaac Sim physics check rejected it) are kept apart from static failures (no valid scene; by stop reason and the
  family of the last validation / build / check error).  A physics failure is **asset-induced** when every failing object is a laptop or a
  transfer case (``e1_physics_objects.py``: the objects behind each failure); a second table splits physics failures by their first error
  code.  ``physics_failure_objects.md/.csv`` list the failing objects of every physics-failed run.
* **Outputs** in ``--out``: ``table1_e1.md`` / ``.tex``, ``runs.csv`` (one row per run), ``by_difficulty.csv``,
  ``acceptance_at_k.*``, ``failure_taxonomy.*``, ``summary.json``, ``eval_<model>.csv``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_prelim as P  # noqa: E402  (helpers)
import e1_physics_objects as PO  # noqa: E402  (failing objects of a physics-failed run)

N_PROMPTS = 30
FIELDS = ("n_benches", "n_robot_cells", "conveyor", "n_racks", "n_doors", "n_people")
RULES = ("bench_access", "conveyor_connected", "storage_reachable", "cell_not_blocking")
DIFFICULTY = ("easy", "medium", "hard")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def boot_mean(v: np.ndarray, B: int, rng) -> tuple[float, float, float]:
    v = v[~np.isnan(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def physics_codes(r: dict) -> list[str]:
    chk = ((r.get("final_checks") or {}).get("check") or {})
    return [i["code"] for i in chk.get("issues", []) if i.get("severity") == "error" and i["code"].startswith("physics.")]


def static_codes(r: dict) -> list[str]:
    """Error codes of the last validate / build / check of a run that never reached a valid scene."""
    out = []
    for k in ("last_validate", "last_build", "last_check"):
        x = (r.get("validation") or {}).get(k)
        if isinstance(x, dict):
            for i in x.get("issues") or x.get("errors") or []:
                if isinstance(i, dict) and i.get("severity", "error") == "error" and i.get("code") and i["code"] not in out:
                    out.append(i["code"])
    return out


def static_family(codes: list[str]) -> str:
    if not codes:
        return "no error recorded"
    c = codes[0]
    if c.startswith("solve.unsatisfiable"):
        return "solver: placement unsatisfiable"
    if c.startswith("solve."):
        return "solver: robot goal / spawn"
    if c.startswith("check."):
        return "static check: " + c.split(".", 1)[1]
    return "spec error: " + c.split(".")[0]


KS = (0, 1, 2, 3, 4, 5, 6, 8, 10)


def at_k(rows: list[dict]) -> dict:
    n = len(rows)
    out = {}
    for name, ok in (("acceptance", lambda r: r["accepted"]), ("static", lambda r: r["final_status"] in ("accepted", "physics_failed"))):
        out[name] = {}
        for k in KS:
            c = sum(1 for r in rows if ok(r) and (r["repairs"] or 0) <= k)
            lo, hi = wilson(c, n)
            out[name][k] = {"n": c, "rate": c / n if n else np.nan, "ci95": [lo, hi]}
    return out


def taxonomy(rows: list[dict]) -> dict:
    from collections import Counter

    tax = Counter()
    for r in rows:
        if r["accepted"]:
            tax["accepted"] += 1
        elif r["final_status"] == "physics_failed":
            tax["physics failed, asset-induced (laptop / transfer case)" if r["asset_induced"] else "physics failed, other object (" + (r["physics_assets"] or "none recorded") + ")"] += 1
        else:
            how = "repair cap reached" if r["stop_reason"] == "max_repairs" else "stopped without a valid scene"
            tax[f"static failure, {how}: " + static_family([c for c in r["static_codes"].split(";") if c])] += 1
    return dict(tax)


def model_runs(ds: Path, variant: str, prompts: dict, out: Path) -> list[dict]:
    """One row per prompt (latest record), the record's effort fields joined to its ``worldc eval`` row."""
    from worldc.eval.run import main as eval_main

    eval_csv = out / f"eval_{ds.name}.csv"
    eval_main(["--runs", str(ds), "--prompts", "prompts/prompts.csv", "--out", str(eval_csv)])
    ev = {r["run_id"]: r for r in P._rows(eval_csv)}
    recs = [json.loads(line) for line in open(ds / "records.jsonl")]
    bad = [r["run_id"] for r in recs if r.get("generator", {}).get("at_freeze") != "true"]
    if bad:
        sys.exit(f"{ds}: records not at the freeze: {bad}")
    latest = {}
    for r in sorted(recs, key=lambda r: r["run_id"]):
        e = ev.get(r["run_id"])
        if e and e["variant"] == variant:
            latest[e["prompt_id"]] = (r, e)
    rows = []
    for pid, (r, e) in sorted(latest.items()):
        u, prm = r.get("usage") or {}, r.get("params") or {}
        objs = PO.failing_objects(ds, r) if r.get("final_status") == "physics_failed" else []
        rows.append({"asset_induced": PO.asset_induced(objs), "physics_assets": ";".join(sorted({o["asset"] for o in objs})), "stop_reason": r.get("stop_reason") or "", "physics_codes": ";".join(physics_codes(r)), "static_codes": ";".join(static_codes(r)),
                     "model": ds.name, "prompt_id": pid, "difficulty": prompts[pid]["difficulty"], "run_id": r["run_id"],
                     "accepted": bool(r["accepted"]), "final_status": r.get("final_status") or "",
                     "error": (r.get("error") or "")[:80], "turns": u.get("turns"), "repairs": u.get("repairs"),
                     "minutes": round(((prm.get("loop_seconds") or 0) + (prm.get("final_checks_seconds") or 0)) / 60, 2),
                     "output_tokens": u.get("output_tokens"), "cost_usd": u.get("est_cost_usd"),
                     "adherence": e.get("adherence", ""), "plausibility": e.get("plausibility", ""),
                     **{f"match_{f}": e.get(f"match_{f}", "") for f in FIELDS},
                     **{f"{x}_rate": e.get(f"{x}_rate", "") for x in RULES}})
    return rows


def summarise(rows: list[dict], B: int, rng) -> dict:
    n, acc = len(rows), [r for r in rows if r["accepted"]]
    med = lambda k: float(np.median([r[k] for r in rows if r[k] is not None])) if rows else np.nan  # noqa: E731
    s = {"runs": n, "accepted": len(acc), "acceptance": len(acc) / n if n else np.nan, "acceptance_ci95": wilson(len(acc), n),
         "first_try": sum(1 for r in acc if r["repairs"] == 0),
         "final_status": dict(Counter(r["final_status"] for r in rows)),
         "median_turns": med("turns"), "median_repairs": med("repairs"), "median_minutes": med("minutes"),
         "median_output_tokens": med("output_tokens"),
         "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 2),
         "hit_repair_cap": sum(1 for r in rows if (r["repairs"] or 0) >= 10)}
    for k in ("adherence", "plausibility"):
        m, lo, hi = boot_mean(np.array([P._num(r[k]) for r in acc]), B, rng)
        s[k], s[f"{k}_ci95"] = m, [lo, hi]
    s["field_match"] = {}
    for f in FIELDS:
        v = [r[f"match_{f}"] for r in acc if r[f"match_{f}"] != ""]
        s["field_match"][f] = {"n": len(v), "rate": sum(x == "True" for x in v) / len(v) if v else None}
    s["rule_rate"] = {}
    for x in RULES:
        v = [P._num(r[f"{x}_rate"]) for r in acc if r[f"{x}_rate"] != ""]
        s["rule_rate"][x] = {"n": len(v), "mean": float(np.mean(v)) if v else None}
    s["at_k"], s["taxonomy"] = at_k(rows), taxonomy(rows)
    s["static_pass"] = sum(1 for r in rows if r["final_status"] in ("accepted", "physics_failed"))
    s["by_difficulty"] = {d: {"runs": sum(r["difficulty"] == d for r in rows),
                              "accepted": sum(r["difficulty"] == d and r["accepted"] for r in rows)} for d in DIFFICULTY}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="dataset/e1")
    ap.add_argument("--models", default="", help="comma-separated dataset folder names (default: every one in --e1)")
    ap.add_argument("--variant", default="constrained")
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.nice(19)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    prompts = {r["id"]: r for r in csv.DictReader(open("prompts/prompts.csv"))}

    names = [m for m in a.models.split(",") if m] or sorted(d.name for d in Path(a.e1).iterdir() if (d / "records.jsonl").exists())
    rows, summary = [], {}
    for m in names:
        rs = model_runs(Path(a.e1) / m, a.variant, prompts, out)
        rows += rs
        summary[m] = summarise(rs, a.boot, rng)

    complete = all(s["runs"] == N_PROMPTS for s in summary.values())
    tag = "" if complete else "PRELIMINARY (" + ", ".join(f"{m} {s['runs']}/{N_PROMPTS}" for m, s in summary.items()) + "), "
    title = f"{tag}E1, {a.variant} prompts, {len(names)} models"

    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "by_difficulty.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "difficulty", "runs", "accepted"])
        w.writerows([m, d, v["runs"], v["accepted"]] for m, s in summary.items() for d, v in s["by_difficulty"].items())
    ak = ["| Model | metric | " + " | ".join(f"@{k}" for k in KS) + " |", "|---|---|" + "---|" * len(KS)]
    flat = [["model", "metric", "k", "count", "n", "rate", "ci_lo", "ci_hi"]]
    for m, s_ in summary.items():
        for name in ("acceptance", "static"):
            ak.append(f"| {m} | {name} | " + " | ".join(f"{100 * s_['at_k'][name][k]['rate']:.0f}% ({s_['at_k'][name][k]['n']})" for k in KS) + " |")
            flat += [[m, name, k, s_["at_k"][name][k]["n"], s_["runs"], round(s_["at_k"][name][k]["rate"], 4), *[round(v, 4) for v in s_["at_k"][name][k]["ci95"]]] for k in KS]
    (out / "acceptance_at_k.md").write_text("\n".join(ak) + "\n\nShare of the prompts accepted (acceptance: static + physics; static: static check only) with at most k repairs; "
                                          "count in brackets; Wilson CIs in acceptance_at_k.csv.\n")
    with open(out / "acceptance_at_k.csv", "w", newline="") as f:
        csv.writer(f).writerows(flat)
    classes = sorted({c for s_ in summary.values() for c in s_["taxonomy"]}, key=lambda c: (c != "accepted", not c.startswith("physics"), c))
    tx = ["| Class | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    tx += [f"| {c} | " + " | ".join(str(s_["taxonomy"].get(c, 0)) for s_ in summary.values()) + " |" for c in classes]
    bycode = collections.Counter()
    for r_ in rows:
        if r_["final_status"] == "physics_failed":
            bycode[(r_["model"], (r_["physics_codes"].split(";")[0] if r_["physics_codes"] else "no code recorded"))] += 1
    codes = sorted({c for _, c in bycode})
    tx += ["", "Physics failures by their first error code (the first listed in the failing check):", "", "| Code | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    tx += [f"| {c} | " + " | ".join(str(bycode.get((m, c), 0)) for m in summary) + " |" for c in codes]
    (out / "failure_taxonomy.md").write_text("\n".join(tx) + "\n\nRuns per outcome class (30 prompts per model). Physics failures: the loop ended with a statically valid scene and the final Isaac Sim "
                                            "physics check rejected it; static failures: no valid scene (stop reason, family of the first error of the last validate / build / check).\n")
    with open(out / "failure_taxonomy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", *summary])
        w.writerows([c, *[s_["taxonomy"].get(c, 0) for s_ in summary.values()]] for c in classes)
    json.dump({"title": title, "variant": a.variant, "boot": a.boot, "models": summary}, open(out / "summary.json", "w"), indent=1)

    pct = lambda s: f"{100 * s['acceptance']:.0f}% ({s['accepted']}/{s['runs']}) [{100 * s['acceptance_ci95'][0]:.0f}, {100 * s['acceptance_ci95'][1]:.0f}]"  # noqa: E731
    ci = lambda s, k: f"{s[k]:.2f} [{s[f'{k}_ci95'][0]:.2f}, {s[f'{k}_ci95'][1]:.2f}]" if s["accepted"] else "n/a"  # noqa: E731
    md = [f"**{title}**", "",
          "| Model | Acceptance [95% CI] | Static pass | Accept. @3 | Accept. @5 | First try | No scene | Physics failed | Median turns | Median repairs | Median min | Adherence [95% CI] | Plausibility [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lcccccccc}", r"\toprule",
           r"Model & Acceptance & First try & No scene & Phys.\ fail & Turns & Repairs & Adherence & Plausibility \\", r"\midrule"]
    for m, s in summary.items():
        fs = s["final_status"]
        md.append(f"| {m} | {pct(s)} | {s['static_pass']}/{s['runs']} | {s['at_k']['acceptance'][3]['n']}/{s['runs']} | {s['at_k']['acceptance'][5]['n']}/{s['runs']} | {s['first_try']} | {fs.get('no_scene', 0)} | {fs.get('physics_failed', 0)} | "
                  f"{s['median_turns']:.0f} | {s['median_repairs']:.0f} | {s['median_minutes']:.1f} | {ci(s, 'adherence')} | {ci(s, 'plausibility')} |")
        tex.append(f"{m.replace('_', r'\_')} & {100 * s['acceptance']:.0f}\\% ({s['accepted']}/{s['runs']}) & {s['first_try']} & "
                   f"{fs.get('no_scene', 0)} & {fs.get('physics_failed', 0)} & {s['median_turns']:.0f} & {s['median_repairs']:.0f} & "
                   f"{s['adherence']:.2f} & {s['plausibility']:.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    md += ["", f"{title}. Accepted = static `check_scene` + final physics check. Static pass = the loop ended with a statically valid scene "
           "(accepted + physics failed). Accept. @k = accepted with at most k repairs (curves in acceptance_at_k.md). Acceptance CI: Wilson. First try: accepted "
           "with 0 repairs. Turns, repairs (capped at 10) and minutes (loop + final checks): medians over all runs. Adherence and "
           "plausibility (`worldc eval`): means over accepted scenes, 95% bootstrap CI.", "",
           "| Model | " + " | ".join(DIFFICULTY) + " | Repair cap hit | Est. cost (USD) |", "|---|---|---|---|---|---|"]
    md += [f"| {m} | " + " | ".join(f"{v['accepted']}/{v['runs']}" for v in s["by_difficulty"].values())
           + f" | {s['hit_repair_cap']} | {s['total_cost_usd']} |" for m, s in summary.items()]
    md += ["", "Est. cost: Claude Code's estimate at Anthropic pricing, real for Claude models only."]
    (out / "table1_e1.md").write_text("\n".join(md) + "\n")
    (out / "table1_e1.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
