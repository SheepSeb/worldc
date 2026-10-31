"""E3 grid: the selected scenes x 3 tiers x seeds x 3 policies, resumable, scenes in parallel.

    uv run python scripts/e3_grid.py --variants-only                 # build the tier variants and verify reachability, no trials
    uv run python scripts/e3_grid.py --seeds 30 --jobs 6             # the full grid (resumes from out/e3/grid/<scene>/trials.jsonl)

Scenes come from ``results/e3_scene_selection/selection.json``; each is labelled by its prompt id and every trial row carries the
source model, family, size, difficulty and the actual occluder count of the hard variant.  Tiers: easy / medium on the ``base``
variant with 1 / 3 workers, hard on the ``hard`` variant (as many occluders as fit, 0-4) with 6 workers.  Policies at the frozen tag
(``sim.POLICIES_TAG``); each row records it.

Per scene, in ``<out>/<scene>/``: ``scenes/{base,hard}`` (rebuilt variants), ``variants_report.json`` (occluder count, why candidates were
rejected, reachability of every robot goal and task point in both variants, worker-path check, replay check), ``map_{base,hard}.png``,
``trials.jsonl`` (appended after every trial; drawn workers, events, trajectory, digest) and ``trials.csv`` (rebuilt at the end).
A trial already in ``trials.jsonl`` is skipped, so an interrupted run resumes exactly where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import e3_one_scene as O  # noqa: E402  (TIERS, base_image, digest, summarize)
from worldc.e3 import sim as S  # noqa: E402
from worldc.e3.tiers import connected, lost_points, make_variants  # noqa: E402
from worldc.e3.walkers import workers  # noqa: E402
from worldc.e3.world import load_world  # noqa: E402


def reach_report(world, base_world=None) -> dict:
    """Robot goals (static legs from the spawn) and task points (door aisle network) of a world.  For the hard variant ``base_world`` is the
    base variant: no task point may have left the 1.2 m aisle network; points never on it (back sides of benches) are counted separately."""
    legs = S.static_legs(world)
    conn = connected(world)
    out = {"goals": len(world.goals), "unreachable_goal_legs": [g["id"] for g, p in zip(world.goals, legs) if p is None],
           "task_points": {k: len(v) for k, v in world.pois.items()}, "task_points_off_aisle_network": sum(not c for _, _, c in conn)}
    if base_world is not None:
        base = connected(base_world)
        out["task_points_lost_vs_base"] = [(k, p) for (k, p, b), (_, _, c) in zip(base, conn) if b and not c]
        out["task_points_reachable"] = not out["task_points_lost_vs_base"]
    else:
        out["task_points_reachable"] = True  # the base variant is the reference
    return out


def build_scene(entry: dict, out: Path, log) -> dict:
    vdir = out / "scenes"
    have = all((vdir / v / "scene" / "e3_variant.json").exists() for v in ("base", "hard"))
    if not have:
        v = make_variants(entry["scene"], vdir, log=log)
        if not (v["base"]["ok"] and v["hard"]["ok"]):
            return {"ok": False, "why": f"variants failed: {json.dumps(v)[:1500]}"}
    hard_meta = json.loads((vdir / "hard" / "scene" / "e3_variant.json").read_text())
    wb, wh = load_world(vdir / "base" / "scene"), load_world(vdir / "hard" / "scene")
    rep = {"scene": entry["prompt"], "model": entry["model"], "source_run": entry["run_id"],
           "occluder_count": hard_meta.get("occluder_count", len(hard_meta.get("occluders", []))), "occluder_candidates": hard_meta.get("candidates"),
           "occluders_rejected": hard_meta.get("rejected"), "build_attempts_hard": len(hard_meta.get("attempts", [])),
           "base": reach_report(wb), "hard": reach_report(wh, wb)}
    # six workers alone (hard tier on the base variant, i.e. 0 occluders): can every walker path be planned?
    try:
        for sd in range(3):
            workers(wb, sd, 6, 60.0, S.DT)
        rep["six_workers_plan"] = "ok"
    except Exception as e:  # noqa: BLE001
        rep["six_workers_plan"] = f"FAILED: {type(e).__name__}: {e}"
    rep["ok"] = (not rep["base"]["unreachable_goal_legs"] and rep["base"]["task_points_reachable"] and not rep["hard"]["unreachable_goal_legs"]
                 and rep["hard"]["task_points_reachable"] and rep["six_workers_plan"] == "ok")
    (out / "variants_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def done_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for ln in open(path):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:  # a half-written last line of an interrupted run
                continue
            keys.add((r["tier"], r["seed"], r["policy"]))
    return keys


def run_scene(args) -> dict:
    entry, root, seeds, variants_only = args
    name = entry["prompt"]
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")
    log = lambda m: (logf.write(m + "\n"), logf.flush())  # noqa: E731
    t0 = time.time()
    rep = build_scene(entry, out, log)
    if not rep.get("ok"):
        log(f"NOT OK: {json.dumps(rep)[:1500]}")
        return {"scene": name, "ok": False, "report": rep}
    if variants_only:
        return {"scene": name, "ok": True, "report": rep}
    vdir = out / "scenes"
    worlds, legs, timeouts = {}, {}, {}
    for variant in ("base", "hard"):
        w = load_world(vdir / variant / "scene")
        w.name = f"{name}/{variant}"
        worlds[variant], legs[variant] = w, S.static_legs(w)
        timeouts[variant] = S.timeout_s(w, legs[variant])
        O.base_image(w).save(out / f"map_{variant}.png")
    timeout = max(timeouts.values())
    prov = S.policy_provenance()
    if prov["policies_changed_since_tag"]:
        log(f"WARNING: policy files differ from {S.POLICIES_TAG}: {prov['policies_changed_since_tag']}")
    meta = {"e3_policies": prov["e3_policies"], "e3_policies_commit": prov["e3_policies_commit"], "code_commit": prov["code_commit"],
            "policies_at_tag": not prov["policies_changed_since_tag"], "scene": name, "model": entry["model"], "source_run": entry["run_id"],
            "family": entry["family"], "size": entry["size"], "difficulty_prompt": entry["difficulty"], "room_w_m": entry["width_m"],
            "room_d_m": entry["depth_m"]}
    jpath = out / "trials.jsonl"
    done = done_keys(jpath)
    n_new = 0
    with open(jpath, "a") as jf:
        for tier, (variant, n) in O.TIERS.items():
            world = worlds[variant]
            occ = rep["occluder_count"] if variant == "hard" else 0
            for seed in range(seeds):
                todo = [p for p in S.POLICIES if (tier, seed, p) not in done]
                if not todo:
                    continue
                H, info = workers(world, seed, n, timeout + 1.0, S.DT)
                for pol in todo:
                    t2 = time.time()
                    res = S.run_trial(world, H, pol, legs[variant], timeout)
                    row = {**meta, "variant": variant, "tier": tier, "workers": n, "occluders": occ, "seed": seed, "timeout_s": round(timeout, 1),
                           "wall_s": round(time.time() - t2, 2), **{k: v for k, v in res.items() if k not in ("events", "arrivals", "trajectory")}}
                    jf.write(json.dumps({**row, "workers_drawn": info, "events": res["events"], "arrivals": res["arrivals"],
                                         "trajectory": res["trajectory"], "digest": O.digest(res)}) + "\n")
                    jf.flush()
                    n_new += 1
    # replay: rebuild one logged trial from its seed in a fresh world and compare digests
    logged = {}
    for ln in open(jpath):
        r = json.loads(ln)
        if r["tier"] == "hard" and r["policy"] == "blind" and r["seed"] == 0:
            logged = r
    if logged:
        w = load_world(vdir / "hard" / "scene")
        H, _ = workers(w, 0, O.TIERS["hard"][1], timeout + 1.0, S.DT)
        again = S.run_trial(w, H, "blind", S.static_legs(w), timeout)
        rep["replay_hard_seed0_blind"] = "identical" if O.digest(again) == logged["digest"] else "DIFFERENT"
        (out / "variants_report.json").write_text(json.dumps(rep, indent=1))
    rows = []
    for ln in open(jpath):
        r = json.loads(ln)
        rows.append({k: v for k, v in r.items() if k not in ("events", "arrivals", "trajectory", "workers_drawn", "digest")})
    rows.sort(key=lambda r: (list(O.TIERS).index(r["tier"]), r["seed"], S.POLICIES.index(r["policy"]) if isinstance(S.POLICIES, (list, tuple)) else 0))
    with open(out / "trials.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    log(f"done {name}: {len(rows)} trials ({n_new} new) in {time.time() - t0:.0f} s")
    return {"scene": name, "ok": True, "trials": len(rows), "new": n_new, "seconds": round(time.time() - t0), "report": rep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e3_scene_selection/selection.json")
    ap.add_argument("--out", default="out/e3/grid")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated prompt ids")
    ap.add_argument("--variants-only", action="store_true")
    a = ap.parse_args()
    chosen = json.loads(Path(a.selection).read_text())["chosen"]
    if a.only:
        chosen = [c for c in chosen if c["prompt"] in a.only.split(",")]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(c, a.out, a.seeds, a.variants_only) for c in chosen]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        results = []
        for r in ex.map(run_scene, jobs):
            results.append(r)
            rep = r["report"]
            extra = (f"{r.get('trials', '')} trials ({r.get('new', '')} new, {r.get('seconds', '')} s)" if not a.variants_only else "")
            print(f"{r['scene']}: {'ok' if r['ok'] else 'NOT OK'} occluders {rep.get('occluder_count')} "
                  f"(of {rep.get('occluder_candidates')} candidates, rejected {rep.get('occluders_rejected')}) {extra}", flush=True)
    (Path(a.out) / ("variants_summary.json" if a.variants_only else "grid_summary.json")).write_text(json.dumps(results, indent=1, default=str))
    print(f"{len(results)} scenes in {time.time() - t0:.0f} s; not ok: {[r['scene'] for r in results if not r['ok']]}")


if __name__ == "__main__":
    main()
