"""Generate seeds, run every validation stage, write warehouse_out/validation_report.{json,md}.

    uv run python -m worldc.run_all --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISAAC = Path(os.environ.get("ISAACSIM_PATH", Path.home() / "isaacsim"))


def usdchecker(usd: Path) -> dict:
    libs = sorted(glob.glob(str(ISAAC / "extscache" / "omni.usd.libs-*")))[-1]
    env = dict(os.environ, PYTHONPATH=libs, PXR_PLUGINPATH_NAME=f"{libs}/bin/usd",
               LD_LIBRARY_PATH=f"{libs}/bin:{libs}/bin/deps:{libs}/bin/usd")
    out = subprocess.run([str(ISAAC / "kit/python/bin/python3.11"), "-I", "-c",
                          f"import sys; sys.path.insert(0, {libs!r}); sys.path.insert(0, {str(ROOT / 'src')!r}); "
                          "import runpy; sys.argv=['x', %r]; runpy.run_module('worldc.usdcheck_cli', run_name='__main__')" % str(usd)],
                         env=env, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("JSON:")]
    if not line:
        return {"errors": ["usdchecker failed to run: " + out.stderr[-500:]], "failed_checks": [], "warnings": [], "ok": False}
    r = json.loads(line[0][5:])
    r["ok"] = not r["errors"] and not r["failed_checks"]
    return r


def kit_log_errors(since: float) -> dict:
    logs = sorted(glob.glob(str(ISAAC / "kit/logs/Kit/Isaac-Sim Python/*/kit_*.log")), key=os.path.getmtime)
    logs = [l for l in logs if os.path.getmtime(l) >= since]
    errs, fatals, phys = [], [], []
    for lg in logs:
        for ln in open(lg, errors="replace"):
            if "[Fatal]" in ln:
                fatals.append(ln.strip()[:300])
            elif "[Error]" in ln:
                errs.append(ln.strip()[:300])
            elif "[Warning]" in ln and "omni.physx" in ln:
                phys.append(ln.strip()[:300])
    return {"log_files": logs, "fatal": fatals[:20], "errors": errs[:30], "n_errors": len(errs), "n_fatal": len(fatals),
            "physx_warnings": phys[:20], "n_physx_warnings": len(phys)}


def run_seed(seed: int, out: Path, skip_isaac=False, sim_seconds=10.0, extra=()) -> dict:
    from . import generate
    from .check import check_scene

    t0 = time.time()
    d = generate.build(seed, out)
    res = {"seed": seed, "dir": str(d)}
    res["usdchecker"] = usdchecker(d / "warehouse.usd")
    res["static"] = check_scene(d)  # writes check.json + nav_occupancy.png
    if not skip_isaac:
        start = time.time()
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        p = subprocess.run([str(ISAAC / "python.sh"), "-m", "worldc.isaac_validate", str(d), "--sim-seconds", str(sim_seconds), *extra],
                           env=env, capture_output=True, text=True)
        (d / "isaac_run.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
        f = d / "isaac_validation.json"
        res["isaac"] = json.loads(f.read_text()) if f.exists() else {"error": "no isaac_validation.json", "returncode": p.returncode}
        res["kit_log"] = kit_log_errors(start)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(r: dict) -> dict:
    s = {"seed": r["seed"], "lighting": json.loads((Path(r["dir"]) / "scene_graph.json").read_text())["lighting"]["setup_name"]}
    s["1_usdchecker"] = r["usdchecker"]["ok"]
    isa = r.get("isaac", {}).get("stages", {})
    kl = r.get("kit_log", {})
    s["2_loads_headless"] = bool(isa.get("load", {}).get("ok") and not kl.get("n_fatal") and not kl.get("n_errors"))
    ph = isa.get("physics", {})
    s["3_settle_fraction"] = ph.get("fraction_displacement_below_2cm")
    s["3_physics_ok"] = bool(ph.get("ok"))
    s["4_articulation_ok"] = bool(isa.get("articulation", {}).get("ok"))
    st = r["static"]
    s["5_nav_ok"] = bool(st["navigation"]["all_reachable"])
    s["5_chokepoint_m"] = next((c["measured_m"] for c in st["clearance"] if c["id"] == "chokepoint"), None)
    s["6_render_ok"] = bool(isa.get("render", {}).get("ok"))
    s["static_clean"] = bool(not st["spawn_interpenetrations"] and not st["floating_bodies"] and not st["hazard_path_conflicts"])
    stages = ("1_usdchecker", "5_nav_ok", "static_clean") if "isaac" not in r else (
        "1_usdchecker", "2_loads_headless", "3_physics_ok", "4_articulation_ok", "5_nav_ok", "6_render_ok", "static_clean")
    s["all_pass"] = all(s[k] for k in stages)
    return s


def write_md(out: Path, summaries, results):
    ok = lambda b: "PASS" if b else "**FAIL**"
    L = ["# Validation report", "", "| seed | lighting | 1 usdchecker | 2 headless load | 3 settled <2cm | 3 physics flags | 4 articulation | 5 nav (all goals) | chokepoint | 6 renders | pre-flight |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        L.append(f"| {s['seed']} | {s['lighting']} | {ok(s['1_usdchecker'])} | {ok(s['2_loads_headless'])} | {100 * (s['3_settle_fraction'] or 0):.2f}% | {ok(s['3_physics_ok'])} | {ok(s['4_articulation_ok'])} | {ok(s['5_nav_ok'])} | {s['5_chokepoint_m']} m | {ok(s['6_render_ok'])} | {ok(s['static_clean'])} |")
    L += ["", "## Details per seed", ""]
    for r in results:
        i = r.get("isaac", {}).get("stages", {})
        ph, ar, st = i.get("physics", {}), i.get("articulation", {}), r["static"]
        L += [f"### seed {r['seed']}", f"- dynamic rigid bodies: {i.get('load', {}).get('n_dynamic_bodies')} ({ph.get('n_bodies_evaluated')} evaluated, {ph.get('excluded_conveyor_payload')} conveyor payload excluded); settled fraction {ph.get('fraction_displacement_below_2cm')}, max displacement {ph.get('max_displacement_m')} m, fell through floor: {len(ph.get('fell_through_floor', []))}, faster than 1 m/s: {len(ph.get('faster_than_1mps', []))}",
              f"- conveyor payload speeds (target 0.5 m/s): " + ", ".join(f"{v['mean_vy_mps']}" for v in ph.get('conveyor_payload_speeds', {}).values()),
              f"- articulations: " + ", ".join(f"{j['name']} ({'ok' if j['ok'] else 'FAIL'}, pen {j['max_interpenetration_mm']} mm)" for j in ar.get("joints", [])),
              f"- spawn interpenetrations >3 mm: {len(st['spawn_interpenetrations'])}; floating: {len(st['floating_bodies'])}; hazard-path conflicts: {len(st['hazard_path_conflicts'])}",
              f"- navigation: {sum(g['reachable'] for g in st['navigation']['goals'])}/{st['navigation']['n_goals']} goals reachable on the 5 cm grid (robot radius {st['navigation']['robot_radius_m']} m); chokepoint clearance exact "
              + ", ".join(f"{c['id']} {c['measured_m']} m" for c in st["clearance"]),
              f"- kit log: {r.get('kit_log', {}).get('n_fatal')} fatal, {r.get('kit_log', {}).get('n_errors')} errors", ""]
    (out / "validation_report.md").write_text("\n".join(L))


def main(argv=None):
    from .isaac_env import gate_or_exit

    gate_or_exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(ROOT / "warehouse_out"))
    ap.add_argument("--skip-isaac", action="store_true")
    ap.add_argument("--sim-seconds", type=float, default=10.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    results, summaries = [], []
    for s in a.seeds:
        r = run_seed(s, out, a.skip_isaac, a.sim_seconds)
        sm = summarize(r)
        results.append(r)
        summaries.append(sm)
        print(json.dumps(sm), flush=True)
    (out / "validation_report.json").write_text(json.dumps({"summary": summaries, "details": results}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    write_md(out, summaries, results)
    print(("ALL PASS" if all(x["all_pass"] for x in summaries) else "SOME FAILED") + (" (Isaac stages skipped)" if a.skip_isaac else ""))


if __name__ == "__main__":
    main()
