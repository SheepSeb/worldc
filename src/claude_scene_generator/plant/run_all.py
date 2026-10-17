"""Generate plant seeds and run every validation stage; writes plant_out/validation_report.{json,md}.

    uv run python -m claude_scene_generator.plant.run_all --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from ..run_all import ISAAC, ROOT, kit_log_errors, usdchecker
from . import gen, static_validate
from .compare import compare_to_reference

DEFAULT_OUT = ROOT / "plant_out"
REF = ROOT / "reference" / "plant.png"


def run_seed(seed: int, out: Path, skip_isaac=False, sim_seconds=10.0, light_scale=1.0) -> dict:
    t0 = time.time()
    d = gen.build(seed, out, light_scale=light_scale)
    res = {"seed": seed, "dir": str(d)}
    res["usdchecker"] = usdchecker(d / "plant.usd")
    res["static"] = static_validate.run(d)
    (d / "static_validation.json").write_text(json.dumps(res["static"], indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    if not skip_isaac:
        start = time.time()
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        p = subprocess.run([str(ISAAC / "python.sh"), "-m", "claude_scene_generator.plant.isaac_validate", str(d), "--sim-seconds", str(sim_seconds)], env=env, capture_output=True, text=True)
        (d / "isaac_run.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
        f = d / "isaac_validation.json"
        res["isaac"] = json.loads(f.read_text()) if f.exists() else {"error": "no isaac_validation.json", "returncode": p.returncode}
        res["kit_log"] = kit_log_errors(start)
        photo = d / "renders" / "photo_view.png"
        if photo.exists():
            res["reference_comparison"] = compare_to_reference(photo, REF, d / "renders" / "photo_vs_reference.png")
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(r: dict) -> dict:
    g = json.loads((Path(r["dir"]) / "scene_graph.json").read_text())
    s = {"seed": r["seed"], "lighting": g["lighting"]["setup_name"]}
    s["1_usdchecker"] = r["usdchecker"]["ok"]
    isa = r.get("isaac", {}).get("stages", {})
    kl = r.get("kit_log", {})
    # Kit logs PhysX [Error] lines for harmless things too, so "no fatal errors" = no [Fatal] and no errors from the physx/usd plugins
    s["2_loads_headless"] = bool(isa.get("load", {}).get("ok") and isa.get("schema", {}).get("ok") and not kl.get("n_fatal") and not kl.get("n_errors"))
    ph = isa.get("physics", {})
    s["3_settled_frac"] = ph.get("fraction_settled_free_or_moving_below_2cm")
    s["3_settled_frac_all"] = ph.get("fraction_settled_all_bodies_below_2cm")
    s["3_n_fell"] = len(ph.get("fell_through_floor", []))
    s["3_n_fast"] = len(ph.get("faster_than_1mps", []))
    s["3_physics_ok"] = bool(ph.get("ok"))
    ar = isa.get("articulation", {})
    s["4_joints_ok"] = f"{ar.get('n_ok')}/{ar.get('n_joints')}"
    s["4_articulation_ok"] = bool(ar.get("ok"))
    nv = r["static"]["navigation"]
    s["5_goals_reachable"] = f"{nv['n_reachable']}/{nv['n_goals']}"
    s["5_nav_ok"] = bool(nv["all_reachable"])
    s["preflight_clean"] = bool(not r["static"]["initial_interpenetrations"] and not r["static"]["floating_loose_items"])
    s["6_render_ok"] = bool(isa.get("render", {}).get("ok"))
    rc = r.get("reference_comparison", {})
    s["6_ref_similarity"] = rc.get("similarity_score")
    dn = isa.get("render", {}).get("reference_view_density", {})
    s["7_density"] = dict(visible_objects_total=dn.get("distinct_visible_objects"), visible_objects_primary=dn.get("distinct_visible_primary_objects_excl_fittings_supports_background"),
                          handwheel_valves=dn.get("valves_with_visible_handwheel"), pipe_segments=dn.get("pipe_segments_visible"), small_instruments=dn.get("small_instruments_visible"),
                          cable_tubing_bundles=dn.get("cable_or_tubing_bundles_visible"), empty_fraction=dn.get("empty_fraction"))
    s["7_density_ok"] = bool(dn and dn["valves_with_visible_handwheel"] >= 25 and dn["pipe_segments_visible"] >= 40 and dn["small_instruments_visible"] >= 30 and dn["cable_or_tubing_bundles_visible"] >= 10
                             and dn["empty_fraction"] <= 0.15 and 150 <= dn["distinct_visible_primary_objects_excl_fittings_supports_background"] <= 250)
    pn = g.get("pipe_network") or {}
    s["8_pipe_network"] = dict(segments=pn.get("n_pipe_segments"), components=pn.get("n_connected_components"), dangling_ends=pn.get("dangling_pipe_ends_without_fitting"), fittings=pn.get("fittings"), supports=pn.get("n_supports"))
    s["8_pipe_network_ok"] = bool(pn and pn["n_connected_components"] == 1 and pn["dangling_pipe_ends_without_fitting"] == 0)
    sc = g.get("scale_check") or {}
    hw, pd = sc.get("handwheel_diameter_m", [0, 9]), sc.get("pipe_diameter_m", [0, 9])
    s["9_scale_ok"] = bool(sc and hw[0] >= 0.2 - 1e-6 and hw[1] <= 0.4 + 1e-6 and pd[0] >= 0.03 - 1e-6 and pd[1] <= 0.5 + 1e-6 and abs(sc["worker_height_m"] - 1.75) < 0.06)
    s["all_pass"] = all(s[k] for k in ("1_usdchecker", "2_loads_headless", "3_physics_ok", "4_articulation_ok", "5_nav_ok", "preflight_clean", "6_render_ok", "7_density_ok", "8_pipe_network_ok", "9_scale_ok")) and (s["3_settled_frac"] or 0) >= 0.95
    return s


def write_md(out: Path, summaries, results):
    ok = lambda b: "PASS" if b else "**FAIL**"
    L = ["# Validation report (plant_out)", "",
         "| seed | lighting | 1 usdchecker | 2 headless load (0 fatal/errors) | 3 settled <2cm (moving/loose) | fell / >1 m/s | 4 joints swept ok | 5 goals reachable (5 cm grid) | pre-flight | 6 renders | 7 density targets | 8 pipe graph | 9 scale | ref. similarity |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        L.append(f"| {s['seed']} | {s['lighting']} | {ok(s['1_usdchecker'])} | {ok(s['2_loads_headless'])} | {100 * (s['3_settled_frac'] or 0):.1f}% ({ok(s['3_physics_ok'])}) | {s['3_n_fell']} / {s['3_n_fast']} | "
                 f"{s['4_joints_ok']} ({ok(s['4_articulation_ok'])}) | {s['5_goals_reachable']} ({ok(s['5_nav_ok'])}) | {ok(s['preflight_clean'])} | {ok(s['6_render_ok'])} | {ok(s['7_density_ok'])} | {ok(s['8_pipe_network_ok'])} | {ok(s['9_scale_ok'])} | {s['6_ref_similarity']} |")
    L += ["", "## Details per seed", ""]
    for r in results:
        i = r.get("isaac", {}).get("stages", {})
        ph, ar, st = i.get("physics", {}), i.get("articulation", {}), r["static"]
        g = json.loads((Path(r["dir"]) / "scene_graph.json").read_text())
        rnd = g["randomization"]
        L += [f"### seed {r['seed']}  ({r['seconds']} s)",
              f"- randomization: lighting={rnd['lighting']}, dirt={rnd['dirt']}, gate valves={rnd['n_gate_valves']}, control valves={rnd['n_control_valves']}, loose items={rnd['n_loose_items']}, label colours={rnd['label_colors']}",
              f"- rigid bodies: {ph.get('n_rigid_bodies')} ({ph.get('n_free_or_moving_bodies')} loose/hose/moving-link bodies evaluated); settled <2 cm: {ph.get('fraction_settled_free_or_moving_below_2cm')} "
              f"(all bodies incl. fixed roots: {ph.get('fraction_settled_all_bodies_below_2cm')}); max displacement {ph.get('max_displacement_m')} m; max speed {ph.get('max_speed_mps_overall')} m/s; "
              f"fell through floor: {len(ph.get('fell_through_floor', []))}; faster than 1 m/s: {len(ph.get('faster_than_1mps', []))}; by kind: {ph.get('by_kind')}",
              f"- joints: {ar.get('n_ok')}/{ar.get('n_joints')} swept through full range with no stuck joint and <5 mm interpenetration ({ar.get('valve_handwheels')} handwheels, {ar.get('control_valve_stems')} control-valve stems); "
              f"failures: {[j['name'] for j in ar.get('joints', []) if not j['ok']]}",
              f"- pre-flight: initial interpenetrations >3 mm: {len(st['initial_interpenetrations'])}; floating loose items: {len(st['floating_loose_items'])}",
              f"- navigation: {st['navigation']['n_reachable']}/{st['navigation']['n_goals']} goals reachable on the 5 cm grid (robot radius {st['navigation']['robot_radius_m']} m), min path clearance {st['navigation']['min_path_clearance_m']} m",
              f"- reference-camera density (instance segmentation, >=30 px): {r.get('isaac', {}).get('stages', {}).get('render', {}).get('reference_view_density')}",
              f"- pipe network: {g.get('pipe_network')}",
              f"- scale check: {g.get('scale_check')}",
              f"- kit log: {r.get('kit_log', {}).get('n_fatal')} fatal, {r.get('kit_log', {}).get('n_errors')} errors", ""]
    (out / "validation_report.md").write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
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
    print("ALL PASS" if all(x["all_pass"] for x in summaries) else "SOME FAILED")


if __name__ == "__main__":
    main()
