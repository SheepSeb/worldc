"""``worldc`` command line.  Every subcommand runs the Isaac Sim gate first (project policy).

    worldc validate spec.yaml [--json]       schema + semantic validation (exit 1 on errors)
    worldc build spec.yaml --seed 3 --out out/scene_3 [--no-check] [--json]
    worldc check out/scene_3                 post-build checks (overlap, floating, hazards, nav, clearance)
    worldc recipe warehouse --seed 3 [--out out/wh_3]   print the recipe's spec (JSON), or build it with --out
    worldc render out/scene_3 --camera overhead [--map] [--width 960 --height 540]
    worldc schema [--section ObjectSpec]     print the JSON schema (or one $defs section)
    worldc assets [--tags door,dock] [--query valve]
    worldc describe rack_row
    worldc isaac-check [--deep]
    worldc drive out/scene_3 [--n-goals 3] [--video]              drive the real robot to its goals in Isaac Sim (+ chase / wide videos)
    worldc generate "prompt" [--model opus] [--max-budget-usd 3]    prompt -> built scene via your Claude Code; saved as a dataset record
    worldc generate --prompts-file prompts.txt                      one prompt per line, run in sequence
    worldc dataset stats | export [--format prompt_spec,sft,records,stats] | show RUN_ID | list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .isaac_env import gate_or_exit


def main(argv=None):
    ap = argparse.ArgumentParser(prog="worldc", description="Declarative scene generator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate a spec (schema + semantic)")
    v.add_argument("spec")
    v.add_argument("--json", action="store_true", help="machine-readable issues")
    b = sub.add_parser("build", help="compile a spec into USD + scene graph")
    b.add_argument("spec")
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--no-check", action="store_true")
    b.add_argument("--json", action="store_true")
    rc = sub.add_parser("recipe", help="print or build a recipe's spec")
    rc.add_argument("name")
    rc.add_argument("--seed", type=int, default=0)
    rc.add_argument("--out", default=None)
    c = sub.add_parser("check", help="post-build checks of a scene directory")
    c.add_argument("scene")
    r = sub.add_parser("render", help="fast proxy render (no Isaac Sim)")
    r.add_argument("scene")
    r.add_argument("--camera", default=None)
    r.add_argument("--map", action="store_true", help="semantic top-down map")
    r.add_argument("--width", type=int, default=960)
    r.add_argument("--height", type=int, default=540)
    s = sub.add_parser("schema", help="print the spec JSON schema")
    s.add_argument("--section", default=None)
    a = sub.add_parser("assets", help="list registered assets")
    a.add_argument("--tags", default="")
    a.add_argument("--query", default=None)
    d = sub.add_parser("describe", help="describe one asset")
    d.add_argument("asset_id")
    dr = sub.add_parser("drive", help="drive the scene's Isaac Sim robot to its goals in Isaac Sim physics")
    dr.add_argument("scene")
    dr.add_argument("--goals", default="", help="comma-separated goal ids (default: the first reachable, separated goals)")
    dr.add_argument("--n-goals", type=int, default=3)
    dr.add_argument("--video", action="store_true", help="also write chase and wide camera videos")
    dr.add_argument("--fps", type=int, default=30)
    dr.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    dr.add_argument("--speed", type=float, default=1.0)
    dr.add_argument("--out", default=None)
    i = sub.add_parser("isaac-check", help="check the Isaac Sim install")
    i.add_argument("--deep", action="store_true")
    g = sub.add_parser("generate", help="prompt -> scene through Claude Code (headless) + the worldc MCP server; collected as a dataset record")
    g.add_argument("prompt", nargs="?", help="scene description (or use --prompt-file / --prompts-file)")
    g.add_argument("--prompt-file", help="read the description from a file")
    g.add_argument("--prompts-file", help="one description per line; each becomes a run")
    g.add_argument("--dataset", default=None, help="dataset folder (default: $WORLDC_DATASET or ./dataset)")
    g.add_argument("--model", default=None, help="Claude Code model alias or id (default: your Claude Code default)")
    g.add_argument("--effort", default=None, help="low|medium|high|xhigh|max (default: Claude Code's)")
    g.add_argument("--max-budget-usd", type=float, default=3.0, help="spend cap per run, enforced by Claude Code")
    g.add_argument("--max-turns", type=int, default=60)
    g.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    g.add_argument("--no-final-checks", action="store_true", help="skip the Isaac physics check + RTX render after the run")
    g.add_argument("--keep-work", action="store_true", help="keep scratch builds from the run")
    g.add_argument("--claude-bin", default="claude")
    ds = sub.add_parser("dataset", help="inspect / export the collected dataset")
    ds.add_argument("action", choices=["stats", "export", "show", "list"])
    ds.add_argument("run_id", nargs="?", help="for `show`: a run id (or unique prefix)")
    ds.add_argument("--dataset", default=None)
    ds.add_argument("--format", default="prompt_spec,sft,records,stats")
    ds.add_argument("--include-rejected", action="store_true")
    ds.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "dataset":  # pure file work: no Isaac needed
        from .datagen.export import export, stats
        from .datagen.records import dataset_root, load_records

        root = dataset_root(args.dataset)
        if args.action == "stats":
            print(json.dumps(stats(load_records(root)), indent=1))
        elif args.action == "list":
            for r in load_records(root):
                print(f"{r.run_id}  {'ACCEPTED' if r.accepted else 'REJECTED':8s} ${r.usage.est_cost_usd:5.2f}  {r.description[:90]}")
        elif args.action == "show":
            from .datagen.view import write_markdown

            runs = [r for r in load_records(root) if r.run_id.startswith(args.run_id or "\0")]
            if len(runs) != 1:
                sys.exit(f"{len(runs)} runs match {args.run_id!r}; `worldc dataset list` shows them")
            print(write_markdown(root / "runs" / runs[0].run_id, runs[0]))
        else:
            for k, p in export(root, [f for f in args.format.split(",") if f], args.include_rejected, args.out).items():
                print(f"{k}: {p}")
        return
    gate_or_exit(deep=getattr(args, "deep", False))
    if args.cmd == "drive":
        from .drive import run_drive

        res = run_drive(args.scene, args.out, goal_ids=[g for g in args.goals.split(",") if g] or None, n_goals=args.n_goals, video=args.video,
                        fps=args.fps, res=tuple(args.res), speed=args.speed)
        for leg in res["legs"]:
            print(f"{'reached' if leg['reached'] else 'MISSED '} {leg['goal']}: {leg['distance_m']} m from the goal at t={leg['time_s']} s")
        print(f"{'OK' if res['ok'] else 'FAILED'}: {len(res['legs'])}/{res.get('planned_legs')} goals in {res['time_s']} s, max tilt {res['max_tilt_deg']} deg"
              + "".join(f"\n  video: {v}" for v in res["videos"].values()))
        sys.exit(0 if res["ok"] else 1)
    if args.cmd == "generate":
        from .datagen.records import dataset_root
        from .datagen.runner import generate

        prompts_ = ([ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
                    if args.prompts_file else [Path(args.prompt_file).read_text().strip() if args.prompt_file else args.prompt])
        if not prompts_ or not prompts_[0]:
            sys.exit("give a prompt, --prompt-file or --prompts-file")
        root = dataset_root(args.dataset)
        recs = [generate(p, root=root, model=args.model, effort=args.effort, max_budget_usd=args.max_budget_usd, max_turns=args.max_turns,
                         timeout_s=args.timeout, final_checks=not args.no_final_checks, keep_work=args.keep_work, claude_bin=args.claude_bin)
                for p in prompts_]
        for r in recs:
            print(json.dumps({"run_id": r.run_id, "accepted": r.accepted, "stop_reason": r.stop_reason, "scene": str(root / r.scene_dir) if r.scene_dir else None,
                              "spec": str(root / "runs" / r.run_id / "spec.json") if r.spec else None, "cost_usd": r.usage.est_cost_usd, "summary": r.summary}))
        print(f"dataset: {root}  (records.jsonl, runs/)", file=sys.stderr)
        sys.exit(0 if all(r.accepted for r in recs) else 1)

    if args.cmd == "validate":
        from .spec.validate import check

        spec, issues = check(args.spec)
        errors = [x for x in issues if x.severity == "error"]
        if args.json:
            print(json.dumps({"ok": not errors, "issues": [x.model_dump() for x in issues]}, indent=1))
        else:
            for x in issues:
                print(x)
            print(f"{'OK' if not errors else 'INVALID'}: {len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
        sys.exit(1 if errors else 0)
    if args.cmd == "build":
        from .compile import compile_scene

        res = compile_scene(args.spec, args.seed, args.out, run_check=not args.no_check)
        if args.json:
            print(json.dumps(res.summary(), indent=1, default=str))
        else:
            for x in res.issues:
                print(x)
            print(f"{'OK' if res.ok else 'FAILED'}: {res.out_dir or '-'} ({res.seconds:.1f} s)")
        sys.exit(0 if res.ok else 1)
    if args.cmd == "recipe":
        from .compile import compile_scene
        from .recipes import RECIPES

        if args.name not in RECIPES:
            sys.exit(f"unknown recipe {args.name!r}; available: {', '.join(sorted(RECIPES))}")
        spec = RECIPES[args.name](args.seed)
        if not args.out:
            print(json.dumps(spec.model_dump(mode="json", by_alias=True, exclude_unset=True), indent=1))
            return
        res = compile_scene(spec, args.seed, args.out)
        print(f"{'OK' if res.ok else 'FAILED'}: {res.out_dir} ({res.seconds:.1f} s)")
        sys.exit(0 if res.ok else 1)
    if args.cmd == "check":
        from .check import check_scene

        r_ = check_scene(args.scene)
        for x in r_["issues"]:
            print(f"[{x['severity']}] {x['code']}: {x['message']} {x['hint']}")
        print("PASSED" if r_["passed"] else "FAILED")
        sys.exit(0 if r_["passed"] else 1)
    if args.cmd == "render":
        from .render import render_proxy, render_topdown_map

        if args.map or not args.camera:
            print(render_topdown_map(args.scene))
        if args.camera:
            print(render_proxy(args.scene, args.camera, width=args.width, height=args.height))
        return
    if args.cmd == "schema":
        from .spec.schema import build_schema, section

        print(json.dumps(section(args.section) if args.section else build_schema(), indent=1))
    elif args.cmd == "assets":
        from .assets import list_assets

        for x in list_assets([t for t in args.tags.split(",") if t], args.query):
            print(f"{x['id']:22s} [{x['source']}] {x['description']}")
    elif args.cmd == "describe":
        from .assets import describe

        print(json.dumps(describe(args.asset_id), indent=1))
    elif args.cmd == "isaac-check":
        print("Isaac Sim OK")


if __name__ == "__main__":
    main()
