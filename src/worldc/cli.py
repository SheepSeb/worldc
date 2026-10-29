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
    worldc generate "prompt" [--model opus] [--max-budget-usd 20]    prompt -> built scene via your Claude Code; saved as a dataset record
    worldc generate --prompts-file prompts.txt                      one prompt per line, run in sequence
    worldc simready search "toaster" | info PKG [--probe] | fetch PKG | index | cached | clean   NVIDIA SimReady (Hugging Face)
    worldc eval --runs DIR --prompts prompts/prompts.csv --out eval.csv [--prompt-id ids.csv]   adherence + plausibility per run
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
    rc.add_argument("--prompts", default=None, help="remanufacturing: one scene per prompts.csv row into --out/<prompt_id>/")
    rc.add_argument("--ids", default="", help="remanufacturing: only these comma-separated prompt ids")
    rc.add_argument("--seeds", type=int, default=30, help="e2_template: build seeds 0..N-1 into --out/<prompt>_seed<NN>/")
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
    g.add_argument("--max-budget-usd", type=float, default=20.0, help="spend cap per run, enforced by Claude Code")
    g.add_argument("--max-turns", type=int, default=120)
    g.add_argument("--allow-unfrozen", action="store_true", help="run a --prompts-file batch off the freeze tag / with a dirty tree")
    g.add_argument("--seed", type=int, default=None, help="pin the build seed for every build_scene of the run (default: the model's choice)")
    g.add_argument("--max-repairs", type=int, default=10, help="stop after this many validate_spec/build_scene calls returned errors (-1: no cap)")
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
    sub.add_parser("simready", help="NVIDIA SimReady assets on Hugging Face: index / search / info / fetch (see -h)", add_help=False)
    sub.add_parser("eval", help="post-hoc adherence + plausibility per run -> CSV (see -h)", add_help=False)
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["simready"]:  # own sub-commands; network / file work only, so no Isaac gate
        from .assets.simready import main as simready_main

        return simready_main(argv[1:])
    if argv[:1] == ["eval"]:  # post-hoc evaluation from scene_graph.json + USD colliders: no Isaac gate
        from .eval.run import main as eval_main

        return eval_main(argv[1:])
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
        if args.prompts_file and not args.allow_unfrozen:  # a batch is data collection: only from the exact frozen checkout
            from .datagen.runner import freeze_info

            fi = freeze_info()
            if fi["at_freeze"] != "true":
                sys.exit(f"refusing a batch: HEAD {fi['head_commit'][:12] or '?'} is not {fi['freeze_tag']} "
                         f"({fi['freeze_commit'][:12] or 'missing'}){' or the tree is dirty' if fi['tree_dirty'] == 'true' else ''}. "
                         f"Check out the tag in a clean worktree (or pass --allow-unfrozen for a trial).")
        root = dataset_root(args.dataset)
        recs = [generate(p, root=root, model=args.model, effort=args.effort, max_budget_usd=args.max_budget_usd, max_turns=args.max_turns,
                         max_repairs=None if args.max_repairs < 0 else args.max_repairs, seed=args.seed, timeout_s=args.timeout, final_checks=not args.no_final_checks, keep_work=args.keep_work, claude_bin=args.claude_bin)
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
    if args.cmd == "recipe" and args.name == "remanufacturing":  # E2 baseline; not in RECIPES (the LLM's list_recipes / get_recipe)
        from .recipes.remanufacturing import build_all

        if not args.prompts or not args.out:
            sys.exit("worldc recipe remanufacturing --prompts prompts/prompts.csv --out out/e2/recipe [--ids P01,P07]")
        res = build_all(args.prompts, args.out, [i for i in args.ids.split(",") if i] or None)
        print(f"{sum(r['accepted'] for r in res)}/{len(res)} accepted ({sum(r['first_try'] for r in res)} first try) -> {args.out}")
        sys.exit(0 if all(r["accepted"] for r in res) else 1)
    if args.cmd == "recipe" and args.name == "e2_template":  # E2 template baseline (specs/e2_template_reman_cell.json)
        from .recipes.remanufacturing import build_template

        if not args.out:
            sys.exit("worldc recipe e2_template --out out/e2/template [--seeds 30]")
        res = build_template(args.out, args.seeds)
        print(f"{sum(r['accepted'] for r in res)}/{len(res)} accepted -> {args.out}")
        sys.exit(0 if all(r["accepted"] for r in res) else 1)
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
