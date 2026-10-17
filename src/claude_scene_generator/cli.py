"""``csg`` command line.  Every subcommand runs the Isaac Sim gate first (project policy).

    csg validate spec.yaml [--json]       schema + semantic validation (exit 1 on errors)
    csg build spec.yaml --seed 3 --out out/scene_3 [--no-check] [--json]
    csg check out/scene_3                 post-build checks (overlap, floating, hazards, nav, clearance)
    csg recipe warehouse --seed 3 [--out out/wh_3]   print the recipe's spec (JSON), or build it with --out
    csg render out/scene_3 --camera overhead [--map] [--width 960 --height 540]
    csg schema [--section ObjectSpec]     print the JSON schema (or one $defs section)
    csg assets [--tags door,dock] [--query valve]
    csg describe rack_row
    csg isaac-check [--deep]
"""
from __future__ import annotations

import argparse
import json
import sys

from .isaac_env import gate_or_exit


def main(argv=None):
    ap = argparse.ArgumentParser(prog="csg", description="Declarative scene generator")
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
    i = sub.add_parser("isaac-check", help="check the Isaac Sim install")
    i.add_argument("--deep", action="store_true")
    args = ap.parse_args(argv)

    gate_or_exit(deep=getattr(args, "deep", False))

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
            print(json.dumps(spec.model_dump(mode="json", by_alias=True, exclude_defaults=True), indent=1))
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
