"""``csg`` command line.  Every subcommand runs the Isaac Sim gate first (project policy).

    csg validate spec.yaml [--json]       schema + semantic validation (exit 1 on errors)
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
