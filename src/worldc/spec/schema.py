"""JSON Schema export for SceneSpec (checked in as spec/schema.json; a test keeps them in sync).

    python -m worldc.spec.schema [--out PATH] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import SPEC_VERSION, SceneSpec

SCHEMA_PATH = Path(__file__).with_name("schema.json")


def build_schema() -> dict:
    s = SceneSpec.model_json_schema(by_alias=True)
    s["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    s["$id"] = f"https://worldc/spec/{SPEC_VERSION}/scene.schema.json"
    s["title"] = "SceneSpec"
    defs = s.setdefault("$defs", {})
    # the generated-asset and texture models are validated by their own validators (specific gen.* codes) but are
    # part of the contract, so expose their schemas here
    from ..assets.generated import GeneratedAssetSpec
    from ..texturegen import TextureSpec

    for model in (GeneratedAssetSpec, TextureSpec):
        sub = model.model_json_schema(ref_template="#/$defs/{model}")
        defs.update(sub.pop("$defs", {}))
        defs[model.__name__] = sub
    defs["GenerateAssetRef"]["properties"]["generate"] = {"$ref": "#/$defs/GeneratedAssetSpec"}
    return s


def section(name: str) -> dict:
    """One $defs entry (plus the defs it references, transitively) -- for token-cheap get_spec_schema(section)."""
    s = build_schema()
    defs = s["$defs"]
    if name in ("SceneSpec", "", None):
        return s
    if name not in defs:
        raise KeyError(f"unknown schema section {name!r}; available: {', '.join(sorted(defs))}")
    out, todo = {}, [name]
    while todo:
        n = todo.pop()
        if n in out:
            continue
        out[n] = defs[n]
        txt = json.dumps(defs[n])
        todo += [d for d in defs if f'"#/$defs/{d}"' in txt]
    return {"$ref": f"#/$defs/{name}", "$defs": out}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCHEMA_PATH))
    ap.add_argument("--check", action="store_true", help="exit 1 if the checked-in schema is stale")
    a = ap.parse_args(argv)
    txt = json.dumps(build_schema(), indent=1, sort_keys=True) + "\n"
    if a.check:
        ok = Path(a.out).exists() and Path(a.out).read_text() == txt
        print("schema up to date" if ok else "schema.json is stale: run python -m worldc.spec.schema")
        sys.exit(0 if ok else 1)
    Path(a.out).write_text(txt)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
