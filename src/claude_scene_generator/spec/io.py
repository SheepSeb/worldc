"""Load / dump / migrate specs and turn pydantic errors into Issues."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from .issues import Issue, error, pointer
from .models import SPEC_VERSION, SceneSpec

# major.minor -> upgrader(dict) -> dict for the next version.  Empty while only 1.0 exists.
MIGRATIONS: dict[str, Any] = {}


class SpecLoader(yaml.SafeLoader):
    """YAML 1.2-ish booleans: only true/false.  YAML 1.1 would turn the placement key ``on:`` (and yes/no/off) into
    booleans, which silently breaks specs."""


SpecLoader.yaml_implicit_resolvers = {k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
                                      for k, v in yaml.SafeLoader.yaml_implicit_resolvers.copy().items()}
SpecLoader.add_implicit_resolver("tag:yaml.org,2002:bool", __import__("re").compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
                                 list("tTfF"))


def load_yaml(text: str):
    return yaml.load(text, Loader=SpecLoader)


def read(path_or_text: str | Path | dict) -> dict:
    """Spec dict from a dict, a .json/.yaml path, or JSON/YAML text."""
    if isinstance(path_or_text, dict):
        return path_or_text
    p = Path(str(path_or_text))
    text = p.read_text() if len(str(path_or_text)) < 4096 and p.exists() else str(path_or_text)
    return load_yaml(text) if not text.lstrip().startswith("{") else json.loads(text)


def migrate(d: dict) -> tuple[dict, list[Issue]]:
    v = str(d.get("spec_version", SPEC_VERSION))
    issues = []
    seen = set()
    while v != SPEC_VERSION:
        if v not in MIGRATIONS or v in seen:
            issues.append(error("schema.version", f"Unsupported spec_version {v!r}; this generator reads {SPEC_VERSION}.",
                                "/spec_version", f"Set spec_version to {SPEC_VERSION!r}."))
            return d, issues
        seen.add(v)
        d, v = MIGRATIONS[v](d)
    return d, issues


def _all_fields(model: type[BaseModel], acc=None) -> set[str]:
    """Every field name (and alias) reachable from ``model`` (for 'did you mean' hints)."""
    acc = set() if acc is None else acc
    schema = model.model_json_schema()

    def walk(s):
        if isinstance(s, dict):
            for k, v in s.get("properties", {}).items():
                acc.add(k)
            for v in s.values():
                walk(v)
        elif isinstance(s, list):
            for v in s:
                walk(v)

    walk(schema)
    return acc


_FIELDS: set[str] | None = None


def _real_path(data, loc, out: list) -> list:
    """Map a pydantic error location onto keys that exist in the input (drops union tags like 'inside', 'rack_aisles')."""
    cur = data
    for i, p in enumerate(loc):
        if isinstance(cur, dict) and isinstance(p, str) and p in cur:
            out.append(p)
            cur = cur[p]
        elif isinstance(cur, (list, tuple)) and isinstance(p, int) and 0 <= p < len(cur):
            out.append(p)
            cur = cur[p]
        elif isinstance(cur, dict) and isinstance(p, str) and i == len(loc) - 1:
            out.append(p)  # missing / extra field: keep the leaf name
        # anything else is a discriminator tag or type label -> skip
    return out


def pydantic_issues(e: ValidationError, data=None, prefix: tuple = ()) -> list[Issue]:
    global _FIELDS
    out = []
    for err in e.errors():
        clean = _real_path(data, err["loc"], list(prefix))
        path = pointer(*clean)
        typ = err["type"]
        hint = ""
        if typ == "extra_forbidden":
            if _FIELDS is None:
                _FIELDS = _all_fields(SceneSpec)
            bad = str(clean[-1]) if clean else ""
            close = difflib.get_close_matches(bad, sorted(_FIELDS), n=3)
            hint = f"Remove '{bad}'" + (f" or rename it to {' / '.join(repr(c) for c in close)}." if close else ".")
            code = "schema.extra_field"
        elif typ == "missing":
            code, hint = "schema.missing_field", f"Add the required field '{clean[-1] if clean else ''}'."
        elif typ in ("literal_error", "enum"):
            code, hint = "schema.bad_value", f"Allowed: {err.get('ctx', {}).get('expected', '')}."
        elif typ in ("placement", "asset_ref"):
            code, hint = f"schema.{typ}", err["msg"]
        else:
            code = f"schema.{typ}"
        out.append(error(code, err["msg"], path, hint, input=_short(err.get("input"))))
    return out


def _short(v, n=200):
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "..."


def parse(src: str | Path | dict) -> tuple[SceneSpec | None, list[Issue]]:
    """Read + migrate + schema-validate.  Returns (spec or None, issues)."""
    try:
        d = read(src)
    except (OSError, ValueError, yaml.YAMLError) as e:
        return None, [error("schema.unreadable", f"Could not read spec: {e}", "", "Provide valid JSON or YAML.")]
    if not isinstance(d, dict):
        return None, [error("schema.not_object", "Spec must be a JSON/YAML object.", "", "Wrap the spec in {...}.")]
    d, issues = migrate(d)
    if issues:
        return None, issues
    try:
        return SceneSpec.model_validate(d), []
    except ValidationError as e:
        return None, pydantic_issues(e, d)


def dump(spec: SceneSpec, fmt: str = "json") -> str:
    d = spec.model_dump(mode="json", by_alias=True, exclude_defaults=False)
    return json.dumps(d, indent=1) if fmt == "json" else yaml.safe_dump(d, sort_keys=False)
