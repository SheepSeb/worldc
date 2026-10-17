"""Small RFC 6902 (JSON Patch) implementation for ``edit_scene``: add, remove, replace, move, copy, test.

    new = apply_patch(doc, [{"op": "replace", "path": "/objects/2/count", "value": 8}])   # doc is not modified

A failing operation raises ``PatchError`` carrying the op index, so the caller can point the LLM at the exact operation.
Paths are RFC 6901 JSON pointers (``~0`` = ``~``, ``~1`` = ``/``); ``-`` appends to an array.
"""
from __future__ import annotations

import copy
from typing import Any

OPS = ("add", "remove", "replace", "move", "copy", "test")


class PatchError(ValueError):
    def __init__(self, index: int, op: str, message: str, hint: str = ""):
        super().__init__(f"patch op {index} ({op}): {message}")
        self.index, self.op, self.message, self.hint = index, op, message, hint


def _tokens(ptr: str) -> list[str]:
    if ptr == "":
        return []
    if not ptr.startswith("/"):
        raise ValueError(f"JSON pointer must be '' or start with '/': {ptr!r}")
    return [t.replace("~1", "/").replace("~0", "~") for t in ptr[1:].split("/")]


def _index(container: list, tok: str, allow_end: bool) -> int:
    if tok == "-" and allow_end:
        return len(container)
    if not tok.isdigit() or (len(tok) > 1 and tok[0] == "0"):
        raise ValueError(f"{tok!r} is not a valid array index")
    i = int(tok)
    if i > len(container) or (i == len(container) and not allow_end):
        raise ValueError(f"array index {i} is out of range (length {len(container)})")
    return i


def _walk(doc: Any, toks: list[str]) -> Any:
    cur = doc
    for t in toks:
        if isinstance(cur, dict):
            if t not in cur:
                raise ValueError(f"no key {t!r}" + (f" (keys: {', '.join(sorted(map(str, cur))[:12])})" if cur else ""))
            cur = cur[t]
        elif isinstance(cur, list):
            cur = cur[_index(cur, t, False)]
        else:
            raise ValueError(f"cannot descend into {type(cur).__name__} with {t!r}")
    return cur


def _get(doc: Any, ptr: str) -> Any:
    return _walk(doc, _tokens(ptr))


def _add(doc: Any, ptr: str, value: Any) -> Any:
    toks = _tokens(ptr)
    if not toks:
        return value
    parent = _walk(doc, toks[:-1])
    last = toks[-1]
    if isinstance(parent, dict):
        parent[last] = value
    elif isinstance(parent, list):
        parent.insert(_index(parent, last, True), value)
    else:
        raise ValueError(f"cannot add into {type(parent).__name__}")
    return doc


def _remove(doc: Any, ptr: str) -> tuple[Any, Any]:
    toks = _tokens(ptr)
    if not toks:
        raise ValueError("cannot remove the document root")
    parent = _walk(doc, toks[:-1])
    last = toks[-1]
    if isinstance(parent, dict):
        if last not in parent:
            raise ValueError(f"no key {last!r} to remove")
        return doc, parent.pop(last)
    if isinstance(parent, list):
        return doc, parent.pop(_index(parent, last, False))
    raise ValueError(f"cannot remove from {type(parent).__name__}")


def apply_patch(doc: Any, patch: list[dict]) -> Any:
    """Return a patched deep copy of ``doc``.  Raises ``PatchError`` on the first bad operation."""
    if not isinstance(patch, list):
        raise PatchError(-1, "", "a patch must be a list of operations", 'e.g. [{"op": "replace", "path": "/objects/0/count", "value": 4}]')
    out = copy.deepcopy(doc)
    for i, op in enumerate(patch):
        name = op.get("op") if isinstance(op, dict) else None
        try:
            if name not in OPS:
                raise ValueError(f"unknown op {name!r}; use one of {', '.join(OPS)}")
            if "path" not in op:
                raise ValueError("missing 'path'")
            path = op["path"]
            if name in ("add", "replace", "test") and "value" not in op:
                raise ValueError("missing 'value'")
            if name in ("move", "copy") and "from" not in op:
                raise ValueError("missing 'from'")
            if name == "add":
                out = _add(out, path, copy.deepcopy(op["value"]))
            elif name == "remove":
                out, _ = _remove(out, path)
            elif name == "replace":
                _get(out, path)  # must exist
                toks = _tokens(path)
                if not toks:
                    out = copy.deepcopy(op["value"])
                else:
                    parent = _walk(out, toks[:-1])
                    key = toks[-1] if isinstance(parent, dict) else _index(parent, toks[-1], False)
                    parent[key] = copy.deepcopy(op["value"])
            elif name == "test":
                if _get(out, path) != op["value"]:
                    raise ValueError(f"value at {path} is {_get(out, path)!r}, expected {op['value']!r}")
            elif name == "copy":
                out = _add(out, path, copy.deepcopy(_get(out, op["from"])))
            else:  # move
                if path.startswith(op["from"] + "/"):
                    raise ValueError("cannot move a value into one of its own children")
                out, val = _remove(out, op["from"])
                out = _add(out, path, val)
        except (ValueError, KeyError, TypeError) as e:
            hint = "Paths are JSON pointers into the spec, e.g. /objects/2/count; '-' appends to a list."
            raise PatchError(i, str(name), str(e).strip("'\""), hint) from e
    return out


def diff_summary(old: Any, new: Any, limit: int = 40) -> list[str]:
    """Human-readable list of the leaf changes between two JSON documents (``/path: old -> new``)."""
    lines: list[str] = []

    def rec(a, b, path):
        if len(lines) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a:
                    lines.append(f"+ {path}/{k}")
                elif k not in b:
                    lines.append(f"- {path}/{k}")
                else:
                    rec(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                rec(x, y, f"{path}/{i}")
        elif a != b:
            if isinstance(a, list) and isinstance(b, list):
                lines.append(f"~ {path}: list length {len(a)} -> {len(b)}")
            else:
                lines.append(f"~ {path}: {_short(a)} -> {_short(b)}")

    rec(old, new, "")
    if len(lines) >= limit:
        lines.append("... (more changes not shown)")
    return lines


def _short(v: Any, n: int = 60) -> str:
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."
