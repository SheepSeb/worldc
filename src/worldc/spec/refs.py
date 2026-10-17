"""Reference parsing (see the grammar in models.Ref).  Shared by validation, the solver and the compiler."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedRef:
    raw: str
    kind: str  # env | wall | clearance | robot | object
    target: str  # object id / env item / wall side / robot item
    instance: int | str | None = None  # int, "*" or None (= all instances)
    anchor: str | None = None  # anchor / port name or pattern

    @property
    def has_wildcard(self) -> bool:
        return self.instance == "*" or (self.anchor is not None and any(c in self.anchor for c in "*?["))


def parse_ref(ref: str) -> ParsedRef:
    """``wall:south`` | ``env.<item>`` | ``robot.<item>`` | ``obj`` | ``obj.N`` | ``obj.*`` | ``obj.anchor`` | ``obj.N.anchor``."""
    if ref.startswith("wall:"):
        return ParsedRef(ref, "wall", ref[5:])
    if ref.startswith("clearance:"):
        return ParsedRef(ref, "clearance", ref[10:])
    parts = ref.split(".")
    if parts[0] == "env":
        return ParsedRef(ref, "env", ".".join(parts[1:]) or "floor")
    if parts[0] == "robot":
        return ParsedRef(ref, "robot", ".".join(parts[1:]) or "spawn")
    obj, rest = parts[0], parts[1:]
    inst = None
    if rest and (rest[0].isdigit() or rest[0] == "*"):
        inst = int(rest[0]) if rest[0].isdigit() else "*"
        rest = rest[1:]
    anchor = ".".join(rest) if rest else None
    return ParsedRef(ref, "object", obj, inst, anchor)


def match(pattern: str, name: str) -> bool:
    return fnmatch.fnmatchcase(name, pattern)
