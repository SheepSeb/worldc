"""Structured, LLM-actionable problems reported by every stage (schema, semantic, solve, build, check).

``code`` is a stable dotted identifier (documented in docs/SPEC.md); ``path`` is a JSON pointer into the spec
(``/objects/3/params/size``) or ``""`` for scene-level issues; ``hint`` suggests a concrete edit.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["error", "warning", "info"]


class Issue(BaseModel):
    severity: Severity = "error"
    code: str
    path: str = ""
    message: str
    hint: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        loc = f" at {self.path}" if self.path else ""
        return f"[{self.severity}] {self.code}{loc}: {self.message}" + (f" Hint: {self.hint}" if self.hint else "")


def error(code: str, message: str, path: str = "", hint: str = "", **data) -> Issue:
    return Issue(severity="error", code=code, path=path, message=message, hint=hint, data=data)


def warning(code: str, message: str, path: str = "", hint: str = "", **data) -> Issue:
    return Issue(severity="warning", code=code, path=path, message=message, hint=hint, data=data)


def has_errors(issues) -> bool:
    return any(i.severity == "error" for i in issues)


def pointer(*parts) -> str:
    """JSON pointer from path parts: pointer("objects", 3, "params") -> "/objects/3/params"."""
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)
