"""MCP server exposing the scene tools (PLAN.md §6).  Needs the ``mcp`` extra and a working Isaac Sim install.

    uv run sceneforge-mcp                       # stdio, for Claude Code / Claude Desktop
    claude mcp add sceneforge -- uv run --directory /path/to/sceneforge sceneforge-mcp

Environment: SCENEFORGE_WORKSPACE (output root, default ./mcp_out), ISAACSIM_PATH, SCENEFORGE_ISAAC_ASSET_ROOT, SCENEFORGE_CATALOG_INDEX.
All tool logic lives in ``tools.py``; this module only registers the functions and attaches images.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from . import tools

INSTRUCTIONS = """Build physically valid 3D scenes (USD for Isaac Sim) from a declarative JSON spec.
Typical loop: search_assets / describe_asset -> write a spec (get_spec_schema for the format; docs/SPEC.md has examples)
-> validate_spec (fix issues using their `path` and `hint`) -> build_scene -> check_scene -> render (proxy is fast, rtx is
realistic) -> edit_scene with a JSON Patch to refine.  Prefer registry assets, then Omniverse catalog assets; generate an
asset only when nothing fits.  Every tool returns {ok, result, issues}; issues carry a JSON-pointer `path` and a `hint`."""


def _content(res: dict) -> list:
    """Tool result as MCP content: the JSON, plus the image if the tool produced one."""
    from mcp.server.mcpserver import Image

    out: list[Any] = [json.dumps(res, indent=1, default=str)]
    img = res.get("result", {}).get("image")
    if img:
        out.append(Image(path=img))
    return out


def build_server():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("sceneforge", instructions=INSTRUCTIONS)
    for name, fn in tools.TOOLS.items():
        def make(fn=fn):
            def call(**kw):
                return _content(fn(**kw))

            return call

        wrapped = make()
        wrapped.__name__, wrapped.__doc__ = name, fn.__doc__
        wrapped.__signature__ = __import__("inspect").signature(fn)
        wrapped.__annotations__ = {k: v for k, v in getattr(fn, "__annotations__", {}).items() if k != "return"}
        server.tool(name=name, description=fn.__doc__, structured_output=False)(wrapped)
    return server


def main(argv=None):
    from .isaac_env import gate_or_exit

    gate_or_exit()  # refuse to start (rather than fail per tool) without a usable Isaac Sim
    build_server().run("stdio")


if __name__ == "__main__":
    sys.exit(main())
