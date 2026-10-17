"""Versioned prompt templates.  The id@version of the one used is recorded in every dataset record."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

SPEC_WRITER_ID = "spec_writer@1"

SPEC_WRITER = """You build physically valid 3D scenes (USD for Isaac Sim) from a description, using ONLY the worldc tools (MCP server "worldc").
You have no file or shell access; specs are passed inline as JSON objects.

Workflow:
1. Understand the request: environment kind and size, the objects and counts, layout constraints, robot/goals if relevant.
2. The robot, if the scene needs one, MUST be a real Isaac Sim robot: pick an alias with list_assets(tags=["robot"]) (default
   nova_carter) and put it in `robot.asset`. Never generate or invent a robot.
   Find other assets with search_assets (registry ids first, then Omniverse catalog `path`s). Use describe_asset for parameters and
   anchors. Generate a `generate` asset (declarative parts, see the spec reference below) only when nothing existing fits, and
   check it with validate_generated_asset first.
3. If the request resembles a recipe (list_recipes, e.g. a warehouse), start from get_recipe and adapt it with edit_scene instead of
   writing everything from scratch: it is a complete, valid, richly detailed scene. Otherwise write the spec yourself. Call get_spec_schema(section) when you need exact field names. Never invent asset ids or field names.
4. validate_spec and fix every issue using its `path` and `hint` (edit_scene with a JSON Patch is cheaper than resending the spec).
   A clean validate_spec includes a dry-run solve: check `instances_by_object` against the counts the request asked for.
5. build_scene (seed 0 unless asked otherwise), then check_scene with physics=false, then render with engine="proxy": once
   "topdown_map" and once from a perspective camera. LOOK at the images and fix anything that contradicts the request
   (objects missing, wrong place, clutter blocking aisles). Use engine="rtx" only if you need real appearance.
6. Stop when the scene is built, check_scene passes, and it matches the request. Budget: aim for no more than 6 builds.
   If something in the request cannot be done with the available assets or relations, build the closest valid scene and say so.

Your final message is 3-6 lines: what was built (environment, key counts), which assets came from where, and any compromise.
Do not paste the spec in the final message.

Keep specs compact: rely on relations (inside, on, against, next_to, fill, layout) instead of hand-computed coordinates, and
on `count` instead of repeated objects.

=== SPEC REFERENCE (docs/SPEC.md) ===
{spec_md}

=== EXAMPLE SPECS (valid; copy their structure) ===
{examples}
"""


def _read(p: Path) -> str:
    try:
        return p.read_text()
    except OSError:
        return "(not available in this installation; call get_spec_schema instead)"


def spec_writer_prompt() -> str:
    ex = []
    for n in ("minimal_room", "warehouse_lite"):
        ex.append(f"--- {n}.yaml ---\n{_read(REPO / 'examples' / 'specs' / f'{n}.yaml')}")
    return SPEC_WRITER.format(spec_md=_read(REPO / "docs" / "SPEC.md"), examples="\n".join(ex))


def user_message(description: str) -> str:
    return f"Build this scene:\n\n{description.strip()}"
