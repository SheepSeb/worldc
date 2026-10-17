# Worldc

**Describe a place. Get a simulation-ready environment you can verify.**

Worldc lets an LLM agent build [Isaac Sim](https://developer.nvidia.com/isaac/sim) environments from a plain-language request,
and proves the result is correct. The agent writes a short declarative *scene spec*; Worldc validates it, solves the layout,
builds the USD scene, checks geometry, navigation and physics, and renders it in Isaac Sim. When something is wrong, the agent gets a
precise, actionable error instead of a silent mistake, so it fixes the scene instead of guessing.

[![Worldc demo](docs/media/worldc_demo_poster.png)](docs/media/worldc_demo.mp4)

## What you get

From one request, with the same spec and seed always producing the same scene:

| output | what it is |
|---|---|
| `scene.usd` | the Isaac Sim scene: geometry, physics, articulations, lights, cameras |
| `scene_graph.json` | every object with its pose, size, mass, joints, goals and cameras |
| `resolved_spec.json` | the fully grounded spec: rebuilding it reproduces the scene exactly |
| `check.json` | the verification report (below) |
| `renders/` | Isaac Sim RTX images from every camera |
| `drive/` | optional: the robot driving to its goals in Isaac Sim, with chase and wide camera videos |

## How it works

```
   "A 24x16 m warehouse with three rack rows, a conveyor, a forklift, and a robot ..."
                                         │
                                         ▼
                ┌───────────────────┐   tools (MCP)    ┌──────────────────────────────┐
                │     LLM agent     │ ───────────────▶ │  Worldc                      │
                │ (any model that   │                  │   search assets · validate   │
                │  can call tools)  │ ◀─────────────── │   solve · build · check ·    │
                └───────────────────┘  issues + hints  │   render                     │
                                                       └───────────────┬──────────────┘
                                                                       ▼
                         scene.usd · scene_graph.json · resolved_spec.json · check.json · renders
```

The agent never writes coordinates by hand. It states *relations* ("pallets inside this zone", "table against the north wall",
"conveyor next to the packing table", "aisles at least 1.2 m"), and the solver finds placements that satisfy them.

## Why you can trust the result

Every scene passes the same gates, and each failure names the exact place in the spec and suggests a fix.

| gate | what is verified |
|---|---|
| **Schema and semantics** | unknown fields are errors; every reference resolves; asset parameters are in range; counts are feasible |
| **Layout solver** | relations are satisfied; nothing overlaps; clearance targets (for example "aisle at least 1.2 m") are measured exactly |
| **Post-build checks** | no interpenetration at spawn, nothing floating, moving hazards stay on clear paths, every goal is reachable for the robot's real footprint on a 5 cm navigation grid |
| **Isaac Sim physics** | the scene is simulated: bodies come to rest, nothing falls through the floor, the robot settles where it was placed |
| **Robot drives the route** | optional and the strongest check: the real robot is driven to its goals in Isaac Sim physics using wheel commands only, while forklifts and people move on their paths |
| **Isaac Sim renders** | every camera is rendered with RTX and checked for black frames, blown-out frames and missing textures |

A validation error looks like this, and is what the agent sees:

```
[error] asset.unknown at /objects/0/asset: Unknown asset id 'palet'. Hint: Use list_assets / search_assets to find an id. Did you mean 'pallet', 'pallet_load'?
```

## Real robots, real assets

- **The robot and the forklift are real Isaac Sim assets.** The robot is always a real Isaac Sim robot (Nova Carter, Carter, iw.hub, Jackal, Dingo, TurtleBot3, Kaya, O3dyn, Ridgeback with
  Franka, Franka, and more). It is referenced from NVIDIA's asset library with its own articulation, joints and sensors, and is
  never generated, and a spec that asks for an invented robot is rejected (see `list_assets(tags=["robot"])`). Forklifts and people are NVIDIA's
  Isaac Sim forklift and characters, driving and walking along waypoint paths as moving hazards whose corridors the checks keep clear.
- **Assets:** a registry of procedural, physics-ready assets (racks, pallets, conveyors, doors, furniture, hazards), your own
  meshes (imported with collision hulls), a searchable offline index of 1,500 NVIDIA Omniverse assets, and declarative generated
  assets for gaps. Search and resolution prefer an existing asset over generating a new one.

## Quickstart

**Requirements:** Python 3.12, [uv](https://docs.astral.sh/uv/), and **Isaac Sim 5.0 or newer** (`ISAACSIM_PATH`, default
`~/isaacsim`). Every command checks for Isaac Sim first and refuses to run without it.

```bash
uv sync --all-extras
uv run worldc isaac-check --deep

uv run worldc validate examples/specs/warehouse_lite.yaml
uv run worldc build examples/specs/warehouse_lite.yaml --seed 3 --out out/lite_3
uv run worldc check out/lite_3                       # the verification report again
uv run worldc render out/lite_3 --camera aisle       # fast preview render
uv run worldc drive out/lite_3 --video               # drive the robot to its goals in Isaac Sim, with chase and wide videos
```

More: `worldc assets --tags robot`, `worldc describe rack_row`, `worldc schema --section ObjectSpec`,
`worldc-import-mesh my_cart.glb --id cart_01 --mass 12`.

## Use it with an LLM agent

`worldc-mcp` is an [MCP](https://modelcontextprotocol.io) server, so any agent or client that supports MCP tools can drive it:

```json
{"mcpServers": {"worldc": {"command": "uv", "args": ["run", "--directory", "/path/to/worldc", "worldc-mcp"]}}}
```

| tool | purpose |
|---|---|
| `list_assets`, `search_assets`, `describe_asset` | find assets (registry first, then the Omniverse catalog) and read their parameters, anchors and physics |
| `get_spec_schema` | the spec format, whole or one section at a time |
| `list_recipes`, `get_recipe` | complete starting-point scenes (for example a warehouse) that the agent adapts instead of starting from nothing |
| `validate_spec` | schema and semantic checks plus a dry-run solve, with per-object placed counts |
| `build_scene` | solve and build the USD scene |
| `check_scene` | the verification report, including the Isaac Sim physics settle |
| `drive_robot` | drive the real robot to its goals in Isaac Sim physics and report whether it got there |
| `render` | fast proxy preview, or Isaac Sim RTX |
| `edit_scene` | apply a JSON Patch to a spec and re-validate |
| `import_asset`, `validate_generated_asset` | bring in your own mesh, or check a generated asset |

The agent loop is: search assets, write a spec, `validate_spec`, fix the issues it reports, `build_scene`, `check_scene`, look at
the renders, then `edit_scene` to refine. Details in [docs/MCP.md](docs/MCP.md).

## A spec, in one screen

```yaml
spec_version: "1.0"
name: minimal_room
environment:
  footprint: {type: rect, size: [6, 5]}
  height: 3.0
  walls: {openings: [{id: door, wall: south, center: 1.5, width: 1.15, height: 2.24, door: swing_door}]}
objects:
  - {id: table, asset: work_table, placement: {against: "wall:north"}}
  - {id: cartons, asset: cardboard_box, count: 2, placement: {on: table.top}}
  - {id: cone, asset: traffic_cone, placement: {near: {target: [-2.0, -1.0], radius: 0.5}}}
robot:
  asset: nova_carter                       # a real Isaac Sim robot
  spawn: {pose: {xy: [1.5, -1.0], yaw: 90}}
  nav_goals: [{id: at_table, type: inspect, target: table.front, approach: port}]
constraints:
  - {type: reachable, to: ["*"]}
```

## Documentation

- [docs/SPEC.md](docs/SPEC.md): the spec reference: relations, constraints, robots, outputs, every issue code
- [docs/ASSETS.md](docs/ASSETS.md): the asset contract and sources (procedural, mesh, user file, generated)
- [docs/CATALOG.md](docs/CATALOG.md): the Isaac Sim asset catalog: offline index, search, and the license notes for NVIDIA content
- [docs/MCP.md](docs/MCP.md): the MCP server and its tools

## Development

```bash
uv run pytest            # about 490 tests
uv run pytest -m slow    # the longer end-to-end ones
```

Layout: `src/worldc/spec` (models, schema, validation), `solver` (layout), `assets` (registry, robots, catalog), `compile.py`
(USD build), `check` (post-build checks), `render` (proxy and RTX), `tools.py` and `mcp_server.py` (the agent tools),
`recipes` (parameterised starting points such as the warehouse), `examples/specs`.
