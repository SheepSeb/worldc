# Sceneforge

Declarative, agent-usable scene generation for Isaac Sim. A **scene spec** (JSON/YAML) describes the environment, zones,
objects and their relations, constraints, moving agents, the robot and its nav goals, cameras and lighting. The compiler
solves it into a physically valid USD scene and a scene graph. The goal is to train an LLM to write specs from natural
language; see [docs/PLAN.md](docs/PLAN.md).

> Work in progress on branch `refactor/spec-pipeline`. Done: spec, assets, solver, compiler, checks, proxy and RTX rendering,
> the warehouse recipe, the Omniverse catalog, the MCP server, and prompt-to-scene generation with dataset collection
> (`sceneforge generate`, see [docs/DATAGEN.md](docs/DATAGEN.md)).

## Requirements
**Isaac Sim ≥ 5.0 must be installed** (`ISAACSIM_PATH`, default `~/isaacsim`). Every command checks this first and refuses
to run without it. Generation itself only uses `usd-core`, numpy, scipy, Pillow, pydantic and PyYAML. Isaac Sim runs as a
subprocess for physics validation and RTX renders.

```bash
uv sync --all-extras          # extras: mesh (trimesh, coacd)
uv run sceneforge isaac-check --deep
```

## Quickstart
```bash
uv run sceneforge validate examples/specs/warehouse_lite.yaml
uv run sceneforge build examples/specs/warehouse_lite.yaml --seed 3 --out out/lite_3
uv run sceneforge render out/lite_3 --camera aisle            # fast preview, no Isaac needed for the render itself
uv run sceneforge recipe warehouse --seed 3 > wh3.json          # the original warehouse, as a spec (or --out DIR to build)
uv run python -m sceneforge.generate --seed 0 1 --out warehouse_out   # legacy CLI (same pipeline)
uv run python -m sceneforge.run_all --seeds 0 --skip-isaac           # + Isaac Sim validation without the flag
uv run sceneforge assets --tags door && uv run sceneforge describe rack_row
uv run sceneforge-import-mesh my_cart.glb --id cart_01 --tags cart,metal --mass 12
uv run pytest                                          # 400 tests; -m slow for the long ones
```

## Docs
- [docs/SPEC.md](docs/SPEC.md): spec reference, relations, constraints, outputs, issue codes
- [docs/ASSETS.md](docs/ASSETS.md): asset contract and sources (procedural, mesh, user file, generated)
- [docs/DATAGEN.md](docs/DATAGEN.md): `sceneforge generate "prompt"` (your Claude Code builds the scene), the collected dataset, exports
- [docs/MCP.md](docs/MCP.md): the `sceneforge-mcp` server (11 tools: assets, validate, build, check, render, edit) and how to register it
- [docs/CATALOG.md](docs/CATALOG.md): Isaac Sim asset catalog (offline index, search, query pinning) and its license findings
- [warehouse_out/README.md](warehouse_out/README.md): the warehouse scenes and how to load them in Isaac Sim / Isaac Lab
