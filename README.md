# claude-scene-generator

Declarative, agent-usable scene generation for Isaac Sim. A **scene spec** (JSON/YAML) describes the environment, zones,
objects and their relations, constraints, moving agents, the robot and its nav goals, cameras and lighting. The compiler
solves it into a physically valid USD scene and a scene graph. The goal is to train an LLM to write specs from natural
language; see [docs/PLAN.md](docs/PLAN.md).

> Work in progress on branch `refactor/spec-pipeline`. Phases 0–5 are done: spec, assets, solver, compiler, checks, the
> proxy renderer and the warehouse recipe. The MCP server and data collection follow.

## Requirements
**Isaac Sim ≥ 5.0 must be installed** (`ISAACSIM_PATH`, default `~/isaacsim`). Every command checks this first and refuses
to run without it. Generation itself only uses `usd-core`, numpy, scipy, Pillow, pydantic and PyYAML. Isaac Sim runs as a
subprocess for physics validation and RTX renders.

```bash
uv sync --all-extras          # extras: mesh (trimesh, coacd)
uv run csg isaac-check --deep
```

## Quickstart
```bash
uv run csg validate examples/specs/warehouse_lite.yaml
uv run csg build examples/specs/warehouse_lite.yaml --seed 3 --out out/lite_3
uv run csg render out/lite_3 --camera aisle            # fast preview, no Isaac needed for the render itself
uv run csg recipe warehouse --seed 3 > wh3.json          # the original warehouse, as a spec (or --out DIR to build)
uv run python -m claude_scene_generator.generate --seed 0 1 --out warehouse_out   # legacy CLI (same pipeline)
uv run python -m claude_scene_generator.run_all --seeds 0 --skip-isaac           # + Isaac Sim validation without the flag
uv run csg assets --tags door && uv run csg describe rack_row
uv run csg-import-mesh my_cart.glb --id cart_01 --tags cart,metal --mass 12
uv run pytest                                          # 400 tests; -m slow for the long ones
```

## Docs
- [docs/SPEC.md](docs/SPEC.md): spec reference, relations, constraints, outputs, issue codes
- [docs/ASSETS.md](docs/ASSETS.md): asset contract and sources (procedural, mesh, user file, generated)
- [warehouse_out/README.md](warehouse_out/README.md): the warehouse scenes and how to load them in Isaac Sim / Isaac Lab
