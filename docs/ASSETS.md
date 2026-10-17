# Assets

All assets implement one contract (`src/worldc/assets/base.py`):

| member | |
|---|---|
| `id`, `description`, `tags`, `source` | what `list_assets` / `describe_asset` show the LLM |
| `Params` | strict pydantic model. Its JSON schema, ranges and defaults are part of `describe()` |
| `footprint(p)`, `anchors(p)`, `keepouts(p)`, `joints(p)` | **pure** functions of params, used by the solver before any USD exists |
| `build(ctx, path, pose, p, rng)` | authors the instance under `path` and returns `Built` (world-frame anchors, keep-outs, info, bodies) |

`pose.pos` z is the height of the supporting surface; assets add their own settle gap. Rigid bodies are never nested.
Dynamic bodies use analytic or convex colliders only, with explicit mass and inertia.

## Sources
| source | where | how to add |
|---|---|---|
| procedural | `assets/procedural/*.py` | subclass `Asset`, decorate with `@register` |
| mesh | `assets/library/**/<id>.asset.yaml` + `.usd` + `.collision.usd` | `worldc-import-mesh in.glb --id cart_01 --tags cart,metal --mass 12` (CoACD convex decomposition; extra `mesh`) |
| user file | anywhere, referenced as `asset: {file: ...}` | imported on first use into a content-addressed cache (`~/.cache/worldc/mesh_assets`) |
| generated | `asset: {generate: {...}}` in a spec; promoted ones in `assets/library/generated/` | declarative parts/links/anchors validated by `assets/generated.py` (`gen.*` issue codes) |
| catalog | Omniverse / Isaac Sim asset index | `catalog: omniverse` + `query` or `path`; see [CATALOG.md](CATALOG.md) |

Materials are named presets (`materials.py`: `VISUAL`, `PHYSICAL`). Procedural textures are either built-in (floor,
cardboard) or declarative `TextureSpec`s (`texturegen.py`).

## Registered assets (phase 1+2)
- **Warehouse:** `building_shell`, `slope_zone`, `rack_row`, `rollup_door`, `swing_door`, `cage`, `conveyor_belt`,
  `conveyor_roller`, `work_table`, `charging_station`, `pallet`, `pallet_load`, `cardboard_box`, `tote`,
  `pallet_jack_tipped`, `traffic_cone`, `trash_bin`
- **Agents:** `human_proxy`, `forklift_proxy`, `robot_placeholder`
- **Industrial** (from the former plant generator): `gate_valve`, `control_valve`, `gauge`, `instrument_box`, `enclosure`,
  `control_station`, `instrument_panel`, `pump_set`, `flexible_hose`, `pipe`, `pipe_run`, `worker_static`, `loose_tool`,
  `ibeam`
- **Mesh example:** `example_crate`

`uv run worldc assets` lists them and `uv run worldc describe <id>` shows the full contract.
