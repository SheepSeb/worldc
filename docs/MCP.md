# MCP server

`sceneforge-mcp` exposes the scene pipeline as 11 tools over stdio, so an agent (Claude Code, Claude Desktop, your own client)
can go from a description to a validated, rendered scene. It needs the `mcp` extra and a working Isaac Sim install; it
refuses to start otherwise (`env.isaac_sim_missing` / `env.isaac_sim_version` / `env.isaac_sim_broken`).

```
uv sync --all-extras
claude mcp add sceneforge -- uv run --directory /path/to/sceneforge sceneforge-mcp
```

Environment: `SCENEFORGE_WORKSPACE` (output root, default `./mcp_out` of the server's working directory), `ISAACSIM_PATH`,
`SCENEFORGE_ISAAC_ASSET_ROOT`, `SCENEFORGE_CATALOG_INDEX`.

## Result format

Every tool returns `{"ok": bool, "result": {...}, "issues": [...]}` and never raises. An issue is
`{severity, code, path, message, hint, data}`: `path` is a JSON pointer into the spec you sent, `hint` suggests the edit. An
unexpected exception becomes one `internal.<ExceptionType>` issue. `spec` arguments take a JSON object, a JSON/YAML string
or a file path. `render` additionally returns the image as MCP image content.

## Tools

| tool | what it does |
|---|---|
| `list_assets(tags?, query?)` | registered assets (AND filter on tags, substring query) |
| `search_assets(query, tags?, sources?, limit?)` | ranked matches over the registry and the Omniverse catalog, each with a ready-to-paste `use` snippet; registry wins ties |
| `describe_asset(id)` | params with ranges and defaults, footprint, anchors, ports, physics, joints, an example |
| `import_asset(file, overrides?)` | import a mesh/USD file as an asset for this session (needs the `mesh` extra) |
| `validate_generated_asset(asset, render?)` | check a declarative `generate` asset alone; returns a proxy render of it |
| `get_spec_schema(section?)` | the JSON Schema, or one `$defs` section (cheaper); examples are in `docs/SPEC.md` |
| `validate_spec(spec, seed?)` | schema + semantic checks, then a dry-run solve (no USD) with per-object placed counts |
| `build_scene(spec, seed?, out_dir?, run_check?)` | compile to USD + `scene_graph.json` + `resolved_spec.json`; output under the workspace (`<spec_hash>_<seed>` by default) |
| `check_scene(path, physics?, sim_seconds?)` | overlap / floating / hazard / reachability / clearance checks, plus an Isaac Sim physics settle in a subprocess |
| `render(path, camera?, width?, height?, engine?)` | `rtx` (Isaac ray tracing, ~20 s) or `proxy` (fast flat shading); a scene-graph camera name, `"topdown_map"`, or `{"position", "look_at", "focal_mm"}` |
| `edit_scene(spec, json_patch)` | apply an RFC 6902 patch, validate, return the new spec + a diff; a bad op is reported by index and nothing is applied |

## Notes

* `out_dir` must stay inside the workspace; scene directories from earlier builds can be passed back from anywhere.
* `import_asset` registers the asset only for the lifetime of the server process. For a spec that must replay later, use
  `{"file": path}` in the spec instead.
* `check_scene`'s `ok` is the pure-Python checks plus "bodies settled" (>= 95% moved under 2 cm, nothing fell or ran away).
  A conveyor cargo that misses belt speed in a short run is a warning (`physics.conveyor_speed`), not a failure.
* A typical loop is in the server's instructions: search assets, write a spec, `validate_spec`, `build_scene`,
  `check_scene`, `render`, then `edit_scene` to refine.
