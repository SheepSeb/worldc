# Scene spec v1.0

> Status: phase 1+2. The spec, its validator and the asset registry exist. The solver and compiler that turn a spec into
> USD land in phase 3+4, so placement semantics below describe the contract the solver implements.

A **scene spec** is a JSON/YAML document that describes a scene declaratively. The machine-readable contract is
[`src/claude_scene_generator/spec/schema.json`](../src/claude_scene_generator/spec/schema.json) (generated from the
pydantic models in `spec/models.py`; `python -m claude_scene_generator.spec.schema` regenerates it).

Full example: [`examples/specs/warehouse_lite.yaml`](../examples/specs/warehouse_lite.yaml).

```bash
uv run csg validate examples/specs/warehouse_lite.yaml       # schema + semantic validation
uv run csg schema --section ObjectSpec                       # one part of the schema
uv run csg assets --tags door                                # find assets
uv run csg describe rack_row                                 # params, footprint, anchors, joints
```

## Conventions
- Metres, kilograms, **degrees** (every angle in a spec), Z up, x east, y north. The origin is the floor centre.
- Unknown fields are errors. In YAML, `on:` is a string key: the loader only treats `true`/`false` as booleans.
- `spec_version` is `"1.0"`. Older versions are migrated on load (none exist yet).
- **Distribution literals** may replace any param value and are sampled per instance:
  `{uniform: [a, b]}`, `{randint: [a, b]}` (inclusive), `{choice: [...], p: [...]}`, `{normal: [mu, sd], clip: [lo, hi]}`.

## Top level

| field | |
|---|---|
| `environment` | **required**. `kind` indoor/outdoor, `footprint {size: [x, y]}`, `height`, `floor` (material, wear, markings, slope_zones), `walls` (thickness, openings with optional `door`), `roof`, `apron_depth` |
| `zones` | named rectangles `[x0, y0, x1, y1]` that relations target. `keep_clear: true` makes a no-placement corridor |
| `objects` | groups of asset instances (below) |
| `constraints` | scene-level constraints (below) |
| `agents` | moving humans / forklifts with waypoint paths (kinematic, driven at runtime) |
| `robot` | spawn placement, footprint radius, `nav_goals` |
| `cameras`, `lighting`, `physics`, `output`, `metadata` | |

## Objects
```yaml
- id: loose_cartons            # unique slug; ids of zones/objects/agents/goals/cameras share one namespace
  asset: cardboard_box         # registry id | {file: ...} | {catalog: omniverse, query: ...} | {generate: {...}}
  count: [4, 8]                # exact or [min, max]; instances above min are optional (the solver may drop them)
  params: {size: {choice: [[0.3, 0.2, 0.2], [0.5, 0.4, 0.3]]}}
  instance_params: null        # optional per-instance overrides
  placement: {inside: [staging, packing]}
  yaw: random                  # degrees | random | aligned | {range: [a, b]} | {face: <ref>}
  jitter: {xy: 0.01, yaw: 1.0}
```

**Asset sources**, in priority order:
1. A user-provided `file` is always used when given.
2. Otherwise a registry asset.
3. Otherwise an Omniverse catalog match (catalog arrives in phase 5b).
4. Only then a `generate` asset: declarative parts, links and anchors. No code is executed.

## References
| form | meaning |
|---|---|
| `packing` | zone (also layout zones such as `racks.aisle_2`) |
| `belt` | all instances of an object |
| `racks.3` | one instance (0-based) |
| `belt.surface`, `racks.3.slot_s0_b1_p0_L2`, `racks.*.slot_*` | anchor / port, with `*` wildcards |
| `env.opening_dock_0`, `env.floor` | environment |
| `wall:south` | a wall |
| `robot.spawn` | the robot spawn |

## Placement relations (exactly one per object)
| relation | meaning |
|---|---|
| `pose: {xy, z?, yaw?, rpy?}` or a list of them | explicit pose(s), the escape hatch |
| `inside: zone \| [zones]`, `margin` | uniform sample, footprint fully inside |
| `on: <support anchor>` | rest on a support surface (`arrange`: random / center / grid) |
| `next_to: {target, side, gap, align}` | beside another object |
| `against: wall:<side> \| <zone>.<edge>`, `along`, `gap` | back against a wall |
| `attach_to: <port>` | snap to a port (door → `env.opening_*`, pipe → nozzle) |
| `fill: {anchors, where, probability, force, max}` | one instance per selected anchor. `count` is ignored; `probability` may be `{by: level, values: {"0": 0.6, ...}}` |
| `layout: {type: rack_aisles \| row \| grid \| along_path, ...}` | generated arrangement. `rack_aisles` also defines zones `<id>.aisle_<k>` |
| `near: {target, radius}` | within a radius of a reference or point |

## Constraints
`no_overlap {between?, margin}`, `min_clearance {between, value}`,
`clearance {zone, axis, span?, target, tolerance, adjust}` (an exact free gap, hit by sliding the `adjust` objects),
`reachable {from, to, robot_radius?}`, `keep_clear {zone | path + width}`, `stable {objects}`,
`count_at_least {object, n}`. Each takes an optional `id` and `hard` (default true).

## Issues
Every stage reports `Issue {severity, code, path, message, hint, data}`. `path` is a JSON pointer into the spec, and
`hint` proposes a concrete edit.

| code | meaning |
|---|---|
| `schema.extra_field` | unknown field (hint: closest valid names) |
| `schema.missing_field`, `schema.bad_value`, `schema.<pydantic type>` | schema violations |
| `schema.placement`, `schema.asset_ref` | placement / asset has none or several of the allowed keys |
| `schema.version`, `schema.unreadable`, `schema.not_object` | document-level problems |
| `asset.unknown` | registry id not found (hint: closest ids) |
| `asset.file_missing` | file reference does not exist |
| `asset.catalog_unavailable` | catalog assets are not usable yet |
| `gen.*` | generated-asset problems (table in `assets/generated.py`), re-pathed under `/objects/i/asset/generate` |
| `param.invalid` | asset param unknown / out of range / wrong type (hint lists valid params) |
| `id.duplicate` | two items share an id |
| `zone.degenerate`, `zone.outside` | bad zone rectangle |
| `ref.unknown`, `ref.anchor_unknown`, `ref.instance_range`, `ref.kind` | dangling or wrong-kind references |
| `relation.cycle` | placement relations depend on each other cyclically |
| `count.pose_mismatch` | number of explicit poses != count |
| `fill.capacity` (warning) | fewer matching anchors than the minimum count |
| `env.outdoor_walls` (warning), `env.opening_outside`, `env.opening_overlap` | environment problems |
| `agent.waypoint_outside` | agent waypoint outside the footprint |
| `env.isaac_sim_missing`, `env.isaac_sim_version`, `env.isaac_sim_broken` | the Isaac Sim gate failed |
