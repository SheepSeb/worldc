# Scene spec v1.0

> Status: phase 5. Specs validate, solve, and compile to USD, `scene_graph.json`, `resolved_spec.json` and `check.json`.
> The original warehouse is now a recipe (`recipes/warehouse.py`, `worldc recipe warehouse --seed 3`). The MCP server comes
> in phase 6.

A **scene spec** is a JSON/YAML document that describes a scene declaratively. The machine-readable contract is
[`src/worldc/spec/schema.json`](../src/worldc/spec/schema.json) (generated from the
pydantic models in `spec/models.py`; `python -m worldc.spec.schema` regenerates it).

Examples (all compile and pass the post-build checks):
- [`minimal_room.yaml`](../examples/specs/minimal_room.yaml)
- [`warehouse_lite.yaml`](../examples/specs/warehouse_lite.yaml)
- [`outdoor_yard.yaml`](../examples/specs/outdoor_yard.yaml) (uses an LLM-style `generate` asset)
- [`industrial_deck.yaml`](../examples/specs/industrial_deck.yaml)

```bash
uv run worldc validate examples/specs/warehouse_lite.yaml       # schema + semantic validation
uv run worldc build examples/specs/warehouse_lite.yaml --seed 3 --out out/lite_3   # solve + USD + checks
uv run worldc render out/lite_3 --camera aisle                  # fast proxy render (+ --map for a semantic top-down map)
uv run worldc check out/lite_3                                  # re-run the post-build checks
uv run worldc schema --section ObjectSpec                       # one part of the schema
uv run worldc assets --tags door                                # find assets
uv run worldc describe rack_row                                 # params, footprint, anchors, joints
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
| `robot` | a real Isaac Sim robot (`asset`, default `nova_carter`), spawn placement, optional navigation radius, `nav_goals` (see Robots) |
| `cameras`, `lighting`, `physics`, `output`, `metadata` | |

## Robots
The scene robot is always a **real Isaac Sim robot**, never a generated one. It is referenced by URL from the Isaac Sim asset root
with its own articulation, joints, drives and sensors left on, and placed at the spawn pose with its base on the floor.

```yaml
robot:
  asset: nova_carter                 # alias; or {catalog: omniverse, path: Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd}
  spawn: {pose: {xy: [12.4, -9.0], yaw: 180}}
  footprint_radius: 0.5              # optional navigation radius; default = the robot's measured footprint radius
  nav_goals: [...]
```
Supported robots are the measured ones in `assets/library/robots/isaac_5.1_robots.json`; list them with `list_assets(tags=["robot"])`
(`worldc assets --tags robot`). Wheeled AMRs (`nova_carter`, `carter_v1`, `iw_hub`, `jackal`, `dingo`, `turtlebot3_burger`, `kaya`,
`o3dyn`), a mobile manipulator (`ridgeback_franka`), an arm (`franka`), and legged / humanoid robots (`spot`, `anymal_d`,
`unitree_go1`, `unitree_h1`; these need a locomotion controller to stand and are not gated by the settle check). Anything else is
rejected: `robot.unknown` for names / paths outside the table, `robot.not_isaac` for generated, file or non-robot assets. Navigation
reachability uses the robot's real footprint. `check_scene` with physics runs the robot's articulation in Isaac Sim and reports
whether it settled where it was spawned.

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
`clearance {zone, axis, span?, side, target, tolerance, adjust}` (an exact free gap, hit by sliding the `adjust` objects).
`side: max | min` anchors the gap to the zone's +/- edge, measured from the blockage to the first real obstacle beyond it
(e.g. the opposite rack). `any` uses the largest gap. Clearances are solved after all structure, fill and support
placements, then refined on the built colliders.
`reachable {from, to, robot_radius?}`, `keep_clear {zone | path + width}`, `stable {objects}`,
`count_at_least {object, n}`. Each takes an optional `id` and `hard` (default true).

## How a spec is solved
The solver is deterministic in (spec, seed). Every random draw uses a generator keyed by (seed, object id, instance index,
purpose), so editing one object leaves the others where they were. It places things in this order:
1. Environment, walls, openings and their doors, slope zones, keep-clear zones.
2. `layout` / `pose` / `against`, then `attach_to` / `fill` / `on`, then `next_to`.
3. Agent corridors, robot spawn and nav-goal stand poses are reserved, so free-floor clutter cannot block them.
4. `inside` / `near`.
5. `clearance` constraints are solved as soon as their `adjust` objects exist.
6. Reachability: optional instances that block a goal are dropped, up to 3 rounds; otherwise `solve.unreachable` names the
   blockers.

Collision is 2.5-D: convex footprint pieces with a z range. An object resting on a support is never tested against that
support. Keep-outs (door swings, dock lanes, agent corridors, goals, keep-clear zones) only block floor-level objects.

Nav goals: `target` may be an object, anchor, port, `clearance:<id>` (the narrowest-gap centre) or `[x, y]`, plus `offset`.
- `approach: port`, or a port of kind approach/parking: the robot stands on the port.
- Anchors with a `face` meta (rack slots): stand `distance` beyond the parent's footprint edge on the face side.
- Otherwise: stand `distance` in front of the target (its +y).

## Outputs (`worldc build` / `compile_scene`)
| file | |
|---|---|
| `<output.usd_name>` | the stage (Z-up, metres, kg, default prim `/World`) |
| `scene_graph.json` | environment, zones, every record (pose, category, mass, articulation, `spec_id`, `instance`), instances (asset, params, pose, root, bodies), goals with reachability, hazards, conveyors, joints, cameras, lighting, clearance checks |
| `resolved_spec.json` | the fully grounded spec: every instance explicit (pose, sampled params), seed pinned. Compiling it reproduces the scene record for record, which makes it the training target |
| `check.json`, `nav_occupancy.png` | post-build checks |
| `textures/` | floor (with markings derived from zones, aisles, agent paths, dock openings and parking) and cardboard |

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
| `solve.unsatisfiable.<relation>` | required instances could not be placed; message gives placed/required and the top blockers (objects, keep-outs such as `dock_lane`, `agent:<id>`, `goal:<id>`) |
| `solve.unsatisfiable.layout` | a layout instance is blocked (names the blockers) |
| `solve.clearance` | the clearance target is not achievable (message gives the best achievable gap) |
| `solve.unreachable` | a nav goal is unreachable for the robot footprint (names the blocking objects) |
| `solve.robot_spawn_blocked`, `solve.goal_target`, `solve.count` | spawn / goal / count_at_least problems |
| `robot.unknown`, `robot.not_isaac` | the robot is not one of the measured, real Isaac Sim robots |
| `physics.robot_unstable` | the robot moved or sank while the scene settled in Isaac Sim |
| `solve.agent_corridor_blocked` (warning) | an agent's corridor overlaps objects placed before it |
| `check.interpenetration`, `check.floating`, `check.hazard_path_blocked` | post-build physics pre-flight (prim paths + spec ids in data) |
| `check.spawn_blocked`, `check.goal_unreachable`, `check.clearance`, `check.no_goals`, `check.no_robot`, `check.load_failed` | post-build navigation and loading |
| `build.failed` | an asset failed to build (generator bug or impossible param combination) |
