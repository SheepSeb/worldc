# Plan: declarative, agent-usable scene generation

Status: **draft, awaiting approval**. Nothing outside this file has been changed.

Goal: turn the repo from two hand-written generators into a general pipeline

```
NL description ──(LLM, later trained)──▶ SceneSpec (JSON) ──validate──▶ solve ──▶ build ──▶ USD + scene_graph.json + textures
                                              ▲                                               │
                                              └──────── structured errors / check_scene ◀────┘
```

The scene spec is the single interface. Recipes, the MCP server, and the data-collection jobs all produce or consume specs.

**Scope change (per your note):** the plant generator is removed. It is not kept as a recipe, CLI, or equivalence target. Its
*generic* equipment builders become registered assets, because they are useful outside that one scene. The plant-specific
glue goes away: the reference-photo comparison, the hard-coded deck/valve-wall layout, `viewstats`, `compare`, `quick_render`,
plant `run_all`/`static_validate`/`isaac_validate`, and `plant_out/`. The warehouse is kept as the single reference recipe
and the regression baseline.

---

## 0. What the code does today (findings that shape the plan)

| Area | Today | Consequence for the refactor |
|---|---|---|
| `usdkit.Kit` | Clean authoring layer: prims, materials, rigid bodies with explicit inertia, joints, fixed-base articulations, `record()` → scene-graph rows. Only `pxr` core. | Keep it almost as is. It becomes the compiler's backend. |
| `geom.py` | Quaternions, OBB + SAT, 2-D oriented-rect overlap, composite inertia, scanline footprints. Pure numpy. | Becomes the solver's geometry kernel. |
| `colliders.collect` | Reads any stage → world OBBs per collider, owner body, category (`warehouse:`/`plant:` namespaces). | Reused by `check_scene` and the post-build validator. Namespace becomes `scene:`, legacy names are still read. |
| `layout.py` | Hard-coded 40×25 m building, aisle math, zone boxes, dock bays, parking, robot spawn, and `Plan` (a 2-D keep-out registry for rejection sampling). | All constants move into the warehouse **recipe** as spec values. `Plan` grows into the solver's occupancy index. |
| `structures.py` | Shell, walls with openings, slope crowns, rack rows (and the slots they expose), roll-up/swing doors, cage + sliding gate, belt/roller conveyors, work table + drawers. Reads `LY.*` globals and a shared `rng`. | Each becomes a ProceduralAsset with explicit params. Rack slots become **anchors**. |
| `objects.Factory` | Pallets, boxes, pallet loading (misalignment jitter), stretch-wrapped loads, totes + items, cones, trash bin, a tipped pallet jack (stable-pose search). | Same: ProceduralAssets. Box/tote "loading" becomes a **fill** policy on support anchors. |
| `generate.build` | ~400 lines of imperative placement: forced pick slots, rack fill probabilities, staging row, a 0.90 m chokepoint (line-searched with exact scanline clearance), cage/packing contents, region-sampled clutter, hazards, robot, goals, lights, cameras. | Becomes `recipes/warehouse.py` (seed → spec). The chokepoint becomes a generic `clearance` constraint the solver hits by line search. |
| `static_validate.py` | Spawn interpenetration, floating bodies, hazard sweep, 5 cm occupancy grid + Dijkstra nav, exact chokepoint clearance. Grid bounds come from `LY`. | Generalised (bounds from the spec footprint) → `check/`. It backs both the solver's post-check and MCP `check_scene`. |
| `runtime.py`, `isaac_validate.py`, `run_all.py` | Isaac-side path followers, articulation sweep, physics settle, renders, usdchecker. | Kept. They read `scene_graph.json`, so its schema stays backward compatible (see §4). |
| `plant/` | 3 k lines, tightly coupled (the pump nozzle position drives rack lanes, which drive pipe routing). | Removed as a generator. `equipment.py`, `pipenet.py`, `common.Art`/`Sh`/`emit` are salvaged into assets (§2.4). |
| RNG | One `np.random.Generator` consumed in program order across the whole build. | Bit-identical reproduction after the refactor is impossible. Equivalence is defined statistically and structurally (§5.2). |
| Repo | No commits yet. Everything is untracked. | **Step 0:** commit the current tree as a baseline on a branch, so equivalence tests and diffs have a reference. |

---

## 1. Scene spec — `src/worldc/spec/`

Pydantic v2 models. `spec/schema.json` is exported from the models (by `python -m worldc.spec export`) and
checked in. A test fails if the two drift apart.

### 1.1 Design rules
- **Relations over poses.** Every placed thing has a `placement`, which is a relation, a region sample, or an explicit pose
  (escape hatch).
- **Groups, not instances.** A warehouse has about 3 k dynamic bodies, so the spec must stay LLM-sized (target < 8 k tokens for
  the warehouse recipe). `count` can be a range, and `fill` policies populate anchors stochastically.
- **Stable IDs.** Every object/group has an `id` (slug). Relations and constraints refer to IDs, and errors cite JSON paths.
- **Units:** metres, kg, degrees in the spec (radians internally), Z-up, right-handed. Yaw-only rotation unless `pose` is given.
- **Strict:** `extra="forbid"` everywhere, so typos fail validation with a "did you mean" hint.
- `spec_version: "1.0"` (semver). `spec/migrate.py` holds upgraders keyed by major.minor, and an older version is upgraded
  on load.

### 1.2 Top-level shape (abridged)

```yaml
spec_version: "1.0"
name: small_warehouse
seed: 0                         # optional; the compiler's seed overrides it
units: {length: m, mass: kg, angle: deg}
environment:
  kind: indoor                  # indoor | outdoor
  footprint: {type: rect, size: [40, 25]}        # or {type: polygon, points: [[x,y],...]}
  height: 8.0                   # indoor only
  floor: {material: concrete_worn, wear: [0.3, 1.0], slope_zones: [...], markings: auto}
  walls: {material: painted_block, thickness: 0.3,
          openings: [{id: dock_0, wall: south, at: -15, width: 3.0, height: 3.2, asset: rollup_door}]}
  roof: {material: steel_deck, trusses: true}     # indoor only
  ground: {material: asphalt}                     # outdoor only
zones:                          # named regions that relations can target
  - {id: packing, rect: [13.8, 1.2, 19.7, 12.4], tags: [packing]}
  - {id: aisle_2, rect: [...], tags: [aisle], keep_clear: false}
  - {id: rack_area, grid: {...}}   # zones can be generated by layout primitives (§1.4)
objects:
  - id: rack_rows
    asset: rack_row
    count: 12
    params: {levels: [0.2, 1.8, 3.4, 5.0], bays: 2, segments: [[-4.3, 1.1], [4.3, 9.7]]}
    placement: {layout: rack_aisles, aisle_width: 3.2, start_x: -19.4, back_gap: 0.1}
  - id: rack_pallets
    asset: pallet_load            # pallet + {boxes|wrapped|totes|empty}
    placement: {fill: {anchors: "rack_rows.*.slot", probability: {by: level, values: [0.55, 0.45, 0.4, 0.35]},
                       variant_weights: {boxes: 0.55, wrapped: 0.28, totes: 0.08, empty: 0.09}}}
  - id: floor_boxes
    asset: cardboard_box
    count: [12, 16]
    params: {size: {min: 0.15, max: 0.6}}
    placement: {inside: [dock_front, east_lane, north_lane, packing], yaw: random}
    constraints: [{no_overlap: true, margin: 0.2}]
  - id: belt_a
    asset: conveyor_belt
    params: {length: 4.0, speed: 0.5}
    placement: {pose: {xy: [15.2, 4.0], yaw: 90}}  # escape hatch
  - id: belt_items
    asset: cardboard_box
    count: 2
    placement: {on: belt_a.surface}
constraints:                     # scene-level constraints
  - {type: min_clearance, between: [rack_rows, "*dynamic"], value: 0.02}
  - {type: reachable, from: robot.spawn, to: [nav_goals.*], robot_radius: 0.38}
  - {type: clearance, id: chokepoint, zone: aisle_2.north, target: 0.90, tolerance: 0.03,
     adjust: [choke_debris]}     # the solver line-searches `adjust` objects to hit the gap
agents:
  - {id: forklift_0, asset: forklift_proxy, path: {waypoints: zone:cross_aisle_loop, mode: loop}, speed: 0.8,
     keep_clear_width: 1.05}
  - {id: worker_0, asset: human_proxy, path: {waypoints: [[-17, 11.2], [8.5, 11.2]], mode: pingpong}, speed: 0.9}
robot:
  spawn: {near: [12.4, -9.0], yaw: 180}  # or {inside: charging}
  footprint_radius: 0.38
  nav_goals:
    - {id: pick_high_0, type: pick, target: rack_pallets[forced=0], approach: front, distance: 0.95}
    - {id: dock_0, type: dock, at: dock_0, offset: [0, 2.0]}
cameras:
  - {id: overhead, mode: top_down, hide: [roof]}
  - {id: aisle_level, look_from: [aisle_1, -8.0, 1.7], look_at: [aisle_1, 8.0, 2.6], focal: 14}
  - {id: robot, mount: robot, offset: [0.2, 0, 1.5], focal: 15}
lighting: {preset: bright_day, scale: 1.0}   # presets: bright_day, mixed_warm, dim_night, overcast, sun, indoor_industrial
physics: {hz: 120, solver: TGS}
output: {usd_name: scene.usd}
```

### 1.3 Placement relations (closed set, discriminated union on the key)
| relation | meaning |
|---|---|
| `pose: {xy, z?, yaw?, quat?}` | explicit pose (escape hatch) |
| `inside: zone_id \| [zone_ids]` | uniform sample inside a zone (footprint fully contained) |
| `on: obj_id.anchor` | rest on a support-surface anchor (z from the anchor, xy sampled inside its polygon) |
| `next_to: {target, side: left/right/front/back/any, gap}` | adjacent to another object's footprint |
| `against: wall_id \| zone_edge` | back face flush with a wall/edge, facing inwards |
| `attach_to: obj_id.port` | snap the asset's own port to a target port (pipes→nozzles, door→opening) |
| `fill: {anchors, probability, variant_weights, max}` | stochastic population of many anchors (rack slots, pallet decks, tote floors) |
| `layout: <primitive>` | generated arrangement (`row`, `grid`, `rack_aisles`, `along_path`) that emits poses and zones |
| `near: {target, radius}` | soft proximity |

Plus per-object `yaw: fixed / random / align_to: wall/aisle / face: obj_id`, and `jitter: {xy, yaw}`.

### 1.4 Constraints
`no_overlap` (default on for everything, with a margin), `min_clearance(a, b, value)`, `clearance(zone, target, tolerance,
adjust)` (exact gap, line search), `reachable(from, to, robot_radius)`, `keep_clear(zone | path, width)` (agent corridors,
door swing arcs, approach cones), `stable` (support check), `count_at_least`. Each constraint has a `hard: bool` (default true)
and an optional `id` that errors use.

### 1.5 Validation layers (all return the same structured `Issue` type)
1. **Schema** (pydantic) → `{code: "schema.extra_field", path: "/objects/3/params/lenght", message, hint: "did you mean 'length'?"}`.
2. **Semantic** (no geometry): unknown asset id, params out of the asset's range (coerced or rejected per the asset param
   schema), dangling references (`belt_a.surface` with no such anchor), cycles in the relation graph, zones outside the
   footprint, impossible counts (e.g. min > anchors available).
3. **Solve** (§3) → unsatisfiable-constraint reports.
4. **Post-build check** (§4) → overlap, floating, nav reachability.

`Issue = {severity, code, path, message, hint, data}`. `code` is a stable dotted identifier, listed in `docs/SPEC.md`, so the
LLM can learn from it.

---

## 2. Asset registry — `src/worldc/assets/`

```
assets/
  base.py          # Asset, ParamSpec, Anchor, Port, PhysicsInfo, ArticulationInfo, BuildContext, BuiltAsset
  registry.py      # register(), get(id), list(tags), describe(id) -> JSON-safe dict
  procedural/      # one module per family
    building.py    # floor_slab, wall (with openings), slope_crown, roof
    racks.py       # rack_row
    doors.py       # rollup_door, swing_door, sliding_gate, cage (fence + gate)
    conveyors.py   # conveyor_belt, conveyor_roller
    furniture.py   # work_table (+ drawers), charger
    logistics.py   # pallet, cardboard_box, pallet_load (boxes|wrapped|totes|empty), tote (+items), pallet_jack_tipped
    clutter.py     # traffic_cone, trash_bin, loose tools (wrench, hammer, bucket, hardhat, toolbox, coil, spool)
    agents.py      # human_proxy (capsule), forklift_proxy (box + forks), robot_placeholder
    industrial.py  # salvaged from plant/: gate_valve, control_valve, gauge, instrument_box, enclosure, control_station,
                   # instrument_panel, pump_set, flexible_hose, pipe (+ pipe_network helper), steel_frame/ibeam
  mesh/
    mesh_asset.py  # MeshAsset: references a USD/GLB file plus a sidecar YAML
    import_mesh.py # CLI: normalise -> decompose -> write sidecar
  library/         # mesh assets live here (empty for now, plus one tiny example)
```

### 2.1 Asset interface
```python
class Asset(Protocol):
    id: str; description: str; tags: set[str]
    params: dict[str, ParamSpec]          # type, range/choices, default, unit, doc; also a "random" sentinel -> sampled from range
    def footprint(self, p) -> Footprint   # local 2-D convex polygon(s) + height, used by the solver before build
    def anchors(self, p) -> list[Anchor]  # support surfaces (polygon @ z, normal, capacity) and attachment ports (pose, kind, radius)
    physics: PhysicsInfo                  # mass model, friction/material, collider type, body type (static/dynamic/kinematic/articulated)
    articulation: ArticulationInfo | None # joints (type, axis, limits, default, drive)
    def build(self, ctx: BuildContext, prim_path, pose, p, rng) -> BuiltAsset   # authors prims via Kit, returns records
```
- `footprint` and `anchors` are **pure** functions of params (no USD). The solver plans with them, and only the compiler
  calls `build`.
- Each asset gets its own `rng` derived from `(scene_seed, object_id, instance_index)` via `np.random.SeedSequence`. Adding
  one object then doesn't reshuffle every other object's randomness, which makes edits and patches local.
- Materials come from a shared `materials.py` registry (named presets → `Kit.material`/`phys_material`), so assets no longer
  receive a `m`/`pm` dict.
- `describe(id)` returns params with ranges and defaults, footprint at defaults, anchors, physics, joints, tags, and a
  one-line example placement. This is what `describe_asset` serves to the LLM.

### 2.2 Refactor of existing builders (behaviour preserved)
Each `Structures.*` / `Factory.*` method becomes `build()` with its constants lifted into params, keeping today's values as
defaults. The bodies are moved mostly verbatim: same prim layout under the instance root, same inertia, same joint
drives, and the same `record()` fields. Global coupling is removed:
- `LY.*` reads → params or `BuildContext.environment` (footprint, wall thickness, height).
- `plan.add(...)` keep-outs → declared by the asset as `keepouts(p)` (door swing arc, gate slide, dock lane), so the solver
  owns the registry.
- Rack `slots` → `anchors` named `slot[s,b,p,level]`, with level/face/aisle metadata so `fill` and nav goals can address them.
- Pallet loading → `pallet_load` asset with `variant`; the `load_boxes` jitter/misalignment logic is unchanged.
- Hazards/robot → agents + robot sections (proxies authored by assets, path metadata as today).

### 2.3 MeshAsset
Sidecar `<name>.asset.yaml`: id, description, tags, source file, scale/up-axis already normalised, bounds, footprint
polygon, mass/friction/material, `collision: {type: convex_hulls, file: <name>.collision.usd, n_hulls}`, anchors (support
surfaces and ports, hand-authored or auto-detected flat top faces), optional articulation (only for USD sources that
already contain joints). `build()` adds a USD reference to the visual, plus the precomputed hull prims with `CollisionAPI`
(`convexHull` approximation). This keeps the "no triangle meshes on dynamic bodies" rule.

`python -m worldc.assets.mesh.import_mesh in.glb --id cart_01 --tags cart,metal --mass 12`:
1. Load (USD via pxr; GLB/OBJ/STL via `trimesh`, optional).
2. Normalise: metres, Z-up, origin at bottom-centre of the footprint.
3. Convex decomposition with CoACD (optional; falls back to a single convex hull with a warning).
4. Write the visual USD, the collision USD, the sidecar YAML, and a thumbnail (top-down footprint PNG).

### 2.4 What is salvaged from `plant/` vs deleted
- **Salvaged into `industrial.py`:** `gate_valve` (wheel + mimic-coupled rising stem), `control_valve`, `gauge`,
  `instrument_box`, `enclosure` (door/latch/drawer), `control_station`, `instrument_panel`, `pump_set` (exposes `nozzle_discharge`/
  `nozzle_suction` ports), `flexible_hose` (dynamic chain between two ports), `pipe` and a `pipe_run` asset that routes
  axis-aligned between two ports with elbows/flanges (simplified `PipeNet`), `worker_static`, loose tools. The `Art`/`Sh`/
  `emit` helpers move to `assets/procedural/_assembly.py` and are shared by every articulated asset.
- **Deleted:** `plant/gen.py`, `run_all.py`, `static_validate.py`, `isaac_validate.py`, `quick_render.py`, `viewstats.py`,
  `compare.py`, `nav.py` (merged into the generic nav), `plant/textures.py` (grating/concrete/sky textures move into the shared
  `textures.py`), and `plant_out/`, `reference/plant.png`.
- No plant recipe. An "industrial deck" example spec in `docs/SPEC.md` shows the salvaged assets composed generically.

### 2.5 Asset sources and resolution order
An object in a spec can name its asset in four ways. When more than one fits, the resolver uses this priority:

| priority | source | spec form | notes |
|---|---|---|---|
| 1 | **User-provided file** | `asset: {file: path/or/url.usd, id?: cart_01, mass?: 12, tags?: [...]}` | If a specific file is given, it is always used. Imported on first use via the `import_mesh` pipeline (§2.3) into a content-addressed cache; the sidecar is generated, and the user's overrides win. |
| 2 | **Registered asset** | `asset: rack_row` | Procedural or already-imported mesh assets in the registry. |
| 3 | **Omniverse / Isaac Sim catalog** | `asset: {catalog: "omniverse", query: "hand truck"}` or a pinned `{catalog: "omniverse", path: ".../HandTruck.usd"}` | See §2.6. A query is resolved once, and the resolved spec pins the exact path + content hash so it replays deterministically. |
| 4 | **LLM-generated asset** | `asset: {generate: {...}}` | See §2.7. Used only when nothing above matches. |

The tool flow nudges the LLM towards this order: `search_assets(query)` returns one ranked list across registry and
catalog with a `source` field. The spec-writer prompt says "use a provided file if one exists, else the best match with
score ≥ threshold, else generate." `validate_spec` emits a `asset.generated_but_match_exists` warning when a generated asset
has a close catalog/registry match. Data collection records which source each object came from, so the model can be
trained on the preference.

### 2.6 Omniverse / Isaac Sim catalog (extra `[omniverse]`, optional)
- **Index, don't query live.** `python -m worldc.assets.catalog index --root <assets root>` walks an Isaac
  Sim / SimReady asset root and writes `catalog.jsonl`: path, name, folder taxonomy, semantic labels found in the USD, bounds
  (metres, Z-up after reading `metersPerUnit`/`upAxis`), whether physics (rigid body / colliders / joints) is authored, and
  a thumbnail if one is shipped. The root can be a local download of the asset packs, a Nucleus server, or the public HTTP
  mirror Isaac Sim uses by default.
- Search is a local lexical + tag search over the index (BM25-style, stdlib), with an optional embedding rerank later.
  The index is versioned and checked into `assets/library/catalog/` per asset-pack version, so tests and datagen don't need
  network access.
- **Using a catalog asset:** the compiler adds a USD *reference* to it (no copy), plus our own collision proxies if the asset
  has none (CoACD at import time, cached by content hash). Physics metadata missing from the asset (mass, friction) comes
  from the tag-based defaults in `materials.py`, and the defaults are flagged in `scene_graph.json`.
- **To verify in that phase:** the exact current asset-root URLs and pack layout for the installed Isaac Sim version
  (`~/isaacsim` is present here), whether Nucleus search (DeepSearch) is usable or we rely only on our own index, and the
  **license terms** for redistributing NVIDIA assets in a training dataset. Renders and specs that reference assets are
  probably fine, but copying meshes into dataset artifacts may not be. I'll report what I find, not assume.

### 2.7 LLM-generated assets and textures
Generated assets are **declarative, not code**. The LLM never writes Python that we execute. It writes a `GeneratedAsset`
spec that a fixed builder compiles:

```yaml
asset:
  generate:
    id: gen/rolling_cart               # namespaced so it can't shadow registry ids
    description: "two-shelf steel utility cart with casters"
    tags: [cart, metal, dynamic]
    body: dynamic                      # static | dynamic | articulated
    mass: 14.0                         # or density + material -> computed
    parts:                             # primitives in the asset frame (box/cylinder/capsule/sphere/cone)
      - {name: top, shape: box, size: [0.9, 0.5, 0.03], pos: [0, 0, 0.9], material: steel_brushed}
      - {name: caster_0, shape: cylinder, radius: 0.05, height: 0.03, pos: [0.4, 0.2, 0.05], axis: y, material: rubber}
    links:                             # optional, articulated only
      - {name: drawer, parts: [...], joint: {type: prismatic, axis: y, limits: [0, 0.35], parent: root}}
    anchors:
      - {name: top_shelf, type: support, polygon: auto_from: top}
```
- The builder reuses `_assembly.py` (`Art`/`emit`), so generated assets get the same inertia, collider, and articulation
  handling as hand-written ones. Validation checks part overlaps within a body, mass > 0, a stable resting pose
  (`stable_orientation` from `objects.py`), joint limits, and size bounds. Errors use the same `Issue` format.
- Accepted generated assets can be **promoted** into the registry (`assets/library/generated/<id>.yaml`) after a check
  pass, so good ones are reused instead of regenerated.
- **Textures:** the LLM writes a `TextureSpec` that our procedural texture code renders deterministically: base material
  preset, colour palette, noise/wear/dirt parameters, decals (labels, stripes, hazard hatching, text, barcodes), and tiling.
  `textures.py` already does most of this for floors and cardboard; it is generalised into layered ops. I can't generate
  raster images myself. If you want photoreal generated textures, that needs an external image-generation model behind an
  optional `TextureProvider` interface; I'd leave that as a hook and not build it now.

---

## 3. Layout solver — `src/worldc/solver/`

```
solver/
  graph.py      # relation graph -> placement order (topological; cycles -> error); expands counts/fills into instances
  index.py      # occupancy index: 2.5-D oriented footprints (xy polygon + z-interval), uniform grid hash, SAT tests (geom.py)
  samplers.py   # one sampler per relation: inside / on / next_to / against / attach_to / fill / layout / pose
  constraints.py# evaluators + repairers: no_overlap, min_clearance, clearance (line search), keep_clear, stable, reachable
  nav.py        # generic occupancy grid + EDT inflation + Dijkstra (from static_validate.py, bounds from the spec)
  solve.py      # solve(spec, seed) -> Solution | SolveFailure
  report.py     # human/LLM-readable failure explanations
```

**Algorithm** (deterministic given the seed; no wall-clock dependence):
1. Expand groups → instances; each instance gets a derived RNG.
2. Order: environment → layout primitives → `pose` → objects that others depend on (supports, attach targets) → `on`/`fill`
   → free-floor (`inside`/`next_to`/`against`) by decreasing footprint area → agents/keep-outs registered first as reserved
   corridors → robot spawn/goals.
3. Per instance: sample up to `K` candidates (default 64) from its relation sampler, then score hard constraints against the
   index; accept the first feasible one (soft constraints are tie-breakers). On failure, backtrack once to the instance's
   own group (re-sample siblings), then give up on that instance. Optional instances (`count` range above the minimum) are
   dropped silently and counted. Required instances produce a failure.
4. Special solvers: `clearance` does a 1-D line search on `adjust` objects (generalised from `right_squeeze`); `attach_to` is
   analytic; `fill` iterates anchors in deterministic order.
5. Global post-pass: nav grid → `reachable` and `keep_clear` checks. If a goal is unreachable, the report names the
   blocking instances along the shortest blocked corridor. Optional clutter is removed and retried (max 3 rounds) before it
   reports.

**Failure report** (what the LLM sees), e.g.:
```json
{"code": "solve.unsatisfiable.inside", "path": "/objects/4", "object": "floor_boxes",
 "message": "Placed 9 of the required 12 'cardboard_box' instances in zones [dock_front, east_lane]: free area after reserved corridors is 3.1 m², needs ≈4.4 m².",
 "hint": "Lower count to ≤9, add zones, or reduce params.size.max (currently 0.6).",
 "data": {"placed": 9, "required": 12, "blocking": ["forklift_0.corridor", "rack_rows"]}}
```
Each constraint type has a template that states *what* failed, *by how much*, *which objects* bound it, and *one or two
concrete edits*.

Performance target: the warehouse solve takes < 3 s (the numpy-vectorised SAT broadphase from `static_validate.broadphase`
is reused).

---

## 4. Compiler — `src/worldc/compile.py`

```python
def compile_scene(spec: SceneSpec | dict | Path, seed: int, out_dir: Path, *, check=True) -> CompileResult
```
1. Load and migrate → validate (schema + semantic). Errors return early with `CompileResult(ok=False, issues=[...])`.
2. `solve(spec, seed)`.
3. Build: `Kit(out_dir/usd_name)`, physics scene, environment (shell assets), each instance's `build()` at its solved pose
   under `/World/<Group>/<id>_<n>`, agents (kinematic proxies with `scene:waypoints` etc. exactly as today, so `runtime.py`
   keeps working), robot placeholder + cameras, lights from presets.
4. Textures: floor (markings derived from zones/agent paths/openings, or explicit), cardboard variants, plus grating/concrete/sky
   when used. They reuse `textures.py` and are seeded from `(seed, "texture", name)`.
5. Write `scene_graph.json`. It is a **superset** of today's warehouse schema (`objects`, `goals`, `hazards`, `conveyors`,
   `articulations`, `robot`, `cameras`, `lighting`, `counts`, `chokepoint` when a `clearance` constraint has id `chokepoint`), plus
   `spec_version`, `spec_hash`, `solver: {dropped_optional, iterations}`, and `instances: {id → paths}`. The resolved spec (all
   ranges sampled, all poses explicit) is written next to it as `resolved_spec.json`. That file is the "fully grounded"
   training target and replays exactly.
6. If `check`, run the post-build checks (§6 `check_scene`) → `check.json`.

Attribute namespace becomes `scene:`; `colliders.collect` reads `scene:`, `warehouse:`, `plant:`.

---

## 5. Recipes — `src/worldc/recipes/`

### 5.1 `recipes/warehouse.py`: `def warehouse(seed: int, **overrides) -> SceneSpec`
- All `layout.py` constants become spec values. Seed-dependent choices that are *structural* (lighting preset `seed % 3`,
  rack palette, box size classes, forced pick slots, staging stacks) are made here with the recipe's RNG and written into
  the spec. Per-instance randomness (fill outcomes, jitter, clutter positions) is left to the compiler.
- The chokepoint is `clearance{id: chokepoint, target: 0.90}` + a `debris_cluster` group (2 fallen pallets, cones, ≤5
  spilled cartons).
- `layout.py`'s `Layout` survives only as a helper used by the `rack_aisles` layout primitive.
- `recipes/__init__.py` holds `RECIPES = {"warehouse": warehouse}` and `list_recipes()` (used by datagen and MCP).

### 5.2 Equivalence test (`tests/test_recipe_equivalence.py`)
Bit-exact output is impossible (RNG order changes), so equivalence is defined as:
- **Baseline fixture:** before touching code, a script records from the *current* generator, for seeds 0–4: per-category
  counts, articulation list (names, joint types, limits), conveyor list, goal names/types, building dims, chokepoint
  clearance, and static-validation pass/fail. It is stored as `tests/fixtures/warehouse_baseline.json`.
- **Exact match** for structural items: rack rows, uprights/beams per row, doors/gates/drawers and their joints, conveyors,
  chargers, walls, hazards, cameras, goal names/types, lighting preset per seed.
- **Within the baseline's observed range** (min–max over seeds 0–4, widened by 10%) for stochastic counts: pallets, boxes,
  totes, tote items, loose boxes, cones, n_dynamic_bodies.
- **Physics validity (no Isaac):** zero spawn interpenetrations > 3 mm, zero floating bodies, all goals reachable, chokepoint
  0.90 ± 0.03 m, hazard paths clear, every dynamic body has positive mass and a valid inertia.
- Runs for 2 seeds in the default suite; all 5 run under `-m slow`.

### 5.3 Old CLI
`python -m worldc.generate --seed N --out warehouse_out` is a thin shim: `compile_scene(warehouse(N), N,
out/f"seed_{N}", usd_name="warehouse.usd")`. Same flags (`--lighting`, `--light-scale`). `run_all.py` keeps working
unchanged. New general CLI: `python -m worldc build spec.yaml --seed 3 --out out/`.

---

## 6. MCP server — `src/worldc/mcp_server.py` (extra `[mcp]`)

Official `mcp` Python SDK (FastMCP), stdio transport by default. Every tool returns
`{ok, result | issues: [Issue...]}`, never a bare traceback; unexpected exceptions become `internal.*` issues with the type
and message.

| tool | behaviour |
|---|---|
| `list_assets(tags?: list[str], query?: str)` | id, one-line description, tags (AND filter on tags; substring query) |
| `search_assets(query, tags?, sources?)` | ranked matches across registry + Omniverse catalog index (§2.6) with `source`, score, bounds, has_physics |
| `import_asset(file, overrides?)` | runs the §2.3 import on a user-provided file and returns the new asset id |
| `validate_generated_asset(asset)` | checks a §2.7 `GeneratedAsset` alone and returns issues + a proxy render of the asset |
| `describe_asset(id)` | full `describe()` (params with ranges/defaults/units, footprint, anchors, ports, physics, joints, example) |
| `get_spec_schema(section?: str)` | JSON Schema (whole, or one `$defs` entry to save tokens) + link to SPEC.md examples |
| `validate_spec(spec)` | schema + semantic + a **dry-run solve** (no USD). Returns issues and a solve summary (placed/dropped counts) |
| `build_scene(spec, seed, out_dir?)` | full compile; returns paths, counts, check summary. Output in a managed workspace dir (`$WORLDC_WORKSPACE`, default `./mcp_out/<spec_hash>_<seed>`) |
| `check_scene(path, physics=true)` | overlap / floating / nav reachability (pure Python) + Isaac Sim physics settle in a subprocess with a timeout (Isaac is guaranteed by the §8 gate) |
| `render(path, camera, width?, height?, engine="rtx"\|"proxy")` | Isaac RTX by default; `proxy` = the fast built-in renderer (below). Returns the image as MCP image content + path |
| `edit_scene(spec, json_patch)` | RFC 6902 patch (small in-house implementation, no dependency) → validate → returns the new spec + issues + a diff summary. Patch errors cite the failing op index |

**Proxy renderer** (`render/proxy.py`, core deps only): rasterises collider OBBs and cylinders with Pillow from any spec camera
(perspective, z-buffer per pixel via numpy triangle rasterisation), flat-shaded by material base colour, plus a top-down
semantic map with zone outlines and agent paths. It isn't photoreal, but it is enough for an LLM judge to see layout errors
and it runs everywhere. RTX renders reuse `isaac_validate`'s camera code via a small `render/isaac_render.py` subprocess.

---

## 7. Data collection — `src/worldc/datagen/` (extra `[datagen]`)

```
datagen/
  client.py      # Anthropic client wrapper: model from config (default claude-opus-5-5), retries, prompt caching of the
                 # schema + asset catalogue, Message Batches submit/poll/collect, cost accounting
  records.py     # pydantic record models (the stable JSONL schema), writer, dedup, split
  prompts/       # versioned prompt templates (jinja-free: str.format), each with an id + version recorded in output
  caption.py     # (a) reverse captioning
  forward.py     # (b) description generation + spec writing via tools + rejection sampling
  agentic.py     # (c) full tool-use loop runner/logger (the MCP tools called in-process, no server needed)
  score.py       # (d) N specs per description, judged with renders + validator metrics -> preference pairs
  cli.py         # python -m worldc.datagen {caption,forward,agentic,score,dedup,split} --config job.yaml
```

- **(a) Reverse captioning:** recipe(seed, overrides) → resolved spec + proxy render (top-down + 1 perspective) → model writes
  K descriptions at levels `terse | moderate | detailed | exhaustive`, plus an `instruction` style ("build me…"). Bulk is done
  via Message Batches. Overrides randomise recipe knobs (footprint, aisle count, clutter level, lighting) for diversity.
- **(b) Forward:** a generator prompt produces diverse descriptions (stratified by env kind, size, density, task focus, and
  asset subsets drawn from the registry). The spec-writer model gets the tools (`list_assets`, `describe_asset`,
  `get_spec_schema`, `validate_spec`) and must end with a spec. It is kept only if `validate_spec` is clean **and**
  `build_scene` + `check_scene` pass. Rejections are stored too (`accepted: false`, with issues) for negative or repair training.
- **(c) Agentic trajectories:** every message, tool call, tool result (images by path reference), and final spec, with token
  usage and stop reasons. Same tool implementations as the MCP server.
- **(d) Scoring:** N (default 4) specs per description at temperature > 0 (or across models). Judge inputs are the
  description, each spec's renders, and the validator metrics (issues, dropped optionals, reachability, counts vs. the
  description's requested counts). The judge produces a rubric score per spec plus pairwise preferences; stored as
  `(chosen, rejected, margin, rationale)`.
- **Record schema** (`docs/DATAGEN.md`, versioned `record_version`): `id` (sha256 of the canonical spec + prompt),
  `kind` (caption|forward|trajectory|preference), `model`, `prompt_id@version`, `params` (temperature etc.), `seed`,
  `env_kind`, `description`, `detail_level`, `spec` (as written), `resolved_spec_path`, `validation` (issues, solve summary,
  check results), `render_paths`, `usage`, `created_at`, `split`.
- **Dedup:** exact (canonical-JSON hash of the spec; normalised-text hash of the description) + near-duplicate (MinHash over
  description shingles, stdlib-only implementation; threshold configurable).
- **Langfuse (optional sink, extra `[langfuse]`).** JSONL stays the **canonical dataset**. Training never depends on a
  running service, and the files are versioned with the repo's job configs. Langfuse is added as a mirror for what it's
  good at:
  - traces of agentic runs (every LLM call and tool call as nested spans), which makes debugging failing trajectories much
    faster than reading JSONL;
  - prompt management/versioning (our `prompt_id@version` maps to Langfuse prompts);
  - human annotation queues and scores, useful for checking the LLM judge in (d) against your own preferences on a sample.
  It's enabled per job with `langfuse: {enabled: true}` + the standard env keys, and is a no-op when it's off or not
  installed. The record `id` is used as the trace id, so records and traces join both ways. Caveats: Message Batches
  results arrive asynchronously, so batch jobs are logged as synthetic traces after collection, not live. Self-hosting vs.
  Langfuse Cloud is your call. I'd self-host if specs or descriptions are proprietary.
- **Held-out split by env type:** the split key is `env_signature` (env kind + recipe/family + dominant asset tags), so
  `test` contains whole environment families never seen in `train` (e.g. hold out "outdoor yard"). Assignment is a
  deterministic hash, and the split config is in the job file.

---

## 8. Packaging, determinism, tests, docs

**pyproject**
- core: `usd-core==24.11`, numpy, scipy, Pillow, **pydantic>=2**, **PyYAML** (sidecars and YAML specs; tiny. Say if you'd
  rather keep JSON-only).
- extras: `mesh = [trimesh, coacd]`, `omniverse = []` (catalog indexing uses only pxr; the extra is reserved for an
  optional embedding reranker), `mcp = [mcp]`, `datagen = [anthropic]`, `langfuse = [langfuse]`, `dev = [pytest]`.
- Console scripts: `worldc` (build/validate/export-schema), `worldc-mcp`, `worldc-datagen`, `worldc-import-mesh`. The existing
  `worldc` stays as the warehouse shim.

**Isaac Sim gate (hard requirement).** `src/worldc/isaac_env.py` has a `require_isaac_sim()` function that runs
before anything else does work:
- Resolve the root from `ISAACSIM_PATH`, falling back to `~/isaacsim`. Require `python.sh` (executable) and `VERSION`
  (parsed, must be ≥ 5.0). Today this resolves to `/home/sebnae/isaacsim`, version `5.1.0-rc.19`.
- `--deep` (and the first run per root+version) also launches a headless `SimulationApp` smoke test in a subprocess with a
  timeout. The result is cached in `~/.cache/worldc/isaac_check.json`, keyed by root + version, so later
  runs only do the cheap file check.
- On failure: exit non-zero with one actionable `Issue` (`env.isaac_sim_missing` / `env.isaac_sim_version` /
  `env.isaac_sim_broken`) saying what was looked for, where, and how to fix it (install, or set `ISAACSIM_PATH`). Nothing
  is written. There is no silent fallback.
- **Called by every entrypoint:** `worldc build/validate`, the `generate` shim, `run_all`, `import_mesh`, the catalog indexer,
  the MCP server at startup (it refuses to start, rather than starting and failing per tool), and every datagen job (checked
  before any API call is made, so no spend happens on a broken setup). `pytest` checks it once per session in `conftest.py`
  and aborts the session if it fails.
- What this changes elsewhere in the plan: `check_scene`'s physics settle and the RTX `render` are **always available**, so
  they become the default rather than a fallback. The proxy renderer stays as a fast preview option (`render(...,
  engine="proxy")`), used in bulk datagen where RTX is too slow. Core *Python* code still imports only usd-core, numpy, scipy,
  Pillow, and pydantic. Isaac runs as a subprocess, so unit tests stay fast. They just won't run on a machine without Isaac.
- **My process:** I run `require_isaac_sim(deep=True)` at the start of each phase and stop if it fails.

**Determinism:** all randomness flows from `SeedSequence(seed)` spawned by stable string keys; set/dict iteration is
always sorted; there are no time-based values in outputs except `generation_seconds` (excluded from hashes). A test builds
the same spec twice and compares `resolved_spec.json` and the USD text for byte equality.

**Tests** (`tests/`, none need Isaac; heavy ones marked `slow`):
- `test_spec_schema.py`: round-trip, exported schema in sync, strictness + hints, migration, example specs in docs validate.
- `test_registry.py`: every asset: params defaults valid, `describe()` JSON-serialisable, footprint/anchors pure, builds into
  a scratch stage with positive mass + valid inertia, articulated assets have consistent joints, deterministic per seed.
- `test_solver.py`: each relation on toy scenes; no_overlap/min_clearance hold; clearance line search hits target; unsatisfiable
  cases yield the expected `code` + hint; determinism.
- `test_compiler.py`: small spec → USD opens, colliders collected, scene_graph keys present, post-checks pass; JSON-patch edits.
- `test_recipe_equivalence.py`: §5.2.
- `test_mcp_tools.py` (skipped without `mcp`): tool functions called directly.
- `test_datagen.py`: record schema, dedup, split, with a fake client (no network).

**Docs:** rewrite `README.md` (overview, install, quickstart, CLI, MCP setup, extras); `docs/SPEC.md` (concepts, every
relation/constraint with an example, the issue-code table, three full examples: minimal room, warehouse-lite, outdoor yard,
and the industrial deck from the salvaged assets); `docs/ASSETS.md` (adding procedural/mesh assets); `docs/DATAGEN.md`
(record schema). Update `warehouse_out/README.md`.

---

## 9. Phases (each ends with: tests green, build warehouse seeds 0 and 1 through the new path, summary, **stop for your review**)

| # | Phase | Deliverables | Exit criteria |
|---|---|---|---|
| 0 | Baseline | Branch `refactor/spec-pipeline`; commit current tree as-is; record `tests/fixtures/warehouse_baseline.json` from the current generator (seeds 0–4) | fixture committed |
| 1+2 | Spec + registry | `spec/` (models, schema export, migrate, issues), `materials.py`, `assets/` with every warehouse builder + salvaged plant equipment, `import_mesh` + user-file assets (§2.5), `GeneratedAsset` builder + `TextureSpec` (§2.7) | schema/registry tests pass; each asset builds standalone; generated-asset examples build and validate; old generator still runs (it is untouched in this phase) |
| 3+4 | Solver + compiler | `solver/`, generic nav/checks (`check/`), `compile.py`, `worldc build` CLI, proxy renderer (needed to eyeball phase output) | solver/compiler tests pass; hand-written example specs build and pass checks |
| 5 | Recipes | `recipes/warehouse.py`, `generate.py` shim, **plant generator removed**, `layout.py`/`structures.py`/`objects.py` reduced to re-exports or deleted | equivalence test passes for seeds 0–1 (0–4 under `slow`); `run_all --skip-isaac` passes; Isaac run optional if you want it |
| 5b | Omniverse catalog | `assets/catalog/` indexer + search, checked-in index for the local Isaac Sim asset version, license findings | index builds offline; a spec using a catalog asset compiles and passes checks |
| 6 | MCP | `mcp_server.py`, JSON-patch, render tool | tool tests pass; manual session with Claude Code builds a scene from NL |
| 7 | Datagen | `datagen/`, docs/DATAGEN.md | fake-client tests pass; a 10-record dry run of each job type (real API, only after you OK the spend) |

Docs are updated incrementally in each phase, and README/SPEC are finalised in phase 7.

---

## 10. Decisions I'm making unless you object
1. **Plant removal** = delete the plant generator/validators/outputs, keep its equipment as generic assets (§2.4). If you
   meant "don't even keep the equipment", say so and `industrial.py` is dropped.
2. **Equivalence** is structural + statistical + physics-valid (§5.2), not bit-exact.
3. **Spec granularity:** groups with count ranges and `fill` policies. The resolved spec (every instance explicit) is a separate
   artifact. The LLM writes the compact form.
4. **PyYAML** added to core deps (sidecars and human-written specs).
5. **Proxy renderer** in core so `render` works without Isaac; RTX when available.
6. **Default datagen model** `claude-opus-5-5`, configurable per job; no API calls until phase 7 and your go-ahead on cost.
7. **Output filename** defaults to `scene.usd`; the warehouse shim keeps `warehouse.usd`.
8. Phase 0 commits the current untracked tree to a new branch (nothing pushed).
9. **Asset priority:** user file > registry > Omniverse catalog > LLM-generated (§2.5). Generated assets and textures are
   declarative specs compiled by fixed code. No LLM-written code is executed.
10. **Langfuse** is an optional mirror for traces, prompts, and human annotation. JSONL stays the canonical dataset.
11. **Isaac Sim is a hard precondition** (§8, "Isaac Sim gate"): every entrypoint and the test session refuse to run without
    it. This replaces the original "tests run without Isaac Sim" constraint. Isaac is still never *imported* by core code
    (it runs as a subprocess), but its presence is required. No bypass flag unless you ask for one (e.g. for CI machines
    without a GPU).

## 11. Risks
- **Spec expressiveness vs. warehouse fidelity:** some imperative tricks (forced pick slots, stacking on lower pallet's
  `load_top`, aisle-edge-aware chokepoint) need specific spec features (`fill` with `force`, `on: <obj>.load_top` anchor,
  `clearance` adjust). If one doesn't generalise cleanly, I'll flag it at the phase-5 review instead of adding a one-off
  feature silently.
- **Solver failure modes on LLM-written specs:** mitigated by optional-instance dropping, bounded backtracking, and explicit
  reports. Quality of hints is tested with a fixed set of broken specs.
- **Runtime:** the warehouse build today is ~15 s, mostly textures + USD authoring. The solver must not add much; budget ≤ 20 s total.
