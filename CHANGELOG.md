# Changelog

## Unreleased: E2 baselines

### Added: E2 template and remanufacturing-recipe baselines (outside the E1 freeze)
`worldc recipe remanufacturing --prompts prompts/prompts.csv --out out/e2/recipe` builds one scene per prompt (seed =
prompt number; a failed static build resamples with seed + 1000 k, up to 10 times); `worldc recipe e2_template --out
out/e2/template --seeds 30` builds the fixed P08 floor plan with seeds 0..29 (props only, no resampling). Static checks
only. Sampling is balanced over the prompt set: main door wall, robot arm and bench arrangement cycle with the seed;
pallet staging and charger are placed before benches; benches are laid out as greedy rows on either axis and either
belt side; shelving can split over two walls. Every scene folder has `e2_attempts.json`: method, attempts, first-try
result, the recipe knobs (with `bench_arrangement` = the *realised* arrangement, `mixed` / `fallback` when the drawn one
could not place every bench, next to `bench_arrangement_drawn`) and provenance (E1 freeze tag and commit, baseline commit,
curated-table sha256, files changed since the freeze).

## e1-freeze-v4 (2026-10-29)

### Fixed: moving parts of articulated SimReady assets rendered at the scene origin
ForgeCAD assets author their articulated links (doors, lids, drawers, turntables, wheels) with `!resetXformStack!` and a
world-frame transform. Referenced under a placed asset, those parts ignored the placement and all sat at the scene origin
(pilot P18: the microwave doors and turntables and the toolbox lids piled up at (0, 0) on the floor; the toolboxes on the
benches had no lids). Affected: `toolbox`, `microwave`, `wheelie_bin`, `pallet_jack`, `planetary_gearbox`, `lidded_tote`.
The import now records each such part's rest pose relative to its parent (`local_xforms` in the asset definition) and the
build authors it as the part's local transform; assets already in the cache get it from their cached source on first use.
Collision hulls, footprints and anchors were already computed in the asset frame and do not change. New tests: a synthetic
reset-xform asset placed off the origin, and every curated scene asset's rendered meshes inside its placed bounds.

### Fixed: one physics scene per stage; the assets' own joints are not simulated (pilot P18 physics timeout)
Every ForgeCAD SimReady asset (KLT bin, toolbox, microwave, wheelie bin, pallet jack, ...) ships its own `PhysicsScene`
(default settings, i.e. GPU dynamics) and its own joints. The importer switched off the asset's rigid bodies and colliders
(prims with physics schemas applied) but not these, which are prim types: a scene with N such assets had N + 1 physics
scenes, and PhysX also built the ForgeCAD articulations and mimic joints between the switched-off links. Depending on the
combination, Isaac Sim's first simulation step (`create_simulation_view` / `update_simulation`) never returned: pilot P18
timed out at 900 s and 1800 s, twice. Bisection on the P18 scene: microwaves + shelving and toolboxes + shelving hang,
each group alone passes; with the fix the full P18 scene settles in 31 s (29 bodies, max drift 0.8 mm). The asset's
joints and physics scenes are now recorded at import (`inactive`) and deactivated in the scene, and its articulation
roots are disabled; the stage keeps exactly one physics scene (ours) and our convex hulls remain the asset's only physics.
The placement test checks that no curated asset keeps an active joint or physics scene.

### Fixed: a timed-out Isaac Sim subprocess kept running
`python.sh` is a shell wrapper around Kit's python; on timeout `subprocess.run` killed only the wrapper, so Kit kept
running on the GPU (orphaned, after the GPU lock was released) alongside the retry and every later job. Physics checks,
RTX renders, drive runs and the install smoke test now run Isaac Sim in its own process group and kill the whole group on
timeout (`isaac_env.run_isaac`). New test: a timed-out wrapper leaves no child behind.

`scripts/run_e1.sh` and the runner's freeze check point at `e1-freeze-v4`.

## e1-freeze-v3 (2026-10-29)

### Changed: two curated benches excluded from the library
`workbench_modular` and `packing_table_deluxe` are no longer registered scene assets (so they never appear under
`workbench` / `table` or anywhere in `list_assets`). They stay in the curated table with `"excluded"` and the reason: their
collision decomposition covers the work surface (a 2 m hull over the whole bench / a top hull ending 15 cm above the
visible top), so nothing can be placed on them. The curated set is now 41 scene assets + 2 `robot_cell` arms.
`scripts/run_e1.sh` and the runner's freeze check point at `e1-freeze-v3`.

## e1-freeze-v2 (2026-10-29)

### Changed: a physics timeout is not a pass
The runner used to treat a final physics check that did not run (e.g. timed out) as passed, so a timed-out scene could be
accepted (pilot run P18). Now a timeout is rerun once with twice the timeout (900 s -> 1800 s, alone on the GPU); a second
timeout makes the run not accepted. Every record has `final_status`: `accepted`, `static_failed`, `physics_failed`,
`physics.timeout`, `no_scene` or `error`; `final_checks.physics_status` is `ok | failed | timeout | not_run`.

### Fixed: curated support anchors sit on their collision hulls
Every support anchor of the curated SimReady assets is fitted to the convex hulls at bake time (`curated.fit_support`): its
height is the top of the hulls carrying it, and its area is cut to the largest rectangle that no hull rises through
(within 0.3 m above). This fixes props sinking 6 mm into `pallet_euro` / `pallet_block` (their tops were measured on the
visual mesh), KLT bins colliding with shelf uprights (board anchors reached into the corner posts) and a partial board hull
standing 1.1 cm above `parts_shelf` board 2. `workbench_modular` and `packing_table_deluxe` have no work-top anchor any
more: their collision decomposition covers the whole work surface (a 2 m hull / a top hull ending 15 cm above the visible
top), so nothing can stand on them in PhysX; their descriptions say so. Baked anchors now replace the importer's in the
cached asset definitions. New test: every support anchor sits on a hull top with nothing rising through it.

### Changed: E1 runner protocol
* `--max-turns` default 120 (was 60); the 10-repair cap stays the binding limit.
* `worldc generate --prompts-file` (a batch) refuses to start unless HEAD is the freeze tag (`WORLDC_FREEZE_TAG`,
  default `e1-freeze-v2`) with a clean tree, untracked files included (`--allow-unfrozen` for trials).
* Isaac Sim subprocesses (physics checks, RTX renders, drives) are queued machine-wide, one at a time on the GPU
  (`isaac_env.isaac_lock`), so several generation loops can run concurrently.
* `scripts/run_e1.sh`: the constrained variant for all models (concurrently); the plain variant is a separate, optional
  pass (`--plain`). It only runs from a clean checkout of `e1-freeze-v2`.

## Unreleased (2026-10-28)

### Changed: convex-hull colliders count as solids in `check_scene`
Before, every `Mesh`-typed collider was skipped by the static checks (interpenetration, floating, hazard sweep, navigation
grid, safety checks). That was meant for raw triangle-mesh colliders (floor / terrain surfaces, whose bounding box is not a
solid), but it also hid the convex-hull colliders of every imported catalog / SimReady / mesh asset: a mesh bench could
never support a prop (`check.floating`), and imported assets were invisible to navigation and the safety checks.

* `colliders.collect` now marks a mesh collider as raw (`is_mesh`) only when its `physics:approximation` is not
  `convexHull`, `convexDecomposition`, `boundingCube` or `boundingSphere`. Hull colliders are solids with their box from
  their own points.
* `Collider.points` keeps a hull's world vertices. An overlap found on the boxes is confirmed by an exact separating-axis
  test on the hull points (`geom.hull_penetration`), because a hull's box over-reports (a yawed drum's box corners).
* Dynamic mesh bodies are lifted by the few millimetres their convex decomposition bulges below the visual mesh.

No threshold or acceptance rule changed. Scenes with imported mesh assets are now actually checked against them, so a scene
that passed before can fail now (and vice versa for props resting on mesh supports). The warehouse recipe (seeds 0-4)
still passes unchanged; it has no imported mesh colliders on its critical paths.

### Added
* Mandatory safety checks in `check_scene` (human reachability from the door, minimum aisle width, robot-cell clearance)
  and enforcement of declared `min_clearance` constraints (`check/safety.py`, docs/SPEC.md).
* `robot_cell` asset (static arm on a pedestal; SimReady safety panels stacked two high by default, or a wire-mesh cage, or
  an unfenced keep-out).
* Curated SimReady subset: 43 asset aliases + 2 FANUC arms for `robot_cell` (`assets/curated.py`, docs/CATALOG.md).
* `{catalog: simready}` references to `nvidia/simready-assets` (index / search / probe / fetch on demand).
* Runner: repair-iteration counter per run and `--max-repairs` (default 10); `--seed` pins the build seed of every
  `build_scene` in a run; each record's `generator` carries the `e1-freeze` tag commit, `at_freeze` and the sha256 of the
  curated asset table; `params` carry `loop_seconds` / `final_checks_seconds`.
* `scripts/run_e1.sh`: the frozen E1 protocol (effort, turns, budget, seed, repairs, timeout) for any list of models.

### E1 protocol values
* Minimum aisle width in the mandatory safety checks: **1.20 m** (was 0.80 m in the first draft of the checks). Human
  radius 0.26 m and robot-cell clearance 0.50 m unchanged. The `warehouse_lite` example was adjusted (table gap 1.3 m,
  loose cartons in the staging zone) to meet it; the warehouse recipe passes as is.
* Spec-writer prompt id `spec_writer@3`.
