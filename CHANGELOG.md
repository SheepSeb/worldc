# Changelog

## e3-fields (2026-10-29)

### Changed: E3 stop-and-wait becomes an ISO 3691-4-style field policy (`fields`)
- **Policy `fields`** replaces `stop`; blind and human-aware are unchanged.
  - It has a protective field (stop) and a warning field (slow to 50 % and flag): rectangles ahead along the travel
    heading, widened while turning, the align turn included.
  - **Lengths:** protective = v·T_react + v²/(2·a_brake) + margin, recomputed every step from the current speed;
    warning = protective + 1.0 m.
  - **Resume:** after 1 s with the protective field clear.
  - **Parameters:** all in `sim.FIELDS`.
- **Robot table:** `brake_decel_mps2` and `brake_decel_source` are new optional fields, and `build_table` keeps them.
  Nova Carter gets 3.0 m/s², which is `worldc drive`'s deceleration limit, not a vendor figure.
- **Collision tagging:** each collision event carries `ahead_x` and `from_behind`. Rows count
  `collisions_robot_moving_from_behind`, and the summary and Table 2 show it.
- **Renamed:** stop-and-wait is now "Fields (ISO 3691-4-style)" in docs and tables.
- **P18 rerun** (same 90 trials): fields has 0.1 / 0.5 / 1.2 robot-moving collisions per trial (easy / medium / hard).
  13 of its 18 are from behind; the other 5 are people stepping into the side of the field at the front corner.
- **Tests:** field lengths and switching, stop and slow, from-behind tagging, and the turn-in-place sweep the fields do
  not see.

## e3-analysis (2026-10-29)

### Added: E3 analysis and scene selection
- **`scripts/e3_analysis.py`:**
  - Cluster-bootstrap CIs (scenes, then seeds).
  - Paired Wilcoxon tests between policies on identical seeds, Holm-corrected, for robot-moving collisions and time to
    goal (unfinished trials count at the timeout).
  - Table 2 (`table2.md` / `.tex`) and the safety-vs-efficiency scatter (SVG plus points).
  - Example on P18: `results/e3_prototype_P18/analysis/`.
- **`scripts/e3_select_scenes.py`:** proposes the E3 scenes from accepted E1 runs, covering the 6 cell-type families and
  S / M / L. It only reads records and grids, so it is safe while E1 runs.
- `results/e3_prototype_P18/README.md`: the remaining stop-and-wait robot-moving collisions are all overtakes from
  behind.

## e3-prototype-2 (2026-10-29)

### Changed: E3 policies guard the align turn; stationary-near-person metric; path direction fix
- **Align turn:** stop-and-wait applies its 1.0 m rule to the turn in place at each goal, and human-aware scales its
  rotation (everywhere, not only while aligning). The align time limit waits while the robot holds, and a hold of 30 s
  during an align turn ends the trial as `frozen`.
- **Metric:** `stationary_in_walkway_s` is replaced by `stationary_near_1m_s` (time standing within 1.0 m of a person).
  The walkway mask is gone.
- **Fix:** a walking path is planned from the smaller task point to the larger and reversed for the other direction. A*
  breaks ties differently each way, so cached paths depended on which direction was asked for first, and with it a
  seed's workers (P18: 10 blind trials changed slightly between runs). Test added.
- **Rows:** each trial row carries `timeout_s`.
- **P18 rerun** (same 90 trials, `results/e3_prototype_P18/`): human-aware has no robot-moving collision in any tier.

## e3-prototype (2026-10-29)

### Added: E3 2D simulator for human-robot safety scenarios (outside the E1 freeze)
`src/worldc/e3/` adds four pieces (details in `docs/E3.md`):
- **Grids and task points:** a built scene becomes robot- and human-inflated 2D grids, plus task points (door, bench,
  storage, cell gate) and human walkways.
- **Walkers:** seeded worker tours, which replay exactly from the trial seed.
- **Robot policies:** blind, stop-and-wait and human-aware, on `isaac_drive`'s planner and pure pursuit.
- **Hard-tier transform:** 2-4 occluding shelving units beside walkways, kept only when the solver places them and the
  rebuild passes `check_scene` (static).

Metrics:
- **Primary:** robot-moving collisions. Turning in place counts as moving.
- **Also reported:** all and robot-stationary collisions, exposure while moving, and stationary time in walkways.
- **Outcomes:** success / collision / frozen / timeout / goal_unreachable / goal_missed.

The robot footprint comes from the robot table and its travel direction from the scene's Isaac drive calibration.
`scripts/e3_one_scene.py` runs one scene. First run: `results/e3_prototype_P18/` (90 trials). Tests: `tests/test_e3.py`.

## e2-baselines-v3 (2026-10-29)

Rechecked against `e1-freeze-v6`: all 29 template scenes passed; 20 of 30 recipe scenes failed
`check.human_path_robot_spawn`. The recipe parked the robot on the main door's axis (2.6 m in from a swing door, 3.8 m from
a roll-up door) and walked workers in from that door, so 27 of the 33 failing legs started at the spawn's own door. The
recipe now parks the robot beside the main door lane: a seeded knob `robot_spawn` (`left_of_lane`, `right_of_lane`,
`near_charger` when there is a charger; cycles across seeds, the next option when the drawn one has no valid spot), on
free floor and clear of the lane and of every worker path by the check distance (+ 5 cm). Worker paths are as before
(43/101 workers end within 3 m of the main door; 37/101 before), none too close to the spawn. Both baselines rebuilt
against `e1-freeze-v7`.

## e1-freeze-v7 (2026-10-29)

### Changed: `check.human_path_robot_spawn` measures to the robot's footprint rectangle
The clearance is now measured from the person's path (its centreline) to the parked robot's footprint: its measured
xy bounds (robot table, centred on the robot origin) as a rectangle rotated by the spawn yaw; required
`HUMAN_RADIUS_M` + `HUMAN_SPAWN_GAP_M` = 0.56 m, i.e. the person's disk stays 0.3 m clear of the robot's body. v6
measured to the footprint-radius circle, which for Nova Carter (0.73 x 0.90 m) reaches up to 13 cm beyond the chassis
along the sides. A robot without measured bounds falls back to its radius circle. On the 11 pilot scenes (v1, v3-v5)
it flags v3 P18 (0.00 m) and v5 P24 (0.00 m) and passes 8; v5 P18 still fails at 0.25 m: its walker would touch the
chassis, which the Isaac check missed only because its 6 s settle ends before the walker gets there (2.9 of 3.5 m).
`check.json` reports per person the clearance to the footprint (`safety.human_paths`). New test: the same path passes
or fails depending on the spawn yaw.

### Documented: the physics settle covers only a 6 s window
The Isaac Sim settle in the final check simulates 6 s with people and forklifts moving from their start phase, so it
only exercises the first few metres of each path; a contact later in a path (v5 P18) is not seen there. A physics pass
is not evidence that a path is clear; whole-path interactions are checked statically (docs/DATAGEN.md, docs/MCP.md).

`scripts/run_e1.sh` and the runner's freeze check point at `e1-freeze-v7`.

## e1-freeze-v6 (2026-10-29)

### Added: mandatory check `check.human_path_robot_spawn`
Every person's waypoint path (the polyline through its waypoints; `loop` paths include the closing leg back to the
start; a single waypoint is a point) must keep at least `HUMAN_RADIUS_M` (0.26) + the robot's footprint radius +
`HUMAN_SPAWN_GAP_M` (0.30) from the robot spawn, i.e. the person's disk stays 0.3 m clear of the robot's footprint disk
(Nova Carter: 1.14 m from the spawn centre). A walker crossing the spawn shoves the parked robot in physics, which until
now only the Isaac Sim check caught (`physics.robot_unstable`: pilot P24 on v5, P18 on v3). The error names the person,
the closest point and the waypoint nearest the spawn, and points at `/agents/<i>/path/waypoints`; `check.json` lists
every person's distance under `safety.human_paths`, and the threshold is in `safety.thresholds`. Tests: a path over the
spawn, a clear path, and a loop whose closing leg passes the spawn (the same points in ping-pong pass).

`scripts/run_e1.sh` and the runner's freeze check point at `e1-freeze-v6`.

## e2-baselines-v2 (2026-10-29)

E2 baselines rebuilt against `e1-freeze-v5` (`results/e2_baselines_v2/`). The template's robot cell moves 0.4 m west
(x 5.4 -> 5.0): the cage is smaller than the panel fence it replaced (whose square was rounded up to whole panels), so the
belt outfeed ended 1.3 m from the cell and `conveyor_connected` failed in every template scene; it is now 0.9 m (>= the
0.5 m robot-cell clearance). The recipe is unchanged.

## e1-freeze-v5 (2026-10-29)

### Changed: robot cells are always caged
`robot_cell` drops the `panels` fence style (and `fence_layers`): the SimReady `barrier_fence_metal_a02` sections it stacked
are an open tubular barrier rail, which does not read as robot guarding. Every fenced cell now uses the wire-mesh cage
with a sliding gate (`fence_style` accepts only `cage`, the default); unfenced keep-out cells are unchanged. The
`safety_fence_panel` asset stays in the library as a standalone barrier.

### Fixed: the solver plans props with footprints that cover their collision hulls
The solver separated props by their visual footprint (margin 0) while `check_scene`'s overlap test uses the convex
collision hulls (3 mm tolerance); in 15 of 23 dynamic curated assets the hulls reach more than 3 mm past the visual
(monitor 14 mm, KLT bin 8 mm, electric motor 56 mm), so siblings on one surface could be placed overlapping and the build
then failed the check (E2 template seeds 18 and 26). Mesh assets now plan with the convex hull of the visual footprint
and the collision hulls (`mesh_asset.planning_polygon`); the curated table's baked footprints are re-baked the same way
(only `baked.footprint` changes; table sha256 changes). New test: template seeds 18 / 26 build without overlaps.

### Changed: `solve.unsatisfiable.on` says why
Repair feedback distinguished nothing: a prop whose support object was never placed got "blocked by: does_not_fit",
which reads as "the surface is too small". The error now names the cause (`data.cause`): `support_missing` ("the
support '<ref>' has no placed instance ...; the props themselves are fine") or `surface_full` (the surface is full or too
small; the blockers are listed, `too_big_for_surface` when a prop exceeds the surface).

`scripts/run_e1.sh` and the runner's freeze check point at `e1-freeze-v5`.

## e2-baselines-v1 (2026-10-29)

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
