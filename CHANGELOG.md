# Changelog

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
