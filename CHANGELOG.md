# Changelog

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
