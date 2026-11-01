# Known issues

Issues found during the experiments that are **not fixed yet** (to be fixed after the paper deadline). Nothing here changes a
reported number; each entry says what is affected and how the experiments work around it.

## `resolved_spec.json` drops the height (`z`) of explicitly posed instances

- **What:** an object with an explicit `placement.pose` list may give each instance a `z` (stacked crates, items on a bench or shelf).
  `resolved_spec.json`, written next to every built scene, keeps `xy` and `yaw` but not `z`.
  Example (E1 run `20261029T134119_7e18efd9`, P01, `battery_crates`): the run's `spec.json` has
  `{"xy": [-3.48, -2.45], "z": 0}, {"xy": [-3.48, -2.45], "z": 0.265}, ...`, the resolved spec has the same `xy` twice with no `z`.
  Same for P23 `returned_monitors` (`z` 0.86 lost).
- **Effect:** rebuilding a scene from its resolved spec puts the stacked / raised instances on the floor at the same `xy`, so the solver
  rejects them (`solve.unsatisfiable.pose ... blocked by: battery_crates`) and the build fails. Rebuilding from the run's `spec.json`
  works.
- **Affected in the experiments:** E3 rebuilds each scene's base variant from `resolved_spec.json`. Three of the 10 E3 scenes
  (P01, P09, P23) did not rebuild this way. For these `worldc.e3.tiers.make_variants` falls back to the run's `spec.json` with the
  agents removed, and records `spec_source` and the instances that moved against the E1 scene in `e3_variant.json`. P01 and P23 are
  identical to the E1 scene; in P09 two scrap bins landed 1.3 m and 2.1 m from their E1 position, because the solver places them
  around the removed agents' paths (P09 passes `check_scene` in both variants, see `docs/E3.md`). Nothing else uses resolved specs
  for rebuilding.
- **Fix:** write `z` into the resolved poses (or resolve to the `on` relation they came from) and add a rebuild round-trip test: a
  resolved spec must reproduce the scene's instance poses exactly.

## Laptop and transfer-case props fail the physics check (`physics.not_settled`, `fell_through_floor`, `unstable`)

- **What:** scenes that put `laptop` (also `isaac_sm_laptop`) or `transfer_case` (also `simready_drivetrain_transfer_case_b01`) instances on
  benches, shelves or the floor often fail the Isaac Sim physics check: the props move more than 2 cm in the 6 s settle
  (`physics.not_settled`, usually 0.1-0.17 m), and some transfer cases and laptops fall through the floor and move faster than 1 m/s
  (`fell_through_floor`, `unstable`; displacements up to 2.9 m). The layouts around them are fine.
- **Evidence:**
  - **E1:** 15 of the 16 `physics_failed` runs fail only on laptops (6 runs) or transfer cases (9 runs), across all three models that
    produced scenes (Sonnet 5, DeepSeek 3, GLM 7); the 16th is a KLT bin placed on a belt (`results/e1/physics_failure_objects.md`).
    Counting these as passes would lift acceptance to at most 29/30 (Sonnet), 30/30 (DeepSeek) and 27/30 (GLM), an upper bound because the
    physics check stops at the first problems it finds.
  - **E2 recipe:** every physics failure of the recipe scenes names `bench_laptop` (P08, P23) or `bench_transfer_case` (P15); P05 and P16
    (no such props) never fail. Physics-accepted samples per attempt: v4 P08 5/19, P15 3/20; v3 P08 4/20, P15 5/19
    (`results/e2_within_v2/physics_failure_causes.md`).
  - **E2 template:** 20 of its 29 built scenes fail, all on `laptops`.
- **Cause:** not diagnosed. The check's own hint points at the collider and spawn height of the asset (`physics.fell_through_floor`:
  "Check its collider and spawn height"); candidates are the assets' collision approximations or masses and the height at which they are
  spawned on a bench top.
- **Affected in the experiments:** E1 acceptance (physics failures are counted as failures, and the E1 taxonomy has an "asset-induced
  (laptop / transfer case)" class for them), and every E2 comparison that needs physics-accepted scenes (the recipe, the template and the
  LLM samples are all subject to it). Nothing is worked around: scenes are reported as the check judged them.
- **Fix (after the deadline):** diagnose the two assets (collider approximation, mass, spawn offset), fix them in the asset library or settle
  such props before the check, then rerun the final physics check on the stored E1 and E2 scenes (no LLM re-run needed) and update the
  acceptance numbers.
