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
