# E2 baselines v2

Built against `e1-freeze-v5` (`3f69259`: caged robot cells, hull-covering planning footprints, `solve.unsatisfiable.on`
with its cause). Recipe scenes built from a clean checkout at the freeze commit (`changed_since_freeze: []`); template
scenes at `2b46810`, which only moves the template's robot cell 0.4 m west so the belt outfeed reaches the cage
(`changed_since_freeze: [specs/e2_template_reman_cell.json]`). Static `check_scene` only; Isaac Sim physics is queued after
the E1 batch (`scripts/e2_physics_queue.py`). Scenes are in `out/e2_v2/` (not tracked). Files as in v1:
`*_records.json` (attempts, knobs with the realised `bench_arrangement`, provenance), `eval_*.csv` (`worldc eval`; template
= P08), `e2_summary.json`. The template's one rejection (seed 7) fails in the solver (`solve.unsatisfiable.on`, cause
`surface_full`: only 2 of 3 KLT bins fit the shelf boards), so it has no scene graph and no eval row (29 rows).
