# E2 baselines v3

Built from a clean checkout of `e1-freeze-v7` (`6f9647d`; `changed_since_freeze: []`, `tree_dirty: false` in every
record): the mandatory `check.human_path_robot_spawn` (clearance to the robot's footprint rectangle), caged robot cells,
hull-covering planning footprints. The recipe parks the robot beside the main door lane (`robot_spawn` knob). Static
`check_scene` only; Isaac Sim physics is queued after the E1 batch (`scripts/e2_physics_queue.py`). Scenes are in
`out/e2_v3/` (not tracked). Files as in v1/v2: `*_records.json`, `eval_*.csv` (`worldc eval`; template = P08),
`e2_summary.json`. Template seed 7 fails in the solver (`solve.unsatisfiable.on`, cause `surface_full`), so it has no
scene graph and no eval row (29 rows).
