# E2 baselines v1

Built at commit `8adce78` (recipe + template code; the asset library, compiler and checks are those of `e1-freeze-v4`,
`adb19e7`; curated table sha256 `17db5bae…`). Static `check_scene` only; Isaac Sim physics is queued after the E1 batch
(`scripts/e2_physics_queue.py`). Scenes themselves are in `out/e2/` (not tracked).

- `template_records.json`, `recipe_records.json`: per scene `e2_attempts.json` (attempts, first-try result, recipe knobs
  with the realised `bench_arrangement`, provenance).
- `eval_template.csv`, `eval_recipe.csv`: `worldc eval` (prompt id from the folder name; template = P08).
- `e2_summary.json`: per-method acceptance, adherence, plausibility and rule rates over accepted scenes.

`provenance.tree_dirty` is `true` in these records: the only change in the working tree during the builds was the
untracked `scripts/e2_physics_queue.py` (not used by the builds); `changed_since_freeze` lists every tracked change.
