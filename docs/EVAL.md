# Post-hoc evaluation (E1 / E2): prompt adherence and layout plausibility

```
worldc eval --runs dataset/e1/<model> --prompts prompts/prompts.csv --out eval.csv          # E1 records
worldc eval --runs baselines/ --prompts prompts/prompts.csv --prompt-id ids.csv --out eval.csv   # E2 scene folders
```

Evaluation only: nothing here runs inside `check_scene`, reaches the generation / repair loop or changes what "accepted"
means (that stays `check_scene` + the final physics check). No Isaac Sim needed: it reads `scene_graph.json`, the resolved
spec and the scene's USD colliders (the occupancy grid is rebuilt with `usd-core`, as `check_scene` builds it).

## Rows

* **Records** (`--runs` holds `records.jsonl`): one row per record, joined to its prompt by the description (equal to the
  row's `text_plain` or `text_constrained`, which also gives `variant`). `repairs` and `at_freeze` come from the record.
* **Baselines** (no `records.jsonl`): one row per folder below `--runs` that holds a `scene_graph.json`. Prompt id from
  `--prompt-id` (CSV `dir,prompt_id`, or JSON `{dir: prompt_id}`), else a `P<nn>` in the folder name. `accepted` =
  `check_scene` passes.
* Non-accepted runs get a row with the metric columns empty.

## Adherence (`eval/adherence.py`)

Counts per scene from registry asset ids in `instances` (`work_table` + the curated benches; `robot_cell`;
`conveyor_belt` / `conveyor_roller`; `rack_row` + the curated shelving; `rollup_door` / `swing_door`) and walking people
from `hazards` (forklift agents ignored). Old warehouse graphs without `instances` fall back to record categories.
Per prompt field (`n_benches, n_robot_cells, conveyor, n_racks, n_doors, n_people`): `req_`, `got_`, `match_`
(+ `abserr_` for counts); -1 = unspecified, not scored. `adherence` = fraction of the specified fields that match.

## Plausibility (`eval/plausibility.py`; thresholds in one block at the top)

| Rule | Per | Passes when |
|---|---|---|
| `bench_access` | workbench | its working side has 0.8 m of free floor along it. Working side: where the asset's `front` port points (`work_table`), else the long side facing the most free floor (curated SimReady benches); `details` say which (`method: port` / `free_space`) |
| `conveyor_connected` | conveyor | (a) an end is within 1.0 m of a workbench, robot cell, pallet, door or other conveyor, **and** (b) a workbench or robot cell is within 1.5 m of the conveyor (end or side; 1.5 m = the mandatory 1.2 m aisle + 0.3 m) |
| `storage_reachable` | rack / shelving unit, staging zone, floor pallet outside one | reachable on foot from a door on the grid inflated by 0.26 m (outdoors: from the site boundary) |
| `cell_not_blocking` | robot cell | its safety footprint is at least 2.0 m from every door opening |

Per rule: `<rule>_n`, `_passed`, `_rate` (empty when the scene has no instance of it). `plausibility` = mean of the rates
that exist: a rule that does not apply is left out, not counted as a pass.
