# E3 grid v1 (tag `e3-grid-v1`)

10 scenes (pooled across three models, `results/e3_scene_selection`) x 3 tiers x 30 seeds x 3 policies = **2,700 trials**, policies at
`e3-policies-v1` (every row records the tag; `policies_at_tag` is true for all 2,700), 2D simulator, no Isaac Sim.
Code: `scripts/e3_grid.py` (runner, resumable), `scripts/e3_analysis.py` (statistics), `src/worldc/e3/`. Protocol: `docs/E3.md`.

| Tier | Workers | Scene variant |
|---|---|---|
| easy | 1 | base |
| medium | 3 | base |
| hard | 6 | base + as many occluders as fit (0-4), 1.2 m aisles kept, every robot goal and task point still reachable |

## Scenes and hard-tier occluders

Rule fixed before running: workers always 1 / 3 / 6; occluders = as many as fit, 0-4; the actual count is recorded.

| Scene | Source model | Family | Size | Room (m) | Occluders | Candidates (rejected: spacing / aisle or goal / solver) | Base from |
|---|---|---|---|---|---|---|---|
| P01 | deepseek-v4.1-flash_cloud | ev_battery_dismantling | S | 8 x 6 | 1 | 6 (2 / 3 / 0) | run spec.json |
| P05 | glm-5.3-flash_cloud | ev_battery_dismantling | M | 14 x 9 | 4 | 156 (60 / 2 / 3) | resolved spec |
| P06 | deepseek-v4.1-flash_cloud | electronics_weee | S | 9 x 6 | 1 | 19 (18 / 0 / 0) | resolved spec |
| P09 | glm-5.3-flash_cloud | electronics_weee | M | 13 x 9 | 4 | 58 (7 / 1 / 0) | run spec.json (2 scrap bins moved 1.3 / 2.1 m) |
| P11 | deepseek-v4.1-flash_cloud | gearbox_reman | S | 10 x 7 | **0** | 64 (0 / 0 / 64) | resolved spec |
| P15 | claude-sonnet-5-5 | gearbox_reman | L | 26 x 14 | 4 | 638 (2 / 0 / 0) | resolved spec |
| P16 | glm-5.3-flash_cloud | appliance_repair | S | 9 x 6 | **0** | 4 (0 / 0 / 4) | resolved spec |
| P19 | claude-sonnet-5-5 | appliance_repair | L | 22 x 12 | 4 | 280 (0 / 0 / 0) | resolved spec |
| P23 | glm-5.3-flash_cloud | sorting_grading | M | 16 x 10 | 1 | 39 (13 / 10 / 15) | run spec.json |
| P28 | claude-sonnet-5-5 | returned_parts_inspection | L | 20 x 12 | 4 | 483 (1 / 0 / 0) | resolved spec |

- **P11 and P16 have no occluders:** the spec's own keep-out zones (`main_aisle`, dock lane, door) cover every candidate, so the solver
  rejects all of them. Their hard tier is the base scene with 6 workers.
- **Reachability:** in all 10 scenes every robot goal is reachable in both variants, and no task point left the door's 1.2 m aisle
  network in the hard variant (`task_points_lost_vs_base` is empty). Task points that were never on the network in the base scene
  (back sides of benches against a wall, 0-15 per scene) are counted in `task_points_off_aisle_network`. Six-worker path planning
  succeeds for every scene. No scene needed a redraw.
- **Replay:** one hard / blind trial per scene rebuilt from its seed gives an identical trajectory and events (digest) in all 10.
- **Pre-check fix:** the first version of the aisle pre-check required every task point, including bench back sides, to be on the
  network, which rejected every candidate in 7 of 10 scenes with zero occluders. It is now relative to the scene without occluders
  (`tiers.py`, commit `aa7e056`).

## Files

- `trials.csv`: all 2,700 rows (scene, source model, family, size, tier, workers, occluders, seed, policy, outcome and metrics, provenance).
- `analysis/`: `table2.md` / `.tex` (moving and stationary collisions, speed at contact), `contact_speed.md` / `.csv` and `collisions.csv` (every collision with the robot's v, w at contact), `tests.md` / `.csv` (paired Wilcoxon, Holm), `scatter.svg`, and the supplementary breakdowns
  `supp_by_size`, `supp_by_scene`, `supp_by_model` (`.md` / `.csv`).
- `scenes/<scene>/`: `variants_report.json`, `e3_variant_{base,hard}.json`, `map_{base,hard}.png`, `trials.jsonl.gz` (drawn workers,
  events, trajectory and digest of every trial; for the later kinematic replay), `run.log`.
- `grid_summary.json`: per-scene run summary.

Regenerate: `uv run python scripts/e3_grid.py --seeds 30 --jobs 8` then
`uv run python scripts/e3_analysis.py out/e3/grid/P*/trials.csv --logs out/e3/grid/P*/trials.jsonl --out results/e3_grid/analysis`.
