# E3 prototype on one scene: P18

P18 (15 x 10 m appliance repair, `e1_pilot_v5` run `20261029T115138_2acd8196`) on the 2D simulator: 3 tiers x 10 seeds x 3
policies = 90 trials in 52 s. Protocol and metric definitions are in `docs/E3.md`. Built from `eb2776e` + the E3 code in
this commit.

```
uv run python scripts/e3_one_scene.py dataset/e1_pilot_v5/deepseek-v4.1-flash_cloud/runs/20261029T115138_2acd8196/scene --name P18 --seeds 10
```

**Variants** (`base_variant.json`, `hard_variant.json`): both rebuilt from the resolved spec without its 3 authored agents.
The hard variant adds 4 shelving units, accepted on the first attempt. Both pass `check_scene` (static).

**Robot frame:** Nova Carter, 0.73 x 0.90 m from the robot table; travel along local +x from this scene's Isaac drive
calibration.

**Replay:** hard seed 4 (blind) rebuilt from its seed in a fresh world gives an identical trajectory and events.

## Results

Primary metric in bold: robot-moving collisions per trial (mean of 10).

| tier / policy | success | moving coll./trial | all / stationary coll. | <0.5 m / <1 m moving (s) | <0.5 m / <1 m all (s) | stationary in walkway (s) | time to goal (median s) | outcomes |
|---|---|---|---|---|---|---|---|---|
| easy / blind | 0.7 | **0.3** | 0.3 / 0.0 | 0.6 / 3.2 | 0.9 / 3.6 | 5.9 | 50.7 | success 7, collision 3 |
| easy / stop | 1.0 | **0.0** | 0.5 / 0.5 | 0.0 / 1.2 | 2.4 / 8.6 | 14.0 | 58.1 | success 10 |
| easy / aware | 1.0 | **0.0** | 0.6 / 0.6 | 0.3 / 2.6 | 2.7 / 5.7 | 13.3 | 62.6 | success 10 |
| medium / blind | 0.0 | **1.6** | 1.7 / 0.1 | 4.7 / 11.7 | 5.7 / 13.7 | 5.9 | 50.7 | collision 10 |
| medium / stop | 0.9 | **0.1** | 1.9 / 1.8 | 0.5 / 4.1 | 6.8 / 23.9 | 26.9 | 68.7 | success 9, collision 1 |
| medium / aware | 1.0 | **0.0** | 1.8 / 1.8 | 0.7 / 7.0 | 4.5 / 18.7 | 27.0 | 82.4 | success 10 |
| hard / blind | 0.0 | **2.7** | 2.9 / 0.2 | 8.4 / 19.7 | 9.7 / 22.7 | 6.0 | 51.3 | collision 10 |
| hard / stop | 0.4 | **0.5** | 4.3 / 3.8 | 2.1 / 9.5 | 14.4 / 50.7 | 46.9 | 91.8 | success 4, collision 4, frozen 2 |
| hard / aware | 0.1 | **0.6** | 6.1 / 5.5 | 1.3 / 14.2 | 29.4 / 67.5 | 68.5 | 136.6 | success 1, collision 4, frozen 5 |

## Notes

- **Blind:** almost all of its collisions happen while it moves. Robot-moving collisions rise from easy to hard
  (0.3, 1.6, 2.7).
- **Stop-and-wait and human-aware:**
  - Most of their collisions are people walking into the robot while it stands, often while it waits at a bench pick-up
    in the bench-front walkway (`stationary in walkway`).
  - 10 of their 12 robot-moving collisions in medium and hard happen while the robot turns in place to align at a goal.
    Neither policy watches for people during that turn.
  - In the hard tier they often freeze (stop 2, aware 5 of 10).
- Files: `summary.md` / `summary.json`, `trials.csv` (one row per trial), `map_{base,hard}.png` (grid, walkways in blue,
  task points, goals in green), and `hard_<policy>_seed<k>.png` (the hard trial with the most robot-moving collisions;
  red crosses = robot moving, grey = stationary). The full `trials.jsonl` and the GIFs are in `out/e3/P18/` (not tracked).
