# E3 prototype on one scene: P18

P18 (15 x 10 m appliance repair, `e1_pilot_v5` run `20261029T115138_2acd8196`) on the 2D simulator: 3 tiers x 10 seeds x 3
policies = 90 trials in 43 s. Protocol and metric definitions are in `docs/E3.md`. Built with the E3 code of this commit.

```
uv run python scripts/e3_one_scene.py dataset/e1_pilot_v5/deepseek-v4.1-flash_cloud/runs/20261029T115138_2acd8196/scene --name P18 --seeds 10
```

**Variants** (`base_variant.json`, `hard_variant.json`): both rebuilt from the resolved spec without its 3 authored agents.
The hard variant adds 4 shelving units, accepted on the first attempt. Both pass `check_scene` (static).

**Robot frame:** Nova Carter, 0.73 x 0.90 m from the robot table; travel along local +x from this scene's Isaac drive
calibration.

**Replay:** hard seed 4 (blind), rebuilt from its seed in a fresh world gives an identical trajectory and events.

## Results

Primary metric in bold: robot-moving collisions per trial (mean of 10).

| tier / policy | success | moving coll./trial | all / stationary coll. | <0.5 m / <1 m moving (s) | <0.5 m / <1 m all (s) | stationary <1 m of a person (s) | time to goal (median s) | outcomes |
|---|---|---|---|---|---|---|---|---|
| easy / blind | 0.7 | **0.3** | 0.3 / 0.0 | 0.6 / 3.2 | 0.9 / 3.6 | 0.4 | 50.7 | success 7, collision 3 |
| easy / stop | 1.0 | **0.0** | 0.5 / 0.5 | 0.0 / 1.2 | 2.4 / 8.6 | 7.4 | 58.1 | success 10 |
| easy / aware | 1.0 | **0.0** | 0.5 / 0.5 | 0.0 / 1.3 | 1.5 / 5.5 | 4.2 | 62.7 | success 10 |
| medium / blind | 0.0 | **1.6** | 1.7 / 0.1 | 4.7 / 11.7 | 5.7 / 13.7 | 1.9 | 50.7 | collision 10 |
| medium / stop | 0.8 | **0.2** | 2.1 / 1.9 | 0.8 / 4.2 | 7.4 / 24.3 | 20.1 | 71.8 | success 8, collision 2 |
| medium / aware | 1.0 | **0.0** | 2.2 / 2.2 | 0.0 / 5.3 | 8.2 / 20.9 | 15.7 | 83.4 | success 10 |
| hard / blind | 0.0 | **2.7** | 2.9 / 0.2 | 8.4 / 19.8 | 9.7 / 22.7 | 2.9 | 51.3 | collision 10 |
| hard / stop | 0.3 | **0.4** | 5.4 / 5.0 | 2.0 / 8.8 | 21.0 / 59.4 | 50.6 | 94.1 | success 3, collision 4, frozen 3 |
| hard / aware | 0.4 | **0.0** | 6.9 / 6.9 | 0.0 / 10.9 | 24.1 / 70.7 | 59.8 | 169.6 | success 4, frozen 6 |

## Notes

- **Policies guard the align turn at goals:** stop-and-wait holds the turn under its 1.0 m rule, and human-aware scales
  its rotation. With this, human-aware has no robot-moving collision in any tier.
- **Blind:** almost all of its collisions happen while it moves. Robot-moving collisions rise from easy to hard
  (0.3, 1.6, 2.7).
- **Stop-and-wait:** 6 robot-moving collisions remain in medium and hard. 5 happen while it drives and 1 while it turns.
  In each, the person comes from beside or behind the robot, which its "ahead" rule does not cover.
- **Stationary collisions:** most collisions of the two cautious policies are people walking into the robot while it
  stands. `stationary <1 m of a person` is the time the robot stands within 1.0 m of someone.
- **Frozen:** in the hard tier the cautious policies often freeze (stop 3, aware 6 of 10). Freezing costs human-aware
  time: its median time to goal is 170 s.
- **Files:**
  - `summary.md` / `summary.json`, and `trials.csv` (one row per trial).
  - `map_{base,hard}.png`: grid, task points, goals in green.
  - `hard_<policy>_seed<k>.png`: the hard trial with the most robot-moving collisions, then the smallest gap. Red crosses
    mark collisions with the robot moving, grey ones with it standing.
  - The full `trials.jsonl` and the GIFs are in `out/e3/P18/` (not tracked).
