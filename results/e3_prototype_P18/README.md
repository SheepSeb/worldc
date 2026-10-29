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

Primary metric in bold: robot-moving collisions per trial (mean of 10). `moving from behind` counts robot-moving
collisions where the person's centre was behind the robot's centre along the travel heading; `already braking` counts
those where the policy was already holding (the robot was braking in a stop).

| tier / policy | success | moving coll./trial | moving from behind (of total) | already braking (of total) | all / stationary coll. | <0.5 m / <1 m moving (s) | <0.5 m / <1 m all (s) | stationary <1 m of a person (s) | time to goal (median s) | outcomes |
|---|---|---|---|---|---|---|---|---|---|---|
| easy / blind | 0.7 | **0.3** | 0 of 3 | 0 of 3 | 0.3 / 0.0 | 0.6 / 3.2 | 0.9 / 3.6 | 0.4 | 50.7 | success 7, collision 3 |
| easy / fields | 0.9 | **0.1** | 1 of 1 | 0 of 1 | 0.2 / 0.1 | 0.5 / 2.9 | 2.1 / 4.8 | 1.9 | 50.7 | success 9, collision 1 |
| easy / aware | 1.0 | **0.0** | 0 of 0 | 0 of 0 | 0.5 / 0.5 | 0.0 / 1.3 | 1.5 / 5.5 | 4.2 | 62.7 | success 10 |
| medium / blind | 0.0 | **1.6** | 2 of 16 | 0 of 16 | 1.7 / 0.1 | 4.7 / 11.7 | 5.7 / 13.7 | 1.9 | 50.7 | collision 10 |
| medium / fields | 0.6 | **0.4** | 3 of 4 | 1 of 4 | 1.7 / 1.3 | 2.1 / 8.9 | 9.8 / 18.6 | 9.7 | 59.2 | success 6, collision 4 |
| medium / aware | 1.0 | **0.0** | 0 of 0 | 0 of 0 | 2.2 / 2.2 | 0.0 / 5.3 | 8.2 / 20.9 | 15.7 | 83.4 | success 10 |
| hard / blind | 0.0 | **2.7** | 4 of 27 | 0 of 27 | 2.9 / 0.2 | 8.4 / 19.8 | 9.7 / 22.7 | 2.9 | 51.3 | collision 10 |
| hard / fields | 0.3 | **1.1** | 6 of 11 | 5 of 11 | 4.2 / 3.1 | 4.6 / 17.0 | 24.6 / 42.1 | 25.1 | 74.2 | success 3, collision 6, frozen 1 |
| hard / aware | 0.4 | **0.0** | 0 of 0 | 0 of 0 | 6.9 / 6.9 | 0.0 / 10.9 | 24.1 / 70.7 | 59.8 | 169.6 | success 4, frozen 6 |

## Notes

- **Policies:** blind, fields (an ISO 3691-4-style stop-and-wait with speed-dependent protective and warning fields)
  and human-aware. Fields replaced the earlier fixed 1.0 m stop-and-wait rule. The earlier numbers are in the
  e3-prototype-2 commit.
- **Field sizes on Nova Carter:**
  - **Parameters:** a_brake 1.5 m/s², T_react 0.1 s, margin 0.25 m, approach 1.6 m/s.
  - **Protective length** = v·T_react + v²/(2·a_brake) + 1.6·(T_react + v/a_brake) + margin. That is 0.41 m standing
    still and 1.91 m at 1.0 m/s. Warning is 1.0 m longer.
  - **Shape:** both rectangles start at the front edge. They are the footprint's width plus 0.25 m a side, plus 0.15 m a
    side while turning.
  - a_brake is a conservative assumption, not a vendor spec.
- **Fields against blind:** fewer robot-moving collisions (hard 1.1 against 2.7) for some time (median 74 s against
  51 s). In hard it freezes once.
- **Where fields collides (16 robot-moving collisions):**
  - **10 from behind:** overtakes and turns with a person behind. The fields only look ahead.
  - **6 at the front or side while already braking.** None happen while the robot drives freely. The remaining ones are
    people crossing into the side of the field close to the robot: the approach term lengthens the field ahead, not to
    the side.
- **Parameter history** (robot-moving collisions per trial, easy / medium / hard):
  - a_brake 3.0 m/s², 0.10 m side margin, no approach term: 0.1 / 0.5 / 1.2.
  - a_brake 1.5 m/s², 0.25 m side margin, no approach term: 0.2 / 0.8 / 1.5, with 13 of 25 hitting while braking.
  - Adding the approach term: 0.1 / 0.4 / 1.1.
- **Human-aware** still has no robot-moving collision in any tier. It freezes in 6 of 10 hard trials and is slowest
  (median 170 s).
- **Stationary collisions:** most collisions of the two cautious policies are people walking into the robot while it
  stands. `stationary <1 m of a person` is the time the robot stands within 1.0 m of someone.
- **Files:**
  - `summary.md` / `summary.json`, and `trials.csv` (one row per trial).
  - `map_{base,hard}.png`: grid, task points, goals in green.
  - `hard_<policy>_seed<k>.png`: the hard trial with the most robot-moving collisions, then the smallest gap. Red crosses
    mark collisions with the robot moving, grey ones with it standing.
  - The full `trials.jsonl` and the GIFs are in `out/e3/P18/` (not tracked).
