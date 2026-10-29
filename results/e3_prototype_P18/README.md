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
collisions where the person's centre was behind the robot's centre along the travel heading.

| tier / policy | success | moving coll./trial | moving from behind (of total) | all / stationary coll. | <0.5 m / <1 m moving (s) | <0.5 m / <1 m all (s) | stationary <1 m of a person (s) | time to goal (median s) | outcomes |
|---|---|---|---|---|---|---|---|---|---|
| easy / blind | 0.7 | **0.3** | 0 of 3 | 0.3 / 0.0 | 0.6 / 3.2 | 0.9 / 3.6 | 0.4 | 50.7 | success 7, collision 3 |
| easy / fields | 0.8 | **0.2** | 1 of 2 | 0.3 / 0.1 | 0.6 / 3.2 | 2.1 / 4.8 | 1.7 | 50.7 | success 8, collision 2 |
| easy / aware | 1.0 | **0.0** | 0 of 0 | 0.5 / 0.5 | 0.0 / 1.3 | 1.5 / 5.5 | 4.2 | 62.7 | success 10 |
| medium / blind | 0.0 | **1.6** | 2 of 16 | 1.7 / 0.1 | 4.7 / 11.7 | 5.7 / 13.7 | 1.9 | 50.7 | collision 10 |
| medium / fields | 0.4 | **0.8** | 3 of 8 | 1.9 / 1.1 | 3.2 / 11.1 | 9.3 / 18.7 | 7.6 | 57.5 | success 4, collision 6 |
| medium / aware | 1.0 | **0.0** | 0 of 0 | 2.2 / 2.2 | 0.0 / 5.3 | 8.2 / 20.9 | 15.7 | 83.4 | success 10 |
| hard / blind | 0.0 | **2.7** | 4 of 27 | 2.9 / 0.2 | 8.4 / 19.8 | 9.7 / 22.7 | 2.9 | 51.3 | collision 10 |
| hard / fields | 0.2 | **1.5** | 8 of 15 | 4.1 / 2.6 | 6.3 / 20.3 | 20.9 / 37.2 | 16.9 | 67.7 | success 2, collision 8 |
| hard / aware | 0.4 | **0.0** | 0 of 0 | 6.9 / 6.9 | 0.0 / 10.9 | 24.1 / 70.7 | 59.8 | 169.6 | success 4, frozen 6 |

## Notes

- **Policies:** blind, fields (an ISO 3691-4-style stop-and-wait with speed-dependent protective and warning fields)
  and human-aware. Fields replaced the earlier fixed 1.0 m stop-and-wait rule. The earlier numbers are in the
  e3-prototype-2 commit.
- **Field sizes on Nova Carter** (a_brake 1.5 m/s², T_react 0.1 s, margin 0.25 m): protective 0.25 m standing still,
  0.68 m at 1.0 m/s. Warning is 1.0 m longer. Both rectangles start at the front edge and are the footprint's width plus
  0.25 m a side, plus 0.15 m a side while turning. a_brake is a conservative assumption, not a vendor spec.
- **Fields against blind:** fewer robot-moving collisions (hard 1.5 against 2.7) at a small time cost (median 68 s
  against 51 s). It collides more than the earlier fixed 1.0 m stop-and-wait rule (hard 0.4).
- **Where fields collides (25 robot-moving collisions):**
  - **13 at the front or side while already braking** in a protective stop. A person enters the field, the robot brakes
    at 1.5 m/s² (up to 0.7 s from 1 m/s), and the non-reactive person keeps walking into it before it stands. The field
    length covers the robot's own stopping distance only, not the person's approach speed during the stop (the K·T
    term of ISO 13855-style sizing).
  - **12 from behind:** 9 overtakes while driving, and 3 corner sweeps or turns with a person behind. The fields only look
    ahead.
  - **Parameters:** with a_brake 3.0 m/s² and a 0.10 m side margin, fields had 0.1 / 0.5 / 1.2 robot-moving collisions
    (easy / medium / hard). Softer braking lengthens each stop, and the wider margin stops the robot for more crossers,
    so more people reach it while it brakes.
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
