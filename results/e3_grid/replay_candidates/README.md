# Kinematic Isaac replay: proposed trials

Rendering only (no physics): the logged robot trajectory and the drawn workers of a trial are replayed as animated prims. Each
proposal is **one hard-tier seed under all three policies** (same scene, same people), so the three clips are directly comparable.
Maps (robot path in red, workers' paths thin, collisions as crosses) are `<scene>_hard_seed<seed>_<policy>.png`; the trial logs are
in `results/e3_grid/scenes/<scene>/trials.jsonl.gz` (digest of each trial below).

## Proposal A (recommended): P23, hard, seed 14

Sorting / grading room, 16 x 10 m, glm-5.3-flash_cloud, 1 occluder.

| Policy | Outcome | Robot-moving coll. | Stationary coll. | Other | Time (s) | Digest |
|---|---|---|---|---|---|---|
| blind | collision | 1 (t 10.5 s, robot at 0.56 m/s) | 0 | | 14.5 | `2b99713d3d786dac` |
| fields | success | 0 | 0 | 3 protective stops, min gap 0.44 m | 18.4 | `5bb5d9ae8411f9ae` |
| aware | success | 0 | 7 (robot stopped, people walk into it) | 35 replans | 41.4 | `cfe83db0d6e754a0` |

Shows the three behaviours in one scene: the blind robot drives into a person, the field robot stops and gets through quickly, the
human-aware robot slows, replans and finishes 3 x later, and the 7 stationary contacts (people walking into the stopped robot) are
visible, which matters for how the 0 moving collisions of human-aware are read (see `docs/E3.md`).

## Proposal B (larger room, 4 occluders): P19, hard, seed 8

Appliance repair, 22 x 12 m, claude-sonnet-5-5, 4 occluders.

| Policy | Outcome | Robot-moving coll. | Stationary coll. | Other | Time (s) | Digest |
|---|---|---|---|---|---|---|
| blind | collision | 2 (16.1 s at 0.84 m/s; 39.1 s turning in place) | 0 | | 41.5 | `c3f8826942f33e1a` |
| fields | success | 0 | 1 | 5 protective stops | 46.7 | `5edd0e1f46c45aa2` |
| aware | success | 0 | 1 | 12 replans | 49.5 | `f80cf4d5e9562126` |

Selection rule for both: hard tier, blind has a robot-moving collision, fields and human-aware succeed, fields has at least one
protective stop; then the shortest blind trial of a medium or large room. 20 seeds qualify; these two read best. 2-3 trials in all:
Proposal A is the three clips of one seed; B is a spare.
