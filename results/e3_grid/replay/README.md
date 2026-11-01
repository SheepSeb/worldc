# Kinematic Isaac Sim replay: P23, hard tier, seed 14, all three policies

Rendering only: no physics. The robot follows its logged trajectory (interpolated from the 0.2 s log to the frame times) and the six
workers their positions recomputed from the trial seed with the trial's own code; heading from the direction of travel. The scene is the
hard variant of P23 (glm-5.3-flash_cloud, 1 occluder) plus six human characters (Isaac Sim, SimReady scene assets) added only for this
replay. Tool: `scripts/e3_replay.py` -> `src/worldc/render/isaac_replay.py` (12 fps, 1280 x 720, RTX); two cameras per trial, `overhead`
(roof hidden) and `corner`, taken from the scene. The time and a COLLISION flag (0.8 s after each logged contact) are drawn on every frame.

| Policy | Outcome | Robot-moving coll. | Stationary coll. | Duration |
|---|---|---|---|---|
| blind | collision | 1 at 10.5 s (0.56 m/s) | 0 | 14.5 s |
| fields | success | 0 | 0 | 18.4 s |
| aware | success | 0 | 7 (people walk into the stopped robot) | 41.4 s |

Files: `P23_hard_seed14_<policy>_<camera>.mp4` and a still (`.png`) at the first contact (or mid-trial when there is none). The logged trials
are in `results/e3_grid/scenes/P23/trials.jsonl.gz` (digests `2b99713d3d786dac`, `5bb5d9ae8411f9ae`, `cfe83db0d6e754a0`).
Notes: the characters stand in a fixed pose and slide (no walking animation); the replay scene's static check flags a 3 cm overlap of
one added character with a bench at its start pose, which does not matter for rendering.
