# Isaac Sim physics of the recipe scenes

Scenes that passed the static check, run through the E1 physics check (`scripts/e2_physics_queue.py`, 6 s settle). Passed / checked:

| Recipe | P05 | P08 | P15 | P16 | P23 | Total |
|---|---|---|---|---|---|---|
| recipe v4 (rules) | 5/5 | 5/18 | 3/20 | 5/5 | 5/7 | 23/55 |
| recipe v3 | 5/5 | 4/20 | 5/18 | 5/5 | 5/5 | 24/53 |

Failure causes by code and object: `physics_failure_causes.md`.
