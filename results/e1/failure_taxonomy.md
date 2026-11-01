| Class | claude-sonnet-5-5 | deepseek-v4.1-flash_cloud | gemma4_31b-cloud | glm-5.3-flash_cloud |
|---|---|---|---|---|
| accepted | 24 | 27 | 7 | 20 |
| physics failed: physics.fell_through_floor | 2 | 3 | 0 | 1 |
| physics failed: physics.not_settled | 3 | 0 | 0 | 7 |
| static failure, repair cap reached: solver: placement unsatisfiable | 1 | 0 | 8 | 1 |
| static failure, repair cap reached: spec error: id | 0 | 0 | 0 | 1 |
| static failure, repair cap reached: static check: human_path_robot_spawn | 0 | 0 | 1 | 0 |
| static failure, repair cap reached: static check: interpenetration | 0 | 0 | 2 | 0 |
| static failure, stopped without a valid scene: no error recorded | 0 | 0 | 11 | 0 |
| static failure, stopped without a valid scene: static check: hazard_path_blocked | 0 | 0 | 1 | 0 |

Runs per outcome class (30 prompts per model). Physics failures: the loop ended with a statically valid scene and the final Isaac Sim physics check rejected it; static failures: no valid scene (stop reason, family of the first error of the last validate / build / check).
