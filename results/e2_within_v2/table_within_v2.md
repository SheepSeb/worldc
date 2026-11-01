**E2 within-prompt study: layout diversity, spatial adherence and acceptance** (prompts P05, P08, P15, P16, P23; physics-accepted samples only)

| Method | Samples used | Within-prompt IoU [95% CI] | Spatial adherence [95% CI] | LLM acceptance per run (static + physics) | Recipe / template: physics-accepted per sample tried | Static builds ok per build attempt | Prompts reaching 5 physics-accepted |
|---|---|---|---|---|---|---|---|
| Sonnet 5.5 | 25 | 0.27 [0.15, 0.52] | 0.85 [0.76, 0.94] | 25/55 (45%) | n/a | n/a | 5/5 |
| DeepSeek v4.1 flash | 25 | 0.20 [0.06, 0.35] | 0.80 [0.71, 0.90] | 25/40 (62%) | n/a | n/a | 5/5 |
| GLM 5.3 flash | 25 | 0.20 [0.09, 0.38] | 0.86 [0.81, 0.93] | 25/36 (69%) | n/a | n/a | 5/5 |
| Recipe v4 (rules) | 23 | 0.25 [0.10, 0.65] | 0.95 [0.87, 1.00] | n/a | 23/56 (41%) | 55/194 (28%) | 4/5 |
| Recipe v3 (no rules) | 24 | 0.05 [0.03, 0.07] | 0.21 [0.10, 0.32] | n/a | 24/54 (44%) | 53/145 (37%) | 4/5 |
| Template | 5 | 1.00 [1.00, 1.00] | 0.88 [0.88, 0.88] | n/a | 5/20 (25%) | 19/20 (95%) | 1/1 |

Recipe attempts per prompt (physics-accepted / samples tried, stop at 5 passes or 20 samples):

| Method | P05 | P08 | P15 | P16 | P23 |
|---|---|---|---|---|---|
| Recipe v4 (rules) | 5/5 | 5/19 | 3/20 | 5/5 | 5/7 |
| Recipe v3 (no rules) | 5/5 | 4/20 | 5/19 | 5/5 | 5/5 |
| Template | - | 5/20 | - | - | - |

Only physics-accepted scenes are used for every method (LLM: accepted = static `check_scene` + the final Isaac Sim physics check; recipe and template: static + the same physics check run by `scripts/e2_physics_queue.py`). IoU: median over pairs of same-prompt samples on native-metre 0.1 m grids (lower = more diverse); a prompt with fewer than 5 samples contributes fewer pairs (none with fewer than 2). Spatial adherence: mean over scenes of the mean rule score (1 = every instance satisfied; an absent object scores 0), `prompts/spatial_rules_e2_within.json`. The template is scored on P08 only, so its spatial CI is degenerate. The recipe is sampled seed by seed (seed base + k, +1000 retries on a failed rule / build / static check) until 5 samples pass physics or 20 samples were tried; every sample is one attempt and the static build retries are counted in the fourth-from-last column.
