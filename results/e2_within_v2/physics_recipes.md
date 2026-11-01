# Isaac Sim physics of the within-prompt recipe scenes

`scripts/e2_physics_queue.py --now out/e2_within/recipe_v4 out/e2_within/recipe` (same rule as the E1 final check: 6 s settle). Run
2026-11-01 on the idle GPU; per scene in `physics_recipes.csv`.

| Recipe | P05 | P08 | P15 | P16 | P23 | Total |
|---|---|---|---|---|---|---|
| v4 (rules) | 5/5 | 1/5 | 0/5 | 5/5 | 3/5 | 14/25 |
| v3 (no rules) | 5/5 | 0/5 | 0/5 | 5/5 | 5/5 | 15/25 |

Failures: `physics.not_settled` (scenes where more than 2 cm of movement remains after 6 s), and in P15 also `fell_through_floor` and
`unstable`. The recipe's static acceptance (25/25 after resampling) therefore overstates what E1 counts as accepted: E1 accepted =
static + physics. The recipe does not resample on physics failures; doing so would be a different baseline.
