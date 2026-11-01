| metric | tier | A vs B | pairs | non-zero | A better / B better | median diff (A-B) | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| moving_collisions | easy | blind vs fields | 300 | 56 | 1 / 55 | 0.00 | 2.79e-11 | 1.12e-10 |
| moving_collisions | easy | blind vs aware | 300 | 57 | 0 / 57 | 0.00 | 5.21e-12 | 2.61e-11 |
| moving_collisions | easy | fields vs aware | 300 | 3 | 0 / 3 | 0.00 | 0.0833 | 0.0833 |
| moving_collisions | medium | blind vs fields | 300 | 139 | 4 / 135 | 0.00 | 2.96e-24 | 1.77e-23 |
| moving_collisions | medium | blind vs aware | 300 | 141 | 0 / 141 | 0.00 | 5.56e-26 | 3.89e-25 |
| moving_collisions | medium | fields vs aware | 300 | 16 | 0 / 16 | 0.00 | 6.33e-05 | 0.000127 |
| moving_collisions | hard | blind vs fields | 300 | 172 | 5 / 167 | 1.00 | 2.68e-29 | 2.14e-28 |
| moving_collisions | hard | blind vs aware | 300 | 180 | 0 / 180 | 1.00 | 1.03e-31 | 9.31e-31 |
| moving_collisions | hard | fields vs aware | 300 | 38 | 0 / 38 | 0.00 | 4.8e-09 | 1.44e-08 |
| time_to_goal | easy | blind vs fields | 300 | 223 | 222 / 1 | -2.60 | 2.49e-38 | 2.49e-38 |
| time_to_goal | easy | blind vs aware | 300 | 298 | 289 / 9 | -10.82 | 2.81e-50 | 1.41e-49 |
| time_to_goal | easy | fields vs aware | 300 | 298 | 283 / 15 | -5.25 | 2.49e-48 | 4.97e-48 |
| time_to_goal | medium | blind vs fields | 300 | 291 | 291 / 0 | -10.48 | 1.8e-49 | 7.22e-49 |
| time_to_goal | medium | blind vs aware | 300 | 300 | 300 / 0 | -49.97 | 5.85e-51 | 4.77e-50 |
| time_to_goal | medium | fields vs aware | 300 | 300 | 299 / 1 | -35.12 | 7.07e-51 | 4.77e-50 |
| time_to_goal | hard | blind vs fields | 300 | 300 | 300 / 0 | -25.68 | 6.08e-51 | 4.77e-50 |
| time_to_goal | hard | blind vs aware | 300 | 300 | 300 / 0 | -92.60 | 5.31e-51 | 4.77e-50 |
| time_to_goal | hard | fields vs aware | 300 | 288 | 288 / 0 | -58.92 | 5.59e-49 | 1.68e-48 |

Supplementary: trial-level pairs (same scene and seed); trials of one scene are not independent, so these p-values are optimistic.
