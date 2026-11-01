| metric | tier | A vs B | pairs | non-zero | A better / B better | median diff (A-B) | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| moving_collisions | easy | blind vs fields | 300 | 144 | 0 / 144 | 0.00 | 7.31e-28 | 3.66e-27 |
| moving_collisions | easy | blind vs aware | 300 | 144 | 0 / 144 | 0.00 | 7.31e-28 | 3.66e-27 |
| moving_collisions | easy | fields vs aware | 300 | 0 | 0 / 0 | 0.00 | 1 | 1 |
| moving_collisions | medium | blind vs fields | 300 | 245 | 0 / 245 | 2.00 | 1.23e-42 | 7.35e-42 |
| moving_collisions | medium | blind vs aware | 300 | 246 | 0 / 246 | 2.00 | 8.46e-43 | 5.92e-42 |
| moving_collisions | medium | fields vs aware | 300 | 2 | 0 / 2 | 0.00 | 0.18 | 0.359 |
| moving_collisions | hard | blind vs fields | 300 | 280 | 0 / 280 | 4.00 | 7.26e-48 | 5.81e-47 |
| moving_collisions | hard | blind vs aware | 300 | 281 | 0 / 281 | 4.00 | 4.95e-48 | 4.45e-47 |
| moving_collisions | hard | fields vs aware | 300 | 5 | 0 / 5 | 0.00 | 0.0339 | 0.102 |
| time_to_goal | easy | blind vs fields | 300 | 233 | 232 / 1 | -69.00 | 5.01e-40 | 2e-39 |
| time_to_goal | easy | blind vs aware | 300 | 298 | 290 / 8 | -92.60 | 2.13e-50 | 1.28e-49 |
| time_to_goal | easy | fields vs aware | 300 | 138 | 115 / 23 | 0.00 | 8.24e-12 | 2.47e-11 |
| time_to_goal | medium | blind vs fields | 300 | 292 | 292 / 0 | -126.45 | 9.99e-50 | 5e-49 |
| time_to_goal | medium | blind vs aware | 300 | 300 | 300 / 0 | -126.45 | 4.74e-51 | 4.15e-50 |
| time_to_goal | medium | fields vs aware | 300 | 40 | 37 / 3 | 0.00 | 4.29e-06 | 8.57e-06 |
| time_to_goal | hard | blind vs fields | 300 | 300 | 300 / 0 | -126.45 | 4.71e-51 | 4.15e-50 |
| time_to_goal | hard | blind vs aware | 300 | 300 | 300 / 0 | -126.45 | 4.61e-51 | 4.15e-50 |
| time_to_goal | hard | fields vs aware | 300 | 10 | 10 / 0 | 0.00 | 0.00506 | 0.00506 |

Supplementary: trial-level pairs (same scene and seed); trials of one scene are not independent, so these p-values are optimistic.
