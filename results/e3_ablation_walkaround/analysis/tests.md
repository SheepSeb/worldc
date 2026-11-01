| metric | tier | A vs B | scenes | non-zero | A better / B better | mean A | mean B | median diff (A-B) | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| moving_collisions | easy | blind vs fields | 10 | 8 | 0 / 8 | 0.25 | 0.01 | 0.22 | 0.00781 | 0.0391 |
| moving_collisions | easy | blind vs aware | 10 | 8 | 0 / 8 | 0.25 | 0.00 | 0.25 | 0.00781 | 0.0391 |
| moving_collisions | easy | fields vs aware | 10 | 3 | 0 / 3 | 0.01 | 0.00 | 0.00 | 0.25 | 0.25 |
| moving_collisions | medium | blind vs fields | 10 | 10 | 0 / 10 | 0.80 | 0.05 | 0.72 | 0.00195 | 0.0176 |
| moving_collisions | medium | blind vs aware | 10 | 10 | 0 / 10 | 0.80 | 0.00 | 0.80 | 0.00195 | 0.0176 |
| moving_collisions | medium | fields vs aware | 10 | 7 | 0 / 7 | 0.05 | 0.00 | 0.05 | 0.0156 | 0.0391 |
| moving_collisions | hard | blind vs fields | 10 | 10 | 0 / 10 | 1.59 | 0.15 | 1.35 | 0.00195 | 0.0176 |
| moving_collisions | hard | blind vs aware | 10 | 10 | 0 / 10 | 1.59 | 0.00 | 1.48 | 0.00195 | 0.0176 |
| moving_collisions | hard | fields vs aware | 10 | 8 | 0 / 8 | 0.15 | 0.00 | 0.12 | 0.00781 | 0.0391 |
| time_to_goal | easy | blind vs fields | 10 | 10 | 10 / 0 | 57.55 | 61.99 | -4.01 | 0.00195 | 0.0176 |
| time_to_goal | easy | blind vs aware | 10 | 10 | 10 / 0 | 57.55 | 74.56 | -14.20 | 0.00195 | 0.0176 |
| time_to_goal | easy | fields vs aware | 10 | 10 | 10 / 0 | 61.99 | 74.56 | -10.46 | 0.00195 | 0.0176 |
| time_to_goal | medium | blind vs fields | 10 | 10 | 10 / 0 | 57.55 | 71.43 | -14.15 | 0.00195 | 0.0176 |
| time_to_goal | medium | blind vs aware | 10 | 10 | 10 / 0 | 57.55 | 115.41 | -55.74 | 0.00195 | 0.0176 |
| time_to_goal | medium | fields vs aware | 10 | 10 | 10 / 0 | 71.43 | 115.41 | -43.57 | 0.00195 | 0.0176 |
| time_to_goal | hard | blind vs fields | 10 | 10 | 10 / 0 | 58.18 | 89.84 | -30.09 | 0.00195 | 0.0176 |
| time_to_goal | hard | blind vs aware | 10 | 10 | 10 / 0 | 58.18 | 164.57 | -104.36 | 0.00195 | 0.0176 |
| time_to_goal | hard | fields vs aware | 10 | 10 | 10 / 0 | 89.84 | 164.57 | -67.76 | 0.00195 | 0.0176 |

Primary: per-scene means over the seeds, paired Wilcoxon across scenes, Holm within each metric over tiers and pairs.
