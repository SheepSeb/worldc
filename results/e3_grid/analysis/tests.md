| metric | tier | A vs B | scenes | non-zero | A better / B better | mean A | mean B | median diff (A-B) | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| moving_collisions | easy | blind vs fields | 10 | 10 | 0 / 10 | 0.93 | 0.24 | 0.57 | 0.00195 | 0.0176 |
| moving_collisions | easy | blind vs aware | 10 | 10 | 0 / 10 | 0.93 | 0.00 | 0.82 | 0.00195 | 0.0176 |
| moving_collisions | easy | fields vs aware | 10 | 10 | 0 / 10 | 0.24 | 0.00 | 0.22 | 0.00195 | 0.0176 |
| moving_collisions | medium | blind vs fields | 10 | 10 | 0 / 10 | 2.99 | 0.80 | 1.73 | 0.00195 | 0.0176 |
| moving_collisions | medium | blind vs aware | 10 | 10 | 0 / 10 | 2.99 | 0.00 | 2.55 | 0.00195 | 0.0176 |
| moving_collisions | medium | fields vs aware | 10 | 10 | 0 / 10 | 0.80 | 0.00 | 0.70 | 0.00195 | 0.0176 |
| moving_collisions | hard | blind vs fields | 10 | 10 | 0 / 10 | 6.04 | 1.59 | 3.45 | 0.00195 | 0.0176 |
| moving_collisions | hard | blind vs aware | 10 | 10 | 0 / 10 | 6.04 | 0.00 | 5.27 | 0.00195 | 0.0176 |
| moving_collisions | hard | fields vs aware | 10 | 10 | 0 / 10 | 1.59 | 0.00 | 1.48 | 0.00195 | 0.0176 |
| time_to_goal | easy | blind vs fields | 10 | 10 | 10 / 0 | 57.55 | 63.95 | -5.58 | 0.00195 | 0.0176 |
| time_to_goal | easy | blind vs aware | 10 | 10 | 10 / 0 | 57.55 | 77.85 | -17.38 | 0.00195 | 0.0176 |
| time_to_goal | easy | fields vs aware | 10 | 10 | 10 / 0 | 63.95 | 77.85 | -11.69 | 0.00195 | 0.0176 |
| time_to_goal | medium | blind vs fields | 10 | 10 | 10 / 0 | 57.55 | 78.91 | -17.87 | 0.00195 | 0.0176 |
| time_to_goal | medium | blind vs aware | 10 | 10 | 10 / 0 | 57.55 | 133.02 | -73.56 | 0.00195 | 0.0176 |
| time_to_goal | medium | fields vs aware | 10 | 10 | 10 / 0 | 78.91 | 133.02 | -56.19 | 0.00195 | 0.0176 |
| time_to_goal | hard | blind vs fields | 10 | 10 | 10 / 0 | 58.18 | 113.08 | -49.02 | 0.00195 | 0.0176 |
| time_to_goal | hard | blind vs aware | 10 | 10 | 10 / 0 | 58.18 | 184.80 | -106.12 | 0.00195 | 0.0176 |
| time_to_goal | hard | fields vs aware | 10 | 10 | 10 / 0 | 113.08 | 184.80 | -50.01 | 0.00195 | 0.0176 |

Primary: per-scene means over the seeds, paired Wilcoxon across scenes, Holm within each metric over tiers and pairs.
