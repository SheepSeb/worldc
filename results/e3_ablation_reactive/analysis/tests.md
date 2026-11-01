| metric | tier | A vs B | scenes | non-zero | A better / B better | mean A | mean B | median diff (A-B) | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| moving_collisions | easy | blind vs fields | 10 | 10 | 0 / 10 | 0.68 | 0.00 | 0.65 | 0.00195 | 0.0176 |
| moving_collisions | easy | blind vs aware | 10 | 10 | 0 / 10 | 0.68 | 0.00 | 0.65 | 0.00195 | 0.0176 |
| moving_collisions | easy | fields vs aware | 10 | 0 | 0 / 0 | 0.00 | 0.00 | 0.00 | 1 | 1 |
| moving_collisions | medium | blind vs fields | 10 | 10 | 0 / 10 | 2.14 | 0.01 | 1.90 | 0.00195 | 0.0176 |
| moving_collisions | medium | blind vs aware | 10 | 10 | 0 / 10 | 2.14 | 0.00 | 1.90 | 0.00195 | 0.0176 |
| moving_collisions | medium | fields vs aware | 10 | 1 | 0 / 1 | 0.01 | 0.00 | 0.00 | 1 | 1 |
| moving_collisions | hard | blind vs fields | 10 | 10 | 0 / 10 | 4.46 | 0.02 | 3.78 | 0.00195 | 0.0176 |
| moving_collisions | hard | blind vs aware | 10 | 10 | 0 / 10 | 4.46 | 0.00 | 3.80 | 0.00195 | 0.0176 |
| moving_collisions | hard | fields vs aware | 10 | 2 | 0 / 2 | 0.02 | 0.00 | 0.00 | 0.5 | 1 |
| time_to_goal | easy | blind vs fields | 10 | 10 | 10 / 0 | 57.55 | 168.53 | -76.63 | 0.00195 | 0.0176 |
| time_to_goal | easy | blind vs aware | 10 | 10 | 10 / 0 | 57.55 | 183.56 | -84.44 | 0.00195 | 0.0176 |
| time_to_goal | easy | fields vs aware | 10 | 10 | 10 / 0 | 168.53 | 183.56 | -13.78 | 0.00195 | 0.0176 |
| time_to_goal | medium | blind vs fields | 10 | 10 | 10 / 0 | 57.55 | 213.56 | -117.01 | 0.00195 | 0.0176 |
| time_to_goal | medium | blind vs aware | 10 | 10 | 10 / 0 | 57.55 | 224.92 | -131.47 | 0.00195 | 0.0176 |
| time_to_goal | medium | fields vs aware | 10 | 7 | 7 / 0 | 213.56 | 224.92 | -4.11 | 0.0156 | 0.0312 |
| time_to_goal | hard | blind vs fields | 10 | 10 | 10 / 0 | 58.18 | 226.12 | -131.47 | 0.00195 | 0.0176 |
| time_to_goal | hard | blind vs aware | 10 | 10 | 10 / 0 | 58.18 | 229.56 | -132.83 | 0.00195 | 0.0176 |
| time_to_goal | hard | fields vs aware | 10 | 5 | 5 / 0 | 226.12 | 229.56 | -1.08 | 0.0625 | 0.0625 |

Primary: per-scene means over the seeds, paired Wilcoxon across scenes, Holm within each metric over tiers and pairs.
