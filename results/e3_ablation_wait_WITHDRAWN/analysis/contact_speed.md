## Robot speed at first contact, per tier x policy

| Tier | Policy | Collisions | Moving | Stationary | |v| mean | median | p90 | max | |v| mean, moving | max, moving | max |w| (rad/s) | v < 0.05, w < 0.2 (stationary) | v < 0.05, turning (w > 0.2) | 0.05 < v <= 0.25 | 0.25 < v <= 0.5 | v > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy | Blind | 214 | 205 | 9 | 0.69 | 0.91 | 1.00 | 1.00 | 0.72 | 1.00 | 1.40 | 9 | 27 | 2 | 25 | 151 |
| easy | Fields (ISO 3691-4-style) | 9 | 0 | 9 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 9 | 0 | 0 | 0 | 0 |
| easy | Human-aware | 9 | 0 | 9 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 9 | 0 | 0 | 0 | 0 |
| medium | Blind | 665 | 643 | 22 | 0.68 | 0.91 | 1.00 | 1.00 | 0.71 | 1.00 | 1.40 | 22 | 74 | 11 | 113 | 445 |
| medium | Fields (ISO 3691-4-style) | 25 | 3 | 22 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.40 | 22 | 3 | 0 | 0 | 0 |
| medium | Human-aware | 22 | 0 | 22 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 22 | 0 | 0 | 0 | 0 |
| hard | Blind | 1379 | 1338 | 41 | 0.68 | 0.90 | 1.00 | 1.00 | 0.71 | 1.00 | 1.40 | 41 | 153 | 19 | 242 | 924 |
| hard | Fields (ISO 3691-4-style) | 47 | 6 | 41 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.40 | 41 | 6 | 0 | 0 | 0 |
| hard | Human-aware | 42 | 0 | 42 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 42 | 0 | 0 | 0 | 0 |

## Checks

- **Events:** 2412 collision events in 2700 logged trials.
- **Tag vs logged v / w:** 0 events where `robot_moving` disagrees with (|v| > 0.05 or |w| > 0.2) beyond the 0.01 rounding of the log; 0 within the rounding band of a threshold.
- **Per-trial counts vs trials.csv:** 0 trials differ (collisions, moving collisions); 0 logged trials missing in the csv.
- **Human-aware:** 73 collisions in total, 0 tagged moving; |v| at contact: max 0.00 m/s, 0 with any non-zero |v|, max |w| 0.00 rad/s.
