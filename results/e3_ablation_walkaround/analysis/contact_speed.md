## Robot speed at first contact, per tier x policy

| Tier | Policy | Collisions | Moving | Stationary | |v| mean | median | p90 | max | |v| mean, moving | max, moving | max |w| (rad/s) | v < 0.05, w < 0.2 (stationary) | v < 0.05, turning (w > 0.2) | 0.05 < v <= 0.25 | 0.25 < v <= 0.5 | v > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy | Blind | 85 | 75 | 10 | 0.61 | 0.88 | 1.00 | 1.00 | 0.69 | 1.00 | 1.40 | 10 | 13 | 0 | 11 | 51 |
| easy | Fields (ISO 3691-4-style) | 20 | 3 | 17 | 0.01 | 0.00 | 0.00 | 0.17 | 0.06 | 0.17 | 1.40 | 17 | 2 | 1 | 0 | 0 |
| easy | Human-aware | 28 | 0 | 28 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 28 | 0 | 0 | 0 | 0 |
| medium | Blind | 265 | 239 | 26 | 0.59 | 0.83 | 1.00 | 1.00 | 0.66 | 1.00 | 1.40 | 26 | 44 | 1 | 41 | 153 |
| medium | Fields (ISO 3691-4-style) | 120 | 16 | 104 | 0.03 | 0.00 | 0.00 | 0.64 | 0.24 | 0.64 | 1.40 | 104 | 6 | 2 | 5 | 3 |
| medium | Human-aware | 89 | 0 | 89 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 89 | 0 | 0 | 0 | 0 |
| hard | Blind | 525 | 477 | 48 | 0.59 | 0.82 | 1.00 | 1.00 | 0.65 | 1.00 | 1.40 | 48 | 86 | 6 | 85 | 300 |
| hard | Fields (ISO 3691-4-style) | 364 | 45 | 319 | 0.03 | 0.00 | 0.00 | 0.67 | 0.20 | 0.67 | 1.40 | 319 | 18 | 10 | 11 | 6 |
| hard | Human-aware | 110 | 0 | 110 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 110 | 0 | 0 | 0 | 0 |

## Checks

- **Events:** 1606 collision events in 2700 logged trials.
- **Tag vs logged v / w:** 0 events where `robot_moving` disagrees with (|v| > 0.05 or |w| > 0.2) beyond the 0.01 rounding of the log; 0 within the rounding band of a threshold.
- **Per-trial counts vs trials.csv:** 0 trials differ (collisions, moving collisions); 0 logged trials missing in the csv.
- **Human-aware:** 227 collisions in total, 0 tagged moving; |v| at contact: max 0.00 m/s, 0 with any non-zero |v|, max |w| 0.00 rad/s.
