## Robot speed at first contact, per tier x policy

| Tier | Policy | Collisions | Moving | Stationary | |v| mean | median | p90 | max | |v| mean, moving | max, moving | max |w| (rad/s) | v < 0.05, w < 0.2 (stationary) | v < 0.05, turning (w > 0.2) | 0.05 < v <= 0.25 | 0.25 < v <= 0.5 | v > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy | Blind | 309 | 279 | 30 | 0.47 | 0.41 | 1.00 | 1.00 | 0.52 | 1.00 | 1.40 | 30 | 78 | 13 | 49 | 139 |
| easy | Fields (ISO 3691-4-style) | 242 | 71 | 171 | 0.08 | 0.00 | 0.33 | 1.00 | 0.27 | 1.00 | 1.40 | 171 | 34 | 6 | 12 | 19 |
| easy | Human-aware | 298 | 0 | 298 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 298 | 0 | 0 | 0 | 0 |
| medium | Blind | 985 | 896 | 89 | 0.49 | 0.42 | 1.00 | 1.00 | 0.54 | 1.00 | 1.40 | 89 | 227 | 32 | 183 | 454 |
| medium | Fields (ISO 3691-4-style) | 1090 | 239 | 851 | 0.08 | 0.00 | 0.31 | 1.00 | 0.35 | 1.00 | 1.40 | 853 | 84 | 33 | 48 | 72 |
| medium | Human-aware | 1244 | 0 | 1244 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 1244 | 0 | 0 | 0 | 0 |
| hard | Blind | 1979 | 1812 | 167 | 0.51 | 0.45 | 1.00 | 1.00 | 0.55 | 1.00 | 1.40 | 167 | 429 | 74 | 365 | 944 |
| hard | Fields (ISO 3691-4-style) | 3043 | 477 | 2566 | 0.04 | 0.00 | 0.04 | 1.00 | 0.28 | 1.00 | 1.40 | 2567 | 174 | 94 | 110 | 98 |
| hard | Human-aware | 2173 | 0 | 2173 | 0.00 | 0.00 | 0.00 | 0.00 | – | – | 0.00 | 2173 | 0 | 0 | 0 | 0 |

## Checks

- **Events:** 11363 collision events in 2700 logged trials.
- **Tag vs logged v / w:** 0 events where `robot_moving` disagrees with (|v| > 0.05 or |w| > 0.2) beyond the 0.01 rounding of the log; 3 within the rounding band of a threshold.
- **Per-trial counts vs trials.csv:** 0 trials differ (collisions, moving collisions); 0 logged trials missing in the csv.
- **Human-aware:** 3715 collisions in total, 0 tagged moving; |v| at contact: max 0.00 m/s, 0 with any non-zero |v|, max |w| 0.00 rad/s.
  - Tag disagrees within rounding: P01 hard fields seed 28 t 7.85: v 0.05, w 0.0, tagged moving True
  - Tag disagrees within rounding: P16 medium fields seed 23 t 67.65: v 0.05, w 0.0, tagged moving True
  - Tag disagrees within rounding: P16 medium fields seed 28 t 39.45: v 0.05, w 0.0, tagged moving True
