| Tier | Policy | Moving coll./trial | of which from behind | of which already braking | Stationary coll./trial | Speed at contact mean / max (m/s) [n] | Success (%) | Frozen (%) | Min gap moving (m) | <0.5 m moving (s) | <1 m moving (s) | Stationary <1 m (s) | Time to goal (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy | Blind | 0.25 [0.11, 0.43] | 0.02 [0.00, 0.04] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.07] | 0.61 / 1.00 [85] | 81 [69, 91] | 0 [0, 0] | 0.29 [0.19, 0.45] | 1.8 [1.2, 2.4] | 3.9 [2.7, 5.4] | 0.5 [0.3, 0.6] | 42 [22, 66] |
|  | Fields (ISO 3691-4-style) | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.02] | 0.00 [0.00, 0.02] | 0.06 [0.02, 0.10] | 0.01 / 0.17 [20] | 99 [97, 100] | 0 [0, 0] | 0.50 [0.38, 0.69] | 0.5 [0.3, 0.8] | 2.6 [1.8, 3.3] | 2.9 [2.0, 3.9] | 45 [26, 71] |
|  | Human-aware | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.09 [0.02, 0.19] | 0.00 / 0.00 [28] | 95 [88, 99] | 5 [1, 11] | 0.77 [0.69, 0.88] | 0.0 [0.0, 0.0] | 1.8 [0.8, 3.1] | 4.1 [2.6, 5.7] | 61 [36, 75] |
| medium | Blind | 0.80 [0.42, 1.23] | 0.04 [0.01, 0.09] | 0.00 [0.00, 0.00] | 0.09 [0.02, 0.19] | 0.59 / 1.00 [265] | 53 [32, 73] | 0 [0, 0] | 0.02 [0.00, 0.16] | 4.9 [3.2, 6.5] | 9.9 [7.0, 13.6] | 1.2 [0.9, 1.5] | 42 [22, 66] |
|  | Fields (ISO 3691-4-style) | 0.05 [0.02, 0.09] | 0.03 [0.01, 0.07] | 0.02 [0.00, 0.04] | 0.35 [0.11, 0.63] | 0.03 / 0.64 [120] | 95 [91, 98] | 0 [0, 0] | 0.31 [0.25, 0.35] | 1.7 [1.1, 2.3] | 7.4 [5.3, 9.6] | 9.5 [6.5, 12.7] | 56 [40, 76] |
|  | Human-aware | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.30 [0.07, 0.59] | 0.00 / 0.00 [89] | 72 [51, 89] | 24 [9, 41] | 0.60 [0.58, 0.68] | 0.0 [0.0, 0.0] | 4.9 [3.1, 7.2] | 19.0 [12.0, 26.6] | 92 [70, 114] |
| hard | Blind | 1.59 [0.78, 2.52] | 0.09 [0.02, 0.18] | 0.00 [0.00, 0.00] | 0.16 [0.03, 0.38] | 0.59 / 1.00 [525] | 40 [19, 62] | 0 [0, 0] | 0.00 [0.00, 0.08] | 8.1 [5.3, 11.2] | 15.7 [10.9, 22.5] | 1.9 [1.4, 2.4] | 42 [22, 66] |
|  | Fields (ISO 3691-4-style) | 0.15 [0.07, 0.26] | 0.09 [0.02, 0.19] | 0.06 [0.02, 0.11] | 1.06 [0.32, 1.92] | 0.03 / 0.67 [364] | 85 [75, 93] | 1 [0, 4] | 0.24 [0.15, 0.30] | 3.3 [2.0, 4.6] | 13.3 [9.6, 17.5] | 23.2 [15.0, 31.2] | 72 [52, 88] |
|  | Human-aware | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.37 [0.07, 0.78] | 0.00 / 0.00 [110] | 40 [16, 64] | 55 [30, 79] | 0.61 [0.57, 0.70] | 0.0 [0.0, 0.0] | 6.9 [2.8, 12.6] | 26.6 [18.5, 35.4] | 126 [105, 393] |

10 scene(s) x 30 seeds per tier and policy. Point estimate with a 95 % cluster-bootstrap CI (scenes, then seeds). Means except min gap and time to goal (medians); time to goal over finished trials only. Moving + stationary = all collisions. Speed at contact: robot |v| at first contact, mean / max over every collision of the cell (stationary ones included, which pulls the mean down), n collisions in brackets, no CI; the split by moving / stationary is in contact_speed.md.

**Primary tests** (per-scene means, paired Wilcoxon across the scenes, Holm-corrected within each metric; A better = lower value):

| Tier | A vs B | Moving coll.: A / B (per-scene means) | A better / B better (scenes) | p (Holm) | Time to goal (s): A / B | A better / B better (scenes) | p (Holm) |
|---|---|---|---|---|---|---|---|
| easy | Blind vs Fields | 0.25 / 0.01 | 0 / 8 | 0.039 | 58 / 62 | 10 / 0 | 0.018 |
| easy | Blind vs Human-aware | 0.25 / 0.00 | 0 / 8 | 0.039 | 58 / 75 | 10 / 0 | 0.018 |
| easy | Fields vs Human-aware | 0.01 / 0.00 | 0 / 3 | 0.250 | 62 / 75 | 10 / 0 | 0.018 |
| medium | Blind vs Fields | 0.80 / 0.05 | 0 / 10 | 0.018 | 58 / 71 | 10 / 0 | 0.018 |
| medium | Blind vs Human-aware | 0.80 / 0.00 | 0 / 10 | 0.018 | 58 / 115 | 10 / 0 | 0.018 |
| medium | Fields vs Human-aware | 0.05 / 0.00 | 0 / 7 | 0.039 | 71 / 115 | 10 / 0 | 0.018 |
| hard | Blind vs Fields | 1.59 / 0.15 | 0 / 10 | 0.018 | 58 / 90 | 10 / 0 | 0.018 |
| hard | Blind vs Human-aware | 1.59 / 0.00 | 0 / 10 | 0.018 | 58 / 165 | 10 / 0 | 0.018 |
| hard | Fields vs Human-aware | 0.15 / 0.00 | 0 / 8 | 0.039 | 90 / 165 | 10 / 0 | 0.018 |
