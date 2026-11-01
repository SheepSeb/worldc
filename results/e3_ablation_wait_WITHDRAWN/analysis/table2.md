| Tier | Policy | Moving coll./trial | of which from behind | of which already braking | Stationary coll./trial | Speed at contact mean / max (m/s) [n] | Success (%) | Frozen (%) | Min gap moving (m) | <0.5 m moving (s) | <1 m moving (s) | Stationary <1 m (s) | Time to goal (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy | Blind | 0.68 [0.47, 0.94] | 0.04 [0.00, 0.09] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.07] | 0.69 / 1.00 [214] | 52 [39, 65] | 0 [0, 0] | 0.04 [0.00, 0.31] | 3.8 [2.8, 4.9] | 6.5 [4.8, 8.8] | 0.9 [0.7, 1.2] | 42 [22, 66] |
|  | Fields (ISO 3691-4-style) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.07] | 0.00 / 0.00 [9] | 42 [28, 55] | 58 [45, 72] | 0.41 [0.41, 0.47] | 0.8 [0.6, 1.1] | 2.5 [1.9, 3.1] | 20.7 [16.2, 25.3] | 44 [15, 64] |
|  | Human-aware | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.07] | 0.00 / 0.00 [9] | 31 [19, 44] | 69 [56, 81] | 0.86 [0.71, 1.00] | 0.0 [0.0, 0.0] | 0.8 [0.3, 1.4] | 20.2 [16.5, 23.6] | 51 [20, 68] |
| medium | Blind | 2.14 [1.48, 3.08] | 0.08 [0.02, 0.16] | 0.00 [0.00, 0.00] | 0.07 [0.01, 0.17] | 0.68 / 1.00 [665] | 18 [8, 28] | 0 [0, 0] | 0.00 [0.00, 0.00] | 9.5 [7.0, 12.7] | 15.2 [11.0, 20.9] | 2.1 [1.7, 2.5] | 42 [22, 66] |
|  | Fields (ISO 3691-4-style) | 0.01 [0.00, 0.04] | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | 0.07 [0.01, 0.17] | 0.00 / 0.00 [25] | 12 [4, 22] | 87 [77, 95] | 0.41 [0.40, 0.41] | 1.2 [1.0, 1.5] | 3.2 [2.6, 4.0] | 30.8 [27.8, 33.5] | 53 [21, 66] |
|  | Human-aware | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.07 [0.01, 0.17] | 0.00 / 0.00 [22] | 4 [1, 9] | 96 [91, 99] | 0.69 [0.62, 0.79] | 0.0 [0.0, 0.0] | 0.9 [0.5, 1.4] | 28.2 [26.5, 29.6] | 49 [14, 98] |
| hard | Blind | 4.46 [2.98, 6.70] | 0.17 [0.06, 0.30] | 0.00 [0.00, 0.00] | 0.14 [0.01, 0.35] | 0.68 / 1.00 [1379] | 6 [2, 12] | 0 [0, 0] | 0.00 [0.00, 0.00] | 15.7 [11.1, 22.1] | 23.4 [15.8, 34.4] | 3.0 [2.3, 3.7] | 42 [22, 66] |
|  | Fields (ISO 3691-4-style) | 0.02 [0.00, 0.07] | 0.01 [0.00, 0.05] | 0.00 [0.00, 0.01] | 0.14 [0.01, 0.35] | 0.00 / 0.00 [47] | 3 [0, 7] | 95 [90, 99] | 0.41 [0.38, 0.41] | 1.4 [1.1, 1.8] | 3.5 [2.7, 4.2] | 33.3 [31.3, 35.2] | 44 [14, 70] |
|  | Human-aware | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.14 [0.02, 0.35] | 0.00 / 0.00 [42] | 1 [0, 2] | 99 [98, 100] | 0.80 [0.70, 0.92] | 0.0 [0.0, 0.0] | 0.6 [0.2, 1.0] | 26.6 [22.7, 29.2] | 81 [45, 116] |

10 scene(s) x 30 seeds per tier and policy. Point estimate with a 95 % cluster-bootstrap CI (scenes, then seeds). Means except min gap and time to goal (medians); time to goal over finished trials only. Moving + stationary = all collisions. Speed at contact: robot |v| at first contact, mean / max over every collision of the cell (stationary ones included, which pulls the mean down), n collisions in brackets, no CI; the split by moving / stationary is in contact_speed.md.

**Primary tests** (per-scene means, paired Wilcoxon across the scenes, Holm-corrected within each metric; A better = lower value):

| Tier | A vs B | Moving coll.: A / B (per-scene means) | A better / B better (scenes) | p (Holm) | Time to goal (s): A / B | A better / B better (scenes) | p (Holm) |
|---|---|---|---|---|---|---|---|
| easy | Blind vs Fields | 0.68 / 0.00 | 0 / 10 | 0.018 | 58 / 169 | 10 / 0 | 0.018 |
| easy | Blind vs Human-aware | 0.68 / 0.00 | 0 / 10 | 0.018 | 58 / 184 | 10 / 0 | 0.018 |
| easy | Fields vs Human-aware | 0.00 / 0.00 | 0 / 0 | 1.000 | 169 / 184 | 10 / 0 | 0.018 |
| medium | Blind vs Fields | 2.14 / 0.01 | 0 / 10 | 0.018 | 58 / 214 | 10 / 0 | 0.018 |
| medium | Blind vs Human-aware | 2.14 / 0.00 | 0 / 10 | 0.018 | 58 / 225 | 10 / 0 | 0.018 |
| medium | Fields vs Human-aware | 0.01 / 0.00 | 0 / 1 | 1.000 | 214 / 225 | 7 / 0 | 0.031 |
| hard | Blind vs Fields | 4.46 / 0.02 | 0 / 10 | 0.018 | 58 / 226 | 10 / 0 | 0.018 |
| hard | Blind vs Human-aware | 4.46 / 0.00 | 0 / 10 | 0.018 | 58 / 230 | 10 / 0 | 0.018 |
| hard | Fields vs Human-aware | 0.02 / 0.00 | 0 / 2 | 1.000 | 226 / 230 | 5 / 0 | 0.062 |
