| Tier | Policy | Moving coll./trial | of which from behind | of which already braking | Stationary coll./trial | Speed at contact mean / max (m/s) [n] | Success (%) | Frozen (%) | Min gap moving (m) | <0.5 m moving (s) | <1 m moving (s) | Stationary <1 m (s) | Time to goal (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy | Blind | 0.93 [0.61, 1.34] | 0.25 [0.13, 0.38] | 0.00 [0.00, 0.00] | 0.10 [0.05, 0.15] | 0.47 / 1.00 [309] | 44 [32, 56] | 0 [0, 0] | 0.00 [0.00, 0.04] | 3.1 [2.3, 4.2] | 5.2 [3.9, 6.7] | 0.7 [0.5, 0.9] | 42 [22, 66] |
|  | Fields (ISO 3691-4-style) | 0.24 [0.14, 0.35] | 0.18 [0.11, 0.26] | 0.05 [0.02, 0.10] | 0.57 [0.36, 0.85] | 0.08 / 1.00 [242] | 78 [69, 86] | 0 [0, 0] | 0.42 [0.18, 0.56] | 0.9 [0.6, 1.2] | 2.9 [2.1, 3.8] | 4.8 [3.5, 6.3] | 46 [28, 71] |
|  | Human-aware | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.99 [0.65, 1.37] | 0.00 / 0.00 [298] | 94 [86, 99] | 5 [1, 11] | 0.76 [0.63, 0.82] | 0.0 [0.0, 0.0] | 1.7 [0.9, 2.9] | 8.1 [5.4, 11.2] | 62 [42, 80] |
| medium | Blind | 2.99 [2.05, 4.22] | 0.76 [0.49, 1.08] | 0.00 [0.00, 0.00] | 0.30 [0.19, 0.42] | 0.49 / 1.00 [985] | 12 [4, 20] | 0 [0, 0] | 0.00 [0.00, 0.00] | 8.4 [6.0, 11.5] | 12.9 [9.3, 17.7] | 1.7 [1.3, 2.2] | 42 [22, 66] |
|  | Fields (ISO 3691-4-style) | 0.80 [0.55, 1.10] | 0.63 [0.43, 0.86] | 0.15 [0.07, 0.24] | 2.84 [2.01, 3.70] | 0.08 / 1.00 [1090] | 45 [32, 58] | 0 [0, 0] | 0.00 [0.00, 0.00] | 2.8 [2.0, 3.8] | 8.4 [6.3, 10.9] | 16.5 [11.8, 21.3] | 64 [44, 82] |
|  | Human-aware | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.15 [3.04, 5.30] | 0.00 / 0.00 [1244] | 63 [42, 81] | 34 [18, 52] | 0.60 [0.57, 0.67] | 0.0 [0.0, 0.0] | 4.0 [2.5, 5.9] | 28.1 [20.1, 36.4] | 101 [74, 140] |
| hard | Blind | 6.04 [4.29, 8.54] | 1.40 [1.00, 1.85] | 0.00 [0.00, 0.00] | 0.56 [0.36, 0.80] | 0.51 / 1.00 [1979] | 2 [0, 4] | 0 [0, 0] | 0.00 [0.00, 0.00] | 13.9 [9.7, 19.6] | 20.3 [14.0, 29.4] | 2.5 [1.9, 3.0] | 42 [22, 66] |
|  | Fields (ISO 3691-4-style) | 1.59 [1.18, 2.07] | 1.26 [0.93, 1.63] | 0.30 [0.17, 0.47] | 8.55 [6.20, 10.94] | 0.04 / 1.00 [3043] | 17 [10, 27] | 8 [3, 14] | 0.00 [0.00, 0.00] | 5.3 [3.9, 6.9] | 13.8 [10.2, 18.5] | 38.8 [28.7, 48.8] | 86 [65, 112] |
|  | Human-aware | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 7.24 [4.78, 10.01] | 0.00 / 0.00 [2173] | 30 [9, 50] | 66 [45, 87] | 0.61 [0.56, 0.76] | 0.0 [0.0, 0.0] | 5.1 [1.6, 9.8] | 36.1 [27.0, 46.7] | 144 [105, 422] |

10 scene(s) x 30 seeds per tier and policy. Point estimate with a 95 % cluster-bootstrap CI (scenes, then seeds). Means except min gap and time to goal (medians); time to goal over finished trials only. Moving + stationary = all collisions. Speed at contact: robot |v| at first contact, mean / max over every collision of the cell (stationary ones included, which pulls the mean down), n collisions in brackets, no CI; the split by moving / stationary is in contact_speed.md.

**Primary tests** (per-scene means, paired Wilcoxon across the scenes, Holm-corrected within each metric; A better = lower value):

| Tier | A vs B | Moving coll.: A / B (per-scene means) | A better / B better (scenes) | p (Holm) | Time to goal (s): A / B | A better / B better (scenes) | p (Holm) |
|---|---|---|---|---|---|---|---|
| easy | Blind vs Fields | 0.93 / 0.24 | 0 / 10 | 0.018 | 58 / 64 | 10 / 0 | 0.018 |
| easy | Blind vs Human-aware | 0.93 / 0.00 | 0 / 10 | 0.018 | 58 / 78 | 10 / 0 | 0.018 |
| easy | Fields vs Human-aware | 0.24 / 0.00 | 0 / 10 | 0.018 | 64 / 78 | 10 / 0 | 0.018 |
| medium | Blind vs Fields | 2.99 / 0.80 | 0 / 10 | 0.018 | 58 / 79 | 10 / 0 | 0.018 |
| medium | Blind vs Human-aware | 2.99 / 0.00 | 0 / 10 | 0.018 | 58 / 133 | 10 / 0 | 0.018 |
| medium | Fields vs Human-aware | 0.80 / 0.00 | 0 / 10 | 0.018 | 79 / 133 | 10 / 0 | 0.018 |
| hard | Blind vs Fields | 6.04 / 1.59 | 0 / 10 | 0.018 | 58 / 113 | 10 / 0 | 0.018 |
| hard | Blind vs Human-aware | 6.04 / 0.00 | 0 / 10 | 0.018 | 58 / 185 | 10 / 0 | 0.018 |
| hard | Fields vs Human-aware | 1.59 / 0.00 | 0 / 10 | 0.018 | 113 / 185 | 10 / 0 | 0.018 |
