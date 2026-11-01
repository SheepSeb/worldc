# Recipe physics failures: causes

Every scene that passed the static check was run through the Isaac Sim physics check (6 s settle, the E1 rule). A scene fails on any error issue.

## recipe v4 (rules)

| Prompt | Checked | Failed | Failure code | Scenes with it | Objects (bodies affected, summed over scenes) |
|---|---|---|---|---|---|
| P05 | 5 | 0 | - |  |  |
| P08 | 18 | 13 | not_settled | 13 | bench_laptop (24, up to 0.12 m) |
| P15 | 20 | 17 | fell_through_floor | 10 | bench_transfer_case (14) |
|  |  |  | not_settled | 16 | bench_transfer_case (84, up to 2.85 m) |
|  |  |  | unstable | 10 | bench_transfer_case (14) |
| P16 | 5 | 0 | - |  |  |
| P23 | 7 | 2 | not_settled | 2 | bench_laptop (2, up to 0.12 m) |

Failed scenes per (code, object): not_settled: bench_transfer_case (16), not_settled: bench_laptop (15), fell_through_floor: bench_transfer_case (10), unstable: bench_transfer_case (10)

## recipe v3

| Prompt | Checked | Failed | Failure code | Scenes with it | Objects (bodies affected, summed over scenes) |
|---|---|---|---|---|---|
| P05 | 5 | 0 | - |  |  |
| P08 | 20 | 16 | fell_through_floor | 1 | bench_laptop (1) |
|  |  |  | not_settled | 16 | bench_laptop (36, up to 1.07 m) |
|  |  |  | unstable | 1 | bench_laptop (1) |
| P15 | 18 | 13 | fell_through_floor | 7 | bench_transfer_case (7) |
|  |  |  | not_settled | 13 | bench_transfer_case (58, up to 1.35 m) |
|  |  |  | unstable | 7 | bench_transfer_case (7) |
| P16 | 5 | 0 | - |  |  |
| P23 | 5 | 0 | - |  |  |

Failed scenes per (code, object): not_settled: bench_laptop (16), not_settled: bench_transfer_case (13), fell_through_floor: bench_transfer_case (7), unstable: bench_transfer_case (7), fell_through_floor: bench_laptop (1), unstable: bench_laptop (1)

## recipe v3, 30 prompts

| Prompt | Checked | Failed | Failure code | Scenes with it | Objects (bodies affected, summed over scenes) |
|---|---|---|---|---|---|
| P01 | 1 | 0 | - |  |  |
| P02 | 1 | 0 | - |  |  |
| P03 | 1 | 0 | - |  |  |
| P04 | 1 | 0 | - |  |  |
| P05 | 1 | 0 | - |  |  |
| P06 | 1 | 0 | - |  |  |
| P07 | 1 | 1 | not_settled | 1 | bench_laptop (3, up to 0.11 m) |
| P08 | 1 | 1 | not_settled | 1 | bench_laptop (1, up to 0.11 m) |
| P09 | 1 | 1 | not_settled | 1 | bench_laptop (7, up to 0.12 m) |
| P10 | 1 | 1 | not_settled | 1 | bench_laptop (6, up to 0.12 m) |
| P11 | 1 | 0 | - |  |  |
| P12 | 1 | 0 | - |  |  |
| P13 | 1 | 0 | - |  |  |
| P14 | 1 | 0 | - |  |  |
| P15 | 1 | 1 | not_settled | 1 | bench_transfer_case (5, up to 0.10 m) |
| P16 | 1 | 0 | - |  |  |
| P17 | 1 | 0 | - |  |  |
| P18 | 1 | 0 | - |  |  |
| P19 | 1 | 0 | - |  |  |
| P20 | 1 | 0 | - |  |  |
| P21 | 1 | 1 | not_settled | 1 | bench_laptop (1, up to 0.11 m) |
| P22 | 1 | 0 | - |  |  |
| P23 | 1 | 0 | - |  |  |
| P24 | 1 | 0 | - |  |  |
| P25 | 1 | 0 | - |  |  |
| P26 | 1 | 1 | not_settled | 1 | returned_units (1, up to 0.09 m) |
| P27 | 1 | 1 | not_settled | 1 | returned_units (2, up to 0.31 m), bench_laptop (1, up to 0.12 m) |
| P28 | 1 | 1 | not_settled | 1 | returned_units (2, up to 0.50 m) |
| P29 | 1 | 1 | not_settled | 1 | returned_units (2, up to 0.12 m), bench_laptop (1, up to 0.11 m) |
| P30 | 1 | 1 | not_settled | 1 | bench_laptop (1, up to 0.12 m), returned_units (1, up to 0.10 m) |

Failed scenes per (code, object): not_settled: bench_laptop (8), not_settled: returned_units (5), not_settled: bench_transfer_case (1)

