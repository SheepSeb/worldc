# E3 scene selection (10 of 10)

From dataset/e1/claude-sonnet-5-5, dataset/e1/deepseek-v4.1-flash_cloud, dataset/e1/glm-5.3-flash_cloud: 70 usable candidates, 20 rejected.

| # | prompt | cell type | size | W x D (m) | difficulty | people | source model | run |
|---|---|---|---|---|---|---|---|---|
| 1 | P01 | ev_battery_dismantling | S | 8 x 6 | easy | 1 | deepseek-v4.1-flash_cloud | `20261029T134119_7e18efd9` |
| 2 | P09 | electronics_weee | M | 13 x 9 | hard | 5 | glm-5.3-flash_cloud | `20261030T010746_ab2f491f` |
| 3 | P19 | appliance_repair | L | 22 x 12 | medium | 4 | claude-sonnet-5-5 | `20261030T103444_99cd458c` |
| 4 | P11 | engine_gearbox_reman | S | 10 x 7 | easy | 2 | deepseek-v4.1-flash_cloud | `20261029T162125_f8a7aee5` |
| 5 | P23 | sorting_grading | M | 16 x 10 | medium | 3 | glm-5.3-flash_cloud | `20261030T040557_ce8ff817` |
| 6 | P28 | returned_parts_inspection | L | 20 x 12 | medium | 4 | claude-sonnet-5-5 | `20261030T124620_e70bc8c5` |
| 7 | P05 | ev_battery_dismantling | M | 14 x 9 | hard | 5 | glm-5.3-flash_cloud | `20261030T000824_3f5c9b00` |
| 8 | P06 | electronics_weee | S | 9 x 6 | easy | 1 | deepseek-v4.1-flash_cloud | `20261029T150501_c3a558df` |
| 9 | P15 | engine_gearbox_reman | L | 26 x 14 | hard | 6 | claude-sonnet-5-5 | `20261030T095745_91fdb0bb` |
| 10 | P16 | appliance_repair | S | 9 x 6 | easy | 1 | glm-5.3-flash_cloud | `20261030T030001_1b71d579` |

Families: {'ev_battery_dismantling': 2, 'electronics_weee': 2, 'appliance_repair': 2, 'gearbox_reman': 2, 'sorting_grading': 1, 'returned_parts_inspection': 1}. Sizes: {'S': 4, 'M': 3, 'L': 3}. Models: {'deepseek-v4.1-flash_cloud': 3, 'glm-5.3-flash_cloud': 4, 'claude-sonnet-5-5': 3}.
**Missing:** families none, sizes none.

Rejected:

- P01 `20261030T075029_7e18efd9`: 1 robot goal(s)
- P06 `20261030T082729_c3a558df`: not accepted (physics_failed, physics failed)
- P08 `20261030T084048_3dc2eed0`: not accepted (physics_failed, physics failed)
- P12 `20261030T092006_75f5e005`: not accepted (physics_failed, physics failed)
- P13 `20261030T093312_5ba1bc22`: not accepted (physics_failed, physics failed)
- P24 `20261030T122519_25cfe6ee`: not accepted (no_scene, physics None)
- P30 `20261030T130236_2a08d72b`: not accepted (physics_failed, physics failed)
- P14 `20261029T170158_ed438709`: not accepted (physics_failed, physics failed)
- P26 `20261029T205751_e8922325`: not accepted (physics_failed, physics failed)
- P30 `20261029T223225_2a08d72b`: not accepted (physics_failed, physics failed)
- P01 `20261029T231422_7e18efd9`: not accepted (no_scene, physics None)
- P06 `20261030T002039_c3a558df`: not accepted (physics_failed, physics failed)
- P08 `20261030T004038_3dc2eed0`: not accepted (physics_failed, physics failed)
- P10 `20261030T012711_a68c6509`: not accepted (physics_failed, physics failed)
- P11 `20261030T014032_f8a7aee5`: not accepted (physics_failed, physics failed)
- P12 `20261030T015353_75f5e005`: not accepted (physics_failed, physics failed)
- P13 `20261030T020822_5ba1bc22`: not accepted (physics_failed, physics failed)
- P14 `20261030T022233_ed438709`: not accepted (no_scene, physics None)
- P25 `20261030T044045_1eadf8f3`: not accepted (physics_failed, physics failed)
- P28 `20261030T050737_e70bc8c5`: not accepted (physics_failed, physics failed)
