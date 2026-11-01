# E1 physics failures: failing objects

16 runs ended `physics_failed` (static check passed, final Isaac Sim physics check failed). Objects are spec object ids with their asset.

| Model | Prompt | Run | Codes | Failing objects (asset; codes; largest displacement) | Props only? |
|---|---|---|---|---|---|
| claude-sonnet-5-5 | P06 | `20261030T082729` | not_settled | bench_laptops_b (laptop; not_settled; 0.114 m); laptops (laptop; not_settled; 0.118 m) | yes |
| claude-sonnet-5-5 | P08 | `20261030T084048` | fell_through_floor, not_settled, unstable | laptops (laptop; fell_through_floor,not_settled,unstable; 0.83 m) | yes |
| claude-sonnet-5-5 | P12 | `20261030T092006` | not_settled | returned_cases (transfer_case; not_settled; 0.104 m) | yes |
| claude-sonnet-5-5 | P13 | `20261030T093312` | not_settled | gearboxes (transfer_case; not_settled; 0.115 m) | yes |
| claude-sonnet-5-5 | P30 | `20261030T130236` | fell_through_floor, unstable | transfer_cases (transfer_case; fell_through_floor,unstable) | yes |
| deepseek-v4.1-flash_cloud | P14 | `20261029T170158` | fell_through_floor, not_settled, unstable | bench_gearboxes_e (transfer_case; fell_through_floor,not_settled,unstable; 1.265 m); bench_gearboxes_w (transfer_case; not_settled; 0.098 m) | yes |
| deepseek-v4.1-flash_cloud | P26 | `20261029T205751` | fell_through_floor, not_settled, unstable | bench_laptop (laptop; fell_through_floor,not_settled,unstable; 0.778 m) | yes |
| deepseek-v4.1-flash_cloud | P30 | `20261029T223225` | fell_through_floor, unstable | bench_gearboxes (transfer_case; fell_through_floor,unstable) | yes |
| glm-5.3-flash_cloud | P06 | `20261030T002039` | not_settled | laptops (laptop; not_settled; 0.134 m) | yes |
| glm-5.3-flash_cloud | P08 | `20261030T004038` | not_settled | bench_laptops (isaac_sm_laptop; not_settled; 0.164 m) | yes |
| glm-5.3-flash_cloud | P10 | `20261030T012711` | not_settled | laptops (isaac_sm_laptop; not_settled; 0.172 m) | yes |
| glm-5.3-flash_cloud | P11 | `20261030T014032` | not_settled | bench_gearboxes (transfer_case; not_settled; 0.099 m); returned_gearboxes (transfer_case; not_settled; 0.097 m) | yes |
| glm-5.3-flash_cloud | P12 | `20261030T015353` | not_settled | incoming_cases (simready_drivetrain_transfer_case_b01; not_settled; 0.098 m) | yes |
| glm-5.3-flash_cloud | P13 | `20261030T020822` | not_settled | bench_cases (transfer_case; not_settled; 0.098 m) | yes |
| glm-5.3-flash_cloud | P25 | `20261030T044045` | not_settled | klt_on_belt (klt_bin; not_settled; 2.395 m) | no |
| glm-5.3-flash_cloud | P28 | `20261030T050737` | fell_through_floor, not_settled, unstable | transfer_cases (transfer_case; fell_through_floor,not_settled,unstable; 1.326 m) | yes |

Runs with a failing laptop: 6; with a failing transfer case: 9; with another object: 1 (of 16).
Runs whose failing objects are all laptops / transfer cases: 15 of 16.

Failing-object families per run: transfer case: 9, laptop: 6, other: 1.
