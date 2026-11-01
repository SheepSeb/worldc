**Table 1 (E2 rows), PRELIMINARY, n = 27 prompts (P01–P29, without P14, P26)**

| Method | Prompts | Acceptance | Adherence | Plausibility | Median IoU [95% CI] |
|---|---|---|---|---|---|
| LLM (E1) | 27 | 93% (27/29) | 0.89 | 0.87 | 0.06 [0.05, 0.10] |
| Recipe | 27 | 100% (27/27) | 1.00 | 0.96 | 0.05 [0.04, 0.07] |
| Template | 1 | 97% (29/30) | 1.00 | 1.00 | 1.00 [1.00, 1.00] |

PRELIMINARY, n = 27 prompts (P01–P29, without P14, P26). E1 = constrained prompts, accepted = static checks + final physics check. Recipe = the same prompt ids; template = all seeds of its one prompt. Baseline acceptance is the static `check_scene` (their Isaac physics check is queued). Adherence and plausibility: means over accepted scenes (`worldc eval`). IoU: pairs of accepted scenes within a method, normalised occupancy grids; lower = more diverse layouts.
