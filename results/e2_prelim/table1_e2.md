**Table 1 (E2 rows), PRELIMINARY, n = 14 prompts (P01–P15, without P14)**

| Method | Prompts | Acceptance | Adherence | Plausibility | Median IoU [95% CI] |
|---|---|---|---|---|---|
| LLM (E1) | 14 | 93% (14/15) | 0.89 | 0.86 | 0.06 [0.03, 0.11] |
| Recipe | 14 | 100% (14/14) | 1.00 | 0.95 | 0.05 [0.03, 0.08] |
| Template | 1 | 97% (29/30) | 1.00 | 1.00 | 1.00 [1.00, 1.00] |

PRELIMINARY, n = 14 prompts (P01–P15, without P14). E1 = constrained prompts, accepted = static checks + final physics check. Recipe = the same prompt ids; template = all seeds of its one prompt. Baseline acceptance is the static `check_scene` (their Isaac physics check is queued). Adherence and plausibility: means over accepted scenes (`worldc eval`). IoU: pairs of accepted scenes within a method, normalised occupancy grids; lower = more diverse layouts.
