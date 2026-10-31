**E2 within-prompt study: layout diversity, spatial adherence and acceptance** (prompts P05, P08, P15, P16, P23)

| Method | Samples | Within-prompt IoU [95% CI] | Spatial adherence [95% CI] | LLM acceptance per run | Recipe acceptance per attempt | Accepted samples per final build |
|---|---|---|---|---|---|---|
| Sonnet 5.5 | 25 | 0.27 [0.15, 0.52] | 0.85 [0.76, 0.94] | 25/55 (45%) | n/a | n/a |
| DeepSeek v4.1 flash | 25 | 0.20 [0.06, 0.35] | 0.80 [0.71, 0.90] | 25/40 (62%) | n/a | n/a |
| GLM 5.3 flash | 25 | 0.20 [0.09, 0.38] | 0.86 [0.81, 0.93] | 25/36 (69%) | n/a | n/a |
| Recipe v4 (rules) | 25 | 0.22 [0.12, 0.58] | 0.96 [0.88, 1.00] | n/a | 25/77 (32%) | 25/25 |
| Recipe v3 (no rules) | 25 | 0.05 [0.03, 0.07] | 0.19 [0.09, 0.31] | n/a | 25/65 (38%) | 25/25 |
| Template | 5 | 1.00 [1.00, 1.00] | 0.88 [0.88, 0.88] | n/a | 29/30 (97%) | n/a |

IoU: median over pairs of same-prompt samples on native-metre 0.1 m grids (lower = more diverse). Spatial adherence: mean over scenes of the mean rule score (1 = every instance of the rule satisfied; a rule whose objects are absent from the scene scores 0), `prompts/spatial_rules_e2_within.json`. The template is scored on P08 only (its prompt), so its spatial CI is degenerate. LLM acceptance: accepted constrained runs of these five prompts (E1 run + extra samples), per run. Recipe acceptance: per build attempt, counting every resampled seed; the last column is the number of the 25 samples that ended accepted. The template builds once per seed.
