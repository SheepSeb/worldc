**E1, constrained prompts, 4 models**

| Model | Acceptance [95% CI] | Static pass | Accept. @3 | Accept. @5 | First try | No scene | Physics failed | Median turns | Median repairs | Median min | Adherence [95% CI] | Plausibility [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet-5-5 | 80% (24/30) [63, 90] | 29/30 | 15/30 | 21/30 | 0 | 1 | 5 | 20 | 3 | 8.1 | 0.93 [0.88, 0.97] | 0.89 [0.83, 0.94] |
| deepseek-v4.1-flash_cloud | 90% (27/30) [74, 97] | 30/30 | 13/30 | 25/30 | 0 | 0 | 3 | 47 | 4 | 17.4 | 0.89 [0.84, 0.94] | 0.87 [0.80, 0.93] |
| gemma4_31b-cloud | 23% (7/30) [12, 41] | 7/30 | 1/30 | 2/30 | 0 | 23 | 0 | 14 | 8 | 1.5 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| glm-5.3-flash_cloud | 67% (20/30) [49, 81] | 28/30 | 5/30 | 11/30 | 0 | 2 | 8 | 38 | 6 | 11.5 | 0.96 [0.92, 0.98] | 0.92 [0.86, 0.97] |

E1, constrained prompts, 4 models. Accepted = static `check_scene` + final physics check. Static pass = the loop ended with a statically valid scene (accepted + physics failed). Accept. @k = accepted with at most k repairs (curves in acceptance_at_k.md). Acceptance CI: Wilson. First try: accepted with 0 repairs. Turns, repairs (capped at 10) and minutes (loop + final checks): medians over all runs. Adherence and plausibility (`worldc eval`): means over accepted scenes, 95% bootstrap CI.

| Model | easy | medium | hard | Repair cap hit | Est. cost (USD) |
|---|---|---|---|---|---|
| claude-sonnet-5-5 | 9/10 | 7/10 | 8/10 | 2 | 14.76 |
| deepseek-v4.1-flash_cloud | 9/10 | 10/10 | 8/10 | 0 | 70.15 |
| gemma4_31b-cloud | 4/10 | 3/10 | 0/10 | 12 | 7.37 |
| glm-5.3-flash_cloud | 7/10 | 6/10 | 7/10 | 4 | 61.24 |

Est. cost: Claude Code's estimate at Anthropic pricing, real for Claude models only.
