# E3 ablation: reactive walkers (tag `e3-ablation-reactive-v1`)

The main grid (`results/e3_grid`) uses walkers that do not react to the robot. Here they do, in the simplest way:

- **Rule:** a walker stops (instantly, no deceleration) while the robot's footprint is within **0.5 m** of it **and ahead of it** (in
  the half-plane of its direction of travel over the next 0.5 s); it walks on **1 s** after that has been false. A walker that is
  dwelling at a task is not moving and is not stopped. **No sidestep** (optional, not implemented). Code: `src/worldc/e3/reactive.py`,
  enabled with `scripts/e3_grid.py --reactive`; rows carry `walkers_reactive`, `walker_stops`, `walker_held_s`.
- **Everything else is the main run:** the same 10 scenes and variants, tiers (1 / 3 / 6 workers), 30 seeds, the same drawn tours, the
  same three policies. Policies at `e3-policies-v1.1` (= v1 plus the optional hook). With the hook off, the new `sim.py` reproduces
  **all 2,700 trial digests of the main grid exactly**, so the main results stand under v1.1.
- **Run:** 2,700 trials, code commit `0851cac`, `policies_at_tag` true for every row.

## Result (`ablation_reactive.md`, `ablation_reactive.csv`)

Per tier x policy, main -> reactive (point estimates; CIs in the table):

| | Moving coll./trial (easy / medium / hard) | Stationary coll./trial | Frozen (%) | Success (%) |
|---|---|---|---|---|
| Blind | 0.93 -> 0.68 / 2.99 -> 2.14 / 6.04 -> 4.46 | 0.10 -> 0.03 / 0.30 -> 0.07 / 0.56 -> 0.14 | 0 -> 0 (all) | 44 -> 52 / 12 -> 18 / 2 -> 6 |
| Fields | 0.24 -> 0.00 / 0.80 -> 0.01 / 1.59 -> 0.02 | 0.57 -> 0.03 / 2.84 -> 0.07 / 8.55 -> 0.14 | 0 -> 58 / 0 -> 87 / 8 -> 95 | 78 -> 42 / 45 -> 12 / 17 -> 3 |
| Human-aware | 0.00 -> 0.00 (all) | 0.99 -> 0.03 / 4.15 -> 0.07 / 7.24 -> 0.14 | 5 -> 69 / 34 -> 96 / 66 -> 99 | 94 -> 31 / 63 -> 4 / 30 -> 1 |

- **Blind:** walkers that wait barely help. The walker stops at 0.5 m but the blind robot keeps driving at about 1 m/s into the standing
  person, so moving collisions fall by about a quarter; those from behind fall most (0.25 -> 0.04 in the easy tier), because a walker
  no longer steps into the robot's path from the front.
- **Fields and human-aware:** collisions of both kinds nearly vanish, **but the robots stop finishing**: 58-95 % (fields) and 69-99 %
  (human-aware) of the trials end *frozen*. The two sides yield to each other: the robot stops at 0.5 m (human-aware's speed scale is
  zero there, the protective field of the fields policy is longer than 0.5 m) and the walker waits as long as the robot is within 0.5 m
  ahead, so neither moves until the robot's 30 s no-progress rule ends the trial. This is a standoff produced by the rule as specified
  (symmetric yielding, no sidestep), not evidence that the cautious policies cope with reactive people; the lower collision counts
  are mostly the same two stopped parties. A sidestep or asymmetric yielding would be needed for a meaningful reactive comparison.
- Time to goal is a median over the few trials that finished and is not comparable with the main run for the cautious policies.

Contacts (`analysis/contact_speed.md`): the same checks as the main run pass (tag agrees with the logged v / w; per-trial counts agree;
all human-aware contacts at |v| = |w| = 0; 0 moving). Blind contacts are faster (mean 0.68-0.69 m/s) because the slow, braking cases
no longer occur.

## Files

`ablation_reactive.md` / `.csv` / `ablation_tests.json` (comparison, Wilcoxon over the 10 scenes per policy x tier, Holm), `trials.csv`
(2,700 rows), `analysis/` (Table 2, tests, scatter, supplementary and contact files for this run), `scenes/<scene>/` (variants report,
gzipped trial logs). Variants are those of the main grid.
Regenerate: `uv run python scripts/e3_grid.py --seeds 30 --jobs 6 --reactive --out out/e3/ablation_reactive --variants-from out/e3/grid`, then
`scripts/e3_ablation_report.py out/e3/grid out/e3/ablation_reactive --out results/e3_ablation_reactive` and
`scripts/e3_analysis.py out/e3/ablation_reactive/P*/trials.csv --logs out/e3/ablation_reactive/P*/trials.jsonl --out results/e3_ablation_reactive/analysis`.
