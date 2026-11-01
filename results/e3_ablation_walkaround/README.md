# E3 ablation: walk-around walkers (tag `e3-ablation-walkaround-v1`)

The main grid (`results/e3_grid`) uses walkers that do not react to the robot. Here a walker treats the robot as an obstacle and walks
around it with A*, **never waiting**:

- **Rule:** every 0.2 s a walker checks the next 2.5 s of its path against the robot's footprint inflated by 0.3 m (plus its own radius,
  i.e. a 0.3 m clearance to the footprint). If blocked, it plans an A* path on the people grid with that region masked to the first clear
  point of its tour (at most 30 s ahead), walks it at its own preferred speed and resumes the tour (replans at most every 0.4 s). A
  walker whose own spot is blocked leaves early (the rest of its dwell is skipped). If no detour exists it keeps its original path
  (no waiting), so contacts remain possible (`walker_no_path`, 1-15 per trial depending on tier and policy).
- **Knowledge:** exact robot pose, no delay, robot only (walkers do not avoid each other, as in the main run).
- **Everything else is the main run:** the same 10 scenes and variants, tiers (1 / 3 / 6 workers), 30 seeds, drawn tours and the same
  three policies, at `e3-policies-v1.1` (v1 plus the optional hook; hook off reproduces all 2,700 main-grid digests). 2,700 trials, code
  commit `a4d9bde`, `policies_at_tag` true in every row.
- **Checks:** walker motion stays continuous (at most 0.085 m per 0.05 s step, the cap of the fixed walkers); the contact checks pass
  as in the main run (`analysis/contact_speed.md`: tag agrees with logged v / w, per-trial counts agree, all 227 human-aware contacts at
  |v| = |w| = 0).
- **Verification:** `verification.md` recomputes both runs per policy x tier from the raw trial logs (`scripts/e3_ablation_verify.py`) and
  re-simulates 48 logged trials (`scripts/e3_ablation_resim_check.py`, all identical). Blind walk-around and fields main-run totals coincide in
  the medium and hard tiers (239 and 477 moving collisions) but the trials and per-scene counts differ: a coincidence of two independent runs.
- **Withdrawn earlier version:** walkers that wait deadlocked by construction and are not reported (`results/e3_ablation_wait_WITHDRAWN`).

## Result (`ablation_reactive.md` / `.csv`; main -> walk-around, easy / medium / hard)

| | Moving coll./trial | Stationary coll./trial | Success (%) | Frozen (%) |
|---|---|---|---|---|
| Blind | 0.93 -> 0.25 / 2.99 -> 0.80 / 6.04 -> 1.59 | 0.10 -> 0.03 / 0.30 -> 0.09 / 0.56 -> 0.16 | 44 -> 81 / 12 -> 53 / 2 -> 40 | 0 / 0 / 0 |
| Fields | 0.24 -> 0.01 / 0.80 -> 0.05 / 1.59 -> 0.15 | 0.57 -> 0.06 / 2.84 -> 0.35 / 8.55 -> 1.06 | 78 -> 99 / 45 -> 95 / 17 -> 85 | 0 -> 0 / 0 -> 0 / 8 -> 1 |
| Human-aware | 0.00 -> 0.00 (all) | 0.99 -> 0.09 / 4.15 -> 0.30 / 7.24 -> 0.37 | 94 -> 95 / 63 -> 72 / 30 -> 40 | 5 -> 5 / 34 -> 24 / 66 -> 55 |

- **Walkers that walk around the robot remove most contacts for every policy.** Moving collisions fall by about 70-90 % (blind) and to
  near zero (fields); stationary collisions (people walking into a stopped robot) fall by 70-95 % for fields and human-aware. The
  ordering of the policies on moving collisions is unchanged (blind > fields > human-aware = 0), and the per-scene Wilcoxon tests over the
  10 scenes give the same direction in every cell (Holm p = 0.018, the minimum with 10 scenes; `ablation_tests.json`).
- **Blind still collides** (1.59 per hard trial): the robot drives at about 1 m/s, a walker replans at most every 0.4 s, and in
  corridors the detour may not exist (no-path events, the walker then continues).
- **Fields** has the largest gain in success (17 -> 85 % in the hard tier) because the main run's failures were collisions.
- **Human-aware** stays slow and still freezes in 24 % (medium) and 55 % (hard) of the trials: its freezing is its own conservatism (it
  stops whenever someone is within 0.5 m, e.g. a person standing at a task spot next to its route), which walkers detouring at 0.3 m do
  not remove. Its moving collisions are 0 by construction in both runs (`docs/E3.md`).
- Time to goal is a median over finished trials; blind's is unchanged because it ignores people.

## Files

`ablation_reactive.md` / `.csv` / `ablation_tests.json` (comparison; the file name is historical), `trials.csv` (2,700 rows),
`analysis/` (Table 2, tests, scatter, supplementary and contact files for this run), `scenes/<scene>/` (variants report, gzipped trial
logs). Variants are those of the main grid.
Regenerate: `uv run python scripts/e3_grid.py --seeds 30 --jobs 6 --walkers walkaround --out out/e3/ablation_walkaround --variants-from out/e3/grid`,
then `scripts/e3_ablation_report.py out/e3/grid out/e3/ablation_walkaround --out results/e3_ablation_walkaround` and
`scripts/e3_analysis.py out/e3/ablation_walkaround/P*/trials.csv --logs out/e3/ablation_walkaround/P*/trials.jsonl --out results/e3_ablation_walkaround/analysis`.
