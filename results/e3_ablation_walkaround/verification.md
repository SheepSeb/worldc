# Verification of the walk-around comparison from the raw trial logs

- main run: `results/e3_grid`, 2700 trials, (walkers, code commit, policies) = {('fixed', 'aa7e056e8c3b', 'e3-policies-v1'): 2700}
- ablation: `results/e3_ablation_walkaround`, 2700 trials, (walkers, code commit, policies) = {('walkaround', 'a4d9bdede851', 'e3-policies-v1.1'): 2700}

## Both runs per policy x tier (300 trials per cell; counts from the collision events)

| Tier | Policy | Main: moving / trial | Main: stationary / trial | Main: moving events | Walk-around: moving / trial | Walk-around: stationary / trial | Walk-around: moving events |
|---|---|---|---|---|---|---|---|
| easy | blind | 0.930 | 0.100 | 279 | 0.250 | 0.033 | 75 |
| easy | fields | 0.237 | 0.570 | 71 | 0.010 | 0.057 | 3 |
| easy | aware | 0.000 | 0.993 | 0 | 0.000 | 0.093 | 0 |
| medium | blind | 2.987 | 0.297 | 896 | 0.797 | 0.087 | 239 |
| medium | fields | 0.797 | 2.837 | 239 | 0.053 | 0.347 | 16 |
| medium | aware | 0.000 | 4.147 | 0 | 0.000 | 0.297 | 0 |
| hard | blind | 6.040 | 0.557 | 1812 | 1.590 | 0.160 | 477 |
| hard | fields | 1.590 | 8.553 | 477 | 0.150 | 1.063 | 45 |
| hard | aware | 0.000 | 7.243 | 0 | 0.000 | 0.367 | 0 |

## Is walk-around blind the same as main fields? (moving collisions)

The totals coincide in the medium and hard tiers (239 and 477 events) but the trials do not:

| Tier | Events: main fields | Events: walk-around blind | Trials with equal count | Correlation of per-trial counts | Identical trial digests |
|---|---|---|---|---|---|
| easy | 71 | 75 | 213 / 300 | 0.17 | 56 |
| medium | 239 | 239 | 140 / 300 | 0.32 | 7 |
| hard | 477 | 477 | 73 / 300 | 0.17 | 0 |

Moving-collision events per scene, hard tier:

| Scene | P01 | P05 | P06 | P09 | P11 | P15 | P16 | P19 | P23 | P28 |
|---|---|---|---|---|---|---|---|---|---|---|
| main fields | 29 | 80 | 39 | 50 | 61 | 81 | 55 | 19 | 35 | 28 |
| walk-around blind | 99 | 60 | 43 | 4 | 46 | 128 | 75 | 4 | 8 | 10 |
| main blind | 141 | 175 | 245 | 135 | 189 | 453 | 189 | 77 | 107 | 101 |

## Re-simulation

Independent of the report: `scripts/e3_ablation_resim_check.py` rebuilds logged walk-around trials from their seeds and compares digests:

- re-simulated 48 logged walk-around trials (scenes P05 P09 P11 P19, medium and hard, blind and fields): 48 identical (digest and moving collisions), 0 different

So the policy labels, the walker model and the counts in the comparison report are what the logs contain; the equal totals in the medium and hard tiers are a coincidence of two independent runs (per-scene sums, per-trial counts and digests all differ).
