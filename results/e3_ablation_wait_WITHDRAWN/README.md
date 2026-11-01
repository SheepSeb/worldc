# WITHDRAWN: E3 ablation "walkers wait for the robot" (not to be reported)

This was the first version of the reactive-walker ablation: a walker stops while the robot is within 0.5 m ahead of it and walks on after
1 s clear. With the cautious policies it deadlocks **by construction**: the robot stops at 0.5 m (human-aware's speed scale is zero
there, the fields policy's protective field is longer than 0.5 m) and the walker waits as long as the robot is that close, so neither moves
and 58-99 % of the trials end *frozen* (`ablation_reactive.md`). That is an artefact of the rule, not a result about the policies, so it
is **not reported**.

Superseded by the walk-around variant (`results/e3_ablation_walkaround`, tag `e3-ablation-walkaround-v1`): walkers treat the robot as an
obstacle and replan around it, no waiting. The files here are kept only so that the earlier tag `e3-ablation-reactive-v1` (pushed before
the withdrawal) stays auditable; the code is `worldc.e3.reactive.ReactiveWalkers` (`scripts/e3_grid.py --walkers wait`).
