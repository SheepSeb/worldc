"""Re-simulate logged walk-around trials from scratch and compare their digests with the logs (labels and policies are what they say).

    python3 scripts/e3_ablation_resim_check.py                # 4 scenes x 2 tiers x 2 policies x 3 trials = 48 trials
"""
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import e3_one_scene as O  # noqa: E402
from worldc.e3 import sim as S  # noqa: E402
from worldc.e3.reactive import WalkAroundWalkers  # noqa: E402
from worldc.e3.walkers import workers  # noqa: E402
from worldc.e3.world import load_world  # noqa: E402

random.seed(1)
same = total = 0
for scene in ("P05", "P09", "P11", "P19"):
    rows = [json.loads(ln) for ln in gzip.open(f"results/e3_ablation_walkaround/scenes/{scene}/trials.jsonl.gz", "rt")]
    for tier, variant, n in (("medium", "base", 3), ("hard", "hard", 6)):
        w = load_world(Path("out/e3/grid") / scene / "scenes" / variant / "scene")
        legs = S.static_legs(w)
        for pol in ("blind", "fields"):
            for r in random.sample([x for x in rows if x["tier"] == tier and x["policy"] == pol], 3):
                H, info = workers(w, r["seed"], n, r["timeout_s"] + 1.0, S.DT)
                res = S.run_trial(w, H, pol, legs, r["timeout_s"], walkers=WalkAroundWalkers(H, w, info, S.DT))
                total += 1
                same += O.digest(res) == r["digest"] and res["collisions_robot_moving"] == r["collisions_robot_moving"]
print(f"re-simulated {total} logged walk-around trials (scenes P05 P09 P11 P19, medium and hard, blind and fields): {same} identical (digest and moving collisions), {total - same} different")
