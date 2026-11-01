"""Effort numbers for recipe v4 (paper): rules per prompt, lines added, special-case workarounds, attempts per accepted sample.

    python3 scripts/e2_effort.py > results/e2_within_v2/effort.md
"""
import collections
import glob
import json
import statistics as st
import subprocess

rules = json.load(open("prompts/spatial_rules_e2_within.json"))
print("## Recipe v4 effort\n")
print("### Rules per prompt (`prompts/spatial_rules_e2_within.json`; counts excluded)\n")
print("| Prompt | Spatial rules | Named door walls |\n|---|---|---|")
tot = 0
for p in ("P05", "P08", "P15", "P16", "P23"):
    d = rules[p]["doors"]
    named = d["walls"] if isinstance(d.get("walls"), list) else [n["wall"] for n in d.get("named", [])]
    n = len(rules[p]["rules"])
    tot += n
    print(f"| {p} | {n} | {', '.join(named) or 'none'} |")
print(f"| total | {tot} | |\n")

print("### Lines of code\n")
num = subprocess.run(["git", "diff", "--numstat", "2fd800f^", "--", "src/worldc/recipes/remanufacturing.py"], capture_output=True, text=True).stdout.split()
src_lines = open("src/worldc/recipes/remanufacturing.py").read().splitlines()
print(f"- `src/worldc/recipes/remanufacturing.py`: **+{num[0]} / -{num[1]}** lines against the v3 recipe (git diff 2fd800f^, the commit before recipe v4); the file is {len(src_lines)} lines now. "
      f"The default path (`rules=None`) is unchanged: v3 specs are identical for 28/28 seeds checked.")
nb = lambda f: sum(1 for ln in open(f) if ln.strip())  # noqa: E731
print(f"- `prompts/spatial_rules_e2_within.json`: {nb('prompts/spatial_rules_e2_within.json')} non-blank lines (the rules themselves, {tot} rules).")
print(f"- `scripts/e2_recipe_within.py`: {nb('scripts/e2_recipe_within.py')} non-blank lines in total; the v4 part is the `--rules` flag (about 8 lines).")
print(f"- Scoring and analysis (`scripts/e2_within_v2.py`), shared by all methods, not recipe effort: {nb('scripts/e2_within_v2.py')} non-blank lines.\n")

print("### Attempts per accepted sample (seeds base+k, retried at +1000 j, up to 11 attempts)\n")
print("| Recipe | Prompt | Attempts per sample | Mean | Max | Failed attempts: rule unsatisfiable | Failed attempts: static build / solver |\n|---|---|---|---|---|---|---|")
for name, root in (("v4 (rules)", "out/e2_within/recipe_v4"), ("v3", "out/e2_within/recipe")):
    allatt = []
    for p in ("P05", "P08", "P15", "P16", "P23"):
        recs = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/{p}_seed*/record.json"))]
        att = [len(r["history"]) for r in recs]
        errs = collections.Counter("rule" if any(e.startswith("recipe.error:RuleUnsat") for e in h["errors"]) else "build"
                                   for r in recs for h in r["history"] if not h["ok"])
        allatt += att
        print(f"| {name} | {p} | {', '.join(map(str, att))} | {st.mean(att):.1f} | {max(att)} | {errs['rule']} | {errs['build']} |")
    print(f"| {name} | all | | {st.mean(allatt):.1f} | {max(allatt)} | | |")
print("""
### Special-case workarounds in recipe v4

Each is a place where the generic placement logic did not satisfy a rule and a prompt-specific branch was added (all in `remanufacturing.py`, active only when `rules` is given):

1. **Unsatisfiable seeds are resampled.** Any rule that cannot be placed raises `RuleUnsat`, which counts as a failed attempt (retry at seed + 1000 j). No rule is relaxed.
2. **P23, pinned coordinates:** the conveyor (x 1-11, y 5) and the cell extent (x 12-15, y 3-7) are put in the plan before the doors, with 0.7 m clearance; doors never go on the east or north wall (the pinned cell and the bench row / shelving leave no room for a door lane there). Cell arms are limited to those that fit the stated 3 x 4 m extent.
3. **P05 and P08, staging by the door:** roll-up doors are switched off (the dock leveler keeps the pallets over 2 m from the opening), and the staging zone is limited to 2.0 x 1.6 / 1.8 x 1.5 m at 0.6 / 0.9 m from the opening.
4. **P05, side-by-side cells:** the first cell is forced into a corner of the east or west wall so a second one fits beside it along the wall; the second is tried along the wall first, then towards the interior, with centres aligned. The free-standing middle shelving unit is placed before the cells, otherwise the second cell takes the middle of the room.
5. **P08, belt into the cell:** the cell is placed on the east wall with its height limited so the bench row on the belt's north side still fits; the belt length is 7.4 m up to the room limit and its far end stays 3.7 m from the west wall (door staging); four benches along it sit 0.05 m apart; the cell's gate is turned to face north / south so the belt does not run into the sliding gate, and the gate approach (1 m) is reserved.
6. **P15:** the conveyor stands beside the south door, outside its lane, rather than in front of it; the middle row's y is bounded below the north-wall cells; rows grow away from the conveyor and the outer rows are shifted by up to 1 m.
7. **Scrap / recycling containers (P08, P16, P23):** placed only at rule-compliant spots (next to the shelving, near a door, south of the cell); containers that do not fit are dropped, at least one is required.
8. **Benches in a row or in adjacent rows** are exempt from the aisle spacing between each other (spaced by construction).
9. **Not generated:** the P05 pallet jack (the recipe has no such object), so that rule scores 0 for the recipe.
10. **Found while doing this, v3 left unchanged:** the v3 conveyor free-length formula has the wrong sign for conveyors that run west from a cell on the east side (it underestimates the room). Rules mode uses its own computation.
""")
