## Recipe v4 effort

### Rules per prompt (`prompts/spatial_rules_e2_within.json`; counts excluded)

| Prompt | Spatial rules | Named door walls |
|---|---|---|
| P05 | 5 | none |
| P08 | 6 | west |
| P15 | 5 | south |
| P16 | 4 | south |
| P23 | 6 | none |
| total | 26 | |

### Lines of code

- `src/worldc/recipes/remanufacturing.py`: **+316 / -13** lines against the v3 recipe (git diff HEAD); the file is 1016 lines now. The default path (`rules=None`) is unchanged: v3 specs are identical for 28/28 seeds checked.
- `prompts/spatial_rules_e2_within.json`: 334 non-blank lines (the rules themselves, 26 rules).
- `scripts/e2_recipe_within.py`: 42 non-blank lines in total; the v4 part is the `--rules` flag (about 8 lines).
- Scoring and analysis (`scripts/e2_within_v2.py`), shared by all methods, not recipe effort: 470 non-blank lines.

### Attempts per accepted sample (seeds base+k, retried at +1000 j, up to 11 attempts)

| Recipe | Prompt | Attempts per sample | Mean | Max | Failed attempts: rule unsatisfiable | Failed attempts: static build / solver |
|---|---|---|---|---|---|---|
| v4 (rules) | P05 | 2, 9, 1, 3, 3 | 3.6 | 9 | 11 | 2 |
| v4 (rules) | P08 | 8, 2, 2, 1, 3 | 3.2 | 8 | 10 | 1 |
| v4 (rules) | P15 | 4, 4, 3, 1, 5 | 3.4 | 5 | 10 | 2 |
| v4 (rules) | P16 | 2, 1, 1, 2, 1 | 1.4 | 2 | 2 | 0 |
| v4 (rules) | P23 | 5, 4, 3, 6, 1 | 3.8 | 6 | 9 | 5 |
| v4 (rules) | all | | 3.1 | 9 | | |
| v3 | P05 | 3, 2, 1, 1, 1 | 1.6 | 3 | 0 | 3 |
| v3 | P08 | 7, 4, 5, 2, 3 | 4.2 | 7 | 0 | 16 |
| v3 | P15 | 4, 5, 5, 1, 1 | 3.2 | 5 | 0 | 11 |
| v3 | P16 | 1, 2, 1, 1, 1 | 1.2 | 2 | 0 | 1 |
| v3 | P23 | 2, 2, 1, 6, 3 | 2.8 | 6 | 0 | 9 |
| v3 | all | | 2.6 | 7 | | |

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

