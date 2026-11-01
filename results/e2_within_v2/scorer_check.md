# E2 within v2: scorer check

## 1. Template, P08, seeds 00-04

Layouts are identical across seeds (the template varies props only), so every seed gives the same result.

| Seed | Rule | Result | Failing instance | Measured | Threshold | Verdict |
|---|---|---|---|---|---|---|
| 00 | pallets.by_west_door | 1/2 | pallet x 2.09-3.31, y 0.24-1.06 | gap to the west door opening 2.09 m (opening y 1.0-2.2 on x = 0) | <= 2.0 m | **real, over by 0.09 m** |
| 00 | conveyor.axis_east | 1/1 | | | | pass |
| 00 | benches.conveyor_north | 3/4 | bench x 1.69-3.51 (centre 2.60), y 5.14-5.86 | centre x 2.60 vs belt x 3.50-10.50 | centre inside the belt's x span | **real: centre 0.90 m west of the belt start** (its east edge touches it) |
| 00 | cell.conveyor_east_end | 1/1 | | | | pass |
| 00 | shelving.east_wall | 1/1 | | | | pass |
| 00 | recycling.next_to_shelving | 2/2 | | | | pass |
| 01 | pallets.by_west_door | 1/2 | pallet x 2.09-3.31, y 0.24-1.06 | gap to the west door opening 2.09 m (opening y 1.0-2.2 on x = 0) | <= 2.0 m | **real, over by 0.09 m** |
| 01 | conveyor.axis_east | 1/1 | | | | pass |
| 01 | benches.conveyor_north | 3/4 | bench x 1.69-3.51 (centre 2.60), y 5.14-5.86 | centre x 2.60 vs belt x 3.50-10.50 | centre inside the belt's x span | **real: centre 0.90 m west of the belt start** (its east edge touches it) |
| 01 | cell.conveyor_east_end | 1/1 | | | | pass |
| 01 | shelving.east_wall | 1/1 | | | | pass |
| 01 | recycling.next_to_shelving | 2/2 | | | | pass |
| 02 | pallets.by_west_door | 1/2 | pallet x 2.09-3.31, y 0.24-1.06 | gap to the west door opening 2.09 m (opening y 1.0-2.2 on x = 0) | <= 2.0 m | **real, over by 0.09 m** |
| 02 | conveyor.axis_east | 1/1 | | | | pass |
| 02 | benches.conveyor_north | 3/4 | bench x 1.69-3.51 (centre 2.60), y 5.14-5.86 | centre x 2.60 vs belt x 3.50-10.50 | centre inside the belt's x span | **real: centre 0.90 m west of the belt start** (its east edge touches it) |
| 02 | cell.conveyor_east_end | 1/1 | | | | pass |
| 02 | shelving.east_wall | 1/1 | | | | pass |
| 02 | recycling.next_to_shelving | 2/2 | | | | pass |
| 03 | pallets.by_west_door | 1/2 | pallet x 2.09-3.31, y 0.24-1.06 | gap to the west door opening 2.09 m (opening y 1.0-2.2 on x = 0) | <= 2.0 m | **real, over by 0.09 m** |
| 03 | conveyor.axis_east | 1/1 | | | | pass |
| 03 | benches.conveyor_north | 3/4 | bench x 1.69-3.51 (centre 2.60), y 5.14-5.86 | centre x 2.60 vs belt x 3.50-10.50 | centre inside the belt's x span | **real: centre 0.90 m west of the belt start** (its east edge touches it) |
| 03 | cell.conveyor_east_end | 1/1 | | | | pass |
| 03 | shelving.east_wall | 1/1 | | | | pass |
| 03 | recycling.next_to_shelving | 2/2 | | | | pass |
| 04 | pallets.by_west_door | 1/2 | pallet x 2.09-3.31, y 0.24-1.06 | gap to the west door opening 2.09 m (opening y 1.0-2.2 on x = 0) | <= 2.0 m | **real, over by 0.09 m** |
| 04 | conveyor.axis_east | 1/1 | | | | pass |
| 04 | benches.conveyor_north | 3/4 | bench x 1.69-3.51 (centre 2.60), y 5.14-5.86 | centre x 2.60 vs belt x 3.50-10.50 | centre inside the belt's x span | **real: centre 0.90 m west of the belt start** (its east edge touches it) |
| 04 | cell.conveyor_east_end | 1/1 | | | | pass |
| 04 | shelving.east_wall | 1/1 | | | | pass |
| 04 | recycling.next_to_shelving | 2/2 | | | | pass |

Map: `maps/template_P08_P08_seed00.png`.

## 2. Sonnet P15, run 20261030T095745: the benches

Room depth 14.0 m, so the central half of y is [3.5, 10.5] m (rule: row centre inside it).

| Row | Centre y (m) | Inside the band? | Margin (m) |
|---|---|---|---|
| 1 (4 benches) | 3.10 | **no** | -0.40 |
| 2 (4 benches) | 5.70 | yes | +2.20 |
| 3 (4 benches) | 8.20 | yes | +2.30 |

The benches' poses in the scene graph are y = -3.9, -1.3, 1.2 in the room-centred frame, i.e. 3.1, 5.7, 8.2 in the SW frame (room centre 7.0): the block sits 1.3 m south of the middle, and the south row's centre is 0.4 m outside the band. **Real.**

## 3. Frame and distances, all 130 scenes

- **SW origin = inner wall face.** `environment.x_range / y_range` equals the inner face of the four wall slabs (wall centre line +/- half thickness) to within 0.0000 m in every scene (wall centre lines sit 0.15 m outside; the slab is 0.3 m thick). Coordinates are `x - x_range[0]`, `y - y_range[0]`, so 0 is the inner face, not the centre line.
- **Door distances go to the opening.** A door is the segment c +/- w/2 on the wall's inner-face line. For the 194 doors, c and w agree with the opening in `resolved_spec.json` (max |delta c| 0.000 m, max |delta w| 0.000 m). The door asset's origin lies at the opening centre, so even the origin would agree, but the gap is to the segment's nearest point, not to the origin.
- **'Against' / 'along' distances go to the inner wall face:** `wall_gap` = rectangle edge minus 0 (or W / D minus the far edge). Recipe v4 shelves, designed 0.12 m off the wall, measure 0.106 m.
- Belt rectangles come from the scene graph's conveyor bbox; against the belt's collider box they differ by at most 0.030 m.

## 4. Rules that fail in (almost) every scene of a model: measured values

**Sonnet P05, cells side by side** (rule: gap <= 2.0 m and centres aligned within 1.0 m)

- 20261030T082013: cells [[8.87, 3.03, 12.13, 6.33], [1.87, 3.03, 5.13, 6.33]], gap 3.74 m, centre offset in y 0.00 m
- 20261030T133156: cells [[2.57, 2.83, 5.83, 6.13], [9.82, 3.28, 12.18, 5.68]], gap 3.99 m, centre offset in y 0.00 m
- 20261030T142415: cells [[1.87, 2.83, 5.13, 6.13], [8.87, 2.83, 12.13, 6.13]], gap 3.74 m, centre offset in y 0.00 m
- 20261031T114803: cells [[2.37, 2.83, 5.63, 6.13], [8.37, 2.83, 11.63, 6.13]], gap 2.74 m, centre offset in y 0.00 m
- 20261031T123733: cells [[2.62, 3.58, 4.98, 5.98], [9.02, 3.58, 11.38, 5.98]], gap 4.03 m, centre offset in y 0.00 m

**DeepSeek P23, pinned conveyor** (rule: x 1 -> 11, y 5, each within 0.25 m)

- 20261029T194939: belt x 1.00 -> 10.40, y centre 5.00  (start off 0.00, end off 0.60)
- 20261029T202336: belt x 1.30 -> 10.90, y centre 5.00  (start off 0.30, end off 0.10)
- 20261029T212605: belt x 1.50 -> 11.50, y centre 5.00  (start off 0.50, end off 0.50)
- 20261030T151853: belt x 1.60 -> 11.00, y centre 5.00  (start off 0.60, end off 0.00)
- 20261031T121006: belt x 2.00 -> 10.20, y centre 5.00  (start off 1.00, end off 0.80)

**Sonnet P08, pallets by the west door** (rule: gap to the west opening <= 2.0 m, per pallet)

- 20261030T134036: pallet gaps [3.01, 2.04, 1.12] m
- 20261031T132403: pallet gaps [2.75, 1.47, 3.1, 2.06] m
- 20261031T141447: pallet gaps [2.82, 4.11, 5.41] m
- 20261031T142740: pallet gaps [2.57, 3.95, 5.33] m
- 20261031T154918: pallet gaps [2.06, 1.0] m

## 5. Threshold sensitivity: near / along / against / conveyor-to-cell / pinned tolerance all +0.25 m

| Method | Spatial adherence (as approved) | Spatial adherence (+0.25 m) |
|---|---|---|
| claude-sonnet-5-5 | 0.853 | 0.883 |
| deepseek-v4.1-flash_cloud | 0.802 | 0.859 |
| glm-5.3-flash_cloud | 0.856 | 0.871 |
| recipe_v4 | 0.960 | 0.960 |
| recipe_v3 | 0.194 | 0.227 |
| template | 0.875 | 0.958 |
