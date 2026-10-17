# Validation report

| seed | lighting | 1 usdchecker | 2 headless load | 3 settled <2cm | 3 physics flags | 4 articulation | 5 nav (all goals) | chokepoint | 6 renders | pre-flight |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | bright_day | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.902 m | PASS | PASS |
| 1 | mixed_warm | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.902 m | PASS | PASS |
| 2 | dim_night | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.9 m | PASS | PASS |
| 3 | bright_day | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.9 m | PASS | PASS |
| 4 | mixed_warm | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.9 m | PASS | PASS |

## Details per seed

### seed 0
- dynamic rigid bodies: 1339 (1334 evaluated, 5 conveyor payload excluded); settled fraction 1.0, max displacement 0.001 m, fell through floor: 0, faster than 1 m/s: 0
- conveyor payload speeds (target 0.5 m/s): 0.5, 0.5, 0.5, 0.5, 0.5
- articulations: RollUp_0 (ok, pen 0.0 mm), RollUp_1 (ok, pen 0.0 mm), RollUp_2 (ok, pen 0.0 mm), Swing_0 (ok, pen 0.0 mm), Swing_1 (ok, pen 0.0 mm), CageGate (ok, pen 0.0 mm), Drawer_0 (ok, pen 0.0 mm), Drawer_1 (ok, pen 0.0 mm)
- spawn interpenetrations >3 mm: 0; floating: 0; hazard-path conflicts: 0
- navigation: 17/17 goals reachable on the 5 cm grid (robot radius 0.38 m); chokepoint clearance exact 0.902 m (grid-quantised 0.75 m)
- kit log: 0 fatal, 0 errors

### seed 1
- dynamic rigid bodies: 1282 (1277 evaluated, 5 conveyor payload excluded); settled fraction 1.0, max displacement 0.0012 m, fell through floor: 0, faster than 1 m/s: 0
- conveyor payload speeds (target 0.5 m/s): 0.5, 0.5, 0.5, 0.5, 0.5
- articulations: RollUp_0 (ok, pen 0.0 mm), RollUp_1 (ok, pen 0.0 mm), RollUp_2 (ok, pen 0.0 mm), Swing_0 (ok, pen 0.0 mm), Swing_1 (ok, pen 0.0 mm), CageGate (ok, pen 0.0 mm), Drawer_0 (ok, pen 0.0 mm), Drawer_1 (ok, pen 0.0 mm)
- spawn interpenetrations >3 mm: 0; floating: 0; hazard-path conflicts: 0
- navigation: 17/17 goals reachable on the 5 cm grid (robot radius 0.38 m); chokepoint clearance exact 0.902 m (grid-quantised 0.75 m)
- kit log: 0 fatal, 0 errors

### seed 2
- dynamic rigid bodies: 1391 (1386 evaluated, 5 conveyor payload excluded); settled fraction 1.0, max displacement 0.0005 m, fell through floor: 0, faster than 1 m/s: 0
- conveyor payload speeds (target 0.5 m/s): 0.5, 0.5, 0.5, 0.5, 0.5
- articulations: RollUp_0 (ok, pen 0.0 mm), RollUp_1 (ok, pen 0.0 mm), RollUp_2 (ok, pen 0.0 mm), Swing_0 (ok, pen 0.0 mm), Swing_1 (ok, pen 0.0 mm), CageGate (ok, pen 0.0 mm), Drawer_0 (ok, pen 0.0 mm), Drawer_1 (ok, pen 0.0 mm)
- spawn interpenetrations >3 mm: 0; floating: 0; hazard-path conflicts: 0
- navigation: 17/17 goals reachable on the 5 cm grid (robot radius 0.38 m); chokepoint clearance exact 0.9 m (grid-quantised 0.8 m)
- kit log: 0 fatal, 0 errors

### seed 3
- dynamic rigid bodies: 1374 (1369 evaluated, 5 conveyor payload excluded); settled fraction 1.0, max displacement 0.0006 m, fell through floor: 0, faster than 1 m/s: 0
- conveyor payload speeds (target 0.5 m/s): 0.5, 0.5, 0.5, 0.5, 0.5
- articulations: RollUp_0 (ok, pen 0.0 mm), RollUp_1 (ok, pen 0.0 mm), RollUp_2 (ok, pen 0.0 mm), Swing_0 (ok, pen 0.0 mm), Swing_1 (ok, pen 0.0 mm), CageGate (ok, pen 0.0 mm), Drawer_0 (ok, pen 0.0 mm), Drawer_1 (ok, pen 0.0 mm)
- spawn interpenetrations >3 mm: 0; floating: 0; hazard-path conflicts: 0
- navigation: 17/17 goals reachable on the 5 cm grid (robot radius 0.38 m); chokepoint clearance exact 0.9 m (grid-quantised 0.8 m)
- kit log: 0 fatal, 0 errors

### seed 4
- dynamic rigid bodies: 1463 (1458 evaluated, 5 conveyor payload excluded); settled fraction 1.0, max displacement 0.0007 m, fell through floor: 0, faster than 1 m/s: 0
- conveyor payload speeds (target 0.5 m/s): 0.5, 0.5, 0.5, 0.5, 0.5
- articulations: RollUp_0 (ok, pen 0.0 mm), RollUp_1 (ok, pen 0.0 mm), RollUp_2 (ok, pen 0.0 mm), Swing_0 (ok, pen 0.0 mm), Swing_1 (ok, pen 0.0 mm), CageGate (ok, pen 0.0 mm), Drawer_0 (ok, pen 0.0 mm), Drawer_1 (ok, pen 0.0 mm)
- spawn interpenetrations >3 mm: 0; floating: 0; hazard-path conflicts: 0
- navigation: 17/17 goals reachable on the 5 cm grid (robot radius 0.38 m); chokepoint clearance exact 0.9 m (grid-quantised 0.8 m)
- kit log: 0 fatal, 0 errors
