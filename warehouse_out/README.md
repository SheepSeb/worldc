# Procedural warehouse (Isaac Sim 5.1 / Isaac Lab)

Seeds 0-4 are generated here; every seed folder holds

| file | what |
|---|---|
| `seed_N/warehouse.usd` | the scene (Z-up, metres, kg). Default prim `/World`. |
| `seed_N/scene_graph.json` | every object: pose, category, mass, friction, articulation info; plus goals, hazards (waypoints), conveyors, cameras, lighting, chokepoint, rack-part lists |
| `seed_N/textures/` | procedural floor (worn concrete + painted lanes) and 5 cardboard/label textures |
| `seed_N/renders/{overhead,aisle_level,robot}.png` | validation renders |
| `seed_N/nav_occupancy.png` | 5 cm occupancy grid, inflated by robot radius, with the planned path to each goal |
| `seed_N/{static,isaac}_validation.json` | raw validation results |
| `validation_report.md/.json` | summary of all seeds (see bottom) |

## Generate more variations

```bash
uv sync
uv run python -m worldc.generate --seed 7 8 9 --out warehouse_out   # recipe -> spec -> compiler
# optional: --lighting {0,1,2} (default seed % 3: bright / mixed warm / dim), --light-scale 1.5
uv run python -m worldc.run_all --seeds 0 1 2 3 4                   # generate + all 6 validations
```
The warehouse is the `warehouse` recipe (`src/worldc/recipes/warehouse.py`): seed -> scene spec, compiled
by the generic spec compiler. `uv run worldc recipe warehouse --seed 7` prints the spec, and each seed folder also holds
`resolved_spec.json` (exact replay) and `check.json`. Generation takes ~9 s per seed (plus ~3 s for the checks) and needs
only `usd-core`, numpy, scipy, Pillow, pydantic and PyYAML (an Isaac Sim install is still required by policy). Validation needs Isaac Sim
(`ISAACSIM_PATH`, default `~/isaacsim`), run headless. `usdchecker` is executed with Isaac's bundled USD because the
pip `usd-core` wheel lacks the shader resources the checker needs.

## Layout (x east, y north, origin = floor centre)
40 x 25 m, 8 m ceiling. 6 rack aisles 3.2 m wide (x = -16.9 ... 8.6), 12 rack rows, 4 beam levels at 0.2 / 1.8 / 3.4 / 5.0 m,
2 cross-aisles (y -7.5..-4.3, 1.1..4.3). Dock (3 roll-up bays + leveler plates) on the south wall; packing zone
(2 belts, roller conveyor, work table with 2 drawers) north-east; charging area with 2 parking spots south-east;
wire-mesh cage with sliding gate east; 2 swing doors (west + north wall). Two slope zones (2-3 % crowns, <= 4 cm).
The chokepoint is aisle 2 (north segment): two fallen pallets + spilled cartons leave **0.90 m** between the debris and
the opposite rack (measured exactly, not on the grid).

## Loading in Isaac Sim
`File > Open` `seed_N/warehouse.usd`, or headless: `omni.usd.get_context().open_stage(path)`. Physics scene:
TGS solver, 120 Hz (`physxScene:timeStepsPerSecond`), CPU dynamics. Use `physics_dt = 1/120`.

## Loading in Isaac Lab
> Not tested: Isaac Lab is not installed on the machine this was built on. Everything below uses only plain USD +
> Isaac Lab spawner APIs; the scene itself was validated in Isaac Sim 5.1 with the tensor API Isaac Lab sits on.

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

USD = "/abs/path/warehouse_out/seed_0/warehouse.usd"

@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    # whole warehouse incl. racks, pallets, boxes, totes, conveyors (surface velocity), doors (joints)
    warehouse = AssetBaseCfg(prim_path="/World/envs/env_.*/Warehouse", spawn=sim_utils.UsdFileCfg(usd_path=USD))
    # (optional) expose doors/drawers as Articulations: root = the world-fixed *Base* prim, 1 DOF each
    # drawer0 = ArticulationCfg(prim_path="/World/envs/env_.*/Warehouse/WorkTable/DrawerBase_0", spawn=None,
    #     actuators={"j": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=3000.0, damping=600.0)})
```
* `SimulationCfg(dt=1/120, ...)`; use `replicate_physics=False` (one big heterogeneous scene) and `num_envs=1`.
* **Hazards are not animated by USD.** Isaac Lab's stepping does not advance USD time samples, so the 2 human capsules
  and the forklift box are kinematic bodies driven by `worldc.runtime.PathFollower`
  (waypoints/speed/mode are also stored on the prims as `warehouse:*` attributes and in `scene_graph.json`):
  ```python
  from worldc.runtime import load_scene_graph, hazard_followers
  followers = hazard_followers(load_scene_graph(".../scene_graph.json"))   # {prim_path: PathFollower}
  x, y, z, yaw = followers["/World/Hazards/Forklift"].pose(sim_time)       # write as kinematic target each step
  ```
* Robot: `/World/Robot/MobileManipulator` is a visual-only placeholder at the spawn pose (`scene_graph.json:robot.spawn`,
  footprint radius 0.38 m); `/World/Robot/MobileManipulator/RobotCamera` is the mounted camera. Spawn your robot at that
  pose. Navigation goals (17, incl. 4 high-shelf picks at 3.4 / 5.0 m with `pick.target_xyz`) are in
  `scene_graph.json:goals`.
* Cameras: `/World/Cameras/Overhead` sits above the roof - make `/World/Building/Roof` invisible to use it;
  `/World/Cameras/AisleLevel`; the robot camera above.

## Physics notes
Dynamic bodies are Xform rigid bodies with analytic box / cylinder / cone colliders only (convex compounds, no triangle
meshes); static racks, walls, doors' frames use static colliders, and the only triangle meshes are the two static
slope-zone crowns. Mass, centre of mass and full inertia tensor (diagonalised, principal axes) are authored from the
compound shapes. Friction: cardboard 0.5, steel 0.6, concrete 0.7, rubber belt 0.9 (wood 0.5, plastic 0.4).
Boxes: 6 sizes per seed (0.15-0.6 m), mass = volume x 60-220 kg/m^3 clamped to 0.2-15 kg, stacked with <= 3 cm / <= 10 deg
random misalignment. "Wrapped" pallets are one rigid body (pallet + consolidated load). Conveyors are kinematic bodies
with `PhysxSurfaceVelocityAPI` (0.5 m/s); the belt texture does not scroll. Each door / drawer / gate is a 1-DOF fixed-base
articulation (tiny world-fixed `ArtBase` link + prismatic/revolute drive), default pose closed.

## Known limitations
* Procedural geometry/materials only (UsdPreviewSurface); no Nucleus/Isaac asset-pack dependencies, so it loads offline.
* Roll-up doors are sliding panels (no coil); the dock leveler is a static tilted plate; human actors are capsules.
* The fallen pallets are flat on the floor (no tilted/leaning pallet) to keep them in stable equilibrium.
* Occupancy/nav is geometric (A* on a 5 cm grid), not a Nav2/Isaac planner run.

## Validation results (seeds 0-4, final run)

| seed | lighting | 1 usdchecker | 2 headless load | 3 settled <2cm | 3 physics flags | 4 articulation | 5 nav (all goals) | chokepoint | 6 renders | pre-flight |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | bright_day | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.902 m | PASS | PASS |
| 1 | mixed_warm | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.902 m | PASS | PASS |
| 2 | dim_night | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.9 m | PASS | PASS |
| 3 | bright_day | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.9 m | PASS | PASS |
| 4 | mixed_warm | PASS | PASS | 100.00% | PASS | PASS | PASS | 0.9 m | PASS | PASS |

Per-seed detail: `validation_report.md`. Settled fraction = bodies whose final position is within 2 cm of the initial pose after 10 s (hazards active); the 5 boxes riding the conveyors are excluded and instead checked for 0.5 m/s carry speed.
