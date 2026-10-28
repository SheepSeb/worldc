# Omniverse / Isaac Sim asset catalog

How the generator finds and uses NVIDIA's Isaac Sim assets (PLAN.md §2.5–2.6), and what the licenses allow.
The index and these findings are for Isaac Sim **5.1.0** (`~/isaacsim/VERSION`).

## Where the assets are

* Isaac Sim 5.1 does not ship the asset packs. Its asset root is configured in
  `exts/isaacsim.storage.native/config/extension.toml` (`persistent.isaac.asset_root.default`):
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1`.
* Layout under that root: `Isaac/{Props,Environments,People,Robots,Sensors,IsaacLab,Samples}`.
  Spec paths are relative to the root and start with `Isaac/`, e.g. `Isaac/Props/Pallet/pallet.usd`.
* **Nucleus search (DeepSearch) is not used.** It needs a Nucleus server. The bucket allows anonymous S3 listing and
  GET, so we index it ourselves and search the index offline. Override the root with `WORLDC_ISAAC_ASSET_ROOT` (a URL or a
  local mirror directory laid out the same way).

## The index

`src/worldc/assets/library/catalog/isaac_5.1.jsonl` (checked in; first line is `_meta`).

```
python -m worldc.assets.catalog_index index [--packs Props Environments People] [--no-deep]
python -m worldc.assets.catalog_index search "hand truck" -k 5
```

* Scope: packs `Props`, `Environments` (everything except the Outdoor/Rivermark terrain) and `People`. Whole-environment
  scene files are indexed but left out of search unless `scenes=True`. Robots, sensors and IsaacLab are out of scope for scene composition and not indexed.
* One entry per asset family: thumbnails, materials, textures, `parts/`, `configuration/` layers are dropped, and numbered
  scene instances (`SM_CardBoxD_04_1169`) collapse into one entry with an `instances` count. Root layers with no default
  prim cannot be referenced and are dropped.
* Per entry: `path`, `name`, `folder`, `pack`, `bytes`, `etag` (content revision), `size_m` (bounds in metres, Z-up),
  `physics` (authored rigid body / collision / articulation / joints), semantic `labels`, `instances`, and `deep` (false
  if the file was unreadable or over 80 MB; then there is no size or physics data).
* Search is BM25 over name (x3), semantic labels (x2) and folder tokens, with a small synonym table
  (`trolley -> cart`, `crate -> box`, ...). `resolve_query` accepts a hit only if at least two thirds of the query words
  match it (so "hand drill" does not pin a hand dryer); otherwise the validator reports `asset.catalog_no_match` and lists the closest paths.

## Using it from a spec

```yaml
objects:
  - id: truck
    asset: {catalog: omniverse, query: "hand truck"}        # resolved once through the index
    placement: {inside: staging}
  - id: pallet
    asset: {catalog: omniverse, path: Isaac/Props/Pallet/pallet.usd}   # pinned
```

* `resolved_spec.json` replaces every `query` with the exact `path` and the index's `rev` (ETag), which is part of the
  download cache key, so a replay uses the same file.
* The asset is downloaded once into `~/.cache/worldc/catalog/`, imported like a user file (footprint,
  bounds, convex-hull colliders) and **referenced by URL** from the scene. The asset's own rigid-body and collision prims
  are switched off by overrides, and our hulls are the only physics. Missing mass and friction come from the tag-based
  defaults and are flagged in `scene_graph.json`.
* `search_assets` (`assets.resolve.search`) returns one ranked list over registry and catalog hits with a `source` field;
  registry assets win ties. `asset.generated_but_match_exists` warns when a generated asset's name matches a catalog
  asset.

## NVIDIA SimReady assets (Hugging Face)

[`nvidia/simready-assets`](https://huggingface.co/datasets/nvidia/simready-assets) holds ~1.7 k SimReady packages
(~250 GB; median ~45 MB, largest ~2.4 GB): props, appliances, tools, medical and industrial equipment, each with one root
USD, physics payloads (PhysX / Newton / MuJoCo), semantic labels (Wikidata class) and a BOM with sha256 per file.
**Nothing is stored up front and nothing is checked in.** Code: `src/worldc/assets/simready.py`.

```
worldc simready index [--revision main]          # metadata only: one HF API call -> ~/.cache/worldc/simready/index.jsonl (~0.7 MB)
worldc simready search "office chair" [--max-mb 100] [--json]
worldc simready info chair_office_amf_black_a01 [--probe]
worldc simready fetch chair_office_amf_black_a01 [--max-mb 1024]
worldc simready cached | clean [PACKAGE]
```

| Step | Downloads | Kept |
|---|---|---|
| `index` | the repo file listing at one commit | one line per package: root USD, bytes, file count, thumbnail path, PhysX payload yes/no |
| `search` | nothing (offline BM25, same scorer as the Omniverse index) | - |
| `info --probe` | the package's USD layers only (no textures), into a temp dir | `probe/<rev>/<pkg>.json`: bounds (m, Z-up), authored physics, semantic labels. Search results show these once probed |
| `fetch`, or a spec that uses the asset | the package minus thumbnails, each file checked against the BOM sha256 | `pkgs/<rev>/<pkg>/` until `clean` |

In a spec:

```yaml
objects:
  - id: chair
    asset: {catalog: simready, query: "office chair"}       # pinned to path + repo commit in resolved_spec.json
  - id: toaster
    asset: {catalog: simready, path: appliance_toaster_a01/simready_usd/sm_appliance_toaster_a01_01.usd}
```

* `validate` resolves queries against the index only and never downloads. The package is fetched at **build** time,
  imported like any catalog asset (footprint, bounds, convex hulls; the asset's own physics is switched off) and
  referenced from the scene **at its cached local path**: HF `resolve` URLs redirect to a CDN, so the package's relative
  payload and texture paths would not resolve against a URL reference. Such a scene opens only on a machine with the same
  cache; rebuild the resolved spec elsewhere (it re-fetches the pinned commit).
* Query pinning is stricter than for Omniverse: at least two thirds of the query words must appear literally in the
  package name or labels. The synonym table only ranks. Otherwise "forklift" would pin a scissor lift table.
* `rev` is the repo commit. A resolved spec pinned at another commit fails with `asset.simready_rev_mismatch`; rebuild
  the index with `--revision <sha>`.
* Error codes: `asset.catalog_unavailable` (no index), `asset.catalog_no_match`, `asset.simready_too_large` (over
  `WORLDC_SIMREADY_MAX_MB`, default 1024), `asset.simready_corrupt` (BOM hash mismatch), `asset.catalog_not_found` /
  `asset.catalog_unreachable` (network).
* MCP `search_assets` lists SimReady hits (`source: simready`, with `mb` = download size) when the index is installed.
  Order at equal score: registry > omniverse > simready.
* Env: `WORLDC_SIMREADY_INDEX` (index file), `WORLDC_SIMREADY_ROOT` (local mirror instead of HF, used by tests),
  `HF_ENDPOINT`, `WORLDC_SIMREADY_MAX_MB`.
* License: **CC-BY-4.0** (per package `.metadata/LICENSE.txt`, kept in the cache). Attribution to NVIDIA is required when
  you publish renders or datasets that use these assets. It is not added to outputs automatically yet.

<!-- curated-simready:start -->
## Curated SimReady subset (remanufacturing cells)

45 assets chosen from the Phase 1 survey (`docs/simready_candidates.md`): 43 scene assets with short aliases and 2 FANUC arms that only `robot_cell` uses. Code: `src/worldc/assets/curated.py`; table: `src/worldc/assets/library/catalog/simready_curated.json` (metadata only, no USD in the repo).

* Every alias is a registered asset (`list_assets(tags=["workbench"])`, `describe_asset("klt_bin")`), solved offline from the footprint / anchors measured by `python -m worldc.assets.curated bake`.
* Building one downloads it at its **pinned commit** into `~/.cache/worldc/catalog/` (once), imports it like any catalog asset (convex-hull colliders; the asset's own physics is switched off) and references the original by its pinned HF URL (`https://huggingface.co/datasets/<repo>/resolve/<sha>/<path>`, never `main`).
* Benches have a `top` support anchor on the work surface and shelves `shelf_0..n` anchors (bottom to top), so props use `placement: {on: bench.top}` / `{on: "shelf.*.shelf_*"}`. Warehouse-01 assets are authored in cm and collision-only; they are converted to metres and built static.

### Licences and attribution

Publishing renders, scenes or datasets that use these assets requires attribution under CC-BY-4.0 (and keeping the licence and notices for the Apache-2.0 FANUC arms). Copy the line for each asset you use.

| Alias | Tags | Licence | Attribution | Pinned USD |
|---|---|---|---|---|
| `workbench_modular` | support, table, workbench, workstation | CC-BY-4.0 | NVIDIA, "furniture_workbench_modular_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/furniture_workbench_modular_a01/simready_usd/sm_furniture_workbench_modular_a01_01.usd) |
| `utility_cart` | cart, workstation | CC-BY-4.0 | NVIDIA, "equipment_utility_cart_e01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/equipment_utility_cart_e01/simready_usd/sm_equipment_utility_cart_e01_01.usd) |
| `packing_table_heavy` | support, table, workbench, workstation | CC-BY-4.0 | NVIDIA, "SM_HeavyDutyPackingTable_C01_01", nvidia/PhysicalAI-SimReady-Warehouse-01 on Hugging Face (rev c7fe115cb79c), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/c7fe115cb79c7ddbd0532630d7768b5736b0ecc4/Props/general/SM_HeavyDutyPackingTable_C01_01/SM_HeavyDutyPackingTable_C01_01_physics.usd) |
| `workbench_steel` | support, table, workbench, workstation | CC-BY-4.0 | NVIDIA, "SM_MetalStainlessTable_A03_01", nvidia/PhysicalAI-SimReady-Warehouse-01 on Hugging Face (rev c7fe115cb79c), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/c7fe115cb79c7ddbd0532630d7768b5736b0ecc4/Props/general/SM_MetalStainlessTable_A03_01/SM_MetalStainlessTable_A03_01_physics.usd) |
| `packing_table_deluxe` | support, table, workbench, workstation | CC-BY-4.0 | NVIDIA, "SM_DeluxePackingTable_A01_01", nvidia/PhysicalAI-SimReady-Warehouse-01 on Hugging Face (rev c7fe115cb79c), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/c7fe115cb79c7ddbd0532630d7768b5736b0ecc4/Props/general/SM_DeluxePackingTable_A01_01/SM_DeluxePackingTable_A01_01_physics.usd) |
| `toolbox` | prop, small, tool | CC-BY-4.0 | ForgeCAD, "toolbox", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/toolbox/simready_usd/toolbox.usda) |
| `bench_grinder` | prop, small, tool | CC-BY-4.0 | ForgeCAD, "bench_grinder", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/bench_grinder/simready_usd/bench_grinder.usda) |
| `parts_shelf` | rack, shelf, storage | CC-BY-4.0 | NVIDIA, "HeavyDutySteelShelving_A06_PR_NVD_01", nvidia/PhysicalAI-SimReady-Warehouse-01 on Hugging Face (rev c7fe115cb79c), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/c7fe115cb79c7ddbd0532630d7768b5736b0ecc4/Props/general/HeavyDutySteelShelving_A06_PR_NVD_01/HeavyDutySteelShelving_A06_PR_NVD_01_physics.usd) |
| `shelf_rack` | rack, shelf, storage | CC-BY-4.0 | NVIDIA, "SM_Rack_F04_01", nvidia/PhysicalAI-SimReady-Warehouse-01 on Hugging Face (rev c7fe115cb79c), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/c7fe115cb79c7ddbd0532630d7768b5736b0ecc4/Props/general/SM_Rack_F04_01/SM_Rack_F04_01_physics.usd) |
| `bulk_storage_rack` | rack, shelf, storage | CC-BY-4.0 | NVIDIA, "SM_BulkStorageRack_A03_01", nvidia/PhysicalAI-SimReady-Warehouse-01 on Hugging Face (rev c7fe115cb79c), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/c7fe115cb79c7ddbd0532630d7768b5736b0ecc4/Props/general/SM_BulkStorageRack_A03_01/SM_BulkStorageRack_A03_01_physics.usd) |
| `klt_bin` | bin, klt, small, storage, tote | CC-BY-4.0 | ForgeCAD, "small_load_carrier", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/small_load_carrier/simready_usd/small_load_carrier.usda) |
| `lidded_tote` | bin, small, storage, tote | CC-BY-4.0 | ForgeCAD, "attached_lid_tote", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/attached_lid_tote/simready_usd/attached_lid_tote.usda) |
| `picking_bin` | bin, small, storage, tote | CC-BY-4.0 | ForgeCAD, "bin_picking_bin", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/bin_picking_bin/simready_usd/bin_picking_bin.usda) |
| `crate_blue` | crate, small, storage, tote | CC-BY-4.0 | NVIDIA, "crate_blue_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/crate_blue_a01/simready_usd/sm_crate_blue_a01_01.usd) |
| `crate_gray` | crate, small, storage, tote | CC-BY-4.0 | NVIDIA, "crate_gray_a08", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/crate_gray_a08/simready_usd/sm_crate_gray_a08_01.usd) |
| `crate_plywood` | crate, storage | CC-BY-4.0 | NVIDIA, "crate_plywood_b02", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/crate_plywood_b02/simready_usd/sm_crate_plywood_b02_01.usd) |
| `drum_steel` | drum, storage | CC-BY-4.0 | NVIDIA, "drum_steel_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/drum_steel_a01/simready_usd/sm_drum_steel_a01_01.usd) |
| `drum_plastic` | drum, small, storage | CC-BY-4.0 | NVIDIA, "drum_plastic_blue_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/drum_plastic_blue_a01/simready_usd/sm_drum_plastic_blue_a01_01.usd) |
| `ibc_tank` | ibc, storage, tank | CC-BY-4.0 | NVIDIA, "ibcTank_a02", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/ibcTank_a02/simready_usd/sm_ibcTank_a02_01.usd) |
| `flammables_cabinet` | cabinet, safety, storage | CC-BY-4.0 | NVIDIA, "cabinet_drum_flammable_a02", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/cabinet_drum_flammable_a02/simready_usd/sm_cabinet_drum_flammable_a02_01.usd) |
| `safety_fence_panel` | barrier, fence, safety | CC-BY-4.0 | NVIDIA, "barrier_fence_metal_a02", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/barrier_fence_metal_a02/simready_usd/sm_barrier_fence_metal_a02_01.usd) |
| `safety_gate` | barrier, fence, gate, safety | CC-BY-4.0 | NVIDIA, "barrier_gate_safetyDock_a02", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/barrier_gate_safetyDock_a02/simready_usd/sm_barrier_gate_safetyDock_a02_01.usd) |
| `barrier_wall` | barrier, safety | CC-BY-4.0 | NVIDIA, "barrier_wall_plastic_orange_a03", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/barrier_wall_plastic_orange_a03/simready_usd/sm_barrier_wall_plastic_orange_a03_01.usd) |
| `fire_extinguisher_stand` | fire, safety | CC-BY-4.0 | NVIDIA, "fireExtinguisher_stand_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/fireExtinguisher_stand_a01/simready_usd/sm_fireExtinguisher_stand_a01_01.usd) |
| `guard_rail` | barrier, guard_rail, safety | CC-BY-4.0 | NVIDIA, "SM_GuardRail_A09_01", nvidia/PhysicalAI-SimReady-Warehouse-01 on Hugging Face (rev c7fe115cb79c), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/c7fe115cb79c7ddbd0532630d7768b5736b0ecc4/Props/general/SM_GuardRail_A09_01/SM_GuardRail_A09_01_physics.usd) |
| `pallet_block` | material_flow, pallet | CC-BY-4.0 | NVIDIA, "SM_BlockPallet_A01_WoodNewA_01", nvidia/PhysicalAI-SimReady-Warehouse-01 on Hugging Face (rev c7fe115cb79c), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/c7fe115cb79c7ddbd0532630d7768b5736b0ecc4/Props/general/SM_BlockPallet_A01_WoodNewA_01/SM_BlockPallet_A01_WoodNewA_01_physics.usd) |
| `pallet_euro` | material_flow, pallet | CC-BY-4.0 | ForgeCAD, "pallet", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/pallet/simready_usd/pallet.usda) |
| `pallet_jack` | material_flow, pallet_jack | CC-BY-4.0 | ForgeCAD, "manual_pallet_jack", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/manual_pallet_jack/simready_usd/manual_pallet_jack.usda) |
| `hand_truck` | hand_truck, material_flow | CC-BY-4.0 | NVIDIA, "handTruck_convertible_steel_orange_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/handTruck_convertible_steel_orange_a01/simready_usd/sm_handTruck_convertible_steel_orange_a01_01.usd) |
| `electric_motor` | drivetrain, ev, motor, part, remanufacturing | CC-BY-4.0 | NVIDIA, "engineAssembly_electric_engine_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/engineAssembly_electric_engine_a01/simready_usd/sm_engineAssembly_electric_engine_a01_01.usd) |
| `transfer_case` | drivetrain, gearbox, part, remanufacturing | CC-BY-4.0 | NVIDIA, "drivetrain_transfer_case_b01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/drivetrain_transfer_case_b01/simready_usd/sm_drivetrain_transfer_case_b01_01.usd) |
| `differential` | drivetrain, gearbox, part, remanufacturing | CC-BY-4.0 | NVIDIA, "drivetrain_differential_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/drivetrain_differential_a01/simready_usd/sm_drivetrain_differential_a01_01.usd) |
| `planetary_gearbox` | gearbox, part, remanufacturing, small | CC-BY-4.0 | ForgeCAD, "planetary_gearbox", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/planetary_gearbox/simready_usd/planetary_gearbox.usda) |
| `washing_machine` | appliance, remanufacturing, returned_unit | CC-BY-4.0 | LG Electronics, "laundry_wm4000hwa", LGElectronics/simready-assets on Hugging Face (rev 96684f4c9c62), CC-BY-4.0 | [usd](https://huggingface.co/datasets/LGElectronics/simready-assets/resolve/96684f4c9c6205aef8f53abc9a09ce0f49234507/laundry_wm4000hwa/simready_usd/laundry_wm4000hwa.usd) |
| `dishwasher` | appliance, remanufacturing, returned_unit | CC-BY-4.0 | LG Electronics, "dishwasher_ldfn4542b", LGElectronics/simready-assets on Hugging Face (rev 96684f4c9c62), CC-BY-4.0 | [usd](https://huggingface.co/datasets/LGElectronics/simready-assets/resolve/96684f4c9c6205aef8f53abc9a09ce0f49234507/dishwasher_ldfn4542b/simready_usd/dishwasher_ldfn4542b.usd) |
| `monitor` | electronics, remanufacturing, returned_unit, small | CC-BY-4.0 | LG Electronics, "monitor_27gx790a", LGElectronics/simready-assets on Hugging Face (rev 96684f4c9c62), CC-BY-4.0 | [usd](https://huggingface.co/datasets/LGElectronics/simready-assets/resolve/96684f4c9c6205aef8f53abc9a09ce0f49234507/monitor_27gx790a/simready_usd/monitor_27gx790a.usd) |
| `laptop` | electronics, remanufacturing, returned_unit, small | CC-BY-4.0 | LG Electronics, "laptop_16z90tl", LGElectronics/simready-assets on Hugging Face (rev 96684f4c9c62), CC-BY-4.0 | [usd](https://huggingface.co/datasets/LGElectronics/simready-assets/resolve/96684f4c9c6205aef8f53abc9a09ce0f49234507/laptop_16z90tl/simready_usd/laptop_16z90tl.usd) |
| `microwave` | appliance, electronics, remanufacturing, returned_unit, small | CC-BY-4.0 | ForgeCAD, "microwave_oven", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/microwave_oven/simready_usd/microwave_oven.usda) |
| `pc_tower` | electronics, remanufacturing, returned_unit, small | CC-BY-4.0 | NVIDIA, "computer_desktop_tower_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/computer_desktop_tower_a01/simready_usd/sm_computer_desktop_tower_a01_01.usd) |
| `scrap_bin_wheeled` | bin, scrap_bin, sorting, waste | CC-BY-4.0 | NVIDIA, "container_recycle_blue_wheeled_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/container_recycle_blue_wheeled_a01/simready_usd/sm_container_recycle_blue_wheeled_a01_01.usd) |
| `scrap_bin` | bin, scrap_bin, sorting, waste | CC-BY-4.0 | NVIDIA, "container_recycle_green_b03", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/container_recycle_green_b03/simready_usd/sm_container_recycle_green_b03_01.usd) |
| `wheelie_bin` | bin, scrap_bin, sorting, waste | CC-BY-4.0 | ForgeCAD, "wheelie_bin", ForgeCAD/simready-assets on Hugging Face (rev b94ea770666b), CC-BY-4.0 | [usd](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/wheelie_bin/simready_usd/wheelie_bin.usda) |
| `parts_washer` | cleaning, remanufacturing, station | CC-BY-4.0 | NVIDIA, "equipment_parts_washer_rotary_a01", nvidia/simready-assets on Hugging Face (rev d94382800d8a), CC-BY-4.0 | [usd](https://huggingface.co/datasets/nvidia/simready-assets/resolve/d94382800d8ab00111bf3f9b9047191475b5de8d/equipment_parts_washer_rotary_a01/simready_usd/sm_equipment_parts_washer_rotary_a01_01.usd) |
| `fanuc_crx10ia` | robot_arm | APACHE-2.0 | FANUC CORPORATION, "crx10ia", FANUC-CORPORATION/SimReady-Assets on Hugging Face (rev cedea1f1a598), APACHE-2.0 | [usd](https://huggingface.co/datasets/FANUC-CORPORATION/SimReady-Assets/resolve/cedea1f1a598d1da8745635ee9d5ac291b89df64/crx10ia/simready_usd/crx10ia.usd) |
| `fanuc_crx20ia_l` | robot_arm | APACHE-2.0 | FANUC CORPORATION, "crx20ia_l", FANUC-CORPORATION/SimReady-Assets on Hugging Face (rev cedea1f1a598), APACHE-2.0 | [usd](https://huggingface.co/datasets/FANUC-CORPORATION/SimReady-Assets/resolve/cedea1f1a598d1da8745635ee9d5ac291b89df64/crx20ia_l/simready_usd/crx20ia_l.usd) |
<!-- curated-simready:end -->

## Licenses (read, not legal advice)

| Pack | Terms | What it means here |
|---|---|---|
| `Props`, `People` and the rest outside `Environments` | NVIDIA Omniverse License Agreement (`~/isaacsim/LICENSE.txt`, `PACKAGE-LICENSES/isaac-sim-assets-LICENSE.txt`) | §2.1 allows installing and using Content, modifying it, and **distributing it "as incorporated into your products or services... but not on a stand-alone basis"**, plus distributing user-generated stills, video and models made with Omniverse. §3 adds distribution requirements (terms at least as protective as NVIDIA's). |
| `Environments` | **Limited Use Content** (`Isaac/Environments/environment-supplement-LICENSE.txt`) | Use only, "without modifications", non-transferable, and the §2.1 distribution grant for Content **does not apply**. |
| Individual prop folders | May carry third-party terms in their own README | `Props/Beaker` is CC-BY 4.0 (author credit required). Only 2 license-like files exist under Props / Environments / People; nobody has audited every folder. |

What the pipeline does with that:

* **Meshes and textures are never copied into outputs.** Scenes reference NVIDIA's files by URL, and only specs,
  `scene_graph.json`, our collision proxies and renders are produced.
  The `asset.catalog_limited_license` warning flags `Environments` assets.
* A dataset of **specs + scene graphs + renders** that use these assets is the lower-risk shape (renders are "user
  generated content"). A dataset that **bundles the USD files** is not covered for `Environments` and is only covered
  for other packs when bundled in a product, not stand-alone.
* Our convex-hull collision files are derived from the meshes. They are small and not renderable, but they are a
  derivative of the geometry; keep them out of anything published until someone with legal standing signs off.
* Not automated yet: attribution for CC-BY props, and a per-asset license field read from folder READMEs.

**Before publishing any dataset or rendered images, have the licensing question checked against your NVIDIA
agreement.** This section summarises the files I read; it does not replace them.
