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
  GET, so we index it ourselves and search the index offline. Override the root with `SCENEFORGE_ISAAC_ASSET_ROOT` (a URL or a
  local mirror directory laid out the same way).

## The index

`src/sceneforge/assets/library/catalog/isaac_5.1.jsonl` (checked in; first line is `_meta`).

```
python -m sceneforge.assets.catalog_index index [--packs Props Environments People] [--no-deep]
python -m sceneforge.assets.catalog_index search "hand truck" -k 5
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
* The asset is downloaded once into `~/.cache/sceneforge/catalog/`, imported like a user file (footprint,
  bounds, convex-hull colliders) and **referenced by URL** from the scene. The asset's own rigid-body and collision prims
  are switched off by overrides, and our hulls are the only physics. Missing mass and friction come from the tag-based
  defaults and are flagged in `scene_graph.json`.
* `search_assets` (`assets.resolve.search`) returns one ranked list over registry and catalog hits with a `source` field;
  registry assets win ties. `asset.generated_but_match_exists` warns when a generated asset's name matches a catalog
  asset.

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
