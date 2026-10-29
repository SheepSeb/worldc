# E2 check (PRELIMINARY while E1 runs)

```
uv run --with matplotlib python scripts/e2_prelim.py --out results/e2_prelim
```

Analysis only: no builds and no Isaac Sim, and the process runs at nice 19. matplotlib comes from an ephemeral `--with`
overlay, so it is not a project dependency. The script docstring holds the method. Rerun the same command after E1:
the "PRELIMINARY" label drops once every prompt has a constrained record.

**Inputs:**
- E1: accepted `constrained` runs in `dataset/e1/deepseek-v4.1-flash_cloud`, evaluated fresh with `worldc eval`.
- Recipe: the scenes for the same prompt ids.
- Template: all accepted seeds, every one from prompt P08.
- Both baselines come from `results/e2_baselines_v3` and `out/e2_v3`.

**Template IoU is exactly 1.0, by construction.** Every floor-standing object in the template (conveyor, robot cell,
benches, shelf, scrap bins, pallets) has a fixed pose in every seed. Seeds only change the items standing on bench,
shelf and pallet tops (laptops, monitors, toolboxes, bins, returned PCs), which add no floor occupancy. The result is
the same at 64, 128 and 256 cells a side. So the template has one floor plan, not a jittered family.

**Reading the IoU:** E1 and recipe pairs cross different prompts (cell types, sizes, door walls), so their low IoU
mostly reflects how different the prompts are. Compare E1 with recipe; both use the same prompt ids. The template is a
single prompt and sets the "no diversity" bound.

Files: `fig4_iou_violins.pdf` (vector) / `.png`, `table1_e2.md` / `.tex`, `iou_pairs.csv`, `scenes.csv`, `summary.json`,
`sanity_pairs.png` (one pair per method, the one closest to its median), `eval_e1.csv`.
