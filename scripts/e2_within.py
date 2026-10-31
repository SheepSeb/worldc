"""E2 within-prompt diversity: how different are several scenes generated for the SAME prompt?  Final Fig. 4 and the IoU
column of Table 1 (cross-prompt IoU from ``e2_prelim.py`` is supplementary only).

    uv run --with matplotlib python scripts/e2_within.py --out results/e2_within

Analysis only (no builds, no Isaac Sim, nice 19), so it can run beside E1 and the sample batch; rerun the same command when
they finish.  Outputs are labelled PRELIMINARY until every prompt has its full set of samples.

* **LLM:** per prompt, the first 5 accepted runs among the constrained E1 run and the ``e2_samples`` runs.
* **Recipe:** per prompt, the accepted scenes of seeds base..base+4 in ``--recipe`` (``<prompt>_seed<k>/``), when built.
* **Template:** seeds 0..4 of its one prompt, the IoU = 1.0 reference.
* **Grid and IoU:** as in ``e2_prelim.py`` (normalised occupancy grids of static nav colliders minus the building shell).
  Pairs within one prompt only (10 per prompt at 5 samples), pooled per method.
* **CI:** 95 % percentile bootstrap clustered by prompt: resample prompts, pool their pairs, take the median.
* **Outputs** in ``--out``: ``fig4_within_iou.pdf`` (vector) / ``.png``, ``table_within.md``, ``within_pairs.csv``,
  ``per_prompt.csv``, ``summary.json``, ``eval_samples.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_prelim as P  # noqa: E402  (occupancy, iou, helpers)

PROMPTS = ("P05", "P08", "P15", "P16", "P23")
SAMPLES = 5
METHODS = ("llm", "recipe", "template")
LABEL = {"llm": "LLM", "recipe": "Recipe", "template": "Template"}
COLOR = {"llm": "#1f77b4", "recipe": "#ff7f0e", "template": "#7f7f7f"}


def eval_runs(runs: Path, out_csv: Path) -> list[dict]:
    from worldc.eval.run import main as eval_main

    eval_main(["--runs", str(runs), "--prompts", "prompts/prompts.csv", "--out", str(out_csv)])
    rows = P._rows(out_csv)
    for r in rows:
        r["dir"] = r["scene_dir"] if Path(r["scene_dir"]).is_absolute() else str(runs / r["scene_dir"])
    return rows


def llm_scenes(e1: Path, samples: Path, variant: str, out: Path) -> dict[str, list[dict]]:
    """prompt -> accepted samples: the E1 run (latest per prompt) first, then the e2_samples runs in run order."""
    e1_rows = [r for r in eval_runs(e1, out / "eval_e1.csv") if r["variant"] == variant and r["prompt_id"] in PROMPTS]
    s_rows = [r for r in eval_runs(samples, out / "eval_samples.csv") if r["variant"] == variant and r["prompt_id"] in PROMPTS]
    latest = {}
    for r in sorted(e1_rows, key=lambda r: r["run_id"]):
        latest[r["prompt_id"]] = r
    by = {p: [] for p in PROMPTS}
    for r in list(latest.values()) + sorted(s_rows, key=lambda r: r["run_id"]):
        by[r["prompt_id"]].append({"name": r["run_id"], "dir": r["dir"], "accepted": r["accepted"] == "True",
                                   "source": "e1" if r in latest.values() else "e2_samples"})
    return by


def recipe_scenes(recipe: Path) -> dict[str, list[dict]]:
    """prompt -> seeds from ``<recipe>/<prompt>_seed<k>/`` with an ``accepted`` flag in ``record.json``."""
    by = {p: [] for p in PROMPTS}
    for d in sorted(recipe.glob("P*_seed*")) if recipe.exists() else []:
        p = d.name.split("_")[0]
        if p in by:
            rec = json.loads((d / "record.json").read_text()) if (d / "record.json").exists() else {}
            by[p].append({"name": d.name, "dir": str(rec.get("scene_dir", d)), "accepted": bool(rec.get("accepted")), "source": "recipe"})
    return by


def template_scenes(e2_results: Path, e2_out: Path) -> dict[str, list[dict]]:
    recs = json.load(open(e2_results / "template_records.json"))
    acc = sorted((r for r in recs if r["accepted"]), key=lambda r: r["scene"])[:SAMPLES]
    p = acc[0]["prompt_id"] if acc else "P08"
    return {p: [{"name": r["scene"], "dir": str(e2_out / "template" / r["scene"]), "accepted": True, "source": "template"} for r in acc]}


def clustered_boot(pairs_by_prompt: dict[str, list[float]], B: int, rng) -> tuple[float, float, float]:
    ps = [p for p, v in pairs_by_prompt.items() if len(v)]
    if not ps:
        return np.nan, np.nan, np.nan
    point = float(np.median(np.concatenate([pairs_by_prompt[p] for p in ps])))
    meds = [np.median(np.concatenate([pairs_by_prompt[ps[k]] for k in rng.integers(0, len(ps), len(ps))])) for _ in range(B)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="dataset/e1/deepseek-v4.1-flash_cloud")
    ap.add_argument("--samples", default="dataset/e2_samples/deepseek-v4.1-flash_cloud")
    ap.add_argument("--variant", default="constrained")
    ap.add_argument("--recipe", default="out/e2_within/recipe", help="recipe seeds <prompt>_seed<k>/ (skipped if absent)")
    ap.add_argument("--e2-results", default="results/e2_baselines_v3")
    ap.add_argument("--e2-out", default="out/e2_v3")
    ap.add_argument("--out", default="results/e2_within")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.nice(19)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    sets = {"llm": llm_scenes(Path(a.e1), Path(a.samples), a.variant, out),
            "recipe": recipe_scenes(Path(a.recipe)),
            "template": template_scenes(Path(a.e2_results), Path(a.e2_out))}

    # the first SAMPLES *accepted* scenes per prompt (a rejected run does not use up a slot)
    for m in ("llm", "recipe"):
        sets[m] = {p: [s for s in ss if s["accepted"]][:SAMPLES] for p, ss in sets[m].items()}
    complete = all(len(sets["llm"][p]) >= SAMPLES for p in PROMPTS) and all(len(sets["recipe"][p]) >= SAMPLES for p in PROMPTS)
    n_llm = sum(len(v) for v in sets["llm"].values())
    n_rec = sum(len(v) for v in sets["recipe"].values())
    tag = "" if complete else f"PRELIMINARY (LLM {n_llm}/{len(PROMPTS) * SAMPLES} samples, recipe {n_rec}/{len(PROMPTS) * SAMPLES}), "
    title = f"{tag}within-prompt IoU, prompts {', '.join(PROMPTS)}"
    print(title)

    pair_rows, prompt_rows, pairs = [], [], {m: {} for m in METHODS}
    for m in METHODS:
        for p, ss in sets[m].items():
            acc = [s for s in ss[:SAMPLES] if s["accepted"]]
            G = [P.occupancy(s["dir"], a.grid) for s in acc]
            for g in G:
                assert P.iou(g, g) == 1.0
            v = []
            for i, j in combinations(range(len(acc)), 2):
                x = P.iou(G[i], G[j])
                v.append(x)
                pair_rows.append({"method": m, "prompt_id": p, "a": acc[i]["name"], "b": acc[j]["name"], "iou": round(x, 4)})
            pairs[m][p] = np.array(v)
            prompt_rows.append({"method": m, "prompt_id": p, "samples": len(ss[:SAMPLES]), "accepted": len(acc), "pairs": len(v),
                                "median_iou": round(float(np.median(v)), 4) if v else ""})
            print(f"  {LABEL[m]:<8} {p}: {len(acc)}/{len(ss[:SAMPLES])} accepted, {len(v):2d} pairs, median IoU "
                  f"{np.median(v):.3f}" if v else f"  {LABEL[m]:<8} {p}: {len(acc)}/{len(ss[:SAMPLES])} accepted, no pairs")

    summary = {"title": title, "prompts": PROMPTS, "grid": a.grid, "boot": a.boot, "methods": {}}
    for m in METHODS:
        med, lo, hi = clustered_boot(pairs[m], a.boot, rng)
        n_pairs = int(sum(len(v) for v in pairs[m].values()))
        n_prompts = sum(1 for v in pairs[m].values() if len(v))
        summary["methods"][m] = {"prompts": n_prompts, "pairs": n_pairs, "median_iou": med, "ci95": [lo, hi]}

    with open(out / "within_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "prompt_id", "a", "b", "iou"])
        w.writeheader()
        w.writerows(pair_rows)
    with open(out / "per_prompt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "prompt_id", "samples", "accepted", "pairs", "median_iou"])
        w.writeheader()
        w.writerows(prompt_rows)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

    fmt = lambda s: f"{s['median_iou']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]" if s["pairs"] else "n/a"  # noqa: E731
    md = [f"**{title}**", "", "| Method | Prompts | Pairs | Median within-prompt IoU [95% CI, clustered by prompt] |", "|---|---|---|---|"]
    md += [f"| {LABEL[m]} | {s['prompts']} | {s['pairs']} | {fmt(s)} |" for m, s in summary["methods"].items()]
    md += ["", "Lower = more diverse layouts for the same prompt. Template: one prompt, the 1.0 reference."]
    (out / "table_within.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [m for m in METHODS if summary["methods"][m]["pairs"]]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for k, m in enumerate(shown):
        v = np.concatenate([x for x in pairs[m].values() if len(x)])
        if np.ptp(v) > 0:
            vp = ax.violinplot([v], positions=[k], showmedians=True, widths=0.8)
            for b in vp["bodies"]:
                b.set_facecolor(COLOR[m]), b.set_alpha(0.35)
        for p, x in pairs[m].items():
            if len(x):
                ax.scatter(np.full(len(x), k) + rng.uniform(-0.12, 0.12, len(x)), x, s=9, color=COLOR[m], alpha=0.8, lw=0)
        ax.hlines(summary["methods"][m]["median_iou"], k - 0.3, k + 0.3, color="k", lw=1.2)
    ax.set_xticks(range(len(shown)), [LABEL[m] for m in shown])
    ax.set_ylabel("within-prompt IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title(title if complete else "PRELIMINARY: within-prompt IoU", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4_within_iou.pdf")
    fig.savefig(out / "fig4_within_iou.png", dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
