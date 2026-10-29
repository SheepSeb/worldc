"""E2 check: LLM scenes (E1) against the recipe and template baselines.  Layout diversity (pairwise IoU of normalised
occupancy grids), Fig. 4 and the E2 rows of Table 1.

    uv run --with matplotlib python scripts/e2_prelim.py --out results/e2_prelim

Analysis only: no builds, no Isaac Sim, and the process lowers its own priority (nice 19), so it can run beside E1.
Rerun the same command after E1 finishes for the final numbers; until every prompt has a record for the variant, every
output is labelled "PRELIMINARY, n = <k> prompts (...)".

* **Scenes:**
  * E1: accepted runs of ``--variant`` (default constrained) in ``--e1``, one per prompt (the latest if repeated).
  * Recipe: the recipe scenes for the same prompt ids.
  * Template: every accepted template scene (all of one prompt).
  Adherence and plausibility come from ``worldc eval``: run fresh on E1, read from ``--e2-results`` for the baselines.
* **Occupancy grid:** the scene's static nav colliders (as ``check_scene`` builds its grid: same z band and ignored
  categories), minus the building shell (``/World/Building``, walls, doors).  Footprints are drawn in a fixed normalised
  frame: the environment bounds map onto the unit square (x right, y up, no rotation), at ``--grid`` cells a side.  A cell is
  occupied when at least half of it is covered (4 x 4 supersampling).
* **IoU:** every pair of scenes within a method.  CI: 95 % percentile bootstrap over scenes (resample scenes, recompute the
  median over the pairs of distinct scenes drawn).
* **Sanity checks:** a scene against itself gives 1.0; one example pair per method, the one closest to that method's
  median, is printed as ASCII and drawn in ``sanity_pairs.png``.
* **Outputs** in ``--out``: ``fig4_iou_violins.pdf`` (vector) and ``.png``, ``table1_e2.md`` / ``.tex``, ``iou_pairs.csv``,
  ``scenes.csv``, ``summary.json``, ``sanity_pairs.png``, ``eval_e1.csv``.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SHELL_CATEGORIES = {"wall", "door_rollup", "door_swing"}
METHODS = ("e1", "recipe", "template")
LABEL = {"e1": "LLM (E1)", "recipe": "Recipe", "template": "Template"}
COLOR = {"e1": "#1f77b4", "recipe": "#ff7f0e", "template": "#7f7f7f"}


# ------------------------------------------------------------------------------------------------------------ scenes
def _rows(path) -> list[dict]:
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def _num(v):
    return float(v) if v not in ("", None) else np.nan


def e1_scenes(e1: Path, variant: str, out: Path) -> tuple[list[dict], list[dict]]:
    """(accepted scenes one per prompt, every eval row of the variant).  Runs ``worldc eval`` on the E1 records."""
    from worldc.eval.run import main as eval_main

    eval_main(["--runs", str(e1), "--prompts", "prompts/prompts.csv", "--out", str(out / "eval_e1.csv")])
    rows = [r for r in _rows(out / "eval_e1.csv") if r["variant"] == variant and r["prompt_id"]]
    by_prompt = {}
    for r in sorted(rows, key=lambda r: r["run_id"]):  # the latest run of a prompt wins
        by_prompt[r["prompt_id"]] = r
    acc = [r for r in by_prompt.values() if r["accepted"] == "True"]
    return [{"method": "e1", "prompt_id": r["prompt_id"], "name": r["run_id"], "dir": str(e1 / r["scene_dir"]) if not Path(r["scene_dir"]).is_absolute() else r["scene_dir"],
             "adherence": _num(r["adherence"]), "plausibility": _num(r["plausibility"])} for r in sorted(acc, key=lambda r: r["prompt_id"])], list(by_prompt.values())


def baseline_scenes(method: str, e2_results: Path, e2_out: Path, prompts: set | None) -> tuple[list[dict], list[dict]]:
    """(accepted scenes, records) of a baseline, limited to ``prompts`` when given."""
    recs = [r for r in json.load(open(e2_results / f"{method}_records.json")) if prompts is None or r["prompt_id"] in prompts]
    ev = {r["run_id"]: r for r in _rows(e2_results / f"eval_{method}.csv")}
    out = []
    for r in recs:
        if not r["accepted"]:
            continue
        e = ev.get(r["scene"], {})
        out.append({"method": method, "prompt_id": r["prompt_id"], "name": r["scene"], "dir": str(e2_out / method / r["scene"]),
                    "adherence": _num(e.get("adherence")), "plausibility": _num(e.get("plausibility"))})
    return sorted(out, key=lambda s: s["name"]), recs


# -------------------------------------------------------------------------------------------------------- occupancy
def occupancy(scene_dir: str, n: int) -> np.ndarray:
    """Normalised occupancy grid (n x n, row 0 = top = max y) of a scene's static content."""
    from PIL import Image, ImageDraw
    from pxr import Usd

    from worldc.check import nav as NV
    from worldc.check import scene as SC
    from worldc.colliders import collect

    _, usd, graph = SC.locate(scene_dir)
    cols = collect(Usd.Stage.Open(str(usd)))
    cfg = SC.nav_config(graph)
    (x0, x1), (y0, y1) = SC.environment_bounds(graph, cols)
    idx = [i for i in NV.nav_colliders(cols, cfg["z_band"], set(cfg["ignore_categories"]))
           if cols[i].category not in SHELL_CATEGORIES and not cols[i].path.startswith("/World/Building")]
    ss = 4
    img = Image.new("L", (n * ss, n * ss), 0)
    dr = ImageDraw.Draw(img)
    for poly in NV.polygons(cols, idx).values():
        P = np.asarray(poly, float)
        u = (P[:, 0] - x0) / (x1 - x0) * n * ss
        v = (y1 - P[:, 1]) / (y1 - y0) * n * ss
        dr.polygon(list(zip(u, v)), fill=1)
    a = np.asarray(img, float).reshape(n, ss, n, ss).mean(axis=(1, 3))
    return a >= 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def iou_matrix(G: list[np.ndarray]) -> np.ndarray:
    M = np.eye(len(G))
    for i, j in combinations(range(len(G)), 2):
        M[i, j] = M[j, i] = iou(G[i], G[j])
    return M


def boot_median(M: np.ndarray, B: int, rng) -> tuple[float, float, float]:
    n = len(M)
    iu = np.triu_indices(n, 1)
    point = float(np.median(M[iu])) if n > 1 else np.nan
    meds = []
    for _ in range(B):
        k = rng.integers(0, n, n)
        sub = M[np.ix_(k, k)][iu]
        sub = sub[k[iu[0]] != k[iu[1]]]  # pairs of distinct scenes only
        if len(sub):
            meds.append(np.median(sub))
    lo, hi = (np.percentile(meds, [2.5, 97.5]) if meds else (np.nan, np.nan))
    return point, float(lo), float(hi)


def ascii_pair(a, b, step=2) -> str:
    ch = lambda g, r: "".join("#" if g[r:r + step, c:c + step].any() else "." for c in range(0, g.shape[1], step))  # noqa: E731
    return "\n".join(f"{ch(a, r)}   {ch(b, r)}" for r in range(0, a.shape[0], step))


# ------------------------------------------------------------------------------------------------------------- main
def prompt_span(ids: list[str], all_ids: list[str]) -> str:
    if not ids:
        return "none"
    nums = sorted(int(i[1:]) for i in ids)
    missing = [f"P{k:02d}" for k in range(nums[0], nums[-1] + 1) if f"P{k:02d}" not in ids]
    return f"P{nums[0]:02d}–P{nums[-1]:02d}" + (f", without {', '.join(missing)}" if missing else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="dataset/e1/deepseek-v4.1-flash_cloud")
    ap.add_argument("--variant", default="constrained")
    ap.add_argument("--e2-results", default="results/e2_baselines_v3", help="E2 records and eval CSVs")
    ap.add_argument("--e2-out", default="out/e2_v3", help="E2 scene folders (<method>/<scene>)")
    ap.add_argument("--out", default="results/e2_prelim")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.nice(19)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_ids = [r["id"] for r in _rows("prompts/prompts.csv")]

    e1, e1_rows = e1_scenes(Path(a.e1), a.variant, out)
    ids = [s["prompt_id"] for s in e1]
    recipe, recipe_recs = baseline_scenes("recipe", Path(a.e2_results), Path(a.e2_out), set(ids))
    template, template_recs = baseline_scenes("template", Path(a.e2_results), Path(a.e2_out), None)
    scenes = {"e1": e1, "recipe": recipe, "template": template}
    attempted = {"e1": len(e1_rows), "recipe": len(recipe_recs), "template": len(template_recs)}
    final = len(e1_rows) >= len(all_ids)
    tag = ("" if final else "PRELIMINARY, ") + f"n = {len(ids)} prompts ({prompt_span(ids, all_ids)})"
    print(f"{tag}\n{a.variant} E1 records: {len(e1_rows)} of {len(all_ids)} prompts; accepted {len(e1)}")

    grids, mats, stats, pairs = {}, {}, {}, []
    rng = np.random.default_rng(a.seed)
    for m in METHODS:
        grids[m] = [occupancy(s["dir"], a.grid) for s in scenes[m]]
        assert all(iou(g, g) == 1.0 for g in grids[m]), f"{m}: a scene against itself is not 1.0"
        assert all(g.any() for g in grids[m]), f"{m}: empty occupancy grid"
        mats[m] = M = iou_matrix(grids[m])
        med, lo, hi = boot_median(M, a.boot, rng)
        iu = np.triu_indices(len(M), 1)
        stats[m] = {"n_scenes": len(M), "n_pairs": len(iu[0]), "median_iou": med, "ci95": [lo, hi],
                    "mean_iou": float(M[iu].mean()) if len(iu[0]) else None, "occupied_fraction_mean": float(np.mean([g.mean() for g in grids[m]]))}
        for i, j in zip(*iu):
            pairs.append({"method": m, "a": scenes[m][i]["name"], "b": scenes[m][j]["name"], "iou": round(float(M[i, j]), 4)})
    print("self-IoU check: every scene against itself = 1.0 (passed)")

    # ---- example pairs (closest to each method's median), ASCII + picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(len(METHODS), 2, figsize=(5.2, 2.5 * len(METHODS)))
    for r, m in enumerate(METHODS):
        M = mats[m]
        iu = np.triu_indices(len(M), 1)
        k = int(np.argmin(np.abs(M[iu] - stats[m]["median_iou"])))
        i, j = iu[0][k], iu[1][k]
        A, B = scenes[m][i], scenes[m][j]
        print(f"\n{LABEL[m]}: {A['prompt_id']} {A['name']}  vs  {B['prompt_id']} {B['name']}  IoU {M[i, j]:.3f} "
              f"(median {stats[m]['median_iou']:.3f})\n{ascii_pair(grids[m][i], grids[m][j])}")
        for c, (s, g) in enumerate(((A, grids[m][i]), (B, grids[m][j]))):
            ax[r, c].imshow(g, cmap="Greys", vmin=0, vmax=1, interpolation="nearest", extent=(0, 1, 0, 1))
            ax[r, c].set_title(f"{LABEL[m]}: {s['prompt_id']} {s['name'][-12:]}", fontsize=8)
            ax[r, c].set_xticks([0, 1]), ax[r, c].set_yticks([0, 1])
        ax[r, 0].set_ylabel(f"IoU {M[i, j]:.2f}")
    fig.suptitle(f"Example pairs, normalised frame ({tag})", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sanity_pairs.png", dpi=130)
    plt.close(fig)

    # ---- Fig. 4: violins of pairwise IoU
    fig, axv = plt.subplots(figsize=(3.4, 2.6))
    data = [mats[m][np.triu_indices(len(mats[m]), 1)] for m in METHODS]
    vp = axv.violinplot(data, showmedians=False, showextrema=False, widths=0.8)
    for body, m in zip(vp["bodies"], METHODS):
        body.set_facecolor(COLOR[m]), body.set_edgecolor("black"), body.set_linewidth(0.5), body.set_alpha(0.6)
    for x, m in enumerate(METHODS, 1):
        s = stats[m]
        axv.vlines(x, *s["ci95"], color="black", lw=1.5)
        axv.plot(x, s["median_iou"], "o", color="white", mec="black", ms=4, zorder=3)
    axv.set_xticks(range(1, len(METHODS) + 1), [f"{LABEL[m]}\n{stats[m]['n_scenes']} scenes" for m in METHODS])
    axv.set_ylabel("Pairwise layout IoU")
    axv.set_ylim(0, 1)
    axv.grid(axis="y", lw=0.3, alpha=0.5)
    axv.set_title(tag, fontsize=7, color="#b00" if not final else "black")
    fig.tight_layout()
    fig.savefig(out / "fig4_iou_violins.pdf")
    fig.savefig(out / "fig4_iou_violins.png", dpi=160)
    plt.close(fig)

    # ---- Table 1, E2 rows
    def mean(m, k):
        v = [s[k] for s in scenes[m] if np.isfinite(s[k])]
        return float(np.mean(v)) if v else np.nan

    rows = []
    for m in METHODS:
        s = stats[m]
        rows.append({"method": LABEL[m], "prompts": len({x["prompt_id"] for x in scenes[m]}), "attempted": attempted[m], "accepted": len(scenes[m]),
                     "acceptance": len(scenes[m]) / attempted[m] if attempted[m] else np.nan,
                     "adherence": mean(m, "adherence"), "plausibility": mean(m, "plausibility"),
                     "median_iou": s["median_iou"], "ci95": s["ci95"]})
    head = ["Method", "Prompts", "Acceptance", "Adherence", "Plausibility", "Median IoU [95% CI]"]
    fmt = lambda r: [r["method"], str(r["prompts"]), f"{r['acceptance']:.0%} ({r['accepted']}/{r['attempted']})", f"{r['adherence']:.2f}",  # noqa: E731
                     f"{r['plausibility']:.2f}", f"{r['median_iou']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"]
    note = (f"{tag}. E1 = {a.variant} prompts, accepted = static checks + final physics check. Recipe = the same prompt ids; "
            "template = all seeds of its one prompt. Baseline acceptance is the static `check_scene` (their Isaac physics "
            "check is queued). Adherence and plausibility: means over accepted scenes (`worldc eval`). IoU: pairs of "
            "accepted scenes within a method, normalised occupancy grids; lower = more diverse layouts.")
    md = [f"**Table 1 (E2 rows), {tag}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(fmt(r)) + " |" for r in rows] + ["", note]
    (out / "table1_e2.md").write_text("\n".join(md) + "\n")
    tex = [f"% {tag}", r"\begin{tabular}{lrrrrr}", r"\toprule", " & ".join(head).replace("%", r"\%") + r" \\", r"\midrule"]
    tex += [" & ".join(c.replace("%", r"\%") for c in fmt(r)) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
    (out / "table1_e2.tex").write_text("\n".join(tex) + "\n")

    with open(out / "iou_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "a", "b", "iou"])
        w.writeheader(), w.writerows(pairs)
    with open(out / "scenes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "prompt_id", "name", "dir", "adherence", "plausibility", "occupied_fraction"])
        w.writeheader()
        for m in METHODS:
            for s, g in zip(scenes[m], grids[m]):
                w.writerow({**s, "occupied_fraction": round(float(g.mean()), 4)})
    (out / "summary.json").write_text(json.dumps({"label": tag, "preliminary": not final, "variant": a.variant, "e1": a.e1,
                                                  "e1_records": len(e1_rows), "prompts": ids, "grid": a.grid, "boot": a.boot,
                                                  "iou": stats, "table1": rows}, indent=1, default=float))
    print("\n" + "\n".join(md))
    for m in METHODS:
        s = stats[m]
        print(f"{LABEL[m]:10s} n scenes {s['n_scenes']:3d}  n pairs {s['n_pairs']:4d}  median IoU {s['median_iou']:.3f} "
              f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
