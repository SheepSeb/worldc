"""E2 within-prompt study, v2: spatial adherence + within-prompt IoU on native-metre grids + acceptance, per method.

    uv run --with matplotlib python scripts/e2_within_v2.py --out results/e2_within_v2

Analysis only (no builds, no Isaac Sim, nice 19).  Methods: the three LLMs (first 5 accepted constrained runs of the prompt among the
E1 run and the ``e2_samples`` runs), recipe v4 (``--recipe-v4``, the prompt's spatial rules as hard constraints), recipe v3
(``--recipe-v3``, seed-drawn layout without rules, reference) and the template.

* **Frame:** every scene is scored in a south-west-origin frame from its own room bounds (``environment.x_range`` / ``y_range``):
  x east, y north, (0, 0) the SW corner.  Walls: south y = 0, north y = depth, west x = 0, east x = width.
* **Rules:** ``prompts/spatial_rules_e2_within.json`` (thresholds in its ``_thresholds``; all distances edge to edge).  A rule with
  several instances scores the fraction of its instances satisfied; a scene's spatial adherence is the mean over the rules it has
  instances for (a rule whose objects are absent is not scored: counts are the count adherence's job).  ``rules_scored`` /
  ``rules_total`` are kept per scene.
* **Footprints:** an instance is the union of the axis-aligned boxes of its colliders (``instances[*].root``).
* **IoU:** occupancy of the static nav colliders (shell excluded, as ``e2_prelim.py``) on a 0.1 m grid in the SW frame.  The grid of a
  prompt spans the largest room among its scenes (cells outside a smaller room are empty), so footprints are never stretched;
  scenes whose room differs from the prompt's stated size are flagged in ``scenes.csv`` (``room_ok``).
* **CI:** 95 % percentile bootstrap clustered by prompt (resample prompts, pool their pairs / scenes, take the median / mean).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_prelim as P  # noqa: E402  (SHELL_CATEGORIES, _rows)

PROMPTS = ("P05", "P08", "P15", "P16", "P23")
SAMPLES = 5
RULES_PATH = Path("prompts/spatial_rules_e2_within.json")
CLASSES = {
    "bench": {"workbench_steel", "work_table", "packing_table_heavy"},
    "shelf": {"parts_shelf", "shelf_rack"},
    "cell": {"robot_cell"},
    "belt": {"conveyor_belt"},
    "pallet": {"pallet", "pallet_euro", "pallet_block", "pallet_load"},
    "jack": {"pallet_jack", "gen/pallet_jack"},
    "scrap": {"scrap_bin", "scrap_bin_wheeled", "wheelie_bin"},
    "washer": {"washing_machine"},
    "door": {"swing_door", "rollup_door"},
}
ASSET_CLASS = {a: c for c, s in CLASSES.items() for a in s}


# ------------------------------------------------------------------------------------------------------------ geometry
def gap(a, b):
    """Edge-to-edge distance of two axis-aligned rectangles (x0, y0, x1, y1)."""
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0.0), max(a[1] - b[3], b[1] - a[3], 0.0))


def pt_gap(r, p):
    return math.hypot(max(r[0] - p[0], p[0] - r[2], 0.0), max(r[1] - p[1], p[1] - r[3], 0.0))


def centre(r):
    return (r[0] + r[2]) / 2, (r[1] + r[3]) / 2


def wall_gap(r, wall, W, D):
    return {"west": r[0], "east": W - r[2], "south": r[1], "north": D - r[3]}[wall]


def door_gap(r, d, W, D):
    """Distance from a rectangle to a door opening (a segment on the wall line)."""
    c, w = d["c"], d["w"]
    pts = []
    for k in range(11):
        t = c - w / 2 + w * k / 10
        pts.append({"west": (0.0, t), "east": (W, t), "south": (t, 0.0), "north": (t, D)}[d["wall"]])
    return min(pt_gap(r, p) for p in pts)


# ------------------------------------------------------------------------------------------------------------ scene
def load_scene(scene_dir: str) -> dict:
    """Classified instance footprints, doors and the occupancy polygons of a built scene, all in the SW-origin frame."""
    from pxr import Usd

    from worldc.check import nav as NV
    from worldc.check import scene as SC
    from worldc.colliders import collect

    _, usd, graph = SC.locate(scene_dir)
    cols = collect(Usd.Stage.Open(str(usd)))
    (x0, x1), (y0, y1) = SC.environment_bounds(graph, cols)
    W, D = x1 - x0, y1 - y0
    by_root = {}
    for i, c in enumerate(cols):
        lo, hi = c.obb.aabb()
        by_root.setdefault("/".join(c.path.split("/")[:4]), []).append((lo[0], lo[1], hi[0], hi[1]))
    exits = {e["id"]: e for e in (graph.get("safety") or {}).get("exits", [])}
    inst = []
    for key, v in graph["instances"].items():
        cls = ASSET_CLASS.get(v["asset"])
        if cls is None:
            continue
        root = v.get("root") or ""
        boxes = [b for k, bs in by_root.items() if k == root or root.startswith(k + "/") or k.startswith(root + "/") for b in bs]
        if not boxes:
            continue
        r = (min(b[0] for b in boxes) - x0, min(b[1] for b in boxes) - y0, max(b[2] for b in boxes) - x0, max(b[3] for b in boxes) - y0)
        item = {"cls": cls, "asset": v["asset"], "key": key, "rect": r, "spec_id": v.get("spec_id")}
        if cls == "door":
            px, py = v["pose"]["xyz"][0] - x0, v["pose"]["xyz"][1] - y0
            wall = min({"west": px, "east": W - px, "south": py, "north": D - py}.items(), key=lambda kv: kv[1])[0]
            near_exit = min(exits.values(), key=lambda e: math.dist(e["xy"], v["pose"]["xyz"][:2]), default=None)
            width = near_exit["width_m"] if near_exit and math.dist(near_exit["xy"], v["pose"]["xyz"][:2]) < 2.0 else (2.4 if v["asset"] == "rollup_door" else 1.15)
            item["wall"], item["c"], item["w"] = wall, (py if wall in ("west", "east") else px), float(width)
        inst.append(item)
    belts = []
    for cv in graph.get("conveyors") or []:
        b = cv["bbox"]
        d = cv.get("direction_world") or [0, 1, 0]
        belts.append({"rect": (b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0), "dir": (d[0], d[1])})
    cfg = SC.nav_config(graph)
    idx = [i for i in NV.nav_colliders(cols, cfg["z_band"], set(cfg["ignore_categories"]))
           if cols[i].category not in P.SHELL_CATEGORIES and not cols[i].path.startswith("/World/Building")]
    polys = [np.asarray(p, float) - np.array([x0, y0]) for p in NV.polygons(cols, idx).values()]
    return {"W": W, "D": D, "inst": inst, "belts": belts, "polys": polys, "doors": [i for i in inst if i["cls"] == "door"]}


# ------------------------------------------------------------------------------------------------------------ rules
def score_rules(sc: dict, pid: str, rules: dict) -> dict[str, dict]:
    """rule id -> {"n": instances, "sat": satisfied, "items": [(rect, ok)], "missing": bool}, for every rule of the prompt.

    A rule whose objects are absent from the scene scores 0 (``missing``): the prompt names them, the scene lacks them."""
    thr = rules["_thresholds"]
    near, along, against = thr["near_m"], thr["along_wall_m"], thr["against_wall_m"]
    W, D = sc["W"], sc["D"]
    of = lambda cls: [i["rect"] for i in sc["inst"] if i["cls"] == cls]  # noqa: E731
    bench, shelf, cell, pallet, jack, scrap, washer = (of(c) for c in ("bench", "shelf", "cell", "pallet", "jack", "scrap", "washer"))
    belts, doors = sc["belts"], sc["doors"]
    out = {}

    def put(rid, objs, flags):
        flags = [bool(f) for f in flags]
        out[rid] = {"n": len(flags), "sat": float(sum(flags)), "items": list(zip(objs, flags)), "missing": not flags}

    def row_clusters(rects, tol=0.4):
        rows = []
        for r in sorted(rects, key=lambda r: centre(r)[1]):
            if rows and abs(centre(r)[1] - np.mean([centre(q)[1] for q in rows[-1]])) <= tol:
                rows[-1].append(r)
            else:
                rows.append([r])
        return [g for g in rows if len(g) >= 2]

    on_pallet = lambda p, o: p[0] <= centre(o)[0] <= p[2] and p[1] <= centre(o)[1] <= p[3]  # noqa: E731
    belt_r = [v["rect"] for v in belts]
    for rule in rules[pid]["rules"]:
        rid = rule["id"]
        if rid == "cells.side_by_side":
            beside = [len(cell) > 1 and any(o is not c and gap(c, o) <= near and (abs(centre(c)[0] - centre(o)[0]) <= 1.0 or abs(centre(c)[1] - centre(o)[1]) <= 1.0)
                                            for o in cell) for c in cell]
            put(rid, cell, beside)
        elif rid == "benches.long_walls":
            on = {w: [b for b in bench if wall_gap(b, w, W, D) <= along] for w in ("north", "south")}
            split = bool(bench) and all(len(v) >= 2 for v in on.values())
            put(rid, bench, [split and any(b in on[w] for w in on) for b in bench])
        elif rid == "shelving.middle":
            put(rid, shelf, [abs(centre(s_)[0] - W / 2) <= W / 6 and abs(centre(s_)[1] - D / 2) <= D / 6 for s_ in shelf])
        elif rid in ("staging.by_door", "palletjack.by_door", "pallets.by_west_door", "scrap.near_door"):
            use = {"staging.by_door": pallet, "palletjack.by_door": jack, "pallets.by_west_door": pallet, "scrap.near_door": scrap}[rid]
            ds = [d for d in doors if d["wall"] == "west"] if rid == "pallets.by_west_door" else doors
            put(rid, use, [any(door_gap(r, d, W, D) <= near for d in ds) for r in use])
        elif rid == "conveyor.axis_east":
            put(rid, belt_r, [abs(b["dir"][0]) > abs(b["dir"][1]) and b["dir"][0] > 0 for b in belts])
        elif rid in ("benches.conveyor_north", "benches.north_of_conveyor"):
            put(rid, bench, [any(centre(b)[1] > centre(v)[1] and v[0] <= centre(b)[0] <= v[2] and gap(b, v) <= near for v in belt_r) for b in bench])
        elif rid == "cell.conveyor_east_end":
            put(rid, cell, [any(centre(c)[0] > centre(v)[0] and gap(c, v) <= near for v in belt_r) for c in cell])
        elif rid in ("shelving.east_wall", "shelving.north_wall", "benches.north_wall"):
            wall, lim = {"shelving.east_wall": ("east", along if pid == "P08" else against), "shelving.north_wall": ("north", against),
                         "benches.north_wall": ("north", against)}[rid]
            objs = bench if rid.startswith("benches") else shelf
            put(rid, objs, [wall_gap(r, wall, W, D) <= lim for r in objs])
        elif rid == "recycling.next_to_shelving":
            put(rid, scrap, [any(gap(r, s_) <= near for s_ in shelf) for r in scrap])
        elif rid == "cells.north_wall":
            put(rid, cell, [wall_gap(c, "north", W, D) <= along for c in cell])
        elif rid == "benches.three_rows":
            rows = row_clusters(bench)  # three rows along x; a bench passes when its row's centre is in the central half of y
            good = [r for g in rows if abs(np.mean([centre(q)[1] for q in g]) - D / 2) <= D / 4 for r in g]
            put(rid, bench, [len(rows) == 3 and r in good for r in bench])
        elif rid == "shelving.east_west":
            put(rid, shelf, [min(wall_gap(s_, "east", W, D), wall_gap(s_, "west", W, D)) <= along for s_ in shelf])
        elif rid == "conveyor.from_south_door":
            souths = [d for d in doors if d["wall"] == "south"]
            put(rid, belt_r, [(v[2] - v[0]) < (v[3] - v[1]) and any(door_gap(v, d, W, D) <= near for d in souths) for v in belt_r])
        elif rid == "conveyor.to_central_row":
            rows = sorted(row_clusters(bench), key=lambda g: np.mean([centre(q)[1] for q in g]))
            mid = rows[1] if len(rows) == 3 else []
            put(rid, belt_r, [any(gap(v, b) <= near for b in mid) for v in belt_r])
        elif rid == "washer.own_pallet":
            put(rid, washer, [sum(1 for o in washer if any(on_pallet(p, o) and on_pallet(p, w) for p in pallet)) == 1 and any(on_pallet(p, w) for p in pallet)
                              for w in washer])
        elif rid == "conveyor.pinned":
            tol, pin = thr["pinned_tol_m"], rule["pinned"]
            put(rid, belt_r, [abs(v[0] - pin["x"][0]) <= tol and abs(v[2] - pin["x"][1]) <= tol and abs(centre(v)[1] - pin["y"]) <= tol for v in belt_r])
        elif rid == "cell.pinned":
            tol, pin = thr["pinned_tol_m"], rule["pinned"]
            put(rid, cell, [c[0] >= pin["x"][0] - tol and c[2] <= pin["x"][1] + tol and c[1] >= pin["y"][0] - tol and c[3] <= pin["y"][1] + tol
                            and abs(centre(c)[0] - sum(pin["x"]) / 2) <= tol and abs(centre(c)[1] - sum(pin["y"]) / 2) <= tol for c in cell])
        elif rid == "conveyor.ends_at_cell":
            put(rid, belt_r, [any(gap(v, c) <= thr["conveyor_end_to_cell_m"] for c in cell) for v in belt_r])
        elif rid == "recycling.south_of_cell":
            put(rid, scrap, [any(centre(r)[1] < c[1] and c[0] - 1.0 <= centre(r)[0] <= c[2] + 1.0 for c in cell) for r in scrap])
        else:
            raise KeyError(rid)
    return out


def rule_score(r: dict) -> float:
    return 0.0 if r["missing"] else r["sat"] / r["n"]


def adherence(res: dict) -> tuple[float, int, int]:
    """(mean rule score, rules with objects present, rules in total); an absent object scores 0."""
    if not res:
        return float("nan"), 0, 0
    return float(np.mean([rule_score(r) for r in res.values()])), sum(1 for r in res.values() if not r["missing"]), len(res)


# ------------------------------------------------------------------------------------------------------------ occupancy
def occupancy(polys, W_max, D_max, res=0.1) -> np.ndarray:
    from PIL import Image, ImageDraw

    nx, ny, ss = int(round(W_max / res)), int(round(D_max / res)), 4
    img = Image.new("L", (nx * ss, ny * ss), 0)
    dr = ImageDraw.Draw(img)
    for p in polys:
        dr.polygon([(x / res * ss, (D_max - y) / res * ss) for x, y in p], fill=1)
    return np.asarray(img, float).reshape(ny, ss, nx, ss).mean(axis=(1, 3)) >= 0.5


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def boot(per_prompt: dict[str, np.ndarray], fn, B, rng):
    ps = [p for p, v in per_prompt.items() if len(v)]
    if not ps:
        return float("nan"), float("nan"), float("nan")
    point = float(fn(np.concatenate([per_prompt[p] for p in ps])))
    vals = [fn(np.concatenate([per_prompt[ps[k]] for k in rng.integers(0, len(ps), len(ps))])) for _ in range(B)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


# ------------------------------------------------------------------------------------------------------------ scene sets
def llm_set(model: str, eval_dir: Path, variant="constrained"):
    """prompt -> first SAMPLES accepted scenes (E1 run first, then the e2_samples runs in run order); plus acceptance over all runs."""
    e1, sm = Path("dataset/e1") / model, Path("dataset/e2_samples") / model
    rows = []
    for runs, f in ((e1, "eval_e1.csv"), (sm, "eval_samples.csv")):
        for r in P._rows(eval_dir / model / f):
            if r["variant"] == variant and r["prompt_id"] in PROMPTS:
                r["dir"] = r["scene_dir"] if Path(r["scene_dir"]).is_absolute() else str(runs / r["scene_dir"])
                r["src"] = "e1" if runs == e1 else "e2_samples"
                rows.append(r)
    e1_latest = {}
    for r in sorted((r for r in rows if r["src"] == "e1"), key=lambda r: r["run_id"]):
        e1_latest[r["prompt_id"]] = r
    runs = list(e1_latest.values()) + sorted((r for r in rows if r["src"] == "e2_samples"), key=lambda r: r["run_id"])
    by = {p: [{"name": r["run_id"], "dir": r["dir"]} for r in runs if r["prompt_id"] == p and r["accepted"] == "True"][:SAMPLES] for p in PROMPTS}
    acc = {"runs": len(runs), "accepted": sum(r["accepted"] == "True" for r in runs)}
    return by, acc


def recipe_set(root: Path):
    by, att, ok_samples = {p: [] for p in PROMPTS}, [], 0
    for d in sorted(root.glob("P*_seed*")):
        pid = d.name.split("_")[0]
        rec = json.loads((d / "record.json").read_text()) if (d / "record.json").exists() else {}
        att.append((len(rec.get("history", [])), bool(rec.get("accepted")), rec.get("final_seed")))
        if pid in by and rec.get("accepted"):
            by[pid].append({"name": d.name, "dir": str(rec.get("scene_dir", d)), "final_seed": rec.get("final_seed")})
    n_att = sum(a for a, _, _ in att)
    acc = {"runs": len(att), "accepted": sum(ok for _, ok, _ in att), "attempts": n_att, "attempts_ok": sum(ok for _, ok, _ in att)}
    return {p: v[:SAMPLES] for p, v in by.items()}, acc


def template_set(e2_results: Path, e2_out: Path):
    recs = json.load(open(e2_results / "template_records.json"))
    ok = sorted((r for r in recs if r["accepted"]), key=lambda r: r["scene"])[:SAMPLES]
    pid = ok[0]["prompt_id"] if ok else "P08"
    by = {p: [] for p in PROMPTS}
    by[pid] = [{"name": r["scene"], "dir": str(e2_out / "template" / r["scene"])} for r in ok]
    return by, {"runs": len(recs), "accepted": sum(r["accepted"] for r in recs)}


# ------------------------------------------------------------------------------------------------------------ maps
def draw_maps(outdir: Path, spec: str, scenes, cache, detail, rules, label):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    outdir.mkdir(parents=True, exist_ok=True)
    cls_color = {"bench": "#8c6d31", "shelf": "#7b4173", "cell": "#3182bd", "belt": "#e6550d", "pallet": "#969696", "jack": "#636363",
                 "scrap": "#31a354", "washer": "#17becf", "door": "#000000"}
    for item in [t for t in spec.split(",") if t]:
        model, pid = item.split(":")
        sel = next((s for s in scenes if s["method"] == model and s["prompt_id"] == pid), None)
        if sel is None:
            continue
        sc, res = cache[sel["dir"]], detail[(model, pid, sel["name"])]
        W, D = sc["W"], sc["D"]
        fig, (ax, tx) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [2.2, 1]})
        ax.add_patch(Rectangle((0, 0), W, D, fill=False, lw=2, ec="k"))
        for poly in sc["polys"]:
            ax.add_patch(Polygon(poly, closed=True, fc="#dddddd", ec="none"))
        for i in sc["inst"]:
            if i["cls"] == "door":
                d = i
                xs, ys = {"west": ([0, 0], [d["c"] - d["w"] / 2, d["c"] + d["w"] / 2]), "east": ([W, W], [d["c"] - d["w"] / 2, d["c"] + d["w"] / 2]),
                          "south": ([d["c"] - d["w"] / 2, d["c"] + d["w"] / 2], [0, 0]), "north": ([d["c"] - d["w"] / 2, d["c"] + d["w"] / 2], [D, D])}[d["wall"]]
                ax.plot(xs, ys, color="#d62728", lw=5, solid_capstyle="butt")
                continue
            r = i["rect"]
            ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fc=cls_color[i["cls"]], alpha=0.35, ec="none"))
        for v in sc["belts"]:
            r = v["rect"]
            ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fc=cls_color["belt"], alpha=0.35, ec="none"))
        lines = []
        for n, rule in enumerate(rules[pid]["rules"], 1):
            r_ = res[rule["id"]]
            if "pinned" in rule:
                pin = rule["pinned"]
                x0, x1 = pin["x"] if isinstance(pin["x"], list) else (0, 0)
                y0, y1 = pin["y"] if isinstance(pin["y"], list) else (pin["y"], pin["y"])
                ax.add_patch(Rectangle((x0, y0), x1 - x0, max(y1 - y0, 0.05), fill=False, ls="--", ec="#1f77b4", lw=1))
            for rect, ok in r_["items"]:
                col = "#2ca02c" if ok else "#d62728"
                ax.add_patch(Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], fill=False, ec=col, lw=2))
                ax.text((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, str(n), ha="center", va="center", fontsize=8, color=col, weight="bold")
            tag = "MISSING (0)" if r_["missing"] else f"{int(r_['sat'])}/{r_['n']}"
            lines.append((f"{n:>2}  {rule['id']}  {tag}", "#d62728" if r_["missing"] or r_["sat"] < r_["n"] else "#2ca02c"))
        ax.set_xlim(-0.5, W + 0.5), ax.set_ylim(-0.5, D + 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{label.get(model, model)}  {pid}  {sel['name']}   spatial adherence {sel['spatial']:.2f}", fontsize=9)
        ax.set_xlabel("x east (m), SW corner = origin; red bar = door; dashed blue = pinned extent", fontsize=7)
        tx.axis("off")
        for k, (t, c) in enumerate(lines):
            tx.text(0.0, 0.97 - k * 0.055, t, fontsize=8, family="monospace", color=c, va="top")
        tx.text(0.0, 0.97 - (len(lines) + 1) * 0.055, "grey = nav obstacles; green / red outline = instance passes / fails the numbered rule\n"
                "colours: bench brown, shelf purple, cell blue, belt orange, pallet grey, scrap green", fontsize=7, va="top")
        fig.tight_layout()
        fig.savefig(outdir / f"{model}_{pid}_{sel['name']}.png", dpi=130)
        plt.close(fig)
        print("map:", outdir / f"{model}_{pid}_{sel['name']}.png")


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="claude-sonnet-5-5,deepseek-v4.1-flash_cloud,glm-5.3-flash_cloud")
    ap.add_argument("--eval-dir", default="results/e2_within", help="per-model eval_e1.csv / eval_samples.csv from e2_within.py")
    ap.add_argument("--recipe-v4", default="out/e2_within/recipe_v4")
    ap.add_argument("--recipe-v3", default="out/e2_within/recipe")
    ap.add_argument("--e2-results", default="results/e2_baselines_v3")
    ap.add_argument("--e2-out", default="out/e2_v3")
    ap.add_argument("--out", default="results/e2_within_v2")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--maps", default="deepseek-v4.1-flash_cloud:P23,claude-sonnet-5-5:P15,glm-5.3-flash_cloud:P08,glm-5.3-flash_cloud:P05,claude-sonnet-5-5:P16",
                    help="model:prompt pairs (the first sample of each) to draw as rule-overlay maps; '' for none")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.nice(19)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rules = json.loads(RULES_PATH.read_text())
    prompt_rows = {r["id"]: r for r in csv.DictReader(open("prompts/prompts.csv"))}

    methods = {}
    for m in a.models.split(","):
        methods[m] = ("llm", *llm_set(m, Path(a.eval_dir)))
    methods["recipe_v4"] = ("recipe", *recipe_set(Path(a.recipe_v4)))
    methods["recipe_v3"] = ("recipe", *recipe_set(Path(a.recipe_v3)))
    methods["template"] = ("template", *template_set(Path(a.e2_results), Path(a.e2_out)))

    scenes, cache, detail = [], {}, {}
    for m, (_, by, _) in methods.items():
        for p, ss in by.items():
            for s in ss:
                if s["dir"] not in cache:
                    cache[s["dir"]] = load_scene(s["dir"])
                sc = cache[s["dir"]]
                pr = prompt_rows[p]
                res = score_rules(sc, p, rules) if p in rules else {}
                detail[(m, p, s["name"])] = res
                adh, present, total = adherence(res)
                scenes.append({"method": m, "prompt_id": p, "name": s["name"], "dir": s["dir"], "W": round(sc["W"], 3), "D": round(sc["D"], 3),
                               "room_ok": abs(sc["W"] - float(pr["width_m"])) < 0.05 and abs(sc["D"] - float(pr["depth_m"])) < 0.05,
                               "spatial": adh, "rules_present": present, "rules_total": total,
                               "rules": {k: [r["n"], r["sat"], r["missing"]] for k, r in res.items()}, "final_seed": s.get("final_seed", "")})
    print(f"{len(scenes)} scenes loaded")

    # recipe footprints identical across seeds (and equal to the stated size)?
    foot = {}
    for s in scenes:
        if s["method"].startswith("recipe"):
            foot.setdefault((s["method"], s["prompt_id"]), set()).add((s["W"], s["D"]))
    foot_ok = {f"{k[0]}/{k[1]}": sorted(v) for k, v in foot.items()}
    print("recipe footprints per prompt:", foot_ok)

    # IoU on native grids: one grid extent per prompt (the largest room among all its scenes)
    ext = {p: (max([s["W"] for s in scenes if s["prompt_id"] == p] or [0]), max([s["D"] for s in scenes if s["prompt_id"] == p] or [0])) for p in PROMPTS}
    pair_rows, ious, spat = [], {m: {p: [] for p in PROMPTS} for m in methods}, {m: {p: [] for p in PROMPTS} for m in methods}
    for m in methods:
        for p in PROMPTS:
            ss = [s for s in scenes if s["method"] == m and s["prompt_id"] == p]
            G = [occupancy(cache[s["dir"]]["polys"], *ext[p]) for s in ss]
            for i, j in combinations(range(len(ss)), 2):
                v = iou(G[i], G[j])
                ious[m][p].append(v)
                pair_rows.append({"method": m, "prompt_id": p, "a": ss[i]["name"], "b": ss[j]["name"], "iou": round(v, 4),
                                  "both_room_ok": ss[i]["room_ok"] and ss[j]["room_ok"]})
            spat[m][p] = [s["spatial"] for s in ss if not math.isnan(s["spatial"])]
    summ = {}
    for m, (kind, by, acc) in methods.items():
        i_med, i_lo, i_hi = boot({p: np.array(v) for p, v in ious[m].items()}, np.median, a.boot, rng)
        s_mean, s_lo, s_hi = boot({p: np.array(v) for p, v in spat[m].items()}, np.mean, a.boot, rng)
        ms = [s for s in scenes if s["method"] == m]
        summ[m] = {"kind": kind, "scenes": len(ms), "prompts": sum(1 for p in PROMPTS if ious[m][p]), "pairs": sum(len(v) for v in ious[m].values()),
                   "iou": [i_med, i_lo, i_hi], "spatial": [s_mean, s_lo, s_hi], "acceptance": acc,
                   "room_mismatch_scenes": sum(1 for s in ms if not s["room_ok"]),
                   "rules_present_mean": float(np.mean([s["rules_present"] for s in ms])) if ms else 0.0}

    with open(out / "within_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        w.writeheader(), w.writerows(pair_rows)
    with open(out / "scenes.csv", "w", newline="") as f:
        cols = ["method", "prompt_id", "name", "dir", "W", "D", "room_ok", "spatial", "rules_present", "rules_total", "final_seed", "rules"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(), w.writerows({**s, "rules": json.dumps(s["rules"])} for s in scenes)
    per_rule = {}
    for s in scenes:
        for k, (n, s_, miss) in s["rules"].items():
            d = per_rule.setdefault((s["method"], s["prompt_id"], k), [0, 0.0, 0, 0])
            d[0] += n; d[1] += s_; d[2] += 1; d[3] += int(miss)  # noqa: E702
    with open(out / "per_rule.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "prompt_id", "rule", "scenes", "scenes_object_missing", "instances", "satisfied", "mean_rule_score"])
        for (m, p, k), (n, s_, c, miss) in sorted(per_rule.items()):
            sc_ = [rule_score({"n": v["rules"][k][0], "sat": v["rules"][k][1], "missing": v["rules"][k][2]}) for v in scenes
                   if v["method"] == m and v["prompt_id"] == p]
            w.writerow([m, p, k, c, miss, n, s_, round(float(np.mean(sc_)), 3)])
    json.dump({"methods": summ, "recipe_footprints": foot_ok, "grid_extent_m": ext}, open(out / "summary.json", "w"), indent=1, default=float)

    fmt = lambda t: "n/a" if math.isnan(t[0]) else f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"  # noqa: E731
    label = {"recipe_v4": "Recipe v4 (rules)", "recipe_v3": "Recipe v3 (no rules)", "template": "Template", "claude-sonnet-5-5": "Sonnet 5.5",
             "deepseek-v4.1-flash_cloud": "DeepSeek v4.1 flash", "glm-5.3-flash_cloud": "GLM 5.3 flash"}
    md = ["**E2 within-prompt study: layout diversity, spatial adherence and acceptance** (prompts " + ", ".join(PROMPTS) + ")", "",
          "| Method | Samples | Within-prompt IoU [95% CI] | Spatial adherence [95% CI] | LLM acceptance per run | Recipe acceptance per attempt | Accepted samples per final build (static) | Recipe samples also passing Isaac Sim physics |",
          "|---|---|---|---|---|---|---|---|"]
    phys = {}
    if (out / "physics_recipes.csv").exists():  # from scripts/e2_physics_queue.py (e2_physics.json of every recipe scene)
        for r in csv.DictReader(open(out / "physics_recipes.csv")):
            m_ = {"recipe v4 (rules)": "recipe_v4", "recipe v3": "recipe_v3"}[r["method"]]
            d_ = phys.setdefault(m_, [0, 0])
            d_[0] += r["physics_status"] == "ok"
            d_[1] += 1
    for m, s_ in summ.items():
        acc = s_["acceptance"]
        per_run = f"{acc['accepted']}/{acc['runs']} ({acc['accepted'] / acc['runs']:.0%})" if s_["kind"] == "llm" else "n/a"
        per_att = (f"{acc['attempts_ok']}/{acc['attempts']} ({acc['attempts_ok'] / acc['attempts']:.0%})" if "attempts" in acc
                   else f"{acc['accepted']}/{acc['runs']} ({acc['accepted'] / acc['runs']:.0%})" if s_["kind"] == "template" else "n/a")
        per_samp = f"{acc['accepted']}/{acc['runs']}" if s_["kind"] == "recipe" else "n/a"
        pp = f"{phys[m][0]}/{phys[m][1]}" if m in phys else "n/a"
        md.append(f"| {label.get(m, m)} | {s_['scenes']} | {fmt(s_['iou'])} | {fmt(s_['spatial'])} | {per_run} | {per_att} | {per_samp} | {pp} |")
    md += ["", "IoU: median over pairs of same-prompt samples on native-metre 0.1 m grids (lower = more diverse). Spatial adherence: mean over scenes of the mean rule "
           "score (1 = every instance of the rule satisfied; a rule whose objects are absent from the scene scores 0), `prompts/spatial_rules_e2_within.json`. "
           "The template is scored on P08 only (its prompt), so its spatial CI is degenerate. LLM acceptance: accepted constrained runs of these five prompts (E1 run + extra "
           "samples), per run. Recipe acceptance: per build attempt, counting every resampled seed; the last column is the number of the 25 samples that ended accepted. "
           "The template builds once per seed. LLM acceptance already includes the final Isaac Sim physics check; the recipe's static acceptance does not, so the last "
           "column gives the recipe samples that also pass it (`physics_recipes.csv`)."]
    (out / "table_within_v2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [m for m in methods if summ[m]["pairs"]]
    colors = {m: plt.get_cmap("tab10")(k) for k, m in enumerate(["template", "recipe_v3", "recipe_v4", *a.models.split(",")])}
    marker = {"template": "s", "recipe_v3": "D", "recipe_v4": "D"}

    # Fig. 4 (main): within-prompt IoU (x, lower = more diverse) against spatial adherence (y), one point per method, 95 % CI bars
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for m in shown:
        (xi, xlo, xhi), (yi, ylo, yhi) = summ[m]["iou"], summ[m]["spatial"]
        ax.errorbar(xi, yi, xerr=[[xi - xlo], [xhi - xi]], yerr=[[yi - ylo], [yhi - yi]], fmt=marker.get(m, "o"), color=colors[m], ms=7,
                    capsize=2.5, lw=1.2, mfc="white" if m == "recipe_v3" else colors[m], label=label.get(m, m))
    ax.set_xlabel("within-prompt IoU  (lower = more diverse)", fontsize=8)
    ax.set_ylabel("spatial adherence  (higher = follows the prompt)", fontsize=8)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="center left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_scatter.pdf")
    fig.savefig(out / "fig4_scatter.png", dpi=200)

    # supplementary: the per-scene / per-pair distributions behind the points
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, data, ylab in ((axs[0], ious, "within-prompt IoU (lower = more diverse)"), (axs[1], spat, "spatial adherence")):
        for k, m in enumerate(shown):
            v = np.concatenate([np.array(x) for x in data[m].values() if len(x)] or [np.array([])])
            if len(v) > 1 and np.ptp(v) > 0:
                vp = ax.violinplot([v], positions=[k], showmedians=False, widths=0.8)
                for b_ in vp["bodies"]:
                    b_.set_facecolor(colors[m]), b_.set_alpha(0.3)
            for p_, x in data[m].items():
                if len(x):
                    ax.scatter(np.full(len(x), k) + rng.uniform(-0.12, 0.12, len(x)), x, s=8, color=colors[m], alpha=0.8, lw=0)
            if len(v):
                ax.hlines(np.median(v) if data is ious else np.mean(v), k - 0.3, k + 0.3, color="k", lw=1.2)
        ax.set_xticks(range(len(shown)), [label.get(m, m) for m in shown], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "figS_violins.pdf")
    fig.savefig(out / "figS_violins.png", dpi=200)

    # manual-check maps: top-down, instances outlined green / red per rule, one panel of rule results
    if a.maps:
        draw_maps(out / "maps", a.maps, scenes, cache, detail, rules, label)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
