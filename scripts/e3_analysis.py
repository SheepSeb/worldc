"""E3 analysis over a full run: cluster-bootstrap CIs, paired Wilcoxon tests between policies, Table 2, and the
safety-vs-efficiency scatter.

    uv run python scripts/e3_analysis.py out/e3/*/trials.csv --out results/e3_analysis [--boot 2000]

Input: the ``trials.csv`` of every scene (``scripts/e3_one_scene.py``), one row per (scene, tier, seed, policy).

* **CIs**: 95 % percentile intervals from a two-stage cluster bootstrap. Each replicate resamples scenes with
  replacement, then seeds within each drawn scene. The same draw is used for every (tier, policy) cell, so cells stay
  paired. With one scene this reduces to a bootstrap over seeds.
* **Tests**: per tier, a paired Wilcoxon signed-rank test for each policy pair on identical (scene, seed) trials, for
  robot-moving collisions and for time to goal. A trial that did not finish (frozen, timeout, ...) counts at its
  ``timeout_s`` (worst case). Zero differences are dropped (``zero_method="wilcox"``); ``n`` is the number of non-zero
  pairs. p-values are Holm-corrected within each metric over all tiers and pairs.
* **Outputs**: ``table2.md``, ``table2.tex``, ``tests.csv``, ``tests.md``, ``scatter.svg`` and ``scatter_points.csv``
  (moving collisions per trial against median time to goal, one point per policy x tier, with CIs), and ``stats.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

TIERS = ("easy", "medium", "hard")
POLICIES = ("blind", "fields", "aware")
POLICY_LABEL = {"blind": "Blind", "fields": "Fields (ISO 3691-4-style)", "aware": "Human-aware"}
PAIRS = (("blind", "fields"), ("blind", "aware"), ("fields", "aware"))


def load(paths) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                for k, v in r.items():
                    if v in ("True", "False"):
                        r[k] = v == "True"
                    elif v == "":
                        r[k] = None
                    else:
                        try:
                            r[k] = float(v)
                        except ValueError:
                            pass
                r["seed"] = int(r["seed"])
                rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------ statistics per cell
def _ttg(r):
    return r["time_to_goal_s"]


METRICS = {  # name -> (per-trial value, aggregate over the trials of a cell)
    "success": (lambda r: float(r["success"]), np.nanmean),
    "moving_collisions": (lambda r: r["collisions_robot_moving"], np.nanmean),
    "moving_collisions_from_behind": (lambda r: r["collisions_robot_moving_from_behind"], np.nanmean),
    "moving_collisions_braking": (lambda r: r["collisions_robot_moving_braking"], np.nanmean),
    "all_collisions": (lambda r: r["collisions"], np.nanmean),
    "stationary_collisions": (lambda r: r["collisions_robot_stationary"], np.nanmean),
    "min_gap_moving": (lambda r: r["min_gap_moving_m"] if r["min_gap_moving_m"] is not None else np.nan, np.nanmedian),
    "within_0.5m_moving": (lambda r: r["time_within_0.5m_moving_s"], np.nanmean),
    "within_1m_moving": (lambda r: r["time_within_1m_moving_s"], np.nanmean),
    "within_0.5m": (lambda r: r["time_within_0.5m_s"], np.nanmean),
    "within_1m": (lambda r: r["time_within_1m_s"], np.nanmean),
    "stationary_near_1m": (lambda r: r["stationary_near_1m_s"], np.nanmean),
    "frozen": (lambda r: float(r["outcome"] == "frozen"), np.nanmean),
    "time_to_goal": (lambda r: _ttg(r) if _ttg(r) is not None else np.nan, np.nanmedian),  # finished trials only
}


def cube(rows, metric):
    """{(tier, policy): array (n_scenes, n_seeds)} of a per-trial metric (NaN where a trial is missing)."""
    scenes = sorted({r["scene"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    si, ki = {s: i for i, s in enumerate(scenes)}, {s: i for i, s in enumerate(seeds)}
    fn = METRICS[metric][0]
    out = {(t, p): np.full((len(scenes), len(seeds)), np.nan) for t in TIERS for p in POLICIES}
    for r in rows:
        out[(r["tier"], r["policy"])][si[r["scene"]], ki[r["seed"]]] = fn(r)
    return out, scenes, seeds


def bootstrap_index(n_scenes, n_seeds, B, rng):
    sc = rng.integers(0, n_scenes, size=(B, n_scenes))
    sd = rng.integers(0, n_seeds, size=(B, n_scenes, n_seeds))
    return sc, sd


def estimate(rows, B=2000, seed=0) -> dict:
    """{metric: {(tier, policy): (point, lo, hi, n)}}"""
    rng = np.random.default_rng(seed)
    out = {}
    idx = None
    for m, (_, agg) in METRICS.items():
        C, scenes, seeds = cube(rows, m)
        if idx is None:
            idx = bootstrap_index(len(scenes), len(seeds), B, rng)
        sc, sd = idx
        out[m] = {}
        for key, M in C.items():
            n = int(np.isfinite(M).sum())
            if n == 0:
                out[m][key] = (np.nan, np.nan, np.nan, 0)
                continue
            point = float(agg(M))
            Mb = M[sc[:, :, None], sd].reshape(B, -1)  # (B, n_scenes * n_seeds)
            with np.errstate(all="ignore"), _quiet():
                boot = agg(Mb, axis=1)
            boot = boot[np.isfinite(boot)]
            lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
            out[m][key] = (point, float(lo), float(hi), n)
    return out


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


# ------------------------------------------------------------------------------------------------------- tests
def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def tests(rows) -> list[dict]:
    by = {(r["scene"], r["tier"], r["seed"], r["policy"]): r for r in rows}
    keys = sorted({(r["scene"], r["seed"]) for r in rows})
    per_trial = {"moving_collisions": lambda r: r["collisions_robot_moving"],
                 "time_to_goal": lambda r: _ttg(r) if _ttg(r) is not None else r["timeout_s"]}
    out = []
    for metric, fn in per_trial.items():
        block = []
        for tier in TIERS:
            for a, b in PAIRS:
                pairs = [(fn(by[(s, tier, k, a)]), fn(by[(s, tier, k, b)])) for s, k in keys
                         if (s, tier, k, a) in by and (s, tier, k, b) in by]
                x, y = np.array(pairs, float).T if pairs else (np.array([]), np.array([]))
                d = x - y
                nz = int((d != 0).sum())
                if nz == 0:
                    stat, p = np.nan, 1.0
                else:
                    res = wilcoxon(x, y, zero_method="wilcox")
                    stat, p = float(res.statistic), float(res.pvalue)
                block.append({"metric": metric, "tier": tier, "a": a, "b": b, "pairs": len(pairs), "n_nonzero": nz,
                              "median_diff": float(np.median(d)) if len(d) else np.nan, "mean_diff": float(np.mean(d)) if len(d) else np.nan,
                              "a_better": int((d < 0).sum()), "b_better": int((d > 0).sum()), "W": stat, "p": p})
        for row, pa in zip(block, holm(np.array([r["p"] for r in block]))):
            row["p_holm"] = float(pa)
        out += block
    return out


# -------------------------------------------------------------------------------------------------------- output
def fmt(v, lo, hi, digits=1, pct=False):
    if not math.isfinite(v):
        return "–"
    f = (lambda z: f"{100 * z:.0f}") if pct else (lambda z: f"{z:.{digits}f}")
    return f"{f(v)} [{f(lo)}, {f(hi)}]"


COLUMNS = [  # (header, metric, digits, pct)
    ("Moving coll./trial", "moving_collisions", 2, False),
    ("of which from behind", "moving_collisions_from_behind", 2, False),
    ("of which already braking", "moving_collisions_braking", 2, False),
    ("All coll./trial", "all_collisions", 1, False),
    ("Success (%)", "success", 0, True),
    ("Frozen (%)", "frozen", 0, True),
    ("Min gap moving (m)", "min_gap_moving", 2, False),
    ("<0.5 m moving (s)", "within_0.5m_moving", 1, False),
    ("<1 m moving (s)", "within_1m_moving", 1, False),
    ("Stationary <1 m (s)", "stationary_near_1m", 1, False),
    ("Time to goal (s)", "time_to_goal", 0, False),
]


def table2(est, n_scenes, n_seeds) -> tuple[str, str]:
    head = ["Tier", "Policy"] + [c[0] for c in COLUMNS]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{tabular}{ll" + "r" * len(COLUMNS) + "}", r"\toprule", " & ".join(head).replace("%", r"\%") + r" \\", r"\midrule"]
    for t in TIERS:
        for i, p in enumerate(POLICIES):
            cells = [fmt(*est[m][(t, p)][:3], digits=d, pct=pc) for _, m, d, pc in COLUMNS]
            md.append(f"| {t if i == 0 else ''} | {POLICY_LABEL[p]} | " + " | ".join(cells) + " |")
            tex.append(" & ".join([t.capitalize() if i == 0 else "", POLICY_LABEL[p]] + [c.replace("–", "--") for c in cells]) + r" \\")
        tex.append(r"\midrule" if t != TIERS[-1] else r"\bottomrule")
    tex.append(r"\end{tabular}")
    note = (f"\n\n{n_scenes} scene(s) x {n_seeds} seeds per tier and policy. Point estimate with a 95 % cluster-bootstrap CI "
            "(scenes, then seeds). Means except min gap and time to goal (medians); time to goal over finished trials only.")
    return "\n".join(md) + note + "\n", "\n".join(tex) + "\n"


def scatter_svg(est, path):
    W, H, L, R, T, Bm = 700, 440, 70, 230, 30, 55
    pts = []
    for t in TIERS:
        for p in POLICIES:
            x = est["time_to_goal"][(t, p)]
            y = est["moving_collisions"][(t, p)]
            if math.isfinite(x[0]) and math.isfinite(y[0]):
                pts.append((t, p, x, y))
    xs = [v for *_, x, _ in pts for v in x[:3] if math.isfinite(v)]
    ys = [v for *_, y in pts for v in y[:3] if math.isfinite(v)]
    xstep, ystep = _nice((max(xs) if xs else 1.0) / 5), _nice((max(ys) if ys else 1.0) / 5)
    x0, x1 = 0.0, xstep * math.ceil((max(xs) if xs else 1.0) * 1.05 / xstep)
    y0, y1 = 0.0, ystep * math.ceil(max(max(ys) if ys else 1.0, 0.5) * 1.05 / ystep)
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    Y = lambda v: H - Bm - (v - y0) / (y1 - y0) * (H - T - Bm)  # noqa: E731
    color = {"blind": "#c0392b", "fields": "#2471a3", "aware": "#1e8449"}
    shape = {"easy": "circle", "medium": "square", "hard": "triangle"}
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<line x1="{L}" y1="{H - Bm}" x2="{W - R}" y2="{H - Bm}" stroke="#333"/><line x1="{L}" y1="{T}" x2="{L}" y2="{H - Bm}" stroke="#333"/>']
    for vx in np.arange(x0, x1 + xstep / 2, xstep):
        s.append(f'<line x1="{X(vx):.1f}" y1="{H - Bm}" x2="{X(vx):.1f}" y2="{H - Bm + 4}" stroke="#333"/>'
                 f'<text x="{X(vx):.1f}" y="{H - Bm + 17}" text-anchor="middle">{vx:g}</text>')
    for vy in np.arange(y0, y1 + ystep / 2, ystep):
        s.append(f'<line x1="{L - 4}" y1="{Y(vy):.1f}" x2="{L}" y2="{Y(vy):.1f}" stroke="#333"/>'
                 f'<line x1="{L}" y1="{Y(vy):.1f}" x2="{W - R}" y2="{Y(vy):.1f}" stroke="#eee"/>'
                 f'<text x="{L - 8}" y="{Y(vy) + 4:.1f}" text-anchor="end">{round(vy, 6):g}</text>')
    s.append(f'<text x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle">Median time to goal (s)</text>')
    s.append(f'<text transform="translate(18,{(T + H - Bm) / 2}) rotate(-90)" text-anchor="middle">Robot-moving collisions per trial</text>')
    for t, p, x, y in pts:
        cx, cy, c = X(x[0]), Y(y[0]), color[p]
        if math.isfinite(x[1]):
            s.append(f'<line x1="{X(x[1]):.1f}" y1="{cy:.1f}" x2="{X(x[2]):.1f}" y2="{cy:.1f}" stroke="{c}" stroke-opacity="0.6"/>')
        s.append(f'<line x1="{cx:.1f}" y1="{Y(y[1]):.1f}" x2="{cx:.1f}" y2="{Y(y[2]):.1f}" stroke="{c}" stroke-opacity="0.6"/>')
        s.append(_marker(shape[t], cx, cy, c))
    lx, ly = W - R + 20, T + 10
    for i, p in enumerate(POLICIES):
        s.append(f'<rect x="{lx}" y="{ly + 18 * i - 8}" width="10" height="10" fill="{color[p]}"/><text x="{lx + 16}" y="{ly + 18 * i + 1}">{POLICY_LABEL[p]}</text>')
    for i, t in enumerate(TIERS):
        yy = ly + 18 * (len(POLICIES) + 1 + i)
        s.append(_marker(shape[t], lx + 5, yy - 3, "#555") + f'<text x="{lx + 16}" y="{yy + 1}">{t}</text>')
    s.append("</svg>")
    Path(path).write_text("\n".join(s) + "\n")
    return pts


def _nice(raw: float) -> float:
    """A round tick step (1, 2 or 5 x 10^k) near ``raw``."""
    e = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    return next(m * e for m in (1, 2, 5, 10) if m * e >= raw)


def _marker(kind, x, y, c):
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{c}"/>'
    if kind == "square":
        return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="{c}"/>'
    return f'<polygon points="{x:.1f},{y - 6.5:.1f} {x - 6:.1f},{y + 4.5:.1f} {x + 6:.1f},{y + 4.5:.1f}" fill="{c}"/>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+", help="trials.csv of each scene")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(a.trials)
    scenes = sorted({r["scene"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    est = estimate(rows, a.boot, a.seed)
    md, tex = table2(est, len(scenes), len(seeds))
    (out / "table2.md").write_text(md)
    (out / "table2.tex").write_text(tex)
    T = tests(rows)
    with open(out / "tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(T[0]))
        w.writeheader()
        w.writerows(T)
    tmd = ["| metric | tier | A vs B | pairs | non-zero | A better / B better | median diff (A-B) | p | p (Holm) |", "|---" * 9 + "|"]
    for r in T:
        tmd.append(f"| {r['metric']} | {r['tier']} | {r['a']} vs {r['b']} | {r['pairs']} | {r['n_nonzero']} | {r['a_better']} / {r['b_better']} | "
                   f"{r['median_diff']:.2f} | {r['p']:.3g} | {r['p_holm']:.3g} |")
    (out / "tests.md").write_text("\n".join(tmd) + "\n")
    pts = scatter_svg(est, out / "scatter.svg")
    with open(out / "scatter_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "policy", "time_to_goal_median_s", "ttg_lo", "ttg_hi", "moving_collisions_per_trial", "mc_lo", "mc_hi"])
        for t, p, x, y in pts:
            w.writerow([t, p, *[round(v, 3) for v in x[:3]], *[round(v, 3) for v in y[:3]]])
    (out / "stats.json").write_text(json.dumps({"inputs": [str(p) for p in a.trials], "scenes": scenes, "seeds": seeds, "bootstrap": a.boot,
                                                "estimates": {m: {f"{t}/{p}": v for (t, p), v in d.items()} for m, d in est.items()}},
                                               indent=1, default=float))
    print(md)
    print("\n".join(tmd))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
