"""Why the recipe scenes fail the Isaac Sim physics check: per method, prompt, failure code and object.

    python3 scripts/e2_physics_causes.py --out results/e2_within_v2

Reads ``e2_physics.json`` of every scene folder under the recipe roots (within-prompt v3, v4, and optionally the 30-prompt baselines).
An object is the spec object id (``/World/<id>/<id>_NNNN``).  ``physics.fell_through_floor`` and ``physics.unstable`` name the bodies in the
issue path; ``physics.not_settled`` names none, so the bodies of ``worst_movers`` that moved more than 2 cm are used.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

ROOTS = {"recipe v4 (rules)": "out/e2_within/recipe_v4", "recipe v3": "out/e2_within/recipe", "recipe v3, 30 prompts": "out/e2_v3/recipe"}


def obj(path: str) -> str:
    p = path.split("/")
    return p[2] if len(p) > 2 else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e2_within_v2")
    a = ap.parse_args()
    out = Path(a.out)
    rows, scenes = [], []
    md = ["# Recipe physics failures: causes", "",
          "Every scene that passed the static check was run through the Isaac Sim physics check (6 s settle, the E1 rule). A scene fails on any error issue.", ""]
    for method, root in ROOTS.items():
        for f in sorted(Path(root).glob("*/e2_physics.json")):
            d = json.loads(f.read_text())
            pid = f.parent.name.split("_")[0]
            codes = collections.defaultdict(collections.Counter)
            for i in d["issues"]:
                if i["severity"] != "error" or not i["code"].startswith("physics."):
                    continue
                if i["path"]:
                    codes[i["code"]][obj(i["path"])] += 1
                elif i["code"] == "physics.not_settled":
                    movers = [m for m in (d.get("physics") or {}).get("worst_movers", []) if m["displacement_m"] > 0.02]
                    for m in movers:
                        codes[i["code"]][obj(m["body"])] += 1
                    if not movers:
                        codes[i["code"]]["(no body above 2 cm listed)"] += 1
                else:
                    codes[i["code"]]["(none)"] += 1
            scenes.append({"method": method, "prompt": pid, "scene": f.parent.name, "status": d["physics_status"], "codes": codes})
    # per method x prompt x code
    for method in ROOTS:
        S = [s for s in scenes if s["method"] == method]
        if not S:
            continue
        md += [f"## {method}", "", "| Prompt | Checked | Failed | Failure code | Scenes with it | Objects (bodies affected, summed over scenes) |", "|---|---|---|---|---|---|"]
        for pid in sorted({s["prompt"] for s in S}):
            P = [s for s in S if s["prompt"] == pid]
            failed = [s for s in P if s["status"] != "ok"]
            allc = sorted({c for s in failed for c in s["codes"]}) or ["-"]
            for n, c in enumerate(allc):
                ob = collections.Counter()
                ns = 0
                for s in failed:
                    if c in s["codes"]:
                        ns += 1
                        ob.update(s["codes"][c])
                md.append(f"| {pid if n == 0 else ''} | {len(P) if n == 0 else ''} | {len(failed) if n == 0 else ''} | {c.replace('physics.', '') if c != '-' else '-'} | {ns or ''} | "
                          + ", ".join(f"{o} ({k})" for o, k in ob.most_common(6)) + " |")
                for o, k in ob.items():
                    rows.append([method, pid, c, o, ns, k])
        # overall by object
        tot = collections.Counter()
        for s in S:
            if s["status"] != "ok":
                for c, ob in s["codes"].items():
                    for o in ob:
                        tot[(c.replace("physics.", ""), o)] += 1
        md += ["", "Failed scenes per (code, object): " + ", ".join(f"{c}: {o} ({k})" for (c, o), k in tot.most_common(10)), ""]
    (out / "physics_failure_causes.md").write_text("\n".join(md) + "\n")
    with open(out / "physics_failure_causes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "prompt", "code", "object", "scenes_with_code", "bodies"])
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
