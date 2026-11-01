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
            disp = {}  # (code, object) -> largest displacement (m) of a body of that object, from worst_movers
            for i in d["issues"]:
                if i["severity"] != "error" or not i["code"].startswith("physics."):
                    continue
                if i["path"]:
                    codes[i["code"]][obj(i["path"])] += 1
                elif i["code"] == "physics.not_settled":
                    movers = [m for m in (d.get("physics") or {}).get("worst_movers", []) if m["displacement_m"] > 0.02]
                    for m in movers:
                        codes[i["code"]][obj(m["body"])] += 1
                        disp[(i["code"], obj(m["body"]))] = max(disp.get((i["code"], obj(m["body"])), 0.0), m["displacement_m"])
                    if not movers:
                        codes[i["code"]]["(no body above 2 cm listed)"] += 1
                else:
                    codes[i["code"]]["(none)"] += 1
            scenes.append({"method": method, "prompt": pid, "scene": f.parent.name, "status": d["physics_status"], "codes": codes, "disp": disp})
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
                mx = {}
                ns = 0
                for s in failed:
                    if c in s["codes"]:
                        ns += 1
                        ob.update(s["codes"][c])
                        for (cc, o), v in s["disp"].items():
                            if cc == c:
                                mx[o] = max(mx.get(o, 0.0), v)
                md.append(f"| {pid if n == 0 else ''} | {len(P) if n == 0 else ''} | {len(failed) if n == 0 else ''} | {c.replace('physics.', '') if c != '-' else '-'} | {ns or ''} | "
                          + ", ".join(f"{o} ({k}{f', up to {mx[o]:.2f} m' if o in mx else ''})" for o, k in ob.most_common(6)) + " |")
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
    with open(out / "physics_recipes.csv", "w", newline="") as f:  # every checked recipe scene: pass / fail and its codes
        w = csv.writer(f)
        w.writerow(["method", "scene", "physics_status", "error_codes"])
        w.writerows([s_["method"], s_["scene"], s_["status"], ";".join(sorted(s_["codes"]))] for s_ in scenes)
    pr = ["# Isaac Sim physics of the recipe scenes", "", "Scenes that passed the static check, run through the E1 physics check (`scripts/e2_physics_queue.py`, 6 s settle). Passed / checked:", "",
          "| Recipe | " + " | ".join(sorted({s_["prompt"] for s_ in scenes if s_["method"] != "recipe v3, 30 prompts"})) + " | Total |", ""]
    pr[-1] = "|---|" + "---|" * (len(pr[-2].split("|")) - 3)
    for method in ROOTS:
        S = [s_ for s_ in scenes if s_["method"] == method and method != "recipe v3, 30 prompts"]
        if S:
            pr.append(f"| {method} | " + " | ".join(f"{sum(s_['status'] == 'ok' for s_ in S if s_['prompt'] == pid)}/{sum(s_['prompt'] == pid for s_ in S)}" for pid in sorted({s_['prompt'] for s_ in S}))
                      + f" | {sum(s_['status'] == 'ok' for s_ in S)}/{len(S)} |")
    pr += ["", "Failure causes by code and object: `physics_failure_causes.md`."]
    (out / "physics_recipes.md").write_text("\n".join(pr) + "\n")
    (out / "physics_failure_causes.md").write_text("\n".join(md) + "\n")
    with open(out / "physics_failure_causes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "prompt", "code", "object", "scenes_with_code", "bodies"])
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
