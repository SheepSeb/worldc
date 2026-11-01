"""E2 baselines: Isaac Sim physics for every built baseline scene, queued to run after the E1 batch.

    uv run python scripts/e2_physics_queue.py out/e2/template out/e2/recipe            # waits for E1 to start and finish
    uv run python scripts/e2_physics_queue.py --now out/e2/template out/e2/recipe      # no E1 to wait for

Low priority: niced, behind the machine-wide Isaac GPU lock, and it never starts a scene while an E1 batch
(scripts/run_e1.sh) is running: it first waits until an E1 batch has been seen and has been gone for --quiet-min
minutes, and pauses again if one starts.  Same physics rule as the E1 final check (6 s settle, 900 s timeout, one rerun
with 1800 s).  Writes <scene>/e2_physics.json (resumable: done scenes are skipped) and <root>/e2_physics_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from worldc import tools
from worldc.datagen.runner import PHYSICS_TIMEOUT_S, physics_status


def e1_running() -> bool:
    # a bash process running the script (a tmux server keeps the command line it was started with in its own argv)
    return subprocess.run(["pgrep", "-f", r"^\S*bash \S*run_e1\.sh"], capture_output=True).returncode == 0


def wait_for_e1(quiet_s: float, seen: bool, log=print) -> bool:
    """Block until an E1 batch has run (unless already seen) and none has run for quiet_s seconds."""
    last = time.time() if e1_running() else 0.0
    while True:
        if e1_running():
            seen, last = True, time.time()
        elif seen and time.time() - last >= quiet_s:
            return seen
        time.sleep(60)


def check(scene: Path) -> dict:
    t0 = time.time()
    c = tools.check_scene(str(scene), physics=True, sim_seconds=6.0, timeout_s=PHYSICS_TIMEOUT_S)
    status, first = physics_status(c), None
    if status == "timeout":
        first = {"status": "timeout", "timeout_s": PHYSICS_TIMEOUT_S}
        c = tools.check_scene(str(scene), physics=True, sim_seconds=6.0, timeout_s=2 * PHYSICS_TIMEOUT_S)
        status = physics_status(c)
    return {"physics_status": status, "physics_first_attempt": first, "ok": c["ok"], "physics": c["result"].get("physics"),
            "issues": c["issues"][:30], "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", help="baseline folders (one scene folder per prompt / seed)")
    ap.add_argument("--now", action="store_true", help="do not wait for an E1 batch")
    ap.add_argument("--quiet-min", type=float, default=10.0, help="minutes without run_e1.sh before starting / resuming")
    a = ap.parse_args(argv)
    os.nice(19)  # inherited by the Isaac Sim subprocesses
    log = lambda s: print(time.strftime("%F %T"), s, flush=True)  # noqa: E731
    scenes = sorted(d for r in a.roots for d in Path(r).iterdir() if (d / "scene.usd").is_file() and ((d / "e2_attempts.json").is_file() or (d / "record.json").is_file()))  # record.json: within-prompt recipe folders
    log(f"{len(scenes)} baseline scenes queued")
    seen = a.now
    for d in scenes:
        if (d / "e2_physics.json").is_file():
            continue
        if not a.now:
            if not seen:
                log("waiting for the E1 batch to start and finish")
            seen = wait_for_e1(a.quiet_min * 60, seen)
        res = check(d)
        (d / "e2_physics.json").write_text(json.dumps(res, indent=1))
        log(f"{d.parent.name}/{d.name}: {res['physics_status']} ({res['seconds']} s)")
    for r in a.roots:
        rows = [{"scene": d.name, **json.loads((d / "e2_physics.json").read_text())} for d in sorted(Path(r).iterdir()) if (d / "e2_physics.json").is_file()]
        (Path(r) / "e2_physics_summary.json").write_text(json.dumps(
            [{k: x[k] for k in ("scene", "physics_status", "ok", "seconds")} for x in rows], indent=1))
    log("done")


if __name__ == "__main__":
    main()
