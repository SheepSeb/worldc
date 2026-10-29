"""Isaac Sim presence gate: every entrypoint calls ``require_isaac_sim()`` before doing any work.

Core code never imports Isaac Sim; it runs as a subprocess.  But the pipeline refuses to run without an install:
``ISAACSIM_PATH`` (default ``~/isaacsim``) must contain an executable ``python.sh`` and a ``VERSION`` >= 5.0.
``deep=True`` additionally boots a headless ``SimulationApp`` once per (root, version) and caches the result.

    python -m worldc.isaac_env [--deep]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

MIN_VERSION = (5, 0)
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "worldc" / "isaac_check.json"
SMOKE = "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); app.update(); print('ISAAC_SMOKE_OK', flush=True); app.close()"


@dataclass
class IsaacSim:
    root: Path
    version: str
    python: Path

    @property
    def key(self) -> str:
        return f"{self.root}@{self.version}"


class IsaacSimUnavailable(RuntimeError):
    """Raised when the Isaac Sim gate fails. ``issue`` is a structured, actionable description."""

    def __init__(self, code: str, message: str, hint: str, data: dict | None = None):
        super().__init__(f"{message} {hint}")
        self.issue = {"severity": "error", "code": code, "path": "", "message": message, "hint": hint, "data": data or {}}


def isaac_root() -> Path:
    return Path(os.environ.get("ISAACSIM_PATH", Path.home() / "isaacsim")).expanduser()


def _parse_version(s: str) -> tuple[int, ...]:
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", s.strip())
    return tuple(int(g) for g in m.groups() if g is not None) if m else ()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def run_isaac(cmd: list[str], *, timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    """subprocess.run for Isaac Sim: python.sh is a shell wrapper around Kit's python, so on timeout subprocess.run would
    kill only the wrapper and leave Kit running on the GPU after the GPU lock is released.  Runs the command
    in its own process group and kills the whole group on timeout, then raises TimeoutExpired."""
    import signal

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()
        raise
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def _smoke(sim: IsaacSim, timeout: float) -> tuple[bool, str]:
    try:
        p = run_isaac([str(sim.python), "-c", SMOKE], timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"headless SimulationApp did not start within {timeout:.0f} s"
    if "ISAAC_SMOKE_OK" in p.stdout:
        return True, ""
    return False, (p.stderr or p.stdout)[-800:]


def require_isaac_sim(deep: bool = False, timeout: float = 300.0) -> IsaacSim:
    """Return the Isaac Sim install or raise ``IsaacSimUnavailable``. Cheap unless ``deep`` and not yet cached."""
    root = isaac_root()
    py = root / "python.sh"
    fix = "Install Isaac Sim >= 5.0 (standalone) or set ISAACSIM_PATH to its root folder."
    if not (py.is_file() and os.access(py, os.X_OK)):
        raise IsaacSimUnavailable("env.isaac_sim_missing", f"Isaac Sim not found: no executable python.sh in {root}.", fix,
                                  {"root": str(root), "env_var": "ISAACSIM_PATH"})
    try:
        version = (root / "VERSION").read_text().strip()
    except OSError:
        raise IsaacSimUnavailable("env.isaac_sim_missing", f"Isaac Sim at {root} has no VERSION file.", fix, {"root": str(root)})
    if _parse_version(version)[:2] < MIN_VERSION:
        raise IsaacSimUnavailable("env.isaac_sim_version", f"Isaac Sim {version} at {root} is older than "
                                  f"{'.'.join(map(str, MIN_VERSION))}.", fix, {"root": str(root), "version": version})
    sim = IsaacSim(root, version, py)
    cache = _load_cache()
    if cache.get(sim.key, {}).get("ok"):
        return sim
    if not deep:
        return sim
    ok, err = _smoke(sim, timeout)
    if not ok:
        raise IsaacSimUnavailable("env.isaac_sim_broken", f"Isaac Sim {version} at {root} failed to start headless.",
                                  "Run `$ISAACSIM_PATH/isaac-sim.compatibility_check.sh` and check GPU drivers.",
                                  {"root": str(root), "version": version, "stderr_tail": err})
    cache[sim.key] = {"ok": True}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))
    return sim


def gate_or_exit(deep: bool = False) -> IsaacSim:
    """CLI helper: print the structured issue and exit(2) instead of raising."""
    try:
        return require_isaac_sim(deep=deep)
    except IsaacSimUnavailable as e:
        print(json.dumps(e.issue, indent=1), file=sys.stderr)
        raise SystemExit(2)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Check that Isaac Sim is installed (and, with --deep, that it boots headless).")
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args(argv)
    sim = gate_or_exit(deep=a.deep)
    print(json.dumps({k: str(v) for k, v in asdict(sim).items()}))


if __name__ == "__main__":
    main()


GPU_LOCK = Path(os.environ.get("WORLDC_ISAAC_LOCK", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "worldc" / "isaac_gpu.lock"))


class isaac_lock:
    """Machine-wide queue for Isaac Sim subprocesses (one at a time on the GPU): concurrent generation loops wait here instead
    of starting a second Isaac Sim.  A subprocess's own timeout starts once it holds the lock."""

    def __enter__(self):
        import fcntl
        import time

        GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        self.f = GPU_LOCK.open("a+")
        t0 = time.time()
        fcntl.flock(self.f, fcntl.LOCK_EX)
        self.waited_s = time.time() - t0
        return self

    def __exit__(self, *exc):
        import fcntl

        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
        return False
