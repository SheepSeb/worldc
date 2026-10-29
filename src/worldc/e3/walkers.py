"""Seeded worker task tours (replacing the LLM-authored waypoint paths).

A trial seed fixes everything random about the workers: worker ``i`` draws from ``default_rng([seed, i])``, so worker 0 of
seed 7 is the same person in every tier and every policy, and the same seed always gives the same trial.

Each worker loops a tour of 3-6 tasks drawn from the scene's task points (a kind first, uniformly over the kinds present,
then a point of that kind; never the same point twice in a row).  Legs are A* on the human-inflated grid, smoothed; along a
leg the speed is the worker's preferred speed, capped at corners and limited by acceleration, from standstill to standstill;
at each task the worker dwells a lognormal time.  A uniform start offset puts workers at different points of their tours.
Workers do not react to the robot or to each other (stated limitation; human-human overlap is reported).
"""
from __future__ import annotations

import math

import numpy as np

from .world import World

SPEED_MEAN, SPEED_SD, SPEED_RANGE = 1.3, 0.2, (0.9, 1.7)  # m/s, preferred walking speed per worker
ACCEL = 0.8  # m/s^2, speeding up and slowing down
CORNER_MIN = 0.35  # fraction of the preferred speed at a 180 degree turn (linear in the turn angle)
DWELL_MEDIAN_S, DWELL_SIGMA = 5.0, 0.6  # lognormal dwell at a task
TASKS = (3, 6)
DS = 0.05  # m, resampling step of a leg


def sample_tour(world: World, rng: np.random.Generator) -> list[tuple[float, float]]:
    kinds = sorted(world.pois)
    tour: list = []
    for _ in range(int(rng.integers(TASKS[0], TASKS[1] + 1))):
        for _ in range(20):
            pts = world.pois[kinds[int(rng.integers(len(kinds)))]]
            p = pts[int(rng.integers(len(pts)))]
            if not tour or p != tour[-1]:
                break
        tour.append(p)
    if len(tour) > 1 and tour[0] == tour[-1]:  # the tour loops: last -> first must be a real leg
        tour.pop()
    return tour


def _leg_profile(path: np.ndarray, v_pref: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every DS along ``path`` and the time at each, standstill at both ends."""
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    n = max(2, int(math.ceil(L / DS)) + 1)
    s = np.linspace(0.0, L, n)
    pts = np.stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])], 1)
    vmax = np.full(n, v_pref)
    for k in range(1, len(path) - 1):  # corner caps at the smoothed path's vertices
        a, b = path[k] - path[k - 1], path[k + 1] - path[k]
        turn = math.acos(float(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9), -1, 1)))
        i = int(np.argmin(abs(s - cum[k])))
        vmax[i] = min(vmax[i], v_pref * (1 - (1 - CORNER_MIN) * turn / math.pi))
    vmax[0] = vmax[-1] = 0.0
    ds = s[1] - s[0]
    for i in range(1, n):
        vmax[i] = min(vmax[i], math.sqrt(vmax[i - 1] ** 2 + 2 * ACCEL * ds))
    for i in range(n - 2, -1, -1):
        vmax[i] = min(vmax[i], math.sqrt(vmax[i + 1] ** 2 + 2 * ACCEL * ds))
    vavg = np.maximum((vmax[1:] + vmax[:-1]) / 2, 0.05)
    return pts, np.concatenate([[0.0], np.cumsum(ds / vavg)])


def worker_trajectory(world: World, seed: int, i: int, horizon_s: float, dt: float) -> tuple[np.ndarray, dict]:
    """(steps, 2) positions at t = 0, dt, ... for worker ``i`` of trial ``seed``, and what was drawn."""
    rng = np.random.default_rng([seed, i])
    tour = sample_tour(world, rng)
    v_pref = float(np.clip(rng.normal(SPEED_MEAN, SPEED_SD), *SPEED_RANGE))
    # one lap: dwell at each task, then walk to the next (a lone task: stand there)
    ts, xs = [], []
    t = 0.0
    dwells = []
    for k, p in enumerate(tour):
        dw = float(rng.lognormal(math.log(DWELL_MEDIAN_S), DWELL_SIGMA))
        dwells.append(round(dw, 2))
        ts += [t, t + dw]
        xs += [p, p]
        t += dw
        if len(tour) > 1:
            pts, tt = _leg_profile(world.human_path(p, tour[(k + 1) % len(tour)]), v_pref)
            ts += list(t + tt[1:])
            xs += [tuple(q) for q in pts[1:]]
            t += tt[-1]
    period = t
    offset = float(rng.uniform(0.0, period))
    ts, xs = np.array(ts), np.array(xs)
    tq = (offset + np.arange(int(math.ceil(horizon_s / dt)) + 1) * dt) % period
    traj = np.stack([np.interp(tq, ts, xs[:, 0]), np.interp(tq, ts, xs[:, 1])], 1)
    return traj, {"tour": [list(p) for p in tour], "speed_mps": round(v_pref, 3), "dwell_s": dwells,
                  "period_s": round(period, 2), "offset_s": round(offset, 2)}


def workers(world: World, seed: int, n: int, horizon_s: float, dt: float) -> tuple[np.ndarray, list[dict]]:
    """(steps, n, 2) positions of ``n`` workers for trial ``seed``."""
    out = [worker_trajectory(world, seed, i, horizon_s, dt) for i in range(n)]
    return np.stack([o[0] for o in out], 1), [o[1] for o in out]
