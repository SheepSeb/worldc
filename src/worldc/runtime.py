"""Runtime helpers for loading a generated warehouse in Isaac Sim / Isaac Lab (no Isaac imports here).

* ``PathFollower`` - deterministic waypoint path (loop or ping-pong) -> pose(t); used to drive the kinematic
  hazard actors (2 human capsules, 1 forklift box) because USD time-samples are not advanced by Isaac Lab's
  physics stepping.
* ``load_scene_graph`` / ``hazard_followers`` - read ``scene_graph.json`` written by the generator.
* ``ArticulationSweep`` - open/close ramp for door / drawer drive targets.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


def load_scene_graph(path) -> dict:
    return json.loads(Path(path).read_text())


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Motion(NamedTuple):
    """Where a path agent is in its stop-and-go cycle (``PathFollower.motion``)."""
    s: float  # arclength along the path
    back: bool  # on the return leg (or pausing at its start)
    speed: float  # m/s right now
    odo: float  # metres walked since the leg (lap) began
    dwell_t: float | None  # seconds into the pause at a path end; None while walking
    turn: float  # 0..1 progress of the turn-around at the end of the pause


def _smooth(f: float) -> float:
    f = min(max(f, 0.0), 1.0)
    return f * f * (3 - 2 * f)


@dataclass
class PathFollower:
    waypoints: list  # [[x, y], ...]
    speed: float
    mode: str = "loop"  # "loop" (closed polygon) | "pingpong"
    z: float = 0.0
    phase_m: float = 0.0
    blend: float = 1.0  # metres over which heading is blended around a vertex
    # human-like ping-pong (all 0 = constant speed and an instant 180 degree flip at the ends, e.g. forklifts): each leg eases in and out
    # over ease_s, the agent pauses dwell_s at each end and turns around (counter-clockwise) during the last turn_s of the pause
    dwell_s: float = 0.0
    ease_s: float = 0.0
    turn_s: float = 0.0

    def __post_init__(self):
        pts = [tuple(map(float, p[:2])) for p in self.waypoints]
        if self.mode == "loop":
            pts = pts + [pts[0]]
        self.pts = np.array(pts)
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.seg_dir = seg / self.seg_len[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total = float(self.cum[-1])

    def _arclen(self, t: float) -> float:
        s = self.phase_m + self.speed * t
        if self.mode == "loop":
            return s % self.total
        s = s % (2 * self.total)
        return s if s <= self.total else 2 * self.total - s

    @property
    def humanized(self) -> bool:
        return self.mode == "pingpong" and (self.dwell_s > 0 or self.ease_s > 0 or self.turn_s > 0)

    def _ease(self, u: float, e: float, leg: float):
        """(distance, speed) u seconds into a leg that accelerates over e, cruises, and brakes over e."""
        v = self.speed
        if e <= 0:
            return v * u, v
        a = v / e
        if u < e:
            return 0.5 * a * u * u, a * u
        r = leg - u
        if r < e:
            return self.total - 0.5 * a * r * r, a * r
        return 0.5 * v * e + v * (u - e), v

    @property
    def period_s(self) -> float:
        """Seconds after which the motion repeats exactly."""
        if self.mode == "loop":
            return self.total / self.speed
        return 2 * (self.total / self.speed + min(self.ease_s, self.total / self.speed) + self.dwell_s)

    def motion(self, t: float) -> Motion:
        """The agent's stop-and-go state at time t (periodic: the animation of a person is authored from it)."""
        v, L = self.speed, self.total
        if self.mode == "loop":
            s = (self.phase_m + v * t) % L
            return Motion(s, False, v, s, None, 0.0)
        e = min(self.ease_s, L / v)
        leg, dwell = L / v + e, self.dwell_s
        k = (t + self.phase_m / v) % (2 * (leg + dwell))
        back = k >= leg + dwell
        k -= (leg + dwell) if back else 0.0
        if k < leg:
            d, sp = self._ease(k, e, leg)
            return Motion(L - d if back else d, back, sp, d, None, 0.0)
        k -= leg
        turn = _smooth((k - (dwell - self.turn_s)) / self.turn_s) if self.turn_s > 0 else float(k >= dwell)
        return Motion(0.0 if back else L, back, 0.0, L, k, turn)

    def _yaw_forward(self, s: float) -> float:
        """Heading walking forward at arclength s, blended across interior vertices."""
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1))
        ang = lambda j: math.atan2(self.seg_dir[j][1], self.seg_dir[j][0])
        w = min(self.blend, 0.5 * self.seg_len[i])
        if i > 0 and s - self.cum[i] < w:
            return ang(i - 1) + (0.5 + 0.5 * (s - self.cum[i]) / w) * _wrap(ang(i) - ang(i - 1))
        if i < len(self.seg_len) - 1 and self.cum[i + 1] - s < w:
            return ang(i) + 0.5 * (1 - (self.cum[i + 1] - s) / w) * _wrap(ang(i + 1) - ang(i))
        return ang(i)

    def _pose_humanized(self, t: float):
        m = self.motion(t)
        i = int(np.clip(np.searchsorted(self.cum, m.s, side="right") - 1, 0, len(self.seg_len) - 1))
        p = self.pts[i] + (m.s - self.cum[i]) / self.seg_len[i] * (self.pts[i + 1] - self.pts[i])
        yaw = self._yaw_forward(m.s) + (math.pi if m.back else 0.0) + (math.pi * _smooth(m.turn) if m.dwell_t is not None else 0.0)
        return float(p[0]), float(p[1]), self.z, float(_wrap(yaw))

    def pose(self, t: float):
        """(x, y, z, yaw) at time t. For ping-pong the heading flips on the return leg."""
        if self.humanized:
            return self._pose_humanized(t)
        s = self._arclen(t)
        back = self.mode == "pingpong" and (self.phase_m + self.speed * t) % (2 * self.total) > self.total
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1))
        u = (s - self.cum[i]) / self.seg_len[i]
        p = self.pts[i] + u * (self.pts[i + 1] - self.pts[i])
        d = self.seg_dir[i] * (-1.0 if back else 1.0)
        yaw = math.atan2(d[1], d[0])
        # smooth heading around vertices (closed loop only)
        if self.mode == "loop":
            w = min(self.blend, 0.5 * self.seg_len[i])
            nxt = self.seg_dir[(i + 1) % len(self.seg_len)]
            prv = self.seg_dir[(i - 1) % len(self.seg_len)]
            if s - self.cum[i] < w:
                f = 0.5 + 0.5 * (s - self.cum[i]) / w
                yaw = math.atan2(prv[1], prv[0]) + f * _wrap(math.atan2(d[1], d[0]) - math.atan2(prv[1], prv[0]))
            elif self.cum[i + 1] - s < w:
                f = 0.5 * (1 - (self.cum[i + 1] - s) / w)
                yaw = math.atan2(d[1], d[0]) + f * _wrap(math.atan2(nxt[1], nxt[0]) - math.atan2(d[1], d[0]))
        return float(p[0]), float(p[1]), self.z, float(yaw)


def hazard_followers(graph: dict) -> dict:
    return {
        h["path"]: PathFollower(h["waypoints"], h["speed_mps"], h["path_mode"], h["z"], h.get("path_phase_m", 0.0),
                     dwell_s=h.get("dwell_s", 0.0), ease_s=h.get("ease_s", 0.0), turn_s=h.get("turn_s", 0.0))
        for h in graph["hazards"]
    }


def yaw_to_quat_wxyz(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


class ArticulationSweep:
    """Linear open -> hold -> close ramp over [lower, upper] for a door/drawer drive target (radians for revolute)."""

    def __init__(self, lower, upper, t_open=3.0, t_hold=0.5):
        self.lo, self.hi, self.t_open, self.t_hold = lower, upper, t_open, t_hold

    @property
    def duration(self):
        return 2 * self.t_open + self.t_hold

    def target(self, t):
        if t < self.t_open:
            return self.lo + (self.hi - self.lo) * t / self.t_open
        if t < self.t_open + self.t_hold:
            return self.hi
        return self.hi - (self.hi - self.lo) * min(1.0, (t - self.t_open - self.t_hold) / self.t_open)
