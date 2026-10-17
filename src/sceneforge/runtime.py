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

import numpy as np


def load_scene_graph(path) -> dict:
    return json.loads(Path(path).read_text())


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class PathFollower:
    waypoints: list  # [[x, y], ...]
    speed: float
    mode: str = "loop"  # "loop" (closed polygon) | "pingpong"
    z: float = 0.0
    phase_m: float = 0.0
    blend: float = 1.0  # metres over which heading is blended around a vertex

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

    def pose(self, t: float):
        """(x, y, z, yaw) at time t. For ping-pong the heading flips on the return leg."""
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
        h["path"]: PathFollower(h["waypoints"], h["speed_mps"], h["path_mode"], h["z"], h.get("path_phase_m", 0.0))
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
