"""Reactive walkers (ablation ``walkers_reactive``): a walker waits for the robot instead of walking on.

The main E3 grid uses walkers that do not react (``walkers.py``).  Here a walker still follows the same seeded tour, but on a private
clock that stops while the robot is *within STOP_GAP_M ahead* of it: the gap from the robot's footprint to the walker's disk is below
STOP_GAP_M and the robot is in front of the walker (in the half-plane of its direction of travel over the next LOOK_S seconds).  The
walker stands still at once (no deceleration, no sidestep) and walks on CLEAR_S after the condition has been false.  A walker that is
dwelling at a task is not moving, so it is not stopped.  The robot's policies are untouched and see the walkers' actual positions.

Used through ``sim.run_trial(..., walkers=ReactiveWalkers(H))``; the default (``walkers=None``) is the main grid's code path.
"""
from __future__ import annotations

import numpy as np

STOP_GAP_M = 0.5
CLEAR_S = 1.0
LOOK_S = 0.5


class ReactiveWalkers:
    def __init__(self, H: np.ndarray, dt: float = 0.05):
        self.H, self.dt = H, dt
        n = H.shape[1]
        self.kv = np.zeros(n, int)  # each walker's private clock, in steps
        self.held = np.zeros(n, bool)
        self.clear_t = np.zeros(n)
        self.stops = 0  # number of times a walker started waiting
        self.held_steps = 0
        self.last = H.shape[0] - 1
        self.look = max(1, int(LOOK_S / dt))

    def step(self, x: float, y: float, yaw: float, half) -> np.ndarray:
        """Positions of the walkers at the current step given the robot's pose; advances their clocks."""
        from .sim import gaps  # sim does not import this module

        idx = np.arange(self.H.shape[1])
        pos = self.H[self.kv, idx]
        gp = gaps(x, y, yaw, half, pos)[0] if len(pos) else np.array([])
        vel = self.H[np.minimum(self.kv + self.look, self.last), idx] - pos
        moving = np.hypot(vel[:, 0], vel[:, 1]) > 1e-6
        ahead = (x - pos[:, 0]) * vel[:, 0] + (y - pos[:, 1]) * vel[:, 1] > 0
        cond = moving & ahead & (gp < STOP_GAP_M)
        newly = cond & ~self.held
        self.stops += int(newly.sum())
        self.held |= cond
        self.clear_t = np.where(cond, 0.0, np.where(self.held, self.clear_t + self.dt, 0.0))
        self.held &= ~(~cond & (self.clear_t >= CLEAR_S))
        self.held_steps += int(self.held.sum())
        advance = ~self.held
        self.kv = np.minimum(self.kv + advance.astype(int), self.last)
        return pos
