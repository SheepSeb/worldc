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


# ------------------------------------------------------------------------------------------------------ walk-around
AROUND_GAP_M = 0.3  # clearance between the robot's footprint and the walker's disk when walking around it
LOOK_AHEAD_S = 2.5  # the part of the path ahead that is checked against the robot
CHECK_S = 0.2  # how often the path ahead is checked
REPLAN_EVERY_S = 0.4  # minimum time between two plans of one walker
REJOIN_MAX_S = 30.0  # how far ahead on the tour a detour may rejoin (a robot parked on a task spot is passed after the dwell)
TRIES = 6  # rejoin candidates tried per plan


class WalkAroundWalkers:
    """Walkers that walk around the robot instead of waiting for it (no waiting at all).

    A walker follows its seeded tour (``H[:, i]``) on a private clock.  Every CHECK_S it checks the next LOOK_AHEAD_S of its path (the tour,
    or the detour it is on) against the robot's footprint inflated by AROUND_GAP_M plus its own radius.  If the robot is in the way it plans an
    A* path on the people grid (HUMAN_RADIUS_M clear of static obstacles, as for the tours) with the inflated footprint masked, to the first
    point of its tour (at least LOOK_AHEAD_S ahead, at most REJOIN_MAX_S) that is clear of the robot, walks it at its own preferred speed and
    resumes the tour from that point.  A walker whose own spot is blocked (the robot parked on it, or on its task spot) leaves early: the rest of
    the dwell is skipped.  If no detour exists it keeps walking the original path (it does not wait), so a contact can still happen.
    Used through ``sim.run_trial(..., walkers=WalkAroundWalkers(H, world, info))``.
    """

    def __init__(self, H: np.ndarray, world, info: list[dict], dt: float = 0.05):
        from ..check.safety import HUMAN_RADIUS_M

        self.H, self.world, self.g, self.dt = H, world, world.hgrid, dt
        n = H.shape[1]
        self.last = H.shape[0] - 1
        self.kv = np.zeros(n, int)
        self.mode = np.zeros(n, int)  # 0 tour, 1 detour
        self.path: list = [None] * n
        self.s = np.zeros(n)
        self.rej = np.zeros(n, int)
        self.pos = H[0].copy()
        self.speed = np.array([float(i.get("speed_mps", 1.3)) for i in info])
        self.next_check = np.zeros(n)
        self.next_plan = np.zeros(n)
        self.clock = 0.0
        self.R = AROUND_GAP_M + HUMAN_RADIUS_M  # walker-centre distance to the robot's rectangle
        self.look = max(2, int(LOOK_AHEAD_S / dt))
        self.stats = {"walker_replans": 0, "walker_detour_s": 0.0, "walker_no_path": 0}

    # -- geometry
    def _dist(self, pts: np.ndarray, rob) -> np.ndarray:
        x, y, yaw, (hx, hy) = rob
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = pts[:, 0] - x, pts[:, 1] - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        return np.hypot(np.maximum(abs(lx) - hx, 0.0), np.maximum(abs(ly) - hy, 0.0))

    def _free(self, rob) -> np.ndarray:
        """The people grid with the robot's inflated footprint masked (only a window around the robot is touched)."""
        g = self.g
        x, y = rob[0], rob[1]
        reach = float(np.hypot(*rob[3])) + self.R + 0.2
        r0, c0 = g.clamp(g.cell(x - reach, y + reach))
        r1, c1 = g.clamp(g.cell(x + reach, y - reach))
        free = g.free.copy()
        rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        pts = np.stack([g.x0 + (cc.ravel() + 0.5) * g.res, g.y1 - (rr.ravel() + 0.5) * g.res], 1)
        free[r0:r1 + 1, c0:c1 + 1] &= (self._dist(pts, rob) > self.R).reshape(rr.shape)
        return free

    def _plan(self, free, a, b):
        from .world import plan

        g = self.g
        r0, c0 = g.clamp(g.cell(*a))
        sl = (slice(max(r0 - 3, 0), r0 + 4), slice(max(c0 - 3, 0), c0 + 4))
        free = free.copy()
        free[sl] = g.free[sl]  # the start may be inside the mask (the robot is on top of the walker): let it out
        return plan(g, free, a, b)

    def _detour(self, i: int, p, rob, free) -> bool:
        """Plan to the first clear point of the tour ahead; True (and the walker is on a detour) if one was found."""
        kv = self.kv[i]
        cand = [j for j in range(min(kv + self.look, self.last), min(kv + int(REJOIN_MAX_S / self.dt), self.last) + 1, 5)]
        if not cand:
            return False
        P = self.H[cand, i]
        clear = [cand[m] for m in np.flatnonzero(self._dist(P, rob) > self.R + 0.1)][:TRIES]
        for j in clear:
            path = self._plan(free, p, self.H[j, i])
            if path is not None and len(path) >= 2:
                # from the tour, p is already this step's position: stay there this step; from a detour, p is the last output: one step on
                self.s[i] = -self.speed[i] * self.dt if self.mode[i] == 0 else 0.0
                self.mode[i], self.path[i], self.rej[i] = 1, path, j
                self.stats["walker_replans"] += 1
                return True
        return False

    def _blocked_ahead(self, i: int, rob) -> bool:
        if self.mode[i] == 0:
            pts = self.H[self.kv[i]:min(self.kv[i] + self.look, self.last) + 1:3, i]
        else:
            path, s = self.path[i], self.s[i]
            seg = np.hypot(*np.diff(path, axis=0).T)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            ss = np.arange(s, min(s + LOOK_AHEAD_S * self.speed[i], cum[-1]) + 0.05, 0.15)
            pts = np.stack([np.interp(ss, cum, path[:, 0]), np.interp(ss, cum, path[:, 1])], 1)
        return bool((self._dist(np.vstack([pts, self.pos[i][None]]), rob) <= self.R).any())

    def step(self, x: float, y: float, yaw: float, half) -> np.ndarray:
        rob = (x, y, yaw, half)
        free = None
        out = self.pos.copy()
        for i in range(self.H.shape[1]):
            p = self.H[self.kv[i], i] if self.mode[i] == 0 else self.pos[i]
            if self.clock >= self.next_check[i]:
                self.next_check[i] = self.clock + CHECK_S
                if self._blocked_ahead(i, rob) and self.clock >= self.next_plan[i]:
                    self.next_plan[i] = self.clock + REPLAN_EVERY_S
                    free = self._free(rob) if free is None else free
                    if not self._detour(i, p, rob, free):
                        self.stats["walker_no_path"] += 1
            if self.mode[i] == 1:
                path = self.path[i]
                seg = np.hypot(*np.diff(path, axis=0).T)
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                self.s[i] += self.speed[i] * self.dt
                self.stats["walker_detour_s"] += self.dt
                if self.s[i] >= cum[-1]:
                    out[i] = self.H[min(self.rej[i], self.last), i]
                    self.mode[i], self.kv[i] = 0, min(self.rej[i] + 1, self.last)
                else:
                    out[i] = (np.interp(self.s[i], cum, path[:, 0]), np.interp(self.s[i], cum, path[:, 1]))
            else:
                out[i] = self.H[self.kv[i], i]
                self.kv[i] = min(self.kv[i] + 1, self.last)
        self.pos = out
        self.clock += self.dt
        return out.copy()

    def summary(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.stats.items()}
