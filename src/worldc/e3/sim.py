"""The 2D trial: a diff-drive robot visits the scene's goals among seeded workers, under one of three policies.

* ``blind``: plan once per leg on the static grid and follow it, ignoring people (the current ``worldc drive``).
* ``stop``: as blind, but halt while a person is within STOP_DIST_M ahead (gap from the footprint, person in front of the
  robot's centre); resume once that has been clear for STOP_CLEAR_S.
* ``aware``: at AWARE_HZ, people are obstacles inflated by AWARE_INFLATE_M; when the path ahead is blocked, replan around
  them (wait if there is no way through); speed scales from full at SLOW_FAR_M to zero at SLOW_NEAR_M from the nearest person.

All three share the planner (A* + smoothing on the robot-inflated grid) and the controller (the pure pursuit of
``render/isaac_drive.py``: look-ahead, turn in place for large heading errors, slew-limited v and w, align at each goal).
The robot is a unicycle whose body is the world's footprint rectangle (``world.robot``); people are kinematic and do not
react.  Distances are gaps between the footprint and a person's disk (radius HUMAN_RADIUS_M); a gap of 0 is a collision.

The robot counts as *moving* when |v| > MOVING_V or |w| > MOVING_W: turning in place counts as moving (the footprint
sweeps).  Since people do not react, collisions are split by whether the robot was moving (the robot's doing; the primary
metric) or standing (a person walked into it).

Outcome of a trial: ``success`` (every goal reached within the timeout and no robot-moving collision); otherwise
``collision`` (a robot-moving collision), ``frozen`` (no progress of FROZEN_M within FROZEN_S while driving),
``timeout``, ``goal_unreachable`` (no static plan) or ``goal_missed`` (finished but a goal was not reached).
"""
from __future__ import annotations

import math

import numpy as np

from ..check.safety import HUMAN_RADIUS_M
from .world import World, plan

DT = 0.05
VMAX, WMAX, LOOK = 1.0, 1.4, 0.8  # as isaac_drive
ACC_V, DEC_V, ACC_W = 2.0, 3.0, 8.0  # slew limits (m/s^2, rad/s^2), as isaac_drive
REACH_M, YAW_TOL, ALIGN_MAX_S, PAUSE_S = 0.4, 0.12, 4.0, 0.9
STOP_DIST_M, STOP_CLEAR_S = 1.0, 1.0
AWARE_HZ, AWARE_INFLATE_M, SLOW_FAR_M, SLOW_NEAR_M, AWARE_LOOKAHEAD_M = 5.0, 0.5, 2.0, 0.5, 4.0
NEAR_BANDS = (0.5, 1.0, 2.0)
MOVING_BANDS = (0.5, 1.0)
MOVING_V, MOVING_W = 0.05, 0.2  # m/s, rad/s
FROZEN_S, FROZEN_M = 30.0, 0.10
POLICIES = ("blind", "stop", "aware")
OUTCOMES = ("success", "collision", "frozen", "timeout", "goal_unreachable", "goal_missed")


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def body_pts(half) -> np.ndarray:
    """Footprint sample points in the robot frame: corners, edge midpoints, centre."""
    hx, hy = half
    return np.array([(sx * hx, sy * hy) for sx in (-1, 0, 1) for sy in (-1, 0, 1)], float)


def gaps(x: float, y: float, yaw: float, half, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gap from the robot rectangle (robot-frame ``yaw``) to each person's disk, and each person's robot-frame x."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = H[:, 0] - x, H[:, 1] - y
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    d = np.hypot(np.maximum(abs(lx) - half[0], 0.0), np.maximum(abs(ly) - half[1], 0.0))
    return np.maximum(d - HUMAN_RADIUS_M, 0.0), lx


class Path:
    def __init__(self, pts: np.ndarray):
        self.pts = pts
        self.seg = np.hypot(*np.diff(pts, axis=0).T)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg)])

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    def closest(self, px, py, s_from):
        best = (1e9, s_from)
        for i in range(max(0, int(np.searchsorted(self.cum, s_from, side="right")) - 2), len(self.seg)):
            if self.cum[i] > s_from + 6.0:
                break
            a, e = self.pts[i], self.pts[i + 1] - self.pts[i]
            t = float(np.clip(((px - a[0]) * e[0] + (py - a[1]) * e[1]) / max(e @ e, 1e-9), 0, 1))
            d = math.hypot(px - (a[0] + t * e[0]), py - (a[1] + t * e[1]))
            if d < best[0]:
                best = (d, self.cum[i] + t * self.seg[i])
        return best[1]

    def point_at(self, s):
        s = min(max(s, 0.0), self.cum[-1])
        i = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.seg) - 1)
        t = (s - self.cum[i]) / max(self.seg[i], 1e-9)
        return self.pts[i] + t * (self.pts[i + 1] - self.pts[i])


def static_legs(world: World) -> list[np.ndarray | None]:
    """Static plans spawn -> goal 1 -> goal 2 ... (None for an unreachable leg)."""
    out, cur = [], world.spawn[:2]
    for g in world.goals:
        out.append(plan(world.grid, world.grid.free, cur, g["pose"][:2]))
        cur = g["pose"][:2]
    return out


def timeout_s(world: World, legs) -> float:
    return 3.0 * sum((Path(p).length if p is not None else 0.0) / VMAX + 6.0 for p in legs) + 30.0


def _human_block(world: World, H: np.ndarray) -> np.ndarray:
    """Cells where the robot centre may not be: within robot radius + person radius + inflation of any person."""
    g = world.grid
    blocked = np.zeros_like(g.free)
    R = world.robot_r + HUMAN_RADIUS_M + AWARE_INFLATE_M
    k = int(math.ceil(R / g.res))
    for hx, hy in H:
        r0, c0 = g.cell(hx, hy)
        rs, cs = slice(max(r0 - k, 0), min(r0 + k + 1, g.ny)), slice(max(c0 - k, 0), min(c0 + k + 1, g.nx))
        rr = (g.y1 - (np.arange(rs.start, rs.stop) + 0.5) * g.res)[:, None]
        cc = (g.x0 + (np.arange(cs.start, cs.stop) + 0.5) * g.res)[None, :]
        blocked[rs, cs] |= (rr - hy) ** 2 + (cc - hx) ** 2 <= R * R
    return blocked


def run_trial(world: World, H: np.ndarray, policy: str, legs=None, timeout: float | None = None) -> dict:
    """Simulate one trial; ``H`` is (steps, n, 2) worker positions at DT.  Returns metrics, events and the robot trajectory.

    ``th`` is the travel heading; the body's yaw is ``th - world.phi0`` (as isaac_drive: heading = robot yaw + phi0)."""
    assert policy in POLICIES
    legs = static_legs(world) if legs is None else legs
    timeout = timeout_s(world, legs) if timeout is None else timeout
    n_steps = min(int(timeout / DT), H.shape[0] - 1)
    g, hg, phi0 = world.grid, world.hgrid, world.phi0
    walk, bpts = world.walkways, body_pts(world.body_half)
    x, y, yaw0 = world.spawn
    th = wrap(yaw0 + phi0)
    v = w = 0.0
    leg, state, state_t = 0, "drive", 0.0
    path = Path(legs[0]) if legs and legs[0] is not None else None
    s_prog = 0.0
    clear_since, holding = None, False
    next_aware, next_replan = 0.0, 0.0
    replans, hold_s = 0, 0.0
    in_contact = np.zeros(H.shape[1], bool)
    collisions, moving_hits, events = 0, 0, []
    min_gap, min_gap_moving = math.inf, math.inf
    near = {b: 0.0 for b in NEAR_BANDS}
    near_moving = {b: 0.0 for b in MOVING_BANDS}
    stationary_s = stationary_walkway_s = 0.0
    hh_overlap = 0.0
    traj, arrivals = [], []
    prog_t, prog_xy = 0.0, (x, y)
    end = None
    t = 0.0
    for k in range(n_steps):
        t = k * DT
        Hk = H[k]
        yaw = th - phi0
        moving = abs(v) > MOVING_V or abs(w) > MOVING_W
        # ---- metrics
        if not moving:
            stationary_s += DT
            c, s = math.cos(yaw), math.sin(yaw)
            if any(walk[hg.clamp(hg.cell(x + c * a - s * b, y + s * a + c * b))] for a, b in bpts):
                stationary_walkway_s += DT
        gp = gaps(x, y, yaw, world.body_half, Hk)[0] if len(Hk) else np.array([])
        if len(gp):
            ahead_x = math.cos(th) * (Hk[:, 0] - x) + math.sin(th) * (Hk[:, 1] - y)  # along the travel heading, from the centre
            gmin = float(gp.min())
            min_gap = min(min_gap, gmin)
            for b in NEAR_BANDS:
                near[b] += DT * (gmin < b)
            if moving:
                min_gap_moving = min(min_gap_moving, gmin)
                for b in MOVING_BANDS:
                    near_moving[b] += DT * (gmin < b)
            hit = gp <= 0.0
            for i in np.flatnonzero(hit & ~in_contact):
                collisions += 1
                moving_hits += moving
                events.append({"type": "collision", "t": round(t, 2), "worker": int(i), "xy": [round(x, 2), round(y, 2)],
                               "robot_v": round(v, 2), "robot_w": round(w, 2), "robot_moving": bool(moving)})
            in_contact = hit
            if len(Hk) > 1:
                dd = np.hypot(*(Hk[:, None, :] - Hk[None, :, :]).transpose(2, 0, 1))
                hh_overlap += DT * float((dd[np.triu_indices(len(Hk), 1)] < 2 * HUMAN_RADIUS_M).any())
        if k % 4 == 0:
            traj.append([round(t, 2), round(x, 3), round(y, 3), round(th, 3), round(v, 3)])
        # ---- done / failed?
        if leg >= len(legs):
            end = "done"
            break
        if path is None:
            end = "goal_unreachable"
            break
        if state == "drive":
            if math.hypot(x - prog_xy[0], y - prog_xy[1]) > FROZEN_M:
                prog_t, prog_xy = t, (x, y)
            elif t - prog_t > FROZEN_S:
                end = "frozen"
                break
        else:
            prog_t, prog_xy = t, (x, y)
        gx, gy, gyaw = world.goals[leg]["pose"]
        # ---- policy: may replace the path (aware) or hold the robot (stop, aware)
        scale = 1.0
        if policy == "stop" and state == "drive" and len(gp):
            if ((gp < STOP_DIST_M) & (ahead_x > 0)).any():
                holding, clear_since = True, None
            elif holding:
                clear_since = t if clear_since is None else clear_since
                holding = t - clear_since < STOP_CLEAR_S
        if policy == "aware" and len(gp):
            scale = float(np.clip((gp.min() - SLOW_NEAR_M) / (SLOW_FAR_M - SLOW_NEAR_M), 0.0, 1.0))
            if state == "drive" and t >= next_aware:
                next_aware = t + 1.0 / AWARE_HZ
                blocked = _human_block(world, Hk)
                ss = np.arange(s_prog, min(s_prog + AWARE_LOOKAHEAD_M, path.length), g.res)
                if any(blocked[g.clamp(g.cell(*path.point_at(s)))] for s in ss):
                    if t >= next_replan:
                        free = g.free & ~blocked
                        r0, c0 = g.clamp(g.cell(x, y))
                        free[max(r0 - 3, 0):r0 + 4, max(c0 - 3, 0):c0 + 4] = g.free[max(r0 - 3, 0):r0 + 4, max(c0 - 3, 0):c0 + 4]
                        p = plan(g, free, (x, y), (gx, gy))
                        replans += 1
                        if p is not None:
                            path, s_prog, holding = Path(p), 0.0, False
                        else:
                            holding, next_replan = True, t + 1.0
                else:
                    holding = False
        # ---- controller (isaac_drive's pure pursuit)
        v_t = w_t = 0.0
        if state == "drive":
            s_prog = path.closest(x, y, s_prog)
            dist_end = math.hypot(gx - x, gy - y)
            tx, ty = path.point_at(min(s_prog + LOOK, path.length))
            alpha = wrap(math.atan2(ty - y, tx - x) - th)
            if dist_end < 0.3:
                state, state_t = "align", t
            elif holding:
                hold_s += DT
            elif abs(alpha) > 0.7:
                w_t = float(np.clip(2.0 * alpha, -WMAX, WMAX))
            else:
                v_t = scale * VMAX * float(np.clip(dist_end / 1.5, 0.25, 1.0)) * (1.0 - 0.6 * abs(alpha))
                w_t = float(np.clip(2.0 * v_t * math.sin(alpha) / LOOK + 0.8 * alpha, -WMAX, WMAX))
        if state == "align":
            err = wrap(gyaw - th)
            w_t = float(np.clip(1.8 * err, -1.0, 1.0)) if abs(err) > YAW_TOL else 0.0
            if abs(err) <= YAW_TOL or t - state_t > ALIGN_MAX_S:
                arrivals.append({"goal": world.goals[leg]["id"], "reached": math.hypot(gx - x, gy - y) < REACH_M, "t": round(t, 2)})
                state, state_t = "pause", t
        if state == "pause" and t - state_t > PAUSE_S:
            leg += 1
            state, s_prog, holding, clear_since = "drive", 0.0, False, None
            if leg < len(legs):
                path = Path(legs[leg]) if legs[leg] is not None else None
                if path is not None and policy == "aware":  # the static plan starts at the goal pose; follow from where the robot is
                    path = Path(np.vstack([[x, y], legs[leg][1:]]))
        v += float(np.clip(v_t - v, -DEC_V * DT, ACC_V * DT))
        w += float(np.clip(w_t - w, -ACC_W * DT, ACC_W * DT))
        th = wrap(th + w * DT)
        x, y = x + v * math.cos(th) * DT, y + v * math.sin(th) * DT
    else:
        t = n_steps * DT
    end = end or "timeout"
    reached = all(a["reached"] for a in arrivals)
    if end == "done" and reached and moving_hits == 0:
        outcome = "success"
    elif moving_hits:
        outcome = "collision"
    elif end == "done":
        outcome = "goal_missed"
    else:
        outcome = end
    fin = lambda m: round(m, 3) if math.isfinite(m) else None  # noqa: E731
    return {"policy": policy, "success": outcome == "success", "outcome": outcome, "end": end, "frozen": end == "frozen",
            "collisions_robot_moving": moving_hits, "collisions": collisions, "collisions_robot_stationary": collisions - moving_hits,
            "time_to_goal_s": round(t, 2) if end == "done" else None, "sim_s": round(t, 2),
            "min_gap_m": fin(min_gap), "min_gap_moving_m": fin(min_gap_moving),
            **{f"time_within_{b:g}m_s": round(near[b], 2) for b in NEAR_BANDS},
            **{f"time_within_{b:g}m_moving_s": round(near_moving[b], 2) for b in MOVING_BANDS},
            "stationary_s": round(stationary_s, 2), "stationary_in_walkway_s": round(stationary_walkway_s, 2),
            "human_human_overlap_s": round(hh_overlap, 2), "goals_reached": sum(a["reached"] for a in arrivals), "n_goals": len(legs),
            "replans": replans, "hold_s": round(hold_s, 2), "events": events, "arrivals": arrivals, "trajectory": traj}
