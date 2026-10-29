"""E3 2D simulator: seeded walkers replay exactly, and the three policies and the collision split behave as defined."""
from __future__ import annotations

import numpy as np
import pytest

from worldc.e3 import sim as S
from worldc.e3.walkers import workers
from worldc.e3.world import World

ROBOT = {"asset": "test", "size_m": [0.73, 0.9], "travel_heading_rad": 0.0, "source": "test"}


@pytest.fixture
def room() -> World:
    """An empty 8 x 4 m room; the robot drives from x = -3 to x = 3 along y = 0."""
    return World("room", (-4.0, 4.0), (-2.0, 2.0), 0.05, {}, [], (-3.0, 0.0, 0.0), ROBOT, 0.577,
                 [{"id": "g", "pose": (3.0, 0.0, 0.0)}], {"door": [(-3.0, 1.5)], "bench": [(3.0, 1.5), (0.0, -1.5), (2.0, -1.5)]})


def standing(x, y, steps=4000):
    return np.tile(np.array([[x, y]], float), (steps, 1, 1))


def test_workers_replay_exactly(room):
    a, info = workers(room, 7, 3, 60.0, S.DT)
    b, _ = workers(room, 7, 3, 60.0, S.DT)
    assert np.array_equal(a, b)
    one, _ = workers(room, 7, 1, 60.0, S.DT)
    assert np.array_equal(one[:, 0], a[:, 0])  # worker 0 of a seed is the same person in every tier
    assert not np.array_equal(workers(room, 8, 3, 60.0, S.DT)[0], a)
    assert all(3 <= len(w["tour"]) <= 6 or len(set(map(tuple, w["tour"]))) < 3 for w in info)


def test_policies_against_a_person_beside_the_route(room):
    H = standing(0.0, -0.6)  # 0.6 m off the straight route: the footprint (half width 0.45) plus the person (0.26) overlap
    blind = S.run_trial(room, H, "blind")
    assert blind["collisions_robot_moving"] >= 1 and blind["outcome"] == "collision"
    stop = S.run_trial(room, H, "stop")
    assert stop["collisions_robot_moving"] == 0 and stop["outcome"] == "frozen" and stop["frozen"]
    aware = S.run_trial(room, H, "aware")
    assert aware["outcome"] == "success" and aware["replans"] >= 1 and aware["min_gap_m"] > 0.3
    assert aware["time_within_0.5m_moving_s"] == 0.0


def test_person_walking_into_a_holding_robot_is_stationary(room):
    H = standing(-1.9, 0.0)  # ahead of the parked robot: stop-and-wait holds
    H[100:140, 0, 0] = np.linspace(-1.9, -3.0, 40)  # then the person walks onto it
    H[140:, 0, 0] = -3.0
    r = S.run_trial(room, H, "stop")
    assert r["collisions"] == 1 and r["collisions_robot_stationary"] == 1 and r["collisions_robot_moving"] == 0
    assert r["outcome"] != "collision" and r["stationary_s"] > 5.0


def test_turning_in_place_counts_as_moving(room):
    turned = World("room", room.xr, room.yr, room.res, {}, [], (-3.0, 0.0, np.pi), ROBOT, 0.577, room.goals, room.raw_pois)
    H = standing(-3.0, 0.75)  # beside the robot: its corner sweeps through the person while it turns around
    r = S.run_trial(turned, H, "blind")
    assert r["events"] and r["events"][0]["robot_moving"] and abs(r["events"][0]["robot_v"]) <= S.MOVING_V
