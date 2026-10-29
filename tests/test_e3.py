"""E3 2D simulator: seeded walkers replay exactly, and the three policies and the collision split behave as defined."""
from __future__ import annotations

import numpy as np
import pytest

from worldc.e3 import sim as S
from worldc.e3.walkers import workers
from worldc.e3.world import World

ROBOT = {"asset": "test", "size_m": [0.73, 0.9], "travel_heading_rad": 0.0, "source": "test", "brake_decel_mps2": 3.0}


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
    fields = S.run_trial(room, H, "fields")
    assert fields["collisions_robot_moving"] == 0 and fields["outcome"] == "frozen" and fields["frozen"]
    aware = S.run_trial(room, H, "aware")
    assert aware["outcome"] == "success" and aware["replans"] >= 1 and aware["min_gap_m"] > 0.3
    assert aware["time_within_0.5m_moving_s"] == 0.0


def test_person_walking_into_a_holding_robot_is_stationary(room):
    H = standing(-1.9, 0.0)  # ahead of the parked robot: the protective field holds
    H[100:140, 0, 0] = np.linspace(-1.9, -3.0, 40)  # then the person walks onto it
    H[140:, 0, 0] = -3.0
    r = S.run_trial(room, H, "fields")
    assert r["collisions"] == 1 and r["collisions_robot_stationary"] == 1 and r["collisions_robot_moving"] == 0
    assert r["outcome"] != "collision" and r["stationary_s"] > 5.0


def test_turning_in_place_counts_as_moving(room):
    turned = World("room", room.xr, room.yr, room.res, {}, [], (-3.0, 0.0, np.pi), ROBOT, 0.577, room.goals, room.raw_pois)
    H = standing(-3.0, 0.75)  # beside the robot: its corner sweeps through the person while it turns around
    r = S.run_trial(turned, H, "blind")
    assert r["events"] and r["events"][0]["robot_moving"] and abs(r["events"][0]["robot_v"]) <= S.MOVING_V


def test_align_turn_is_guarded(room):
    """A person steps beside the goal just as the robot starts its align turn there (goal yaw: turn around)."""
    turn = World("room", room.xr, room.yr, room.res, {}, [], room.spawn, ROBOT, 0.577, [{"id": "g", "pose": (3.0, 0.0, np.pi)}],
                 room.raw_pois)
    far = standing(-3.6, 1.6)
    first = S.run_trial(turn, far, "blind")
    t_align = next(r[0] for r in first["trajectory"] if np.hypot(3.0 - r[1], r[2]) < 0.3)
    H = far.copy()
    H[int(t_align / S.DT) + 4:] = (3.0, 0.75)
    blind = S.run_trial(turn, H, "blind")
    assert blind["collisions_robot_moving"] >= 1 and abs(blind["events"][0]["robot_v"]) <= S.MOVING_V  # swept while turning
    r = S.run_trial(turn, H, "aware")
    assert r["collisions_robot_moving"] == 0 and r["hold_s"] > 0 and r["outcome"] == "frozen", r["outcome"]
    assert r["stationary_near_1m_s"] > 25.0
    # fields: the person is beside the footprint, never in the (widened) fields ahead, so the corner sweep hits them
    r = S.run_trial(turn, H, "fields")
    assert r["collisions_robot_moving"] >= 1 and r["protective_stops"] == 0


def test_fields_stop_slow_and_switch_with_speed(room):
    prot, warn = S.field_lengths(1.0, 3.0)
    assert prot == pytest.approx(0.1 + 1 / 6 + 0.25) and warn == pytest.approx(prot + 1.0)
    assert S.field_lengths(0.0, 3.0)[0] == pytest.approx(0.25)
    H = standing(0.0, -0.6)
    r = S.run_trial(room, H, "fields")
    assert r["protective_stops"] >= 1 and r["field_warning_s"] > 0 and r["min_gap_moving_m"] > 0
    far = standing(0.0, -1.5)  # clear of both fields: drives as blind
    assert S.run_trial(room, far, "fields")["trajectory"] == S.run_trial(room, far, "blind")["trajectory"]


def test_collision_from_behind_is_tagged(room):
    H = standing(-3.9, 0.0)
    H[:, 0, 0] = np.minimum(-3.9 + 1.8 * S.DT * np.arange(len(H)), 3.9)  # walks along the route faster than the robot
    r = S.run_trial(room, H, "blind")
    assert r["events"][0]["from_behind"] and r["events"][0]["robot_moving"] and r["collisions_robot_moving_from_behind"] >= 1


def test_walkers_do_not_depend_on_path_query_order(room):
    a, _ = workers(room, 3, 3, 60.0, S.DT)
    other = World("room", room.xr, room.yr, room.res, {}, [], room.spawn, ROBOT, 0.577, room.goals, room.raw_pois)
    pts = [p for v in other.pois.values() for p in v]
    for p in pts[::-1]:  # ask for every path the other way round first
        for q in pts:
            if p != q:
                other.human_path(p, q)
    assert np.array_equal(workers(other, 3, 3, 60.0, S.DT)[0], a)
