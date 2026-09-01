import numpy as np

from shortstop.env import Obstacle, ReachAvoid2D
from shortstop.experiment import make_scenario
from shortstop.expert import (
    _bypass_waypoints,
    _segment_blocked,
    generate_demo_pair,
    plan_waypoints,
    rollout_expert,
)


def test_bypass_waypoints_keep_constant_perpendicular_clearance_and_sides_are_opposite():
    obstacle = Obstacle(center=[3.0, 1.0], radius=0.6)
    travel_direction = np.array([1.0, 0.0])  # axis-aligned, so y-offset == perpendicular offset
    margin = 0.1
    r_safe = obstacle.radius + margin

    entry_u, exit_u = _bypass_waypoints(travel_direction, obstacle, margin=margin, mode="upper")
    entry_l, exit_l = _bypass_waypoints(travel_direction, obstacle, margin=margin, mode="lower")

    assert np.isclose(entry_u[1] - obstacle.center[1], r_safe)
    assert np.isclose(exit_u[1] - obstacle.center[1], r_safe)
    assert np.isclose(entry_l[1] - obstacle.center[1], -r_safe)
    assert exit_u[0] > entry_u[0]  # exit is further along travel_direction than entry


def test_segment_blocked_detects_direct_hit_and_clear_miss():
    p, q = np.array([0.0, 0.0]), np.array([2.0, 0.0])
    blocked, t = _segment_blocked(p, q, center=[1.0, 0.0], radius=0.3)
    assert blocked and np.isclose(t, 0.5)

    blocked, _ = _segment_blocked(p, q, center=[1.0, 5.0], radius=0.3)
    assert not blocked


def test_plan_waypoints_upper_and_lower_diverge_around_a_blocking_obstacle():
    start, goal = np.array([-4.0, 0.0]), np.array([4.0, 0.0])
    obstacles = [Obstacle(center=[0.0, 0.0], radius=0.6)]

    upper = plan_waypoints(start, goal, obstacles, margin=0.1, mode="upper")
    lower = plan_waypoints(start, goal, obstacles, margin=0.1, mode="lower")

    assert upper[1][1] > 0  # detours above the obstacle
    assert lower[1][1] < 0  # detours below the obstacle


def test_planner_reaches_goal_avoids_obstacles_and_is_multimodal_across_many_scenarios():
    """Statistical verification (Stage 6a, step 1) over many randomized
    scenarios (same generator as the ablation/baseline experiments).

    `make_scenario`'s 3 obstacles (radius 0.4-0.8, centers in a 4x3 box)
    overlap each other in a large fraction of draws -- this single-obstacle-
    at-a-time detour planner isn't meant to solve that general case, it
    just needs to *cleanly refuse* it (RuntimeError) rather than emit an
    unsafe path, which is why success is checked well below 100%:
    `generate_demo_pair` (step 2's building block) already drops whatever
    fails here, so building the dataset just means oversampling scenario
    seeds. Whenever a scenario *does* succeed for both modes, the two
    resulting trajectories should be clearly distinct -- not near-
    duplicates -- confirming real multimodality, and no successful rollout
    should ever violate (the planner's margin exists precisely so scripted-
    expert demos are never the source of unsafe training data).
    """
    n_scenarios = 200
    reach_ok = {"upper": 0, "lower": 0}
    violated_ct = {"upper": 0, "lower": 0}
    contested = 0
    separations = []

    for i in range(n_scenarios):
        rng = np.random.default_rng(1000 + i)
        start, goal, obstacles = make_scenario(rng)

        for mode in ("upper", "lower"):
            try:
                wp = plan_waypoints(start, goal, obstacles, margin=0.1, mode=mode)
            except RuntimeError:
                continue
            env = ReachAvoid2D(
                start=start, goal=goal, obstacles=obstacles,
                rng=np.random.default_rng(2000 + i),
            )
            result = rollout_expert(env, wp)
            if result["violated"]:
                violated_ct[mode] += 1

        pair = generate_demo_pair(start, goal, obstacles, rng=rng)
        for mode in pair:
            reach_ok[mode] += 1

        if "upper" in pair and "lower" in pair:
            u, l = pair["upper"]["states"], pair["lower"]["states"]
            n = min(len(u), len(l))
            if not np.allclose(u[:n], l[:n], atol=1e-6):
                contested += 1
                separations.append(float(np.mean(np.linalg.norm(u[:n] - l[:n], axis=1))))

    for mode in ("upper", "lower"):
        assert violated_ct[mode] == 0
        assert reach_ok[mode] >= 0.5 * n_scenarios

    assert contested >= 0.2 * n_scenarios  # obstacles are sparse but should block sometimes
    assert np.mean(separations) > 0.3  # modes are meaningfully different, not near-duplicates
