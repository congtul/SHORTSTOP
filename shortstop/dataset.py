"""Behavior-cloning dataset construction for Stage 6a (see shortstop/expert.py).

Turns scripted-expert rollouts into (state, obstacle_vector) -> action_chunk
training windows for a small diffusion/flow policy. Every window comes from
a trajectory that `generate_demo_pair` already confirmed reached the goal
without violating any obstacle -- there is no other filter, so nothing that
failed to reach the goal (or collided) ever enters the dataset.
"""
import numpy as np

from .env import encode_obstacles
from .experiment import make_scenario
from .expert import generate_demo_pair

__all__ = ["encode_obstacles", "windows_from_demo", "build_dataset"]


def windows_from_demo(demo, obstacle_vec, horizon):
    """Slice one successful rollout into (state, obstacle_vec, action_chunk)
    windows. `demo["actions"][t]` was chosen by the expert while looking at
    `demo["states"][t]` (see rollout_expert), so that pairing is exact.
    """
    states, actions = demo["states"], demo["actions"]
    for t in range(len(actions) - horizon + 1):
        yield states[t], obstacle_vec, actions[t:t + horizon]


def build_dataset(target_demos=500, horizon=8, seed_start=0, margin=0.1, max_seeds=20_000):
    """Generate scripted-expert demos until `target_demos` successful
    (scenario, mode) trajectories are collected, then slice every one into
    BC windows.

    Scenario seeds are drawn sequentially from `seed_start`; each scenario
    can contribute 0, 1, or 2 demos (upper/lower) depending on how many
    modes the planner/rollout succeeded on (see shortstop/expert.py and
    scripts/verify_expert.py -- roughly 65% per mode, so this naturally
    oversamples seeds past `target_demos`).
    """
    states_out, obs_out, chunks_out = [], [], []
    n_demos = 0
    n_scenarios = 0
    per_mode = {"upper": 0, "lower": 0}
    seed = seed_start

    while n_demos < target_demos and seed < seed_start + max_seeds:
        rng = np.random.default_rng(seed)
        start, goal, obstacles = make_scenario(rng)
        obstacle_vec = encode_obstacles(obstacles)
        pair = generate_demo_pair(start, goal, obstacles, margin=margin, rng=rng)
        n_scenarios += 1

        for mode, demo in pair.items():
            per_mode[mode] += 1
            n_demos += 1
            for s, ov, chunk in windows_from_demo(demo, obstacle_vec, horizon):
                states_out.append(s)
                obs_out.append(ov)
                chunks_out.append(chunk)
        seed += 1

    if n_demos < target_demos:
        raise RuntimeError(f"only collected {n_demos}/{target_demos} demos within {max_seeds} seeds")

    return {
        "states": np.asarray(states_out),
        "obstacle_vecs": np.asarray(obs_out),
        "action_chunks": np.asarray(chunks_out),
        "horizon": horizon,
        "n_scenarios_attempted": n_scenarios,
        "n_demos": n_demos,
        "n_demos_upper": per_mode["upper"],
        "n_demos_lower": per_mode["lower"],
        "seed_start": seed_start,
        "seed_end": seed,
    }
