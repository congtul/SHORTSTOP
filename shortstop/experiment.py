"""Shared episode-running harness.

Every stage script (Reach-only, +STL-to-go, +CE search, +Repair, ...) uses
the exact same scenario generator and Propose -> (shield) -> execute loop;
only the shield implementation differs across ablation rows. Keeping this in
one place means every stage is run under identical conditions (same seeds,
same env/policy config), so results are directly comparable -- which is the
whole point of an ablation table.
"""
import time

import numpy as np

from .env import Obstacle, ReachAvoid2D
from .policy import GaussianChunkPolicy
from .reach import nominal_rollout


def make_scenario(rng):
    start = np.array([-4.0, 0.0])
    goal = np.array([4.0, 0.0])
    obstacles = [
        Obstacle(center=rng.uniform([-2.0, -1.5], [2.0, 1.5]), radius=rng.uniform(0.4, 0.8))
        for _ in range(3)
    ]
    return start, goal, obstacles


def nominal_violates(state, chunk, dt, obstacles):
    """Privileged check: would this (rejected) chunk actually have hit an obstacle?

    Uses the noise-free nominal rollout as ground truth -- "privileged" here
    means we get to look at the outcome directly instead of only the shield's
    conservative reachtube bound.
    """
    path = nominal_rollout(state, chunk, dt)
    return any(o.contains(p) for p in path[1:] for o in obstacles)


def run_episode(
    shield_factory,
    rng,
    horizon=8,
    n_candidates=8,
    dt=0.1,
    w_bar=0.02,
    shield_w_bar=None,
    record_decisions=False,
):
    """Run one episode under `shield_factory`.

    `shield_factory(goal, obstacles, dt, w_bar) -> shield instance`, or
    `None` for the Unshielded baseline (first action of the first candidate
    is executed with no filtering at all).

    `w_bar` is the *true* disturbance bound used to generate the env's
    actual noise. `shield_w_bar`, if given, is what the shield is told to
    certify against instead -- defaults to `w_bar` (the shield gets the
    ground truth for free), but pass a value from
    shortstop.calibration.calibrate_w_bar() to certify against a realistic,
    data-driven estimate instead (see Table VII's calibration recipe).
    """
    start, goal, obstacles = make_scenario(rng)
    env = ReachAvoid2D(start=start, goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar, rng=rng)
    policy = GaussianChunkPolicy(goal=goal, horizon=horizon, n_candidates=n_candidates, rng=rng)
    certified_w_bar = w_bar if shield_w_bar is None else shield_w_bar
    shield = shield_factory(goal, obstacles, dt, certified_w_bar) if shield_factory is not None else None

    state = env.reset()
    trajectory = [state.copy()]
    activations = 0
    violated = False
    reached = False
    latencies_ms = []
    rejected_total = 0
    rejected_truly_unsafe = 0
    repair_attempts = 0
    repair_successes = 0
    decisions = [] if record_decisions else None

    for _ in range(env.max_steps):
        candidates = policy.propose(state)
        if shield is not None:
            t0 = time.perf_counter()
            action_chunk, info = shield.select(state, candidates)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            if info["fallback"] or info["n_admissible"] < len(candidates):
                activations += 1

            for chunk, ok in zip(candidates, info["admissible_mask"]):
                if not ok:
                    rejected_total += 1
                    if nominal_violates(state, chunk, dt, obstacles):
                        rejected_truly_unsafe += 1

            if info.get("repair_attempted"):
                repair_attempts += 1
                if info.get("repair_succeeded"):
                    repair_successes += 1

            if record_decisions:
                chosen_idx = next((i for i, c in enumerate(candidates) if c is action_chunk), None)
                decisions.append({
                    "state": state.copy(),
                    "candidates": candidates,
                    "mask": info["admissible_mask"],
                    "chosen_idx": chosen_idx,
                    "fallback": info["fallback"],
                })

            first_action = action_chunk[0]
        else:
            first_action = candidates[0][0]

        state, done, step_info = env.step(first_action)
        trajectory.append(state.copy())
        violated = violated or step_info["violated"]
        reached = reached or step_info["reached"]
        if done:
            break

    return {
        "violated": violated,
        "reached": reached,
        "shield_activations": activations,
        "steps": len(trajectory) - 1,
        "latencies_ms": latencies_ms,
        "rejected_total": rejected_total,
        "rejected_truly_unsafe": rejected_truly_unsafe,
        "repair_attempts": repair_attempts,
        "repair_successes": repair_successes,
        "trajectory": np.array(trajectory),
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "decisions": decisions,
        "dt": dt,
        "w_bar": w_bar,
        "shield_w_bar": certified_w_bar,
    }
