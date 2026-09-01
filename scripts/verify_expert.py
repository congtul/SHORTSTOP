"""Verification report for the Stage 6a scripted expert (shortstop/expert.py).

Runs the "upper"/"lower" perpendicular-offset circumnavigation planner over
many randomized scenarios (same generator as the ablation/baseline
experiments, shortstop.experiment.make_scenario) and reports:
  - plan-converged rate and, among converged plans, violation rate (must be
    ~0 -- a scripted expert with a safety margin should never be the source
    of unsafe demos) and reach rate
  - what fraction of scenarios are "contested" (the two modes actually
    diverge, i.e. some obstacle genuinely blocks the direct path)
  - how far apart the two modes' trajectories are on contested scenarios
    (confirms real multimodality, not near-duplicate paths)

Plan non-convergence is expected on a large minority of scenarios: this
planner detours around one obstacle at a time and simply refuses (rather
than emitting an unsafe path) when make_scenario's 3 obstacles (radius
0.4-0.8 in a 4x3 box) happen to overlap heavily. Building the dataset
(step 2) just means oversampling scenario seeds past whatever this script
reports as the per-scenario success rate.

Usage:
    .venv/Scripts/python.exe scripts/verify_expert.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shortstop.env import ReachAvoid2D
from shortstop.experiment import make_scenario
from shortstop.expert import generate_demo_pair, plan_waypoints, rollout_expert


def main(n_scenarios=500, margin=0.1, seed_offset=5000):
    plan_ok = {"upper": 0, "lower": 0}
    violated = {"upper": 0, "lower": 0}
    reached = {"upper": 0, "lower": 0}
    contested = 0
    separations = []

    for i in range(n_scenarios):
        rng = np.random.default_rng(seed_offset + i)
        start, goal, obstacles = make_scenario(rng)

        for mode in ("upper", "lower"):
            try:
                wp = plan_waypoints(start, goal, obstacles, margin=margin, mode=mode)
            except RuntimeError:
                continue
            plan_ok[mode] += 1
            env = ReachAvoid2D(
                start=start, goal=goal, obstacles=obstacles,
                rng=np.random.default_rng(10_000 * seed_offset + i),
            )
            result = rollout_expert(env, wp)
            if result["violated"]:
                violated[mode] += 1
            if result["reached"]:
                reached[mode] += 1

        pair = generate_demo_pair(start, goal, obstacles, margin=margin, rng=rng)
        if "upper" in pair and "lower" in pair:
            u, l = pair["upper"]["states"], pair["lower"]["states"]
            n = min(len(u), len(l))
            if not np.allclose(u[:n], l[:n], atol=1e-6):
                contested += 1
                separations.append(float(np.mean(np.linalg.norm(u[:n] - l[:n], axis=1))))

    print(f"scenarios: {n_scenarios}")
    for mode in ("upper", "lower"):
        print(
            f"  [{mode}] plan converged: {plan_ok[mode]}/{n_scenarios} "
            f"({100 * plan_ok[mode] / n_scenarios:.1f}%)  "
            f"| of those: reached {reached[mode]}/{plan_ok[mode]}, "
            f"violated {violated[mode]}/{plan_ok[mode]}"
        )
    print(
        f"contested (upper != lower, both usable): {contested}/{n_scenarios} "
        f"({100 * contested / n_scenarios:.1f}%)"
    )
    if separations:
        print(
            f"mean pointwise separation on contested scenarios: "
            f"{np.mean(separations):.3f} (std {np.std(separations):.3f})"
        )


if __name__ == "__main__":
    main()
