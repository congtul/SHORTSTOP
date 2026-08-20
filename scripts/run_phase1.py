"""Phase 1 runner: Reach-Avoid-2D env + Reach-only rejector.

Usage:
    .venv/Scripts/python.exe scripts/run_phase1.py

Note: the exact violation/success numbers in Table III of the paper depend on
an env configuration (obstacle layout, w_bar, action bounds) that isn't fully
specified in the text. This script uses plausible placeholder values so the
pipeline (propose -> reach -> reject -> execute) runs end-to-end; tune
n_episodes / w_bar / noise_std / n_candidates once you're ready to calibrate
against the paper's numbers.

Metrics: 6 of the paper's 7 metrics are computed here (safety-violation rate,
task success, shield-activation rate, intervention precision, latency,
conservatism cost). Recovery rate is not computable yet -- it measures the
value of repair over plain reject, and there is no repair loop until Phase 4.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shortstop.env import Obstacle, ReachAvoid2D
from shortstop.metrics import aggregate, conservatism_cost
from shortstop.policy import GaussianChunkPolicy
from shortstop.reach import nominal_rollout, propagate_tube
from shortstop.shield import ReachOnlyShield


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


def run_episode(shielded, rng, horizon=8, n_candidates=8, dt=0.1, w_bar=0.02, record_decisions=False):
    start, goal, obstacles = make_scenario(rng)
    env = ReachAvoid2D(start=start, goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar, rng=rng)
    policy = GaussianChunkPolicy(goal=goal, horizon=horizon, n_candidates=n_candidates, rng=rng)
    shield = ReachOnlyShield(goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar)

    state = env.reset()
    trajectory = [state.copy()]
    activations = 0
    violated = False
    reached = False
    latencies_ms = []
    rejected_total = 0
    rejected_truly_unsafe = 0
    decisions = [] if record_decisions else None

    for _ in range(env.max_steps):
        candidates = policy.propose(state)
        if shielded:
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
        "trajectory": np.array(trajectory),
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "decisions": decisions,
        "dt": dt,
        "w_bar": w_bar,
    }


def find_example(reached_target, seed_start=0, max_tries=300):
    """Search seeds seed_start.. for a shielded episode matching reached_target."""
    for seed in range(seed_start, seed_start + max_tries):
        result = run_episode(True, np.random.default_rng(seed))
        if result["reached"] == reached_target:
            return seed, result
    raise RuntimeError(f"No example with reached={reached_target} found in {max_tries} seeds")


def plot_episode(result, path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    for obs in result["obstacles"]:
        ax.add_patch(plt.Circle(obs.center, obs.radius, color="red", alpha=0.4))
    traj = result["trajectory"]
    ax.plot(traj[:, 0], traj[:, 1], "-o", color="tab:blue", markersize=3, label="trajectory")
    ax.plot(*result["start"], "ks", label="start")
    ax.plot(*result["goal"], "g*", markersize=15, label="goal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_snapshot_step(decisions):
    """Prefer the first step with at least one rejected chunk, so the
    reachtube-vs-obstacle picture actually shows a reject happening."""
    for t, d in enumerate(decisions):
        if any(not ok for ok in d["mask"]):
            return t
    return 0


def plot_decision_snapshot(result, step, path, suptitle):
    """One subplot per candidate chunk at a single decision step: its nominal
    trajectory and reachtube boxes, colored by admissible (green) vs
    rejected (red), so it's visible *why* the shield accepted/rejected it.
    """
    decision = result["decisions"][step]
    state = decision["state"]
    candidates = decision["candidates"]
    mask = decision["mask"]
    chosen_idx = decision["chosen_idx"]
    obstacles = result["obstacles"]
    goal = result["goal"]
    dt = result["dt"]
    w_bar = result["w_bar"]

    n = len(candidates)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    half_width = 2.5

    for i, (chunk, ok) in enumerate(zip(candidates, mask)):
        ax = axes[i]
        color = "tab:green" if ok else "tab:red"

        for obs in obstacles:
            ax.add_patch(plt.Circle(obs.center, obs.radius, color="red", alpha=0.3))

        tube = propagate_tube(state, chunk, dt, w_bar)
        for box in tube[1:]:
            w = box.high[0] - box.low[0]
            h = box.high[1] - box.low[1]
            ax.add_patch(plt.Rectangle(box.low, w, h, edgecolor=color, facecolor=color, alpha=0.15))

        path_pts = np.array(nominal_rollout(state, chunk, dt))
        ax.plot(path_pts[:, 0], path_pts[:, 1], "-o", color=color, markersize=3)
        ax.plot(*state, "k^", markersize=8, label="current state")
        ax.plot(*goal, "g*", markersize=12, label="goal")

        status = "ADMISSIBLE" if ok else "REJECTED"
        marker = " [chosen]" if i == chosen_idx else ""
        ax.set_title(f"chunk {i}: {status}{marker}", color=color, fontsize=10)
        ax.set_xlim(state[0] - half_width, state[0] + half_width)
        ax.set_ylim(state[1] - half_width, state[1] + half_width)
        ax.set_aspect("equal")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    n_episodes = 500
    unshielded_logs = [run_episode(False, np.random.default_rng(1000 + i)) for i in range(n_episodes)]
    shielded_logs = [run_episode(True, np.random.default_rng(1000 + i)) for i in range(n_episodes)]

    print("=== Unshielded ===")
    print(aggregate(unshielded_logs))
    print("=== Reach-only rejector (Phase 1) ===")
    print(aggregate(shielded_logs))

    cons_cost = conservatism_cost(unshielded_logs, shielded_logs)
    print(f"Conservatism cost (success drop on benign episodes): {cons_cost}")
    print("Recovery rate: TODO -- needs the repair loop (Phase 4); reject-only has nothing to compare against.")

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    seed_success, example_success = find_example(True)
    print(f"Example that reaches the goal: seed={seed_success}")
    plot_episode(
        example_success,
        results_dir / "phase1_example_success.png",
        f"Reach-Avoid-2D -- Phase 1, reached goal (seed={seed_success})",
    )

    seed_failure, example_failure = find_example(False)
    print(f"Example that does NOT reach the goal: seed={seed_failure}")
    plot_episode(
        example_failure,
        results_dir / "phase1_example_failure.png",
        f"Reach-Avoid-2D -- Phase 1, did not reach goal (seed={seed_failure})",
    )

    # Re-run the same two seeds with decision recording on, to draw a
    # per-candidate reachtube-vs-obstacle snapshot at one decision step.
    for label, seed in (("success", seed_success), ("failure", seed_failure)):
        example = run_episode(True, np.random.default_rng(seed), record_decisions=True)
        step = pick_snapshot_step(example["decisions"])
        had_reject = any(not ok for ok in example["decisions"][step]["mask"])
        print(
            f"Decision snapshot ({label}): seed={seed}, step={step}, "
            f"has a rejected chunk={had_reject}"
        )
        plot_decision_snapshot(
            example,
            step,
            results_dir / f"phase1_decision_{label}.png",
            f"Decision snapshot ({label}) -- seed={seed}, step={step}",
        )


if __name__ == "__main__":
    main()
