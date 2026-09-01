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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shortstop.experiment import run_episode as _run_episode
from shortstop.metrics import aggregate, conservatism_cost
from shortstop.reach import nominal_rollout, propagate_tube
from shortstop.shield import ReachOnlyShield


def _stage1_shield(goal, obstacles, dt, w_bar):
    return ReachOnlyShield(goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar)


def run_episode(shielded, rng, horizon=8, n_candidates=8, dt=0.1, w_bar=0.02, record_decisions=False):
    """Thin Stage-1-specific wrapper around shortstop.experiment.run_episode:
    keeps this script's original `shielded: bool` signature so the plotting
    helpers below don't need to change.
    """
    shield_factory = _stage1_shield if shielded else None
    return _run_episode(
        shield_factory,
        rng,
        horizon=horizon,
        n_candidates=n_candidates,
        dt=dt,
        w_bar=w_bar,
        record_decisions=record_decisions,
    )


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
