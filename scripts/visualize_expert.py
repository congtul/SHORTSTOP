"""Visualize the Stage 6a scripted expert for one scenario seed.

Plots the scenario (start, goal, 3 obstacles) together with, for both the
"upper" and "lower" bypass modes: the planned waypoints (dashed) and the
actual executed trajectory under the real ReachAvoid2D (with disturbance
noise) driven by the pure-pursuit controller (solid).

Usage:
    .venv/Scripts/python.exe scripts/visualize_expert.py [seed] [out_path]
    (defaults: seed=0, out_path=results/expert_seed<seed>.png)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from shortstop.env import ReachAvoid2D
from shortstop.experiment import make_scenario
from shortstop.expert import plan_waypoints, rollout_expert


def plot_scenario(seed, out_path):
    rng = np.random.default_rng(seed)
    start, goal, obstacles = make_scenario(rng)

    fig, ax = plt.subplots(figsize=(8, 6))
    for obs in obstacles:
        ax.add_patch(plt.Circle(obs.center, obs.radius, color="0.6", alpha=0.6, zorder=1))
        ax.add_patch(
            plt.Circle(obs.center, obs.radius, color="0.4", fill=False, linestyle=":", zorder=1)
        )
    ax.plot(*start, "s", color="black", markersize=10, label="start", zorder=3)
    ax.plot(*goal, "*", color="black", markersize=15, label="goal", zorder=3)

    colors = {"upper": "tab:blue", "lower": "tab:orange"}
    for mode in ("upper", "lower"):
        try:
            waypoints = plan_waypoints(start, goal, obstacles, mode=mode)
        except RuntimeError as e:
            print(f"[{mode}] plan failed: {e}")
            continue
        wp = np.array(waypoints)
        ax.plot(
            wp[:, 0], wp[:, 1], "--", color=colors[mode], alpha=0.5,
            label=f"{mode}: planned waypoints", zorder=2,
        )

        env = ReachAvoid2D(
            start=start, goal=goal, obstacles=obstacles,
            rng=np.random.default_rng(seed + 1),
        )
        result = rollout_expert(env, waypoints)
        states = result["states"]
        ax.plot(
            states[:, 0], states[:, 1], "-", color=colors[mode], linewidth=2,
            label=f"{mode}: executed (reached={result['reached']}, "
            f"violated={result['violated']})",
            zorder=4,
        )

    ax.set_aspect("equal")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(f"Reach-Avoid-2D scripted expert -- seed={seed}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"results/expert_seed{seed}.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plot_scenario(seed, out_path)
