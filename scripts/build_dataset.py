"""Build the Stage 6a behavior-cloning dataset (see shortstop/dataset.py).

Every window's action chunk comes from a scripted-expert rollout that
reached the goal without violating any obstacle (shortstop.expert.
generate_demo_pair filters everything else out before this ever sees it).

Usage:
    .venv/Scripts/python.exe scripts/build_dataset.py [target_demos] [out_path]
    (defaults: target_demos=500, out_path=results/expert_dataset.npz)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from shortstop.dataset import build_dataset


def main(target_demos=500, horizon=8, out_path="results/expert_dataset.npz"):
    data = build_dataset(target_demos=target_demos, horizon=horizon)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        states=data["states"],
        obstacle_vecs=data["obstacle_vecs"],
        action_chunks=data["action_chunks"],
        horizon=data["horizon"],
    )

    print(
        f"scenarios attempted: {data['n_scenarios_attempted']} "
        f"(seeds {data['seed_start']}-{data['seed_end']})"
    )
    print(
        f"successful demos: {data['n_demos']} "
        f"(upper={data['n_demos_upper']}, lower={data['n_demos_lower']})"
    )
    print(
        f"training windows: {len(data['states'])}  "
        f"(state dim=2, obstacle_vec dim={data['obstacle_vecs'].shape[1]}, "
        f"chunk shape=({data['horizon']}, 2))"
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    target_demos = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/expert_dataset.npz"
    main(target_demos=target_demos, out_path=out_path)
