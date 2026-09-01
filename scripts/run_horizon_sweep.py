"""Conservatism-horizon trade-off sweep (Prop. 1, Fig. 4 in the paper).

The paper predicts and observes a specific shape as the certified horizon H
grows: violation falls sharply then saturates (longer certification catches
more of the danger the policy can walk into), while activation and latency
rise monotonically (a longer tube means more candidates get rejected, and
propagating/certifying a longer tube costs more per decision) -- "too short
and the shield is myopic ... too long and it is needlessly conservative"
(Prop. 1's conservatism-horizon bound). Fig. 4 reports this saturating near
H=8 for their setup.

This script reproduces that sweep on Reach-Avoid-2D: for each H in HORIZONS,
run every stage in SWEEP_STAGES for n_episodes and report violation/
activation/latency, so you can check whether this prototype shows the same
saturating shape (it need not land on the same H -- the paper's own H=8
"sweet spot" comes from its specific obstacle/policy scale, not a universal
constant).

Uses the ground-truth (privileged) w_bar, not the calibrated one from
shortstop.calibration -- this script isolates the horizon's effect alone;
see run_ablation.py for the calibrated-w_bar comparison.

Usage:
    .venv/Scripts/python.exe scripts/run_horizon_sweep.py

To change what's compared: edit HORIZONS (list of H values) or
SWEEP_STAGES (subset of scripts.run_ablation.STAGES's keys) below.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ablation import STAGES, TRUE_W_BAR
from shortstop.experiment import run_episode
from shortstop.metrics import aggregate

HORIZONS = [4, 6, 8, 10, 12, 16]
SWEEP_STAGES = ["stage1_reach_only", "stage4_repair"]
N_EPISODES = 300


def run_stage_at_horizon(shield_factory, horizon, n_episodes, seed_offset=2000):
    return [
        run_episode(shield_factory, np.random.default_rng(seed_offset + i), horizon=horizon, w_bar=TRUE_W_BAR)
        for i in range(n_episodes)
    ]


def fmt_pct(x):
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def fmt_num(x, digits=2):
    return f"{x:.{digits}f}" if x is not None else "n/a"


def main():
    results = {}
    for stage_name in SWEEP_STAGES:
        factory = STAGES[stage_name]
        rows = []
        for horizon in HORIZONS:
            logs = run_stage_at_horizon(factory, horizon, N_EPISODES)
            m = aggregate(logs)
            rows.append({"horizon": horizon, **m})
        results[stage_name] = rows

    for stage_name, rows in results.items():
        print(f"\n{stage_name}")
        header = f"{'H':<6}{'violation':<12}{'success':<10}{'activation':<12}{'latency_ms':<12}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['horizon']:<6}"
                f"{fmt_pct(row['violation_rate']):<12}"
                f"{fmt_pct(row['success_rate']):<10}"
                f"{fmt_pct(row['shield_activation_rate']):<12}"
                f"{fmt_num(row['latency_ms_median']):<12}"
            )

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "horizon_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved machine-readable results to {results_dir / 'horizon_sweep.json'}")


if __name__ == "__main__":
    main()
