"""Ablation runner: compares every implemented shield stage side by side.

Mirrors the paper's Table III ablation (Reach-only -> +STL-to-go ->
+CE search -> +Repair), plus an Unshielded baseline. Every stage is run
with the exact same paired seeds (via shortstop.experiment.run_episode), so
rows are directly comparable -- no stage-specific script drift.

Usage:
    .venv/Scripts/python.exe scripts/run_ablation.py

Runtime: pure-Python, unvectorized episode loop -- expect roughly the same
per-stage cost as scripts/run_phase1.py (~1-2 min per 500-episode stage), so
the full 5-stage sweep below takes on the order of 10-15 minutes. Lower
n_episodes in main() for a quicker sanity-check pass during development.

To add a new stage: add one entry to STAGES below. Nothing else needs to
change -- run_episode() in shortstop/experiment.py already treats every
shield_factory identically.

Note: as documented in shortstop/shield.py, Stage 2 and Stage 3 (STLShield
vs. CEShield) make the *same* accept/reject decisions in this 2D prototype --
CEShield only adds counterexample diagnostics on top. The safety numbers only
move again at Stage 4 (RepairShield), where rejected candidates can actually
be fixed instead of discarded. This is a modeling choice specific to this
prototype (see docstrings), not a bug -- the paper's own exact percentages
depend on an env configuration it doesn't fully specify (see run_phase1.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shortstop.experiment import run_episode
from shortstop.metrics import aggregate, conservatism_cost
from shortstop.shield import CEShield, ReachOnlyShield, RepairShield, STLShield

# Shared knobs across every certified stage (Stage 2+), so the only thing that
# changes row to row is the shield *logic*, not its tuning.
EPSILON = 0.05           # STL certification margin
TRUST_REGION = 0.3       # ||delta_a|| bound per repair step (Stage 4)
MAX_REPAIR_ITERS = 1     # Algorithm 1's default: one gradient step, no retry.
                         # >1 is a CEGIS-style extension beyond the paper --
                         # see RepairShield's docstring in shortstop/shield.py.

STAGES = {
    "unshielded": None,
    "stage1_reach_only": lambda goal, obstacles, dt, w_bar: ReachOnlyShield(
        goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar
    ),
    "stage2_stl_to_go": lambda goal, obstacles, dt, w_bar: STLShield(
        goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar, epsilon=EPSILON
    ),
    "stage3_ce_search": lambda goal, obstacles, dt, w_bar: CEShield(
        goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar, epsilon=EPSILON
    ),
    "stage4_repair": lambda goal, obstacles, dt, w_bar: RepairShield(
        goal=goal,
        obstacles=obstacles,
        dt=dt,
        w_bar=w_bar,
        epsilon=EPSILON,
        trust_region=TRUST_REGION,
        max_repair_iters=MAX_REPAIR_ITERS,
    ),
}


def run_stage(shield_factory, n_episodes, seed_offset=1000):
    return [run_episode(shield_factory, np.random.default_rng(seed_offset + i)) for i in range(n_episodes)]


def fmt_pct(x):
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def fmt_num(x, digits=3):
    return f"{x:.{digits}f}" if x is not None else "n/a"


def main():
    n_episodes = 500

    logs_by_stage = {name: run_stage(factory, n_episodes) for name, factory in STAGES.items()}
    metrics_by_stage = {name: aggregate(logs) for name, logs in logs_by_stage.items()}

    baseline_logs = logs_by_stage["unshielded"]
    for name, logs in logs_by_stage.items():
        if name == "unshielded":
            continue
        metrics_by_stage[name]["conservatism_cost"] = conservatism_cost(baseline_logs, logs)

    columns = [
        ("stage", 20, lambda m: m),
        ("violation", 10, lambda m: fmt_pct(m["violation_rate"])),
        ("success", 10, lambda m: fmt_pct(m["success_rate"])),
        ("activation", 11, lambda m: fmt_pct(m["shield_activation_rate"])),
        ("latency_ms", 11, lambda m: fmt_num(m["latency_ms_median"])),
        ("precision", 10, lambda m: fmt_pct(m["intervention_precision"])),
        ("recovery", 10, lambda m: fmt_pct(m["recovery_rate"])),
        ("cons.cost", 10, lambda m: fmt_pct(m.get("conservatism_cost"))),
    ]
    header = "".join(f"{name:<{width}}" for name, width, _ in columns)
    print(header)
    print("-" * len(header))
    for stage_name, m in metrics_by_stage.items():
        row = f"{stage_name:<20}"
        for _, width, extract in columns[1:]:
            row += f"{extract(m):<{width}}"
        print(row)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "ablation.json", "w") as f:
        json.dump(metrics_by_stage, f, indent=2)
    print(f"\nSaved machine-readable results to {results_dir / 'ablation.json'}")


if __name__ == "__main__":
    main()
