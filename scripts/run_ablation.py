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

Note: with USE_CALIBRATED_W_BAR=True (default), every certified stage
certifies against a *calibrated* disturbance bound (shortstop.calibration,
Table VII's "high-quantile residual x safety factor" recipe) rather than the
ground-truth TRUE_W_BAR handed to it for free. This is more realistic (a
real deployed shield does not know the true noise bound) but does mean
recovery/activation numbers move a little run to run depending on the
calibration sample; set it to False to go back to the privileged/ground-
truth w_bar used before calibration was added.

Note: STEP_SIZE/TRUST_REGION (Eq. 4's eta/delta) default here to values
*tuned for this prototype's scale*, not the paper's literal Table VII
numbers (eta=0.05, delta=0.1) -- this prototype's units (obstacle radius
~0.4-0.8, action magnitude ~1.0) are not calibrated to the paper's
real-world cm scale (see run_phase1.py's "reproducibility note" below), so
the paper's literal eta=0.05 step rarely clears an obstacle here. See the
comment above STEP_SIZE's definition for the sweep that picked these
defaults, and how to switch back to the paper's literal values.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shortstop.baselines import CBFShield, ConfThreshShield, MPCFilterShield, STLMonitorShield
from shortstop.calibration import calibrate_w_bar
from shortstop.env import ReachAvoid2D
from shortstop.experiment import make_scenario, run_episode
from shortstop.metrics import aggregate, conservatism_cost
from shortstop.policy import DiffusionChunkPolicy
from shortstop.shield import CEShield, ReachOnlyShield, RepairShield, STLShield

# "gaussian" (default) = every stage/baseline sees the same GaussianChunkPolicy
# stand-in as always (unchanged output: results/ablation.json). "diffusion" =
# swap in the Stage 6a trained policy (scripts/train_diffusion_policy.py's
# checkpoint) for every row instead, and save to results/ablation_diffusion.json
# so the two runs stay side by side for comparison, rather than one overwriting
# the other.
#   .venv/Scripts/python.exe scripts/run_ablation.py diffusion
POLICY = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
DIFFUSION_CHECKPOINT = "results/diffusion_policy.pt"


def build_policy_factory(policy_name):
    if policy_name == "gaussian":
        return None  # run_episode's own default: builds GaussianChunkPolicy
    if policy_name == "diffusion":
        from shortstop.diffusion_policy import load_checkpoint

        model, schedule, cond_mean, cond_std, info = load_checkpoint(DIFFUSION_CHECKPOINT)
        print(
            f"Loaded {DIFFUSION_CHECKPOINT} (trained {info['n_steps']} steps, "
            f"final_val_loss={info['final_val_loss']:.4f})\n"
        )

        def factory(goal, obstacles, horizon, n_candidates, rng):
            return DiffusionChunkPolicy(
                model, schedule, cond_mean, cond_std, obstacles, n_candidates=n_candidates, rng=rng
            )

        return factory
    raise ValueError(f"unknown POLICY {policy_name!r}, expected 'gaussian' or 'diffusion'")

# Shared knobs across every certified stage (Stage 2+), so the only thing that
# changes row to row is the shield *logic*, not its tuning.
EPSILON = 0.05           # STL certification margin (paper's fixed "margin epsilon", Table VII)

# eta/delta (Eq. 4): the paper's literal Table VII values are eta=0.05,
# delta=0.1. Kept here at values *tuned for this prototype's scale* instead
# (obstacle radius ~0.4-0.8, action magnitude ~1.0 -- not the paper's cm
# scale), so Stage 4 actually demonstrates repair on this environment rather
# than almost always falling back. A 150-episode sweep at max_repair_iters=1
# (see shortstop/shield.py's RepairShield docstring for what eta/delta mean):
#   step_size=0.05 (paper's eta), trust_region=0.3  -> recovery  6.1%
#   step_size=0.15,               trust_region=0.5  -> recovery 11.6%
#   step_size=0.30,               trust_region=1.0  -> recovery 17.1%  <- picked
#   step_size=0.50,               trust_region=1.5  -> recovery 26.8%
#   step_size=0.60,               trust_region=2.0  -> recovery 32.5%
# violation_rate stayed ~0.7-1.3% across all of them -- a bigger step makes
# repair succeed more often, it does not weaken certification (every repaired
# chunk is still re-certified before being accepted). Raise these two if you
# want Stage 4 to show a stronger recovery effect; set them to 0.05/0.1 to
# reproduce the paper's literal hyperparameters instead.
TRUST_REGION = 1.0
STEP_SIZE = 0.3
MAX_REPAIR_ITERS = 1     # Algorithm 1's default: one gradient step, no retry.
                         # >1 is a CEGIS-style extension beyond the paper --
                         # see RepairShield's docstring in shortstop/shield.py.

# Table VII's calibration recipe applied to this prototype's disturbance bound
# (see shortstop/calibration.py for why w_bar rather than a model-error term).
TRUE_W_BAR = 0.02              # ground-truth noise level the env actually uses
USE_CALIBRATED_W_BAR = True    # False = shields get TRUE_W_BAR for free (old behavior)
CALIBRATION_EPISODES = 200
CALIBRATION_QUANTILE = 0.99
CALIBRATION_SAFETY_FACTOR = 1.25

# Table II's five comparison baselines (shortstop/baselines.py). Each is its
# own dict entry below like the Stage 1-4 rows -- comment one out to drop it
# from the table, or tune its knob here without touching baselines.py.
CONF_THRESH_DISAGREEMENT_THRESHOLD = 0.15  # reject if a candidate's endpoint
                                            # is this far from the K-candidate centroid.
                                            # See ConfThreshShield's docstring: this proxy is
                                            # measured to be *uncorrelated* with real danger on
                                            # this scenario at any threshold (0.5 down to 0.06
                                            # all gave ~0.81 violation, same as Unshielded) --
                                            # tuning this only changes activation rate, not safety.
MPC_MAX_ACTION_NORM = 1.0                  # QP action bound, matches env's own clip
CBF_ALPHA = 1.0                            # class-K gain in the barrier condition

STAGES = {
    "unshielded": None,
    # --- Table II baselines: comment out any of these 4 lines to drop it ---
    "conf_thresh": lambda goal, obstacles, dt, w_bar: ConfThreshShield(
        goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar,
        disagreement_threshold=CONF_THRESH_DISAGREEMENT_THRESHOLD,
    ),
    "mpc_filter": lambda goal, obstacles, dt, w_bar: MPCFilterShield(
        goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar, max_action_norm=MPC_MAX_ACTION_NORM,
    ),
    "cbf_shield": lambda goal, obstacles, dt, w_bar: CBFShield(
        goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar, alpha=CBF_ALPHA,
    ),
    "stl_monitor": lambda goal, obstacles, dt, w_bar: STLMonitorShield(
        goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar,
    ),
    # --- ShortStop's own Stage 1-4 ablation ---
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
        step_size=STEP_SIZE,
        max_repair_iters=MAX_REPAIR_ITERS,
    ),
}


def run_stage(shield_factory, n_episodes, seed_offset=1000, shield_w_bar=None, policy_factory=None):
    return [
        run_episode(
            shield_factory,
            np.random.default_rng(seed_offset + i),
            w_bar=TRUE_W_BAR,
            shield_w_bar=shield_w_bar,
            policy_factory=policy_factory,
        )
        for i in range(n_episodes)
    ]


def fmt_pct(x):
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def fmt_num(x, digits=3):
    return f"{x:.{digits}f}" if x is not None else "n/a"


def main():
    n_episodes = 500
    policy_factory = build_policy_factory(POLICY)
    print(f"Policy: {POLICY}\n")

    shield_w_bar = None
    if USE_CALIBRATED_W_BAR:
        calib_rng = np.random.default_rng(999)

        def make_env():
            start, goal, obstacles = make_scenario(calib_rng)
            return ReachAvoid2D(
                start=start, goal=goal, obstacles=obstacles, dt=0.1, w_bar=TRUE_W_BAR, rng=calib_rng
            )

        shield_w_bar = calibrate_w_bar(
            make_env,
            n_episodes=CALIBRATION_EPISODES,
            quantile=CALIBRATION_QUANTILE,
            safety_factor=CALIBRATION_SAFETY_FACTOR,
            rng=calib_rng,
        )
        print(
            f"Calibrated w_bar = {shield_w_bar:.4f} (true w_bar = {TRUE_W_BAR}) from "
            f"{CALIBRATION_EPISODES} held-out episodes "
            f"[{CALIBRATION_QUANTILE:.0%} quantile x {CALIBRATION_SAFETY_FACTOR} safety factor]\n"
        )

    logs_by_stage = {
        name: run_stage(factory, n_episodes, shield_w_bar=shield_w_bar, policy_factory=policy_factory)
        for name, factory in STAGES.items()
    }
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
        ("lat_median", 11, lambda m: fmt_num(m["latency_ms_median"])),
        ("lat_mean", 10, lambda m: fmt_num(m["latency_ms_mean"])),
        ("lat_p95", 9, lambda m: fmt_num(m["latency_ms_p95"])),
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

    out_name = "ablation.json" if POLICY == "gaussian" else f"ablation_{POLICY}.json"
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / out_name, "w") as f:
        json.dump(metrics_by_stage, f, indent=2)
    print(f"\nSaved machine-readable results to {results_dir / out_name}")


if __name__ == "__main__":
    main()
