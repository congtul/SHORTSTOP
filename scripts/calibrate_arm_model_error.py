"""Calibrates `model_error` for the arm/CALVIN reachtube (`shortstop.
arm_reach.propagate_arm_tube`) from real rollout residuals -- closes the
gap flagged in docs/PARAMETERS_REFERENCE.md's "model_error" entry: every
Arm*Shield so far has used a hardcoded default (0.02), never measured
against this checkpoint's actual Jacobian-pseudo-inverse linearization
error the way `shortstop.calibration.calibrate_w_bar` is designed to for
the 2D prototype's disturbance bound.

Not a reuse of `shortstop.calibration` -- that module's
`collect_disturbance_residuals` is hard-wired to `ReachAvoid2D`'s flat-
array state, `env.max_action_norm`/`dt`/`low`/`high`, and a
`state + action*dt` nominal model. CALVIN's env has a dict obs
(`obs["robot_obs_raw"]`) and the nominal model is the Jacobian-pseudo-
inverse step (`shortstop.arm_reach._step_joint_config`), not `x + a*dt` --
this script measures the SAME kind of residual with the arm's own nominal
model instead (see `shortstop.arm_reach.step_prediction_residual`), then
applies the identical statistical recipe: `quantile(residuals, 0.99) *
1.25` (Table VII's own "model-error quantile: 99th (x1.25)").

No shield, no obstacle -- this is a pure measurement pass: `policy.propose
(...)`'s first candidate is executed directly (same as the unshielded
harness), and at every real step this records `step_prediction_residual`
(joint_angles before the step, the actual action row executed, joint_angles
after) -- purely a bookkeeping addition around the unshielded loop, not a
new rollout pattern.

Run from WSL2, inside the `mdt_env` conda environment (see
docs/CALVIN_SETUP.md) -- like every other real-data CALVIN script, NOT
runnable in the sandbox this was written in:

    cd SHORTSTOP
    python scripts/calibrate_arm_model_error.py

One-shot, no --tuning/eval split: nothing here is chosen by comparing
candidate values against a metric (unlike disagreement_threshold/
epsilon/radius) -- it's a direct measurement, so it can run once on
`cfg.num_sequences` sequences (uses the TUNING cohort, idx 0..N-1, purely
by convention -- there is no held-out confirmation step needed for a
measurement, unlike a value chosen by comparing tradeoffs).

Prints residual percentiles (mean/median/p10/p90/p99/max) and the
calibrated `model_error = quantile(residuals, 0.99) * 1.25`. This number
is NOT wired in automatically -- manually plug it into whichever future
`ArmReachOnlyShield`/`ArmSTLShield`/`ArmRepairShield` construction needs a
real `model_error` (not `ArmSTLMonitorShield` -- its `model_error=0.0` is
fixed by definition, unaffected by this calibration), the same
documented-constant pattern `OBSTACLE_RADIUS`/`CHOSEN_THRESHOLD` already
follow in the other CALVIN scripts.
"""
import sys
from datetime import datetime
from pathlib import Path

import calvin_env
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from mdt.evaluation.multistep_sequences import get_sequences  # noqa: E402
from mdt.evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition  # noqa: E402
from mdt.utils.utils import get_last_checkpoint  # noqa: E402

from shortstop.arm_reach import step_prediction_residual  # noqa: E402
from shortstop.calvin_experiment import _joint_angles_from_obs, _lang_goal, _to_action_tensor  # noqa: E402
from shortstop.mdt_policy_client import ForwardOnlyPolicy  # noqa: E402

# Same shared values every other CALVIN script uses -- see scripts/
# run_calvin_shielded.py's own comments for the full reasoning behind each.
N_CANDIDATES = 8
REPLAN_STEPS = 10

# Table VII's own calibration recipe (also used by
# shortstop.calibration.calibrate_w_bar for the 2D disturbance bound).
QUANTILE = 0.99
SAFETY_FACTOR = 1.25

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "calibrate_arm_model_error_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"

_LOG_FILE = None


def _log(msg):
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def _residual_percentiles(residuals):
    values = np.asarray(residuals)
    return {
        "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)), "min": float(values.min()), "max": float(values.max()),
    }


def _run_subtask_and_collect_residuals(env, policy, task_oracle, lang_embeddings, subtask, val_annotations, ep_len):
    """Same P-execute loop as run_calvin_unshielded_subtask (no shield, no
    obstacle) but also records step_prediction_residual for every real
    step -- appended to `residuals` in place."""
    obs = env.get_obs()
    goal = _lang_goal(lang_embeddings, val_annotations, subtask)
    start_info = env.get_info()

    residuals = []
    reached = False
    steps_taken = 0
    ep_len_budget = ep_len
    while steps_taken < ep_len_budget:
        joint_angles = _joint_angles_from_obs(obs)
        candidates = policy.propose({**obs, "goal": goal})
        chunk = candidates[0]

        for action_row in chunk[:REPLAN_STEPS]:
            obs, _, _, current_info = env.step(_to_action_tensor(action_row))
            steps_taken += 1
            next_joint_angles = _joint_angles_from_obs(obs)
            residuals.append(step_prediction_residual(joint_angles, action_row, next_joint_angles))
            joint_angles = next_joint_angles

            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                reached = True
                break
            if steps_taken >= ep_len_budget:
                break
        if reached:
            break
    return residuals


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    global _LOG_FILE
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(RUN_OUTPUT_DIR / "run.log", "w", encoding="utf-8")
    _log(f"[run] writing log to: {RUN_OUTPUT_DIR}")
    _log(f"[run] config: num_sequences={cfg.num_sequences} ep_len={cfg.ep_len} replan_steps={REPLAN_STEPS} "
         f"sampler_type={cfg.sampler_type} num_sampling_steps={cfg.num_sampling_steps} n_candidates={N_CANDIDATES}")

    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler
    model.eval()

    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    policy = ForwardOnlyPolicy(model, n_candidates=N_CANDIDATES)

    # Tuning cohort (idx 0..N-1) -- see module docstring for why no eval
    # split is needed for a pure measurement pass.
    N = cfg.num_sequences
    eval_sequences = get_sequences(2 * N)[0:N]
    SEQUENCE_SEED_BASE = 1000

    all_residuals = []
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(SEQUENCE_SEED_BASE + idx, workers=True)
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        for subtask in eval_sequence:
            residuals = _run_subtask_and_collect_residuals(
                env, policy, task_oracle, lang_embeddings, subtask, val_annotations, ep_len=cfg.ep_len,
            )
            all_residuals.extend(residuals)

    stats = _residual_percentiles(all_residuals)
    _log(
        f"[residuals] n={stats['n']}  mean={stats['mean']:.5f}  median={stats['median']:.5f}  "
        f"p10={stats['p10']:.5f}  p90={stats['p90']:.5f}  p99={stats['p99']:.5f}  "
        f"min={stats['min']:.5f}  max={stats['max']:.5f}"
    )
    calibrated_model_error = stats["p99"] * SAFETY_FACTOR
    _log(f"[calibrated] model_error = p99 * {SAFETY_FACTOR} = {calibrated_model_error:.5f}")
    _log(
        "[run] DONE -- manually plug the calibrated model_error above into any future "
        "ArmReachOnlyShield/ArmSTLShield/ArmRepairShield construction that needs one "
        "(ArmSTLMonitorShield's model_error=0.0 is fixed by definition, unaffected)."
    )
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
