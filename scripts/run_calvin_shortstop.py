"""ShortStop baseline (the paper's own full method): shortstop.arm_shield.
ArmRepairShield wired into the CALVIN eval loop -- Propose K candidates ->
Certify (STL robustness-to-go over the full 9-frame+8-link reachtube) ->
Repair (counterexample-guided, one gradient step + re-certify, Algorithm 1)
-> Select by shortstop.calvin_progress's g(a) among whatever repaired/
already-admissible candidates remain.

Filter freq is decoupled from policy freq here too, automatically --
ArmRepairShield inherits ArmReachOnlyShield.recertify() (see docs/
PARAMETERS_REFERENCE.md's "tach tan suat filter khoi policy" table, which
already lists ShortStop/Repair as the shield best positioned to do this,
since Reach/Certify/Repair are all cheap -- no diffusion re-sample needed).

Three tunable knobs, swept together (see docs/PARAMETERS_REFERENCE.md's
"epsilon"/"trust_region & step_size" entries for the individual trade-offs
each one controls):
  - `epsilon`: STL safety margin (Eq. 2) -- too small lets geometric/
    linearization error turn "just barely safe" into a real violation;
    too large rejects candidates that were actually fine.
  - `trust_region`/`step_size`: repair's own gradient-step size and how
    far a repaired candidate may drift from the original (Eq. 4) -- too
    small can't escape the violation in `max_repair_iters=1` step; too
    large risks a repaired candidate that no longer serves the task (or,
    for the arm, drives a joint past JOINT_LIMITS -- ArmRepairShield's
    own `_repair` already re-checks this, see arm_shield.py).

`model_error` is the ONE arm-specific number every ShortStop-family shield
(this one included) needs calibrated for real, not left at ArmReachOnlyShield's
generic 0.02 default -- see scripts/calibrate_arm_model_error.py.
`w_bar=0.0` throughout (see run_calvin_mpc_filter.py's own docstring for
why: CALVIN's real disturbance sources are captured end-to-end by
model_error's own calibration residual, so a separate w_bar budget is
redundant here, unlike the 2D prototype where the environment injects an
explicit, separately-known random disturbance).

Run from WSL2, inside `mdt_env` (see docs/CALVIN_SETUP.md):

    cd SHORTSTOP
    python scripts/run_calvin_shortstop.py            # eval cohort (default): runs CHOSEN_PARAMS once
    python scripts/run_calvin_shortstop.py --tuning    # tuning cohort: sweep PARAMS_TO_SWEEP

Tuning/eval cohort split: identical mechanism to every other run_calvin_
shielded_*.py script (see docs/TUNING_WORKFLOW.md muc 0).
"""
import sys
from pathlib import Path

import calvin_env  # noqa: F401 -- import side effects only, matches every other CALVIN script
import hydra
from pytorch_lightning import seed_everything

TUNING_MODE = "--tuning" in sys.argv
if TUNING_MODE:
    sys.argv.remove("--tuning")

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from shortstop.arm_shield import ArmRepairShield  # noqa: E402
from shortstop.calvin_baseline_runner import (  # noqa: E402
    clearance_stats, cohort_sequences, fallback_rate, intervention_precision, latency_stats, log_clearance_debug,
    make_logger, make_run_output_dir, rank_violating_sequence_idxs_by_length, save_debug_videos,
    setup_env_and_policy, shield_activation_rate, write_results_json,
)
from shortstop.calvin_experiment import run_calvin_shielded_sequence  # noqa: E402
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates, recovery_rate  # noqa: E402
from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk  # noqa: E402

# Shared with every other CALVIN baseline script -- see run_calvin_
# shielded.py's own comments for the full reasoning/history.
N_CANDIDATES = 8
REPLAN_STEPS = 10
OBSTACLE_RADIUS = 0.08

# model_error -- PLACEHOLDER (ArmReachOnlyShield's generic default) until
# scripts/calibrate_arm_model_error.py has run for real. w_bar=0.0 always
# (see module docstring).
MODEL_ERROR = 0.02

# Candidate (epsilon, trust_region, step_size) triples to sweep in
# --tuning mode -- PLACEHOLDER, anchored at Table VII's own defaults
# (epsilon=0.02, trust_region=0.05, step_size=0.02, see arm_shield.
# ArmRepairShield's constructor) plus a spread around each, one axis at a
# time (varying all 3 simultaneously in a full grid would be 4x4x4=64 runs
# -- start with a coordinate sweep, expand to a joint grid only if the
# real numbers show strong interaction between them).
PARAMS_TO_SWEEP = [
    {"epsilon": 0.02, "trust_region": 0.05, "step_size": 0.02},  # Table VII defaults
    {"epsilon": 0.0, "trust_region": 0.05, "step_size": 0.02},
    {"epsilon": 0.05, "trust_region": 0.05, "step_size": 0.02},
    {"epsilon": 0.02, "trust_region": 0.02, "step_size": 0.02},
    {"epsilon": 0.02, "trust_region": 0.1, "step_size": 0.02},
    {"epsilon": 0.02, "trust_region": 0.05, "step_size": 0.01},
    {"epsilon": 0.02, "trust_region": 0.05, "step_size": 0.05},
]

# Final, chosen params -- PLACEHOLDER (Table VII defaults) until a real
# --tuning sweep picks the actual trade-off point. Only read in eval mode.
CHOSEN_PARAMS = {"epsilon": 0.02, "trust_region": 0.05, "step_size": 0.02}

RUN_NAME = "calvin_shortstop_runs"


def _run_one_config(
    log, run_output_dir, label, shield, params, env, policy, task_oracle, lang_embeddings, val_annotations,
    get_env_state_for_initial_condition, eval_sequences, cfg, sequence_seed_base,
):
    obstacle_fn = lambda joint_angles, chunk: sample_obstacle_from_reference_chunk(  # noqa: E731
        joint_angles, chunk, radius=OBSTACLE_RADIUS,
    )

    sequence_results = []
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(sequence_seed_base + idx, workers=True)
        attempts = run_calvin_shielded_sequence(
            env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
            get_env_state_for_initial_condition, shield, ep_len=cfg.ep_len, replan_steps=REPLAN_STEPS,
            obstacle_fn=obstacle_fn,
        )
        sequence_results.append(attempts)

    slots = build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5)
    violation_rate, success_rate = fixed_cohort_rates(slots)
    activation_rate = shield_activation_rate(sequence_results)
    total_fallback_rate = fallback_rate(sequence_results)
    lat_stats = latency_stats(sequence_results)
    precision = intervention_precision(sequence_results)
    recovery = recovery_rate(sequence_results)
    log(
        f"[{label}] violation_rate={violation_rate:.3f}  success_rate={success_rate:.3f}  "
        f"shield_activation_rate={activation_rate:.3f}  fallback_rate={total_fallback_rate:.3f}  "
        f"recovery_rate={'n/a' if recovery is None else f'{recovery:.3f}'}  "
        f"(avg_seq_len={success_rate * 5:.2f}/5, n_sequences={cfg.num_sequences})"
    )
    precision_str = "n/a" if precision is None else f"{precision:.3f}"
    if lat_stats is not None:
        log(
            f"  latency_ms_mean={lat_stats['mean']:.3f}  latency_ms_median={lat_stats['median']:.3f}  "
            f"latency_ms_p95={lat_stats['p95']:.3f}  intervention_precision={precision_str}"
        )
    else:
        log("  (no decisions recorded any latency)")

    entry = {
        "label": label,
        "epsilon": params["epsilon"],
        "trust_region": params["trust_region"],
        "step_size": params["step_size"],
        "model_error": MODEL_ERROR,
        "violation_rate": violation_rate,
        "success_rate": success_rate,
        "shield_activation_rate": activation_rate,
        "fallback_rate": total_fallback_rate,
        "recovery_rate": recovery,
        "avg_seq_len": success_rate * 5,
        "n_sequences": cfg.num_sequences,
        "latency_ms": lat_stats,
        "intervention_precision": precision,
        "clearance_stats": None,
        "video_paths": None,
        "video_skip_reason": None,
        "sequence_results": sequence_results,
    }

    if cfg.debug:
        stats = clearance_stats(sequence_results)
        entry["clearance_stats"] = stats
        log_clearance_debug(log, label, stats)
        vis_idxs = rank_violating_sequence_idxs_by_length(sequence_results, cfg.num_videos)
        if not vis_idxs:
            reason = "no sequence violated at this config -- skipping video"
            log(f"  [debug] {label}: {reason}")
            entry["video_skip_reason"] = reason
        else:
            safe_label = (
                f"e{params['epsilon']}_tr{params['trust_region']}_ss{params['step_size']}"
            ).replace(".", "p").replace("-", "neg")
            video_paths = []
            for vis_idx in vis_idxs:
                video_paths += save_debug_videos(
                    run_output_dir, run_output_dir / "videos", vis_idx, safe_label, obstacle_fn, shield, env,
                    policy, task_oracle, lang_embeddings, val_annotations, get_env_state_for_initial_condition,
                    cfg, eval_sequences, sequence_seed_base, REPLAN_STEPS,
                )
            entry["video_paths"] = video_paths

    return entry


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    run_output_dir = make_run_output_dir(REPO_ROOT, RUN_NAME, TUNING_MODE)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_output_dir / "run.log")
    log(f"[run] writing log + results.json (+ videos/ if any) to: {run_output_dir}")
    log(f"[run] cohort: {'TUNING (idx 0..N-1)' if TUNING_MODE else 'EVAL (idx N..2N-1)'} "
        f"-- pass --tuning to switch (see docs/TUNING_WORKFLOW.md muc 0)")
    log(f"[run] config: num_sequences={cfg.num_sequences} ep_len={cfg.ep_len} replan_steps={REPLAN_STEPS} "
        f"sampler_type={cfg.sampler_type} num_sampling_steps={cfg.num_sampling_steps} debug={cfg.debug} "
        f"n_candidates={N_CANDIDATES} obstacle_radius={OBSTACLE_RADIUS} model_error={MODEL_ERROR}")

    seed_everything(0, workers=True)
    env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, N_CANDIDATES)
    from mdt.evaluation.utils import get_env_state_for_initial_condition

    eval_sequences, sequence_seed_base, cohort_offset, N = cohort_sequences(cfg, TUNING_MODE)

    results_path = run_output_dir / "results.json"

    def _write_progress(results):
        # Re-written after EVERY sweep config, not just once at the end --
        # see run_calvin_mpc_filter.py's identical helper for why.
        write_results_json(results_path, {
            "tuning_mode": TUNING_MODE,
            "cohort_sequence_idx_range": [cohort_offset, cohort_offset + N],
            "n_candidates": N_CANDIDATES,
            "obstacle_radius": OBSTACLE_RADIUS,
            "model_error": MODEL_ERROR,
            "params_to_sweep": PARAMS_TO_SWEEP if TUNING_MODE else None,
            "chosen_params": None if TUNING_MODE else CHOSEN_PARAMS,
            "results": results,
        })

    results = []
    configs = PARAMS_TO_SWEEP if TUNING_MODE else [CHOSEN_PARAMS]
    for params in configs:
        label = (
            f"epsilon={params['epsilon']},trust_region={params['trust_region']},step_size={params['step_size']}"
        )
        if not TUNING_MODE:
            label = "FINAL " + label
        shield = ArmRepairShield(
            obstacles=[], w_bar=0.0, model_error=MODEL_ERROR,
            epsilon=params["epsilon"], trust_region=params["trust_region"], step_size=params["step_size"],
        )
        entry = _run_one_config(
            log, run_output_dir, label, shield, params, env, policy, task_oracle, lang_embeddings,
            val_annotations, get_env_state_for_initial_condition, eval_sequences, cfg, sequence_seed_base,
        )
        results.append(entry)
        _write_progress(results)
        log(f"  [progress] wrote {len(results)}/{len(configs)} config(s) so far to: {results_path}")

    log(f"[run] wrote structured results to: {results_path}")
    log(f"[run] DONE -- zip up {run_output_dir} and send it back for tuning analysis")
    log.file.close()


if __name__ == "__main__":
    main()
