"""MPC-Filter baseline: shortstop.arm_shield.ArmMPCFilterShield wired into
the CALVIN eval loop -- Propose K candidates -> take candidates[0] only,
correct it via a single QP (H-step, full 9-frame+8-link scope, see that
class's own docstring for the linearization this needs that the 2D
version doesn't) -- Select is a no-op here (MPC-Filter doesn't rank
K candidates, see g(a)'s absence from this script entirely, matching
`ArmMPCFilterShield.select()`'s own `del scores`).

Filter freq is decoupled from policy freq here too, automatically --
ArmMPCFilterShield inherits ArmReachOnlyShield.recertify() (see that
class's own docstring for why reusing it, unchanged, is correct even
though the chunk came from QP correction, not best-of-K selection).

Unlike Conf-Thresh/STL-Monitor, this baseline has no single named
hyperparameter of its own to sweep (no `disagreement_threshold`/
`epsilon`) -- its only tunable knobs are `w_bar`/`model_error`, shared
with every other real ShortStop-family shield and calibrated the SAME
way (see scripts/calibrate_arm_model_error.py and docs/PARAMETERS_
REFERENCE.md's "model_error"/"w_bar" entries -- CALVIN's own real
disturbance sources, e.g. calvin_env's own IK solver differing from this
repo's Jacobian-pinv model, plus real position-controller force/velocity
limits, are captured end-to-end by that one calibration residual, so
`w_bar=0.0` and `model_error` alone carries the full budget). `--tuning`
still sweeps a small grid of (w_bar, model_error) pairs, in case MPC-
Filter's own QP-based conservatism behaves differently from ShortStop's
reachtube at the SAME calibrated numbers -- not because this baseline
needs its own independently-tuned margin.

Run from WSL2, inside `mdt_env` (see docs/CALVIN_SETUP.md):

    cd SHORTSTOP
    python scripts/run_calvin_mpc_filter.py            # eval cohort (default): runs CHOSEN_* once
    python scripts/run_calvin_mpc_filter.py --tuning    # tuning cohort: sweep WBAR_MODEL_ERROR_TO_SWEEP

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

from shortstop.arm_shield import ArmMPCFilterShield  # noqa: E402
from shortstop.calvin_baseline_runner import (  # noqa: E402
    clearance_stats, cohort_sequences, fallback_rate, intervention_precision, latency_stats, log_clearance_debug,
    make_logger, make_run_output_dir, rank_violating_sequence_idxs_by_length, save_debug_videos,
    setup_env_and_policy, shield_activation_rate, write_results_json,
)
from shortstop.calvin_experiment import run_calvin_shielded_sequence  # noqa: E402
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates  # noqa: E402
from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk  # noqa: E402

# Shared with every other CALVIN baseline script -- see run_calvin_
# shielded.py's own comments for the full reasoning/history.
N_CANDIDATES = 8
REPLAN_STEPS = 10
OBSTACLE_RADIUS = 0.08

# Candidate (w_bar, model_error) pairs to sweep in --tuning mode --
# PLACEHOLDER, anchored at w_bar=0.0 (see module docstring: CALVIN's real
# disturbance is captured end-to-end by model_error's own calibration, so
# a separate w_bar budget is redundant, not "missing") across a spread of
# model_error values bracketing ArmReachOnlyShield's own generic default
# (0.02) -- revise once scripts/calibrate_arm_model_error.py has a real
# number to anchor around instead of guessing the spread.
WBAR_MODEL_ERROR_TO_SWEEP = [
    {"w_bar": 0.0, "model_error": 0.0},
    {"w_bar": 0.0, "model_error": 0.02},
    {"w_bar": 0.0, "model_error": 0.05},
    {"w_bar": 0.0, "model_error": 0.1},
]

# Final, chosen (w_bar, model_error) -- PLACEHOLDER until a real --tuning
# sweep (and scripts/calibrate_arm_model_error.py) pick real numbers. Only
# read in eval mode (no --tuning flag).
CHOSEN_W_BAR = 0.0
CHOSEN_MODEL_ERROR = 0.02

RUN_NAME = "calvin_mpc_filter_runs"


def _run_one_config(
    log, run_output_dir, label, shield, w_bar, model_error, env, policy, task_oracle, lang_embeddings,
    val_annotations, get_env_state_for_initial_condition, eval_sequences, cfg, sequence_seed_base,
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
    log(
        f"[{label}] violation_rate={violation_rate:.3f}  success_rate={success_rate:.3f}  "
        f"shield_activation_rate={activation_rate:.3f}  fallback_rate={total_fallback_rate:.3f}  "
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
        "w_bar": w_bar,
        "model_error": model_error,
        "violation_rate": violation_rate,
        "success_rate": success_rate,
        "shield_activation_rate": activation_rate,
        "fallback_rate": total_fallback_rate,
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
            safe_label = f"wbar{w_bar}_me{model_error}".replace(".", "p").replace("-", "neg")
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
        f"n_candidates={N_CANDIDATES} obstacle_radius={OBSTACLE_RADIUS}")

    seed_everything(0, workers=True)
    env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, N_CANDIDATES)
    from mdt.evaluation.utils import get_env_state_for_initial_condition

    eval_sequences, sequence_seed_base, cohort_offset, N = cohort_sequences(cfg, TUNING_MODE)

    # obstacles=[] is a placeholder: run_calvin_shielded_subtask overwrites
    # shield.obstacles from candidates[0] before this shield's very first
    # decision of each subtask (see its own docstring) -- never read as [].
    results_path = run_output_dir / "results.json"

    def _write_progress(results):
        # Re-written after EVERY sweep config, not just once at the end --
        # a tuning sweep is slow enough (each config = a full cohort
        # rollout) that losing everything to a crash/interrupt partway
        # through, or having to wait for the whole sweep before seeing any
        # number, is real pain. Cheap to redo: `results` only grows one
        # entry at a time, and json.dump just overwrites the same file.
        write_results_json(results_path, {
            "tuning_mode": TUNING_MODE,
            "cohort_sequence_idx_range": [cohort_offset, cohort_offset + N],
            "n_candidates": N_CANDIDATES,
            "obstacle_radius": OBSTACLE_RADIUS,
            "wbar_model_error_to_sweep": WBAR_MODEL_ERROR_TO_SWEEP if TUNING_MODE else None,
            "chosen_w_bar": None if TUNING_MODE else CHOSEN_W_BAR,
            "chosen_model_error": None if TUNING_MODE else CHOSEN_MODEL_ERROR,
            "results": results,
        })

    results = []
    configs = WBAR_MODEL_ERROR_TO_SWEEP if TUNING_MODE else [{"w_bar": CHOSEN_W_BAR, "model_error": CHOSEN_MODEL_ERROR}]
    for config in configs:
        w_bar, model_error = config["w_bar"], config["model_error"]
        label = f"w_bar={w_bar},model_error={model_error}" if TUNING_MODE else f"FINAL w_bar={w_bar},model_error={model_error}"
        shield = ArmMPCFilterShield(obstacles=[], w_bar=w_bar, model_error=model_error)
        entry = _run_one_config(
            log, run_output_dir, label, shield, w_bar, model_error, env, policy, task_oracle, lang_embeddings,
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
