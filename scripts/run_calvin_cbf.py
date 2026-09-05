"""CBF-Shield baseline: shortstop.arm_shield.ArmCBFShield wired into the
CALVIN eval loop -- Propose K candidates -> take candidates[0] only,
correct ONLY its first action via a tiny, single-step control-barrier-
function QP (see that class's own docstring for exactly what happens on
each cycle, and why this is a fundamentally different scope from MPC-
Filter's own whole-horizon QP) -- Select is a no-op here (CBF doesn't rank
K candidates, see g(a)'s absence from this script entirely, matching
`ArmCBFShield.select()`'s own `del scores`).

Filter freq is decoupled from policy freq here too, automatically --
ArmCBFShield inherits ArmReachOnlyShield.recertify() (unused: this class
defines its own `resolve()`, which the harness always prefers when a
shield has one -- see `run_calvin_shielded_subtask`'s own docstring).
Unlike ArmMPCFilterShield (whose `resolve()` is what makes IT genuinely
receding-horizon), CBF's `select()` and `resolve()` are the EXACT SAME
tiny single-step QP call by construction -- there is no "remaining
horizon" to re-plan here at all, matching the literature's own framing of
CBF-QP as a real-time, per-control-cycle controller rather than a
chunk-level planner.

`alpha` (the class-K gain, docs/PARAMETERS_REFERENCE.md's own "alpha
(CBF-Shield gain)" entry) is CBF's one distinguishing hyperparameter --
`w_bar`/`model_error` stay fixed at the SAME calibrated values every other
ShortStop-family shield uses (see run_calvin_mpc_filter.py's own docstring
for why CALVIN's real disturbance sources are captured end-to-end by
model_error's own calibration, so a separate w_bar budget is redundant),
so a violation/success difference between CBF and MPC-Filter/ShortStop at
the SAME budget reflects a genuine difference in filtering STRATEGY
(pointwise-reactive vs whole-horizon-predictive), not an unfair difference
in how much disturbance each one is allowed to assume away.

Run from WSL2, inside `mdt_env` (see docs/CALVIN_SETUP.md):

    cd SHORTSTOP
    python scripts/run_calvin_cbf.py            # eval cohort (default): runs CHOSEN_ALPHA once
    python scripts/run_calvin_cbf.py --tuning    # tuning cohort: sweep ALPHAS_TO_SWEEP

Tuning/eval cohort split: identical mechanism to every other run_calvin_
shielded_*.py script (see docs/TUNING_WORKFLOW.md muc 0).
"""
import sys
from pathlib import Path

import calvin_env  # noqa: F401 -- import side effects only, matches every other CALVIN script
import hydra
import numpy as np
from pytorch_lightning import seed_everything

TUNING_MODE = "--tuning" in sys.argv
if TUNING_MODE:
    sys.argv.remove("--tuning")

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from shortstop.arm_shield import ArmCBFShield  # noqa: E402
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
OBSTACLE_RADIUS = 0.06
# Must match run_calvin_unshielded.py's OBSTACLE_OFFSET_MAX exactly, not
# sample_obstacle_from_reference_chunk's own default -- see that
# function's docstring / run_calvin_mpc_filter.py's identical comment for
# why relying on the shared default would silently make this baseline's
# obstacle placement a DIFFERENT difficulty than the other baselines it's
# compared against.
OBSTACLE_OFFSET_MAX = 0.6

# w_bar=0.0, model_error=REAL calibrated value -- fixed, not swept here,
# same budget every other ShortStop-family shield uses (see module
# docstring). scripts/calibrate_arm_model_error.py, real run 2026-09-05,
# n=45632 residuals, p99=0.00777*1.25 -- see docs/PARAMETERS_REFERENCE.md's
# "model_error" entry.
W_BAR = 0.0
MODEL_ERROR = 0.00972

# Candidate alpha (class-K gain) values to sweep in --tuning mode --
# PLACEHOLDER, not yet independently swept for real. Anchored at alpha=1.0
# (the literature's own simplest/most common choice -- demands the next
# state be safe outright, see ArmCBFShield's own docstring on what alpha=
# 1.0 means), bracketed by a more conservative (0.3, intervenes earlier/
# farther out) and two more aggressive values (2.0, 4.0 -- permits the
# margin to drop faster/closer to the boundary before intervening).
ALPHAS_TO_SWEEP = [0.3, 0.5, 1.0, 2.0, 4.0]

# Final, chosen alpha -- PLACEHOLDER (the literature's own simplest
# choice) until a real --tuning sweep picks alpha* from the actual
# violation/success/activation trade-off. Only read in eval mode (no
# --tuning flag).
CHOSEN_ALPHA = 1.0

RUN_NAME = "calvin_cbf_runs"


def _run_one_config(
    log, run_output_dir, label, shield, alpha, env, policy, task_oracle, lang_embeddings,
    val_annotations, get_env_state_for_initial_condition, eval_sequences, cfg, sequence_seed_base,
):
    sequence_results = []
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(sequence_seed_base + idx, workers=True)
        # See scripts/run_calvin_shielded.py's identical block for why
        # this is a fresh, caller-seeded, non-global rng per sequence.
        obstacle_rng = np.random.default_rng(sequence_seed_base + idx)
        obstacle_fn = lambda joint_angles, chunk, rng=obstacle_rng: sample_obstacle_from_reference_chunk(  # noqa: E731
            joint_angles, chunk, radius=OBSTACLE_RADIUS, rng=rng, offset_max=OBSTACLE_OFFSET_MAX,
        )
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
        "alpha": alpha,
        "w_bar": W_BAR,
        "model_error": MODEL_ERROR,
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
            safe_label = f"alpha{alpha}".replace(".", "p").replace("-", "neg")
            video_paths = []
            for vis_idx in vis_idxs:
                # See scripts/run_calvin_shielded.py's identical block for
                # why this is rebuilt fresh here, seeded to reproduce the
                # exact placement that sequence actually saw.
                video_obstacle_rng = np.random.default_rng(sequence_seed_base + vis_idx)
                video_obstacle_fn = lambda joint_angles, chunk, rng=video_obstacle_rng: sample_obstacle_from_reference_chunk(  # noqa: E731,E501
                    joint_angles, chunk, radius=OBSTACLE_RADIUS, rng=rng, offset_max=OBSTACLE_OFFSET_MAX,
                )
                video_paths += save_debug_videos(
                    run_output_dir, run_output_dir / "videos", vis_idx, safe_label, video_obstacle_fn, shield, env,
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
        f"n_candidates={N_CANDIDATES} obstacle_radius={OBSTACLE_RADIUS} "
        f"obstacle_offset_max={OBSTACLE_OFFSET_MAX} w_bar={W_BAR} model_error={MODEL_ERROR}")

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
        # see scripts/run_calvin_mpc_filter.py's identical helper for why.
        write_results_json(results_path, {
            "tuning_mode": TUNING_MODE,
            "cohort_sequence_idx_range": [cohort_offset, cohort_offset + N],
            "n_candidates": N_CANDIDATES,
            "obstacle_radius": OBSTACLE_RADIUS,
            "w_bar": W_BAR,
            "model_error": MODEL_ERROR,
            "alphas_to_sweep": ALPHAS_TO_SWEEP if TUNING_MODE else None,
            "chosen_alpha": None if TUNING_MODE else CHOSEN_ALPHA,
            "results": results,
        })

    results = []
    alphas = ALPHAS_TO_SWEEP if TUNING_MODE else [CHOSEN_ALPHA]
    for alpha in alphas:
        label = f"alpha={alpha}" if TUNING_MODE else f"FINAL alpha={alpha}"
        shield = ArmCBFShield(obstacles=[], w_bar=W_BAR, model_error=MODEL_ERROR, alpha=alpha)
        entry = _run_one_config(
            log, run_output_dir, label, shield, alpha, env, policy, task_oracle, lang_embeddings,
            val_annotations, get_env_state_for_initial_condition, eval_sequences, cfg, sequence_seed_base,
        )
        results.append(entry)
        _write_progress(results)
        log(f"  [progress] wrote {len(results)}/{len(alphas)} config(s) so far to: {results_path}")

    log(f"[run] wrote structured results to: {results_path}")
    log(f"[run] DONE -- zip up {run_output_dir} and send it back for tuning analysis")
    log.file.close()


if __name__ == "__main__":
    main()
