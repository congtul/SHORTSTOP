"""ShortStop ablation on CALVIN -- Stage 1 (ArmReachOnlyShield, binary
reach/no-reach) vs Stage 2 (ArmSTLShield, STL margin) vs Stage 4
(ArmRepairShield, counterexample-guided repair -- the paper's own full
method, see scripts/run_calvin_shortstop.py for its dedicated tuning).
Stage 3 (counterexample search) is NOT its own row here -- see
shortstop/arm_shield.py's module docstring for why (Stage 2 and Stage 3
produce identical accept/reject decisions on the 2D prototype; CE search
is still exercised internally, by ArmRepairShield's own Repair step, just
not surfaced as a separate comparison row).

Every stage shares the SAME (w_bar=0.0, model_error=MODEL_ERROR) budget
(see run_calvin_shortstop.py's own docstring for why w_bar=0.0) -- the
ablation studies what EACH STAGE'S OWN MECHANISM adds on top of that
shared foundation, not a confound from different shields using different
disturbance assumptions.

Adding a new stage later (e.g. once MPC-Filter/CBF-Shield are wired in, if
you want them in the SAME ablation table instead of their own standalone
scripts): register one new entry in the `STAGES` dict below -- a
`factory(w_bar, model_error, params) -> shield`, a `chosen_params` dict
(used in eval mode and whenever `--tuning` has no `params_to_sweep` for
that stage), and an optional `params_to_sweep` list (used only in
`--tuning` mode, `None` for a stage with nothing of its own to sweep,
e.g. Stage 1's binary test has no margin/threshold). Nothing else in this
script needs to change -- the main loop iterates `STAGES` generically.

Run from WSL2, inside `mdt_env` (see docs/CALVIN_SETUP.md):

    cd SHORTSTOP
    python scripts/run_calvin_shortstop_ablation.py            # eval cohort: each stage's own chosen_params, once
    python scripts/run_calvin_shortstop_ablation.py --tuning    # tuning cohort: each stage's own params_to_sweep

Tuning/eval cohort split: identical mechanism to every other run_calvin_
shielded_*.py script (see docs/TUNING_WORKFLOW.md muc 0). Note this
script's own `--tuning` is NOT the primary place to fine-tune Stage 4's
own (epsilon, trust_region, step_size) -- that's scripts/run_calvin_
shortstop.py's job (a wider, dedicated sweep); this script's per-stage
`params_to_sweep` lists are intentionally small, meant to sanity-check
the ablation story (does each added stage actually help?) rather than
to re-derive each stage's own best hyperparameters from scratch.
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

from shortstop.arm_shield import ArmReachOnlyShield, ArmRepairShield, ArmSTLShield  # noqa: E402
from shortstop.calvin_baseline_runner import (  # noqa: E402
    clearance_stats, cohort_sequences, fallback_rate, intervention_precision, latency_stats, log_clearance_debug,
    make_logger, make_run_output_dir, rank_violating_sequence_idxs_by_length, save_debug_videos,
    setup_env_and_policy, shield_activation_rate, write_results_json,
)
from shortstop.calvin_experiment import run_calvin_shielded_sequence  # noqa: E402
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates, recovery_rate  # noqa: E402
from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk  # noqa: E402

N_CANDIDATES = 8
REPLAN_STEPS = 10
OBSTACLE_RADIUS = 0.08

# Shared foundation every stage below builds on -- PLACEHOLDER until
# scripts/calibrate_arm_model_error.py has a real number (see
# run_calvin_shortstop.py's own docstring for why w_bar stays 0.0).
MODEL_ERROR = 0.02

# STAGES registry -- see module docstring for the contract each entry
# must satisfy. Order here is the order rows print/serialize in, so keep
# it Stage 1 -> Stage 2 -> Stage 4 (paper's own progression).
STAGES = {
    "stage1_reach_only": {
        "factory": lambda w_bar, model_error, params: ArmReachOnlyShield(
            obstacles=[], w_bar=w_bar, model_error=model_error,
        ),
        "chosen_params": {},
        # Nothing to sweep -- a binary robustness>=0 test has no margin/
        # threshold of its own (unlike Stage 2's epsilon).
        "params_to_sweep": None,
    },
    "stage2_stl": {
        "factory": lambda w_bar, model_error, params: ArmSTLShield(
            obstacles=[], w_bar=w_bar, model_error=model_error, epsilon=params["epsilon"],
        ),
        "chosen_params": {"epsilon": 0.02},
        "params_to_sweep": [{"epsilon": e} for e in (0.0, 0.02, 0.05, 0.1)],
    },
    "stage4_repair": {
        "factory": lambda w_bar, model_error, params: ArmRepairShield(
            obstacles=[], w_bar=w_bar, model_error=model_error,
            epsilon=params["epsilon"], trust_region=params["trust_region"], step_size=params["step_size"],
        ),
        # Mirrors run_calvin_shortstop.py's own CHOSEN_PARAMS -- keep the
        # two in sync if that script's own tuning changes this.
        "chosen_params": {"epsilon": 0.02, "trust_region": 0.05, "step_size": 0.02},
        "params_to_sweep": [
            {"epsilon": 0.02, "trust_region": 0.05, "step_size": 0.02},  # Table VII defaults
            {"epsilon": 0.02, "trust_region": 0.02, "step_size": 0.02},
            {"epsilon": 0.02, "trust_region": 0.1, "step_size": 0.02},
        ],
    },
}

RUN_NAME = "calvin_shortstop_ablation_runs"


def _run_one_config(
    log, run_output_dir, label, shield, stage_name, params, env, policy, task_oracle, lang_embeddings,
    val_annotations, get_env_state_for_initial_condition, eval_sequences, cfg, sequence_seed_base,
):
    sequence_results = []
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(sequence_seed_base + idx, workers=True)
        # See scripts/run_calvin_shielded.py's identical block for why
        # this is a fresh, caller-seeded, non-global rng per sequence.
        obstacle_rng = np.random.default_rng(sequence_seed_base + idx)
        obstacle_fn = lambda joint_angles, chunk, rng=obstacle_rng: sample_obstacle_from_reference_chunk(  # noqa: E731
            joint_angles, chunk, radius=OBSTACLE_RADIUS, rng=rng,
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
        "stage": stage_name,
        "params": params,
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
            safe_label = f"{stage_name}_{params}".replace(".", "p").replace("-", "neg")
            safe_label = "".join(c for c in safe_label if c.isalnum() or c in "_")
            video_paths = []
            for vis_idx in vis_idxs:
                # See scripts/run_calvin_shielded.py's identical block for
                # why this is rebuilt fresh here, seeded to reproduce the
                # exact placement that sequence actually saw.
                video_obstacle_rng = np.random.default_rng(sequence_seed_base + vis_idx)
                video_obstacle_fn = lambda joint_angles, chunk, rng=video_obstacle_rng: sample_obstacle_from_reference_chunk(  # noqa: E731,E501
                    joint_angles, chunk, radius=OBSTACLE_RADIUS, rng=rng,
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
        f"n_candidates={N_CANDIDATES} obstacle_radius={OBSTACLE_RADIUS} model_error={MODEL_ERROR} "
        f"stages={list(STAGES.keys())}")

    seed_everything(0, workers=True)
    env, policy, task_oracle, val_annotations, lang_embeddings = setup_env_and_policy(cfg, N_CANDIDATES)
    from mdt.evaluation.utils import get_env_state_for_initial_condition

    eval_sequences, sequence_seed_base, cohort_offset, N = cohort_sequences(cfg, TUNING_MODE)

    results_path = run_output_dir / "results.json"

    def _write_progress(results):
        # Re-written after EVERY (stage, params) config, not just once at
        # the end -- see run_calvin_mpc_filter.py's identical helper for
        # why. Especially valuable here: 3 stages x their own sweep sizes
        # can be a long run, and a later stage's crash shouldn't cost the
        # earlier stages' already-finished results.
        write_results_json(results_path, {
            "tuning_mode": TUNING_MODE,
            "cohort_sequence_idx_range": [cohort_offset, cohort_offset + N],
            "n_candidates": N_CANDIDATES,
            "obstacle_radius": OBSTACLE_RADIUS,
            "model_error": MODEL_ERROR,
            "stages": {
                name: {"chosen_params": spec["chosen_params"], "params_to_sweep": spec["params_to_sweep"]}
                for name, spec in STAGES.items()
            },
            "results": results,
        })

    results = []
    total_configs = sum(
        len(spec["params_to_sweep"]) if TUNING_MODE and spec["params_to_sweep"] is not None else 1
        for spec in STAGES.values()
    )
    for stage_name, spec in STAGES.items():
        if TUNING_MODE and spec["params_to_sweep"] is not None:
            configs = spec["params_to_sweep"]
        else:
            configs = [spec["chosen_params"]]

        for params in configs:
            label = f"{stage_name} {params}"
            if not TUNING_MODE:
                label = "FINAL " + label
            shield = spec["factory"](0.0, MODEL_ERROR, params)
            entry = _run_one_config(
                log, run_output_dir, label, shield, stage_name, params, env, policy, task_oracle,
                lang_embeddings, val_annotations, get_env_state_for_initial_condition, eval_sequences, cfg,
                sequence_seed_base,
            )
            results.append(entry)
            _write_progress(results)
            log(f"  [progress] wrote {len(results)}/{total_configs} config(s) so far to: {results_path}")

    log(f"[run] wrote structured results to: {results_path}")
    log(f"[run] DONE -- zip up {run_output_dir} and send it back for tuning analysis")
    log.file.close()


if __name__ == "__main__":
    main()
