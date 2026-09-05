"""Shared plumbing for the CALVIN baseline driver scripts (scripts/run_
calvin_*.py) -- setup boilerplate, tuning/eval cohort split, and the
metric helpers every one of those scripts otherwise re-implements
identically (confirmed: run_calvin_shielded.py/run_calvin_stl_monitor.py
both carry byte-for-byte copies of `_clearance_stats`/
`_shield_activation_rate`/`_latency_stats`/`_intervention_precision`/
`_rank_violating_sequence_idxs_by_length`, each one's own docstring saying
"Identical to scripts/run_calvin_shielded.py's own helper").

Deliberately NOT retrofitted into the 3 EXISTING scripts (run_calvin_
unshielded.py/run_calvin_shielded.py/run_calvin_stl_monitor.py) -- those
already have real, cited eval numbers tied to their exact current source;
touching them risks a silent behavior change in already-validated code for
no real benefit. This module exists so every NEW baseline script (MPC-
Filter, ShortStop, the stage ablation, ...) shares ONE copy instead of
starting a fourth/fifth/sixth copy-paste.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from shortstop.calvin_experiment import run_calvin_shielded_sequence
from shortstop.calvin_obstacle_viz import save_sequence_video


def setup_env_and_policy(cfg, n_candidates):
    """The identical hydra/checkpoint/model/env/policy bootstrap every
    run_calvin_*.py script's own `main(cfg)` starts with -- returns
    `(env, policy, task_oracle, lang_embeddings)`. Sampler overrides
    (`num_sampling_steps`/`sampler_type`/`sigma_min`/`sigma_max`/
    `noise_scheduler`) are applied from `cfg`, matching every existing
    script's own identical block. Deliberately does NOT set
    `model.multistep` -- that attribute only gates `MDTVAgent.step()`'s
    own internal replan counter, never called by `ForwardOnlyPolicy`."""
    from mdt.evaluation.utils import get_default_beso_and_env
    from mdt.utils.utils import get_last_checkpoint
    from shortstop.mdt_policy_client import ForwardOnlyPolicy

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

    import hydra
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    policy = ForwardOnlyPolicy(model, n_candidates=n_candidates)
    return env, policy, task_oracle, val_annotations, lang_embeddings


def cohort_sequences(cfg, tuning_mode):
    """Tuning cohort = idx 0..N-1, eval cohort = idx N..2N-1, disjoint --
    identical mechanism/convention to every existing run_calvin_*.py
    script (see docs/TUNING_WORKFLOW.md muc 0 / [[feedback_tuning_cohort_split]]).
    Returns `(eval_sequences, sequence_seed_base, cohort_offset, N)`."""
    from mdt.evaluation.multistep_sequences import get_sequences

    N = cfg.num_sequences
    cohort_offset = 0 if tuning_mode else N
    eval_sequences = get_sequences(2 * N)[cohort_offset:cohort_offset + N]
    sequence_seed_base = 1000 + cohort_offset
    return eval_sequences, sequence_seed_base, cohort_offset, N


def clearance_stats(sequence_results):
    clearances = [
        a["min_clearance"] for attempts in sequence_results for a in attempts
        if a["min_clearance"] is not None
    ]
    if not clearances:
        return None
    clearances = np.asarray(clearances)
    return {
        "n": len(clearances),
        "mean": float(clearances.mean()),
        "median": float(np.median(clearances)),
        "p10": float(np.percentile(clearances, 10)),
        "p90": float(np.percentile(clearances, 90)),
        "min": float(clearances.min()),
        "max": float(clearances.max()),
    }


def log_clearance_debug(log, label, stats):
    if stats is None:
        log(f"  [debug] {label}: no attempted subtasks had an obstacle to measure clearance against")
        return
    log(
        f"  [debug] {label}: min_clearance over {stats['n']} attempted subtasks -- "
        f"mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
        f"p10={stats['p10']:.4f}  p90={stats['p90']:.4f}  "
        f"min={stats['min']:.4f}  max={stats['max']:.4f}"
    )


def shield_activation_rate(sequence_results):
    """Pooled ratio over every decision, not a mean of per-subtask rates."""
    n_decisions = sum(a["n_decisions"] for attempts in sequence_results for a in attempts)
    n_activated = sum(a["n_activated"] for attempts in sequence_results for a in attempts)
    if n_decisions == 0:
        return None
    return n_activated / n_decisions


def fallback_rate(sequence_results):
    """Pooled ratio of decisions that were a TOTAL fallback (n_admissible
    == 0), distinct from shield_activation_rate (which also counts a
    PARTIAL rejection) -- see calvin_experiment.run_calvin_shielded_subtask's
    'n_fallback' field for why this is tracked separately (the fallback-
    window-shrink fix's own diagnostic)."""
    n_decisions = sum(a["n_decisions"] for attempts in sequence_results for a in attempts)
    n_fallback = sum(a.get("n_fallback", 0) for attempts in sequence_results for a in attempts)
    if n_decisions == 0:
        return None
    return n_fallback / n_decisions


def latency_stats(sequence_results):
    all_latencies = [t for attempts in sequence_results for a in attempts for t in a["latencies_ms"]]
    if not all_latencies:
        return None
    values = np.asarray(all_latencies)
    return {
        "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def intervention_precision(sequence_results):
    """Pooled rejected_truly_unsafe/rejected_total across every decision."""
    total_rejected = sum(a["rejected_total"] for attempts in sequence_results for a in attempts)
    total_truly_unsafe = sum(a["rejected_truly_unsafe"] for attempts in sequence_results for a in attempts)
    if total_rejected == 0:
        return None
    return total_truly_unsafe / total_rejected


def rank_violating_sequence_idxs_by_length(sequence_results, top_k):
    """Sequence indices with at least one violated attempt, longest-running first."""
    violating = [idx for idx, attempts in enumerate(sequence_results) if any(a["violated"] for a in attempts)]
    violating.sort(key=lambda idx: len(sequence_results[idx]), reverse=True)
    return violating[:top_k]


def save_debug_videos(
    run_output_dir, vis_output_dir, sequence_idx, safe_label, obstacle_fn, shield, env, policy, task_oracle,
    lang_embeddings, val_annotations, get_env_state_for_initial_condition, cfg, eval_sequences, sequence_seed_base,
    replan_steps,
):
    """Re-runs sequence `sequence_idx` alone (same per-sequence seed the
    main pass already used for it) with recording on, merges every
    subtask attempt into ONE mp4. `safe_label` must already be filesystem-
    safe (caller's responsibility -- e.g. replace '.'/'-' the way every
    existing script's own epsilon/threshold label does)."""
    from pytorch_lightning import seed_everything

    initial_state, eval_sequence = eval_sequences[sequence_idx]
    seed_everything(sequence_seed_base + sequence_idx, workers=True)
    attempts = run_calvin_shielded_sequence(
        env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
        get_env_state_for_initial_condition, shield, ep_len=cfg.ep_len, replan_steps=replan_steps,
        obstacle_fn=obstacle_fn, record_trajectory=True, record_camera_frames=True,
    )
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    subtask_records = []
    for subtask, attempt in zip(eval_sequence, attempts):
        if attempt["violated"]:
            outcome = "violated"
        elif attempt["reached"]:
            outcome = "reached"
        else:
            outcome = "failed"
        subtask_records.append({
            "subtask": subtask,
            "frames": attempt["camera_frames"],
            "depth_frames": attempt["depth_frames"],
            "obstacle": attempt["obstacle"],
            "outcome": outcome,
        })

    static_camera = next(cam for cam in env.env.cameras if cam.name == "static")
    out_path = vis_output_dir / f"seq{sequence_idx}_{safe_label}.mp4"
    save_sequence_video(
        subtask_records, static_camera, str(out_path),
        base_position=env.env.robot.base_position, base_orientation=env.env.robot.base_orientation,
    )
    return [str(out_path.relative_to(run_output_dir))]


def make_run_output_dir(repo_root, run_name, tuning_mode):
    return repo_root / "outputs" / run_name / f"run_{datetime.now():%Y%m%d_%H%M%S}_{'tuning' if tuning_mode else 'eval'}"


def make_logger(log_file_path):
    """Opens `log_file_path` for writing and returns a `log(msg)` function
    that both prints and appends to it -- same tee-to-file pattern every
    existing script implements as a module-level `_log`/`_LOG_FILE` pair,
    packaged as a closure instead so multiple scripts can each get their
    own independent logger from this one shared module."""
    log_file = open(log_file_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log.file = log_file
    return log


def write_results_json(results_path, payload):
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
