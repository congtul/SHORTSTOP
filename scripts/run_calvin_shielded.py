"""Shielded baseline (Conf-Thresh): shortstop.arm_shield.ArmConfThreshShield
wired into the CALVIN eval loop -- Propose K candidates -> reject on
sampler-ensemble disagreement (docs/main (3).txt Sec. V.D) -> Select by
shortstop.calvin_progress's g(a) (a kinematic goal-distance proxy, see
that module's docstring). The privileged obstacle (radius=0.08, already
tuned -- see docs/PARAMETERS_REFERENCE.md) is still injected exactly as
in scripts/run_calvin_unshielded.py, purely for ground-truth measurement
-- Conf-Thresh's own filter never sees it (see shortstop.arm_shield.
ArmConfThreshShield's docstring).

Run from WSL2, inside `mdt_env` (see docs/CALVIN_SETUP.md):

    cd SHORTSTOP
    python scripts/run_calvin_shielded.py            # eval cohort (default): runs CHOSEN_THRESHOLD once
    python scripts/run_calvin_shielded.py --tuning    # tuning cohort: diagnostic + THRESHOLDS_TO_SWEEP

Tuning/eval cohort split: identical mechanism to
scripts/run_calvin_unshielded.py's own `--tuning` flag (see its
docstring and docs/TUNING_WORKFLOW.md muc 0) -- copy-pasted rather than
shared, since these are still two small, mostly independent scripts.

Three phases:
  --tuning:
    1. Diagnostic (disagreement_threshold=inf): every candidate passes
       the filter, but select() still runs Select-by-g(a) among all K --
       this is NOT equivalent to the unshielded baseline (which always
       executes candidates[0]); it's a genuine "select-only" comparison
       row in its own right, reported alongside the sweep. Also logs
       percentile stats of every disagreement value seen (collected via
       a thin instrumentation wrapper around this one pass, not a second
       re-run), to inform THRESHOLDS_TO_SWEEP below (same role
       _log_clearance_debug played for the radius sweep).
    2. Sweep (THRESHOLDS_TO_SWEEP): pick tau* from the violation/success/
       shield-activation trade-off once real numbers are in -- this list
       starts as a placeholder, same as RADII_TO_SWEEP once did.
  (no flag, default) eval:
    3. Final (CHOSEN_THRESHOLD, run once on the disjoint held-out
       cohort) -- the number that actually gets reported.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import calvin_env
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

from mdt.evaluation.multistep_sequences import get_sequences  # noqa: E402
from mdt.evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition  # noqa: E402
from mdt.utils.utils import get_last_checkpoint  # noqa: E402

from shortstop.arm_shield import ArmConfThreshShield  # noqa: E402
from shortstop.calvin_experiment import run_calvin_shielded_sequence  # noqa: E402
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates  # noqa: E402
from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk  # noqa: E402
from shortstop.calvin_obstacle_viz import save_sequence_video  # noqa: E402
from shortstop.mdt_policy_client import ForwardOnlyPolicy  # noqa: E402

# K candidates per replan decision -- paper's own starting value (Report_2:
# "K nho (paper dung K=8)"), NOT re-tuned here: doing so needs a latency
# measurement this pass doesn't add. Revisit if per-decision latency (K
# sequential diffusion samples) turns out to matter on real hardware.
N_CANDIDATES = 8

# Replan cadence -- MUST match scripts/run_calvin_unshielded.py's own
# REPLAN_STEPS exactly (same reasoning: pure harness-level chunk slicing,
# not a model property, see that script's comment) for the unshielded-
# vs-shielded comparison to stay apples-to-apples. Tried =5 (ratio 0.5,
# Diffusion Policy convention) on 2026-09-04, reverted same day: real
# sweep showed a ~10pp baseline success_rate cost from more frequent
# replanning alone (independent noise per propose() call = more "seams"
# between chunks), with no corresponding benefit for Conf-Thresh (its
# filter frequency is tied to policy frequency regardless -- see docs/
# PARAMETERS_REFERENCE.md's "multistep / replan_steps" entry for the
# full writeup). Back to =10 (ratio 1.0, matching act_window_size).
REPLAN_STEPS = 10

# Obstacle radius: already tuned at this same REPLAN_STEPS=10, see
# docs/PARAMETERS_REFERENCE.md / scripts/run_calvin_unshielded.py's own
# RADII_TO_SWEEP comment. Re-tune there (not here) if the checkpoint/
# dataset or REPLAN_STEPS ever change again.
OBSTACLE_RADIUS = 0.08

# Candidate disagreement_threshold values to sweep (meters, same units as
# the endpoint-vs-centroid distance ArmConfThreshShield computes) --
# REVISED 2026-09-05 from the diagnostic pass's real percentiles (real
# run, n_sequences=100, REPLAN_STEPS=10, radius=0.08): disagreement over
# 17856 candidate-level values -- mean=0.368 median=0.349 p10=0.146
# p90=0.609 p99=0.906 min=0.003 max=1.387. The original placeholder
# [0.02, 0.05, 0.1] was far below even p10 -- every one of them rejected
# essentially every candidate (shield_activation_rate 0.996-1.0,
# success_rate ~0), so no real tradeoff curve was visible. Spans p10 ->
# median -> p90 -> p99 this time to actually see violation/success/
# activation move. Only read in --tuning mode.
THRESHOLDS_TO_SWEEP = [0.15, 0.35, 0.6, 0.9]

# Skip Phase 1 (disagreement_threshold=inf diagnostic) in --tuning runs --
# it's fully deterministic given the same cohort/checkpoint/K/radius/
# REPLAN_STEPS, so re-running it just reproduces the same percentiles
# already used to pick THRESHOLDS_TO_SWEEP above (real numbers logged
# there). Flip back to True if K/radius/REPLAN_STEPS/checkpoint change
# and the percentiles need refreshing.
RUN_DIAGNOSTIC = False

# Final, already-chosen threshold -- PLACEHOLDER until the tuning pass
# above actually picks tau*. Only read in eval mode (no --tuning flag).
CHOSEN_THRESHOLD = 0.05

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "calvin_shielded_runs" / (
    f"run_{datetime.now():%Y%m%d_%H%M%S}_{'tuning' if TUNING_MODE else 'eval'}"
)
VIS_OUTPUT_DIR = RUN_OUTPUT_DIR / "videos"

_LOG_FILE = None  # opened at the top of main(), see _log()


def _log(msg):
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def _clearance_stats(sequence_results):
    """Same shape as run_calvin_unshielded.py's own helper -- percentile
    stats of min_clearance over every attempted subtask, None if none had
    an obstacle to measure against."""
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


def _log_clearance_debug(label, stats):
    if stats is None:
        _log(f"  [debug] {label}: no attempted subtasks had an obstacle to measure clearance against")
        return
    _log(
        f"  [debug] {label}: min_clearance over {stats['n']} attempted subtasks -- "
        f"mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
        f"p10={stats['p10']:.4f}  p90={stats['p90']:.4f}  "
        f"min={stats['min']:.4f}  max={stats['max']:.4f}"
    )


def _shield_activation_rate(sequence_results):
    """Pooled ratio over every replan decision across every attempted
    subtask -- sum(n_activated)/sum(n_decisions), NOT a mean of each
    subtask's own rate (that would implicitly weight a 3-decision subtask
    the same as a 30-decision one). See docs/TUNING_WORKFLOW.md."""
    n_decisions = sum(a["n_decisions"] for attempts in sequence_results for a in attempts)
    n_activated = sum(a["n_activated"] for attempts in sequence_results for a in attempts)
    if n_decisions == 0:
        return None
    return n_activated / n_decisions


def _disagreement_percentiles(disagreement_samples):
    if not disagreement_samples:
        return None
    values = np.asarray(disagreement_samples)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _log_disagreement_debug(label, stats):
    if stats is None:
        _log(f"  [debug] {label}: no decisions recorded any disagreement value")
        return
    _log(
        f"  [debug] {label}: disagreement over {stats['n']} decisions -- "
        f"mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
        f"p10={stats['p10']:.4f}  p90={stats['p90']:.4f}  p99={stats['p99']:.4f}  "
        f"min={stats['min']:.4f}  max={stats['max']:.4f}"
    )


def _rank_violating_sequence_idxs_by_length(sequence_results, top_k):
    """Same as scripts/run_calvin_unshielded.py's own helper -- sequence
    indices with at least one violated attempt, longest-running first."""
    violating = [idx for idx, attempts in enumerate(sequence_results) if any(a["violated"] for a in attempts)]
    violating.sort(key=lambda idx: len(sequence_results[idx]), reverse=True)
    return violating[:top_k]


def _save_debug_videos(
    sequence_idx, threshold, obstacle_fn, shield, env, policy, task_oracle, lang_embeddings, val_annotations,
    get_env_state_for_initial_condition, cfg, eval_sequences, sequence_seed_base,
):
    """Shielded analogue of run_calvin_unshielded.py's own
    _save_debug_videos -- re-runs sequence `sequence_idx` alone (same
    per-sequence seed the main sweep already used for it) with recording
    on, merges every subtask attempt into ONE mp4."""
    initial_state, eval_sequence = eval_sequences[sequence_idx]
    seed_everything(sequence_seed_base + sequence_idx, workers=True)
    attempts = run_calvin_shielded_sequence(
        env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
        get_env_state_for_initial_condition, shield, ep_len=cfg.ep_len, replan_steps=REPLAN_STEPS,
        obstacle_fn=obstacle_fn, record_trajectory=True, record_camera_frames=True,
    )
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_threshold = str(threshold).replace(".", "p")

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
    out_path = VIS_OUTPUT_DIR / f"seq{sequence_idx}_t{safe_threshold}.mp4"
    save_sequence_video(
        subtask_records, static_camera, str(out_path),
        base_position=env.env.robot.base_position, base_orientation=env.env.robot.base_orientation,
    )
    return [str(out_path.relative_to(RUN_OUTPUT_DIR))]


def _run_one_threshold(
    label, shield, threshold, env, policy, task_oracle, lang_embeddings, val_annotations,
    get_env_state_for_initial_condition, eval_sequences, cfg, sequence_seed_base,
    disagreement_sink=None,
):
    """Runs `shield` over every sequence in `eval_sequences`, returns the
    results.json-style entry for it. `disagreement_sink`, if given, is
    extended in-place with every `select()` call's per-candidate
    disagreement values during this same pass (no second re-run needed
    to harvest them) -- undone before any later (video) use of `shield`.
    """
    obstacle_fn = lambda joint_angles, chunk: sample_obstacle_from_reference_chunk(  # noqa: E731
        joint_angles, chunk, radius=OBSTACLE_RADIUS,
    )

    original_select = shield.select
    if disagreement_sink is not None:
        def _instrumented_select(joint_angles, candidates, scores):
            action, info = original_select(joint_angles, candidates, scores)
            disagreement_sink.extend(info["disagreement"])
            return action, info
        shield.select = _instrumented_select

    sequence_results = []
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(sequence_seed_base + idx, workers=True)
        attempts = run_calvin_shielded_sequence(
            env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
            get_env_state_for_initial_condition, shield, ep_len=cfg.ep_len, replan_steps=REPLAN_STEPS,
            obstacle_fn=obstacle_fn,
        )
        sequence_results.append(attempts)

    shield.select = original_select  # undo instrumentation before any further (video) use of this shield

    slots = build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5)
    violation_rate, success_rate = fixed_cohort_rates(slots)
    activation_rate = _shield_activation_rate(sequence_results)
    _log(
        f"[{label}] violation_rate={violation_rate:.3f}  success_rate={success_rate:.3f}  "
        f"shield_activation_rate={activation_rate:.3f}  (avg_seq_len={success_rate * 5:.2f}/5, "
        f"n_sequences={cfg.num_sequences})"
    )

    entry = {
        "label": label,
        "disagreement_threshold": threshold,
        "violation_rate": violation_rate,
        "success_rate": success_rate,
        "shield_activation_rate": activation_rate,
        "avg_seq_len": success_rate * 5,
        "n_sequences": cfg.num_sequences,
        "clearance_stats": None,
        "video_paths": None,
        "video_skip_reason": None,
    }

    if cfg.debug:
        clearance_stats = _clearance_stats(sequence_results)
        entry["clearance_stats"] = clearance_stats
        _log_clearance_debug(label, clearance_stats)
        vis_idxs = _rank_violating_sequence_idxs_by_length(sequence_results, cfg.num_videos)
        if not vis_idxs:
            reason = (
                "no sequence violated at this threshold -- shield may be filtering out the "
                "violating candidates entirely, or the threshold is loose enough nothing gets "
                "rejected; skipping video"
            )
            _log(f"  [debug] {label}: {reason}")
            entry["video_skip_reason"] = reason
        else:
            video_paths = []
            for vis_idx in vis_idxs:
                video_paths += _save_debug_videos(
                    vis_idx, threshold, obstacle_fn, shield, env, policy, task_oracle, lang_embeddings,
                    val_annotations, get_env_state_for_initial_condition, cfg, eval_sequences, sequence_seed_base,
                )
            entry["video_paths"] = video_paths

    return entry


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    global _LOG_FILE
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(RUN_OUTPUT_DIR / "run.log", "w", encoding="utf-8")
    _log(f"[run] writing log + results.json (+ videos/ if any) to: {RUN_OUTPUT_DIR}")
    _log(f"[run] cohort: {'TUNING (idx 0..N-1)' if TUNING_MODE else 'EVAL (idx N..2N-1)'} "
         f"-- pass --tuning to switch (see docs/TUNING_WORKFLOW.md muc 0)")
    _log(f"[run] config: num_sequences={cfg.num_sequences} ep_len={cfg.ep_len} replan_steps={REPLAN_STEPS} "
         f"sampler_type={cfg.sampler_type} num_sampling_steps={cfg.num_sampling_steps} debug={cfg.debug} "
         f"n_candidates={N_CANDIDATES} obstacle_radius={OBSTACLE_RADIUS}")

    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    # NOT model.multistep = cfg.multistep: see scripts/run_calvin_
    # unshielded.py's identical comment -- that attribute only gates
    # MDTVAgent.step()'s own internal replan counter, never called here.
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

    # Tuning cohort = idx 0..N-1, eval cohort = idx N..2N-1, disjoint --
    # identical mechanism to scripts/run_calvin_unshielded.py, see its
    # docstring / docs/TUNING_WORKFLOW.md muc 0.
    N = cfg.num_sequences
    COHORT_OFFSET = 0 if TUNING_MODE else N
    eval_sequences = get_sequences(2 * N)[COHORT_OFFSET:COHORT_OFFSET + N]
    SEQUENCE_SEED_BASE = 1000 + COHORT_OFFSET

    results = []

    if TUNING_MODE:
        # Phase 1: diagnostic -- threshold=inf, every candidate admissible,
        # select() still runs Select-by-g(a) among all K (a genuine
        # "select-only" row, not equivalent to unshielded). Disagreement
        # values from this same pass are logged as percentiles to inform
        # THRESHOLDS_TO_SWEEP. Fully deterministic given the same cohort/
        # checkpoint/K/radius/REPLAN_STEPS (seed_everything reseeds every
        # sequence identically) -- re-running it just reproduces the same
        # numbers, so RUN_DIAGNOSTIC skips it once its percentiles are
        # already known (see THRESHOLDS_TO_SWEEP's own comment for the
        # real numbers this run already produced on 2026-09-05: mean=
        # 0.368 median=0.349 p10=0.146 p90=0.609 p99=0.906). Flip back to
        # True only if K/radius/REPLAN_STEPS/checkpoint change again and
        # the percentiles need refreshing.
        if RUN_DIAGNOSTIC:
            label = "select-only (disagreement_threshold=inf)"
            disagreement_samples = []
            shield = ArmConfThreshShield(disagreement_threshold=float("inf"), replan_steps=REPLAN_STEPS)
            entry = _run_one_threshold(
                label, shield, float("inf"), env, policy, task_oracle, lang_embeddings, val_annotations,
                get_env_state_for_initial_condition, eval_sequences, cfg, SEQUENCE_SEED_BASE,
                disagreement_sink=disagreement_samples,
            )
            results.append(entry)
            _log_disagreement_debug(label, _disagreement_percentiles(disagreement_samples))

        # Phase 2: sweep -- pick tau* from these once real numbers are in.
        for threshold in THRESHOLDS_TO_SWEEP:
            shield = ArmConfThreshShield(disagreement_threshold=threshold, replan_steps=REPLAN_STEPS)
            entry = _run_one_threshold(
                f"disagreement_threshold={threshold}", shield, threshold, env, policy, task_oracle,
                lang_embeddings, val_annotations, get_env_state_for_initial_condition, eval_sequences, cfg,
                SEQUENCE_SEED_BASE,
            )
            results.append(entry)
    else:
        # Phase 3: final -- CHOSEN_THRESHOLD, held-out cohort, once.
        shield = ArmConfThreshShield(disagreement_threshold=CHOSEN_THRESHOLD, replan_steps=REPLAN_STEPS)
        entry = _run_one_threshold(
            f"FINAL disagreement_threshold={CHOSEN_THRESHOLD}", shield, CHOSEN_THRESHOLD, env, policy,
            task_oracle, lang_embeddings, val_annotations, get_env_state_for_initial_condition, eval_sequences,
            cfg, SEQUENCE_SEED_BASE,
        )
        results.append(entry)

    results_path = RUN_OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "tuning_mode": TUNING_MODE,
            "cohort_sequence_idx_range": [COHORT_OFFSET, COHORT_OFFSET + N],
            "n_candidates": N_CANDIDATES,
            "obstacle_radius": OBSTACLE_RADIUS,
            "thresholds_to_sweep": THRESHOLDS_TO_SWEEP if TUNING_MODE else None,
            "chosen_threshold": None if TUNING_MODE else CHOSEN_THRESHOLD,
            "results": results,
        }, f, indent=2)
    _log(f"[run] wrote structured results to: {results_path}")
    _log(f"[run] DONE -- zip up {RUN_OUTPUT_DIR} and send it back for tuning analysis")
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
