"""Shielded baseline (Conf-Thresh): shortstop.arm_shield.ArmConfThreshShield
wired into the CALVIN eval loop -- Propose K candidates -> reject on
sampler-ensemble disagreement (docs/main (3).txt Sec. V.D) -> Select by
shortstop.calvin_progress's g(a) (a kinematic goal-distance proxy, see
that module's docstring). The privileged obstacle (radius=0.08,
offset_max=0.6, already tuned -- see docs/PARAMETERS_REFERENCE.md) is
still injected exactly as in scripts/run_calvin_unshielded.py, purely for
ground-truth measurement
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

# Obstacle radius/offset: already tuned at this same REPLAN_STEPS=10, see
# docs/PARAMETERS_REFERENCE.md / scripts/run_calvin_unshielded.py's own
# RADII_TO_SWEEP/OBSTACLE_OFFSET_MAX comments. Re-tune there (not here) if
# the checkpoint/dataset or REPLAN_STEPS ever change again. Both must be
# passed explicitly (not left to sample_obstacle_from_reference_chunk's
# own defaults) -- found 2026-09-06 that this script was silently using a
# stale offset_max=0.3 (the function's old default) while
# run_calvin_unshielded.py had already moved its own sweep to 0.6, which
# would have made this baseline's obstacle placement meaningfully harder
# than the unshielded run it's supposed to be compared against.
OBSTACLE_RADIUS = 0.08
OBSTACLE_OFFSET_MAX = 0.6

# Candidate disagreement_threshold values to sweep (meters, same units as
# the endpoint-vs-centroid distance ArmConfThreshShield computes).
#
# STALE (2026-09-06) -- everything below (the percentiles, the 4-point
# list, CHOSEN_THRESHOLD and its eval-cohort confirmation) was measured
# BEFORE the CALVIN_ACTION_SCALE fix (see arm_reach.py / docs/PARAMETERS_
# REFERENCE.md's "model_error" entry) -- `ArmConfThreshShield._endpoint()`
# calls `propagate_arm_tube`, the EXACT function that fix touched, so the
# old percentiles (mean=0.368 median=0.349 p10=0.146 p90=0.609 p99=0.906
# min=0.003 max=1.387, n=17856) were measured on a candidate-endpoint
# spread ~50x too large (the same ratio CALVIN_ACTION_SCALE=0.02
# corrects). Also predates the g(a) base-frame fix, the obstacle self-
# collision fix, and the offset_max fairness fix -- none of those touch
# `_endpoint()`'s own geometry, but they do change what episode state
# each decision sees, so a from-scratch re-sweep is the only safe option
# here, not a reuse of the old list.
#
# Naive rescaling predicts the new distribution lands around
# old_value/50 -- roughly p10~0.003, median~0.007, p90~0.012, p99~0.018,
# max~0.028. That's an ESTIMATE (this ratio assumes the ONLY thing that
# changed is CALVIN_ACTION_SCALE, which is true for this specific
# computation, but hasn't been measured for real yet). To avoid the same
# "guessed range was entirely off the real distribution, needed a second
# --tuning run after seeing the diagnostic" failure mode from before
# (also hit while tuning STL-Monitor/epsilon) in a single pass this time,
# this list DELIBERATELY spans both the predicted new low range AND the
# old high range as a hedge -- whichever regime turns out real, at least
# several points should land inside the actual distribution on this
# FIRST run. RUN_DIAGNOSTIC=True below logs the real percentiles from
# THIS run regardless, to confirm/refute the /50 estimate directly.
THRESHOLDS_TO_SWEEP = [0.003, 0.007, 0.012, 0.02, 0.03, 0.15, 0.35, 0.9]

# Re-enabled 2026-09-06 (was False) -- the old percentiles this flag's
# "already logged, no need to reproduce" reasoning relied on are exactly
# the ones now stale (see THRESHOLDS_TO_SWEEP's own comment). Needs a
# real diagnostic pass again to get trustworthy percentiles under the
# current, fully-fixed pipeline. Flip back to False once this run's real
# percentiles are captured in a comment here, same pattern as before.
RUN_DIAGNOSTIC = True

# PLACEHOLDER (2026-09-06) -- the previously-chosen 0.9 and its eval-
# cohort confirmation (violation=0.140/success=0.466) are STALE for the
# same reasons as THRESHOLDS_TO_SWEEP above; kept only as the prior
# baseline's value until a fresh --tuning sweep (this widened list) picks
# a real one. Only read in eval mode (no --tuning flag) -- do NOT run eval
# with this value before re-tuning on the widened sweep above.
CHOSEN_THRESHOLD = 0.9

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


def _latency_stats(sequence_results):
    """mean/median/p95 latency_ms over every decision across every
    attempted subtask -- mirrors shortstop.metrics.aggregate's own
    latency_ms_mean/median/p95, computed here at CALVIN granularity from
    run_calvin_shielded_subtask's new latencies_ms field."""
    all_latencies = [t for attempts in sequence_results for a in attempts for t in a["latencies_ms"]]
    if not all_latencies:
        return None
    values = np.asarray(all_latencies)
    return {
        "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _intervention_precision(sequence_results):
    """sum(rejected_truly_unsafe)/sum(rejected_total) pooled across every
    decision -- the paper's own intervention_precision metric (Table II),
    not previously computed for any CALVIN baseline. None if nothing was
    ever rejected (nothing to compute a precision over)."""
    total_rejected = sum(a["rejected_total"] for attempts in sequence_results for a in attempts)
    total_truly_unsafe = sum(a["rejected_truly_unsafe"] for attempts in sequence_results for a in attempts)
    if total_rejected == 0:
        return None
    return total_truly_unsafe / total_rejected


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
        # Built fresh per sequence, seeded by sequence_seed_base+idx -- an
        # independent numpy Generator (does NOT touch the global numpy/
        # torch RNG state seed_everything() just reseeded above), so the
        # SAME random obstacle offset is reused across every baseline
        # comparison run at this same sequence idx, while never desyncing
        # the policy's own diffusion noise draws. See sample_obstacle_
        # from_reference_chunk's own docstring (2026-09-05 fix) for why
        # this must be a caller-seeded, non-global RNG.
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

    shield.select = original_select  # undo instrumentation before any further (video) use of this shield

    slots = build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5)
    violation_rate, success_rate = fixed_cohort_rates(slots)
    activation_rate = _shield_activation_rate(sequence_results)
    latency_stats = _latency_stats(sequence_results)
    intervention_precision = _intervention_precision(sequence_results)
    _log(
        f"[{label}] violation_rate={violation_rate:.3f}  success_rate={success_rate:.3f}  "
        f"shield_activation_rate={activation_rate:.3f}  (avg_seq_len={success_rate * 5:.2f}/5, "
        f"n_sequences={cfg.num_sequences})"
    )
    precision_str = "n/a" if intervention_precision is None else f"{intervention_precision:.3f}"
    if latency_stats is not None:
        _log(
            f"  latency_ms_mean={latency_stats['mean']:.3f}  latency_ms_median={latency_stats['median']:.3f}  "
            f"latency_ms_p95={latency_stats['p95']:.3f}  intervention_precision={precision_str}"
        )
    else:
        _log("  (no decisions recorded any latency)")

    entry = {
        "label": label,
        "disagreement_threshold": threshold,
        "violation_rate": violation_rate,
        "success_rate": success_rate,
        "shield_activation_rate": activation_rate,
        "avg_seq_len": success_rate * 5,
        "n_sequences": cfg.num_sequences,
        "latency_ms": latency_stats,
        "intervention_precision": intervention_precision,
        "clearance_stats": None,
        "video_paths": None,
        "video_skip_reason": None,
        # Full per-sequence, per-subtask-attempt records -- not just the
        # aggregates above -- so conservatism_cost/recovery_rate (and
        # anything else) can be recomputed later without a re-run (see
        # docs/PARAMETERS_REFERENCE.md's metrics-gap note: the earlier lack
        # of this is exactly why those 2 metrics couldn't be backfilled
        # for Conf-Thresh's already-finalized eval run).
        "sequence_results": sequence_results,
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
                # Rebuilt fresh here (not reusing the main loop's own
                # obstacle_fn, whose rng belongs to -- and has already
                # been consumed by -- the LAST sequence of the main pass)
                # -- same seed=sequence_seed_base+vis_idx as that
                # sequence originally got, so the video reproduces the
                # EXACT SAME obstacle placement actually measured, not a
                # different random draw.
                video_obstacle_rng = np.random.default_rng(sequence_seed_base + vis_idx)
                video_obstacle_fn = lambda joint_angles, chunk, rng=video_obstacle_rng: sample_obstacle_from_reference_chunk(  # noqa: E731,E501
                    joint_angles, chunk, radius=OBSTACLE_RADIUS, rng=rng, offset_max=OBSTACLE_OFFSET_MAX,
                )
                video_paths += _save_debug_videos(
                    vis_idx, threshold, video_obstacle_fn, shield, env, policy, task_oracle, lang_embeddings,
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
         f"n_candidates={N_CANDIDATES} obstacle_radius={OBSTACLE_RADIUS} "
         f"obstacle_offset_max={OBSTACLE_OFFSET_MAX}")

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

    results_path = RUN_OUTPUT_DIR / "results.json"

    def _write_progress(results):
        # Re-written after EVERY sweep entry (diagnostic + each threshold),
        # not just once at the end -- a tuning sweep is slow enough (each
        # entry = a full cohort rollout) that losing everything to a
        # crash/interrupt partway through, or having to wait for the whole
        # sweep before seeing any number, is real pain.
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
            _write_progress(results)
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
            _write_progress(results)
            _log(f"  [progress] wrote {len(results)}/{len(THRESHOLDS_TO_SWEEP) + int(RUN_DIAGNOSTIC)} "
                 f"entry(ies) so far to: {results_path}")
    else:
        # Phase 3: final -- CHOSEN_THRESHOLD, held-out cohort, once.
        shield = ArmConfThreshShield(disagreement_threshold=CHOSEN_THRESHOLD, replan_steps=REPLAN_STEPS)
        entry = _run_one_threshold(
            f"FINAL disagreement_threshold={CHOSEN_THRESHOLD}", shield, CHOSEN_THRESHOLD, env, policy,
            task_oracle, lang_embeddings, val_annotations, get_env_state_for_initial_condition, eval_sequences,
            cfg, SEQUENCE_SEED_BASE,
        )
        results.append(entry)
        _write_progress(results)

    _log(f"[run] wrote structured results to: {results_path}")
    _log(f"[run] DONE -- zip up {RUN_OUTPUT_DIR} and send it back for tuning analysis")
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
