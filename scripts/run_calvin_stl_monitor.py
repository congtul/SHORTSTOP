"""STL-Monitor baseline: shortstop.arm_shield.ArmSTLMonitorShield wired
into the CALVIN eval loop -- Propose K candidates -> reject any whose
nominal STL robustness (no reachtube, no repair) is below `epsilon` ->
Select by shortstop.calvin_progress's g(a). Per the real paper text
(docs/main (3).txt Sec. V.D/IV): "STL-Monitor, which computes nominal STL
robustness on the fˆ rollout and rejects if negative (no reachtube, no
counterexample search)".

Unlike Conf-Thresh, this shield DOES see the privileged obstacle (it
must, to certify against it) -- see ArmSTLMonitorShield's own docstring
and shortstop.calvin_experiment.run_calvin_shielded_subtask's note on
obstacle-aware shields (the obstacle is wired into `shield.obstacles`
before that subtask's first decision, reusing the same obstacle every
other baseline places from candidates[0]).

Filter freq is also decoupled from policy freq here, automatically --
ArmSTLMonitorShield inherits ArmReachOnlyShield.recertify(), so the
harness re-checks this shield's already-selected chunk against the REAL
state after every executed row (not just every REPLAN_STEPS), abandoning
the rest of it and re-proposing early on failure (see
run_calvin_shielded_subtask's own docstring and docs/
PARAMETERS_REFERENCE.md's "tach tan suat filter khoi policy" entry).
Nothing in this script needs to opt into that -- it falls out of reusing
ArmSTLMonitorShield as-is.

`epsilon` IS tuned here, unlike an earlier revision of this script --
the paper's own text gives two conflicting readings of it (see
ArmSTLMonitorShield's docstring in shortstop/arm_shield.py): "rejects if
negative" (epsilon=0.0) right next to "All model-based baselines use the
identical f-hat, epsilon and fallback for a fair comparison" (epsilon =
ShortStop's own calibrated margin, 0.02 in this codebase). Rather than
guessing which the paper means, this is resolved the same way Conf-
Thresh's disagreement_threshold was: run --tuning, look at the real
violation/success/activation trade-off across a spread of epsilon values
anchored at both readings, pick tau* on the tuning cohort, confirm once
on the held-out eval cohort.

Run from WSL2, inside `mdt_env` (see docs/CALVIN_SETUP.md):

    cd SHORTSTOP
    python scripts/run_calvin_stl_monitor.py            # eval cohort (default): runs CHOSEN_EPSILON once
    python scripts/run_calvin_stl_monitor.py --tuning    # tuning cohort: diagnostic + EPSILONS_TO_SWEEP

Tuning/eval cohort split: identical mechanism to
scripts/run_calvin_shielded.py's own `--tuning` flag (see its docstring
and docs/TUNING_WORKFLOW.md muc 0).

Three phases:
  --tuning:
    1. Diagnostic (epsilon=-inf): every candidate passes the filter, but
       select() still runs Select-by-g(a) among all K -- a genuine
       "select-only" comparison row, reported alongside the sweep. Also
       logs percentile stats of every nominal-robustness value seen
       (collected via a thin instrumentation wrapper around this one
       pass), to confirm/refine EPSILONS_TO_SWEEP below.
    2. Sweep (EPSILONS_TO_SWEEP): pick epsilon* from the violation/
       success/shield-activation trade-off once real numbers are in --
       this list starts anchored at the paper's two readings (0.0 and
       0.02) plus two more spread points, same placeholder-then-revise
       pattern THRESHOLDS_TO_SWEEP went through.
  (no flag, default) eval:
    3. Final (CHOSEN_EPSILON, run once on the disjoint held-out cohort)
       -- the number that actually gets reported.
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

from shortstop.arm_reach import arm_robustness_to_go, propagate_arm_tube  # noqa: E402
from shortstop.arm_shield import ArmSTLMonitorShield  # noqa: E402
from shortstop.calvin_experiment import run_calvin_shielded_sequence  # noqa: E402
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates  # noqa: E402
from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk  # noqa: E402
from shortstop.calvin_obstacle_viz import save_sequence_video  # noqa: E402
from shortstop.mdt_policy_client import ForwardOnlyPolicy  # noqa: E402

# Shared with scripts/run_calvin_shielded.py -- every baseline in the same
# comparison table must use the same K/replan cadence/obstacle radius
# (paper's own "All shields wrap the same frozen checkpoint" + one shared
# Table VII hyperparameter set, see docs/PARAMETERS_REFERENCE.md). See
# that script's own comments for the full reasoning/history behind each
# of these values -- copy-pasted rather than shared, same as its own
# relationship to run_calvin_unshielded.py.
N_CANDIDATES = 8
REPLAN_STEPS = 10
OBSTACLE_RADIUS = 0.08

# Candidate epsilon values to sweep (meters, same units as
# arm_robustness_to_go -- a signed distance) -- PLACEHOLDER, not yet
# revised from a real diagnostic pass (see run_calvin_shielded.py's own
# THRESHOLDS_TO_SWEEP for the pattern this will follow once real
# percentiles are in). Anchored at the paper's two conflicting readings:
# 0.0 ("rejects if negative", literal) and 0.02 (ShortStop's own
# calibrated margin -- "All model-based baselines use the identical
# f-hat, epsilon and fallback", shared reading), plus two more spread
# points to see the trend between/around them.
EPSILONS_TO_SWEEP = [0.0, 0.02, 0.05, 0.1]

# Run Phase 1 (epsilon=-inf diagnostic) -- True until a real run has
# logged the nominal-robustness percentiles at least once (mirrors
# run_calvin_shielded.py's RUN_DIAGNOSTIC, which flipped to False only
# after its own real numbers were captured in a comment).
RUN_DIAGNOSTIC = True

# Final, chosen epsilon -- PLACEHOLDER (literal "rejects if negative"
# reading) until a real --tuning sweep picks epsilon* from the actual
# violation/success/activation trade-off. Only read in eval mode (no
# --tuning flag).
CHOSEN_EPSILON = 0.0

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "calvin_stl_monitor_runs" / (
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
    """Identical to scripts/run_calvin_shielded.py's own helper."""
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
    """Identical to scripts/run_calvin_shielded.py's own helper -- pooled
    ratio over every decision, not a mean of per-subtask rates."""
    n_decisions = sum(a["n_decisions"] for attempts in sequence_results for a in attempts)
    n_activated = sum(a["n_activated"] for attempts in sequence_results for a in attempts)
    if n_decisions == 0:
        return None
    return n_activated / n_decisions


def _robustness_percentiles(robustness_samples):
    if not robustness_samples:
        return None
    values = np.asarray(robustness_samples)
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


def _log_robustness_debug(label, stats):
    if stats is None:
        _log(f"  [debug] {label}: no decisions recorded any nominal-robustness value")
        return
    _log(
        f"  [debug] {label}: nominal robustness over {stats['n']} candidates -- "
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
    sequence_idx, epsilon, obstacle_fn, shield, env, policy, task_oracle, lang_embeddings, val_annotations,
    get_env_state_for_initial_condition, cfg, eval_sequences, sequence_seed_base,
):
    """Re-runs sequence `sequence_idx` alone (same per-sequence seed the
    main pass already used for it) with recording on, merges every
    subtask attempt into ONE mp4."""
    initial_state, eval_sequence = eval_sequences[sequence_idx]
    seed_everything(sequence_seed_base + sequence_idx, workers=True)
    attempts = run_calvin_shielded_sequence(
        env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
        get_env_state_for_initial_condition, shield, ep_len=cfg.ep_len, replan_steps=REPLAN_STEPS,
        obstacle_fn=obstacle_fn, record_trajectory=True, record_camera_frames=True,
    )
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_epsilon = str(epsilon).replace(".", "p").replace("-", "neg")

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
    out_path = VIS_OUTPUT_DIR / f"seq{sequence_idx}_e{safe_epsilon}.mp4"
    save_sequence_video(
        subtask_records, static_camera, str(out_path),
        base_position=env.env.robot.base_position, base_orientation=env.env.robot.base_orientation,
    )
    return [str(out_path.relative_to(RUN_OUTPUT_DIR))]


def _run_one_epsilon(
    label, shield, epsilon, env, policy, task_oracle, lang_embeddings, val_annotations,
    get_env_state_for_initial_condition, eval_sequences, cfg, sequence_seed_base,
    robustness_sink=None,
):
    """Runs `shield` over every sequence in `eval_sequences`, returns the
    results.json-style entry for it. `robustness_sink`, if given, is
    extended in-place with every candidate's nominal-robustness value
    during this same pass -- done by swapping in an instrumented
    `_admissible` that computes the same tube/robustness the real one
    would (no duplicate rollout), undone before any later (video) use of
    `shield`.
    """
    obstacle_fn = lambda joint_angles, chunk: sample_obstacle_from_reference_chunk(  # noqa: E731
        joint_angles, chunk, radius=OBSTACLE_RADIUS,
    )

    original_admissible = shield._admissible
    if robustness_sink is not None:
        def _instrumented_admissible(joint_angles, task_chunk):
            tube = propagate_arm_tube(joint_angles, task_chunk, shield.w_bar, shield.model_error)
            robustness = arm_robustness_to_go(tube, shield.obstacles)
            robustness_sink.append(robustness)
            return robustness >= shield.epsilon
        shield._admissible = _instrumented_admissible

    sequence_results = []
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(sequence_seed_base + idx, workers=True)
        attempts = run_calvin_shielded_sequence(
            env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
            get_env_state_for_initial_condition, shield, ep_len=cfg.ep_len, replan_steps=REPLAN_STEPS,
            obstacle_fn=obstacle_fn,
        )
        sequence_results.append(attempts)

    shield._admissible = original_admissible  # undo instrumentation before any further (video) use of this shield

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
        "epsilon": epsilon,
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
                "no sequence violated at this epsilon -- STL-Monitor may be filtering out the "
                "violating candidates entirely, or epsilon is loose enough nothing gets rejected; "
                "skipping video"
            )
            _log(f"  [debug] {label}: {reason}")
            entry["video_skip_reason"] = reason
        else:
            video_paths = []
            for vis_idx in vis_idxs:
                video_paths += _save_debug_videos(
                    vis_idx, epsilon, obstacle_fn, shield, env, policy, task_oracle, lang_embeddings,
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

    # obstacles=[] is a placeholder: run_calvin_shielded_subtask overwrites
    # shield.obstacles from candidates[0] before this shield's very first
    # decision of each subtask (see its own docstring) -- never read as [].
    results = []

    if TUNING_MODE:
        # Phase 1: diagnostic -- epsilon=-inf, every candidate admissible,
        # select() still runs Select-by-g(a) among all K (a genuine
        # "select-only" row, not equivalent to unshielded). Nominal-
        # robustness values from this same pass are logged as percentiles
        # to confirm/refine EPSILONS_TO_SWEEP.
        if RUN_DIAGNOSTIC:
            label = "select-only (epsilon=-inf)"
            robustness_samples = []
            shield = ArmSTLMonitorShield(obstacles=[], epsilon=float("-inf"))
            entry = _run_one_epsilon(
                label, shield, float("-inf"), env, policy, task_oracle, lang_embeddings, val_annotations,
                get_env_state_for_initial_condition, eval_sequences, cfg, SEQUENCE_SEED_BASE,
                robustness_sink=robustness_samples,
            )
            results.append(entry)
            _log_robustness_debug(label, _robustness_percentiles(robustness_samples))

        # Phase 2: sweep -- pick epsilon* from these once real numbers are in.
        for epsilon in EPSILONS_TO_SWEEP:
            shield = ArmSTLMonitorShield(obstacles=[], epsilon=epsilon)
            entry = _run_one_epsilon(
                f"epsilon={epsilon}", shield, epsilon, env, policy, task_oracle,
                lang_embeddings, val_annotations, get_env_state_for_initial_condition, eval_sequences, cfg,
                SEQUENCE_SEED_BASE,
            )
            results.append(entry)
    else:
        # Phase 3: final -- CHOSEN_EPSILON, held-out cohort, once.
        shield = ArmSTLMonitorShield(obstacles=[], epsilon=CHOSEN_EPSILON)
        entry = _run_one_epsilon(
            f"FINAL epsilon={CHOSEN_EPSILON}", shield, CHOSEN_EPSILON, env, policy,
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
            "epsilons_to_sweep": EPSILONS_TO_SWEEP if TUNING_MODE else None,
            "chosen_epsilon": None if TUNING_MODE else CHOSEN_EPSILON,
            "results": results,
        }, f, indent=2)
    _log(f"[run] wrote structured results to: {results_path}")
    _log(f"[run] DONE -- zip up {RUN_OUTPUT_DIR} and send it back for tuning analysis")
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
