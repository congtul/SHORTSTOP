"""Unshielded baseline: violation_rate/success_rate with vs. without the
privileged obstacle, over `cfg.num_sequences` CALVIN sequences -- swept
over a handful of candidate obstacle radii (see RADII_TO_SWEEP below),
since the radius itself is a free parameter with no paper reference (it
is a synthetic X_u we invented for CALVIN, not something the paper
specifies): too small and the obstacle is almost never hit (floor effect,
nothing for a shield to demonstrate improvement on); too large and almost
every attempt hits it (ceiling effect). See docs/PARAMETERS_REFERENCE.md
muc 1's "radius" entry for the full reasoning.

Run from WSL2, inside the `mdt_env` conda environment set up per
docs/CALVIN_SETUP.md (needs a real GPU + the mdt_policy checkpoint +
debug dataset -- NOT runnable/tested in the dev sandbox this was written
in; treat this as a carefully-reasoned first draft to debug against the
real checkout, not a guaranteed-working script):

    cd SHORTSTOP
    python scripts/run_calvin_unshielded.py            # eval cohort (default)
    python scripts/run_calvin_unshielded.py --tuning    # tuning cohort

Tuning/eval cohort split (see docs/TUNING_WORKFLOW.md muc 0 -- applies to
every swept parameter in this repo, not just radius): sequence idx
0..N-1 is the TUNING cohort (pick RADII_TO_SWEEP's final value here),
idx N..2N-1 is a disjoint EVAL cohort, meant to be run exactly once with
that already-chosen value for the number that actually gets
reported -- never re-picked based on the eval run's own numbers. Default
(no flag) is eval, since RADII_TO_SWEEP is normally already narrowed to
the one chosen value by the time this script gets run; pass `--tuning`
only while still sweeping candidate values.

Reuses mdt_evaluate.py's own model/env setup (`get_default_beso_and_env`
+ the same sampler/EMA overrides `main()` applies) rather than
shortstop.mdt_policy_client.MDTPolicyClient -- that class only calls
`MDTVAgent.load_from_checkpoint()` directly and is missing the
sampler_type/num_sampling_steps/sigma/EMA-weight overrides the real eval
script applies afterwards; use it only for structural/mocked testing
(tests/test_mdt_policy_client.py), not for a real run.

When `cfg.debug` (patched default: True -- see patches/mdt_policy_shortstop.patch;
flip to False only when told to prep for release):
  - prints (and logs, see below) min_clearance stats (mean/median/p10/
    p90/min over every attempted subtask) per radius -- a finer-grained
    signal than the binary violated/not-violated rate for judging whether
    a radius is in a sane range before committing to it.
  - re-runs the first sequence that actually violated at each radius
    (same per-sequence seed the main sweep already used for it, so it's
    the exact attempt those metrics/stats already reflect) with
    trajectory recording on, and merges every subtask attempt of that
    sequence into ONE mp4 -- matching how the real CALVIN eval pipeline
    records one continuous video per sequence, not a separate file per
    subtask -- see shortstop/calvin_obstacle_viz.py.

Everything printed is *also* written to a run-specific output directory
(printed at the very start and end so it's easy to find) meant to be
zipped up and handed back for analysis on a machine without this
env/checkpoint/dataset:
  outputs/calvin_unshielded_runs/run_<timestamp>/
    run.log       -- exact copy of everything printed to stdout
    results.json  -- same numbers, structured (radius -> violation_rate,
                      success_rate, clearance stats, which sequence/subtask
                      each video came from) -- read this one programmatically
    videos/*.mp4  -- from the debug obstacle visualization above
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import calvin_env
import hydra
import numpy as np
from pytorch_lightning import seed_everything

# Popped out of sys.argv here, before hydra.main() ever parses it (hydra's
# own CLI only understands `key=value` overrides and a handful of reserved
# `--` flags -- an unrecognized `--tuning` would otherwise make hydra error
# out). See module docstring's "Tuning/eval cohort split" for what this
# controls.
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

from shortstop.calvin_experiment import run_calvin_unshielded_sequence  # noqa: E402
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates  # noqa: E402
from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk  # noqa: E402
from shortstop.calvin_obstacle_viz import save_sequence_video  # noqa: E402
from shortstop.mdt_policy_client import ForwardOnlyPolicy  # noqa: E402

# Replan cadence: how many rows of each proposed (act_window_size=10)
# chunk actually get executed before propose() is called again -- a pure
# harness-level slicing choice (chunk[:REPLAN_STEPS] inside
# run_calvin_unshielded_sequence), NOT a model/checkpoint property, so
# changing it needs no retraining (see docs/PARAMETERS_REFERENCE.md's
# "multistep / replan_steps" entry). NOT `cfg.multistep` (that hydra knob
# only feeds MDTVAgent.step()'s own internal counter, which this harness
# never calls; see ForwardOnlyPolicy's docstring on calling forward()
# directly instead of step()).
#
# Tried =5 (ratio 0.5 against act_window_size=10, matching Diffusion
# Policy's Ta=8/Tp=16 execution/prediction ratio) on 2026-09-04, reverted
# same day: real sweep on this checkpoint showed the "without obstacle"
# baseline success_rate drop from 0.930 (REPLAN_STEPS=10) to 0.834
# (REPLAN_STEPS=5) -- a ~10pp cost from more frequent replanning alone,
# before any obstacle. Each propose() call draws fresh independent noise
# (no continuity with the previous chunk), so replanning more often means
# more "seams" between independently-sampled chunks -- exactly why
# Diffusion Policy's own ablation didn't pick the smallest possible Ta
# either. That 0.5 ratio doesn't transfer to this checkpoint, and nothing
# in this project currently benefits from more frequent replanning
# (Conf-Thresh's filter frequency is tied to policy frequency either way,
# and no baseline with a decouplable certify step exists yet -- see
# docs/PARAMETERS_REFERENCE.md's decoupling-feasibility table) enough to
# justify paying that cost. Back to =10 (ratio 1.0).
REPLAN_STEPS = 10

# Candidate obstacle radii to sweep (meters) -- edit this list directly
# while tuning; no config knob for it yet since we're still narrowing the
# range by hand, see module docstring.
#
# STALE (2026-09-05) -- the [0.02, 0.05, 0.08, 0.12] -> r=0.08 sweep below
# was invalidated by TWO independent, since-fixed bugs: (1) CALVIN_ACTION_
# SCALE (see arm_reach.py) -- sample_obstacle_from_reference_chunk's own
# propagate_arm_tube call placed the obstacle ~50x farther along the
# chunk's direction than the arm's real per-window reach, so every radius
# below was tested against an unrealistically-distant placement, not the
# "obstacle at this window's real destination" the design intends; (2)
# GRIPPER_TIP_OFFSET (see robot_geometry.py) -- the ground-truth capsule
# check near the gripper used a primitive radius 0.04m too small, so
# violation_rate below under-counted collisions in the outer 4cm of the
# fingertip's real reach. Both changed the actual difficulty at any given
# radius value, in an a-priori unclear direction (placement is now much
# closer/more "real", the gripper-region primitive is now bigger) --
# re-sweep from scratch, don't just re-run r=0.08.
#
# NEW SWEEP (2026-09-05, pending a real run): widened AND shifted down
# from the old range, since GRIPPER_TIP_OFFSET's fix alone means even
# radius=0 now carries a real ~0.20m margin near the gripper
# (GRIPPER_TIP_OFFSET+GRIPPER_TIP_RADIUS = 0.14+0.06) where the old sweep
# had 0.16 -- worth re-checking for a floor/ceiling effect at both ends
# rather than assuming last time's "no floor at 0.02, no ceiling at 0.12"
# still holds now that placement is realistic instead of ~50x overshot.
# Includes 0.0 (a point obstacle) as a new baseline to isolate how much
# violation_rate the arm's own capsule geometry alone now produces.
# cfg.debug's steps_taken diagnostic (_violated_steps_taken_percentiles,
# added after the LAST real sweep and never yet checked against real
# numbers) is worth reading closely this run -- it directly answers "does
# the arm get a real chance to move before this obstacle is hit", the
# exact concern that motivated re-checking this list at all.
RADII_TO_SWEEP = [0.0, 0.02, 0.04, 0.08, 0.12, 0.16]

# OFF for this re-sweep (2026-09-05): confirmed neither the action-scale
# nor the GRIPPER_TIP_OFFSET bug can touch the "without obstacle" number
# at all -- _clearance(obs, obstacle) returns None immediately when
# obstacle is None (calvin_experiment.py), before touching any of the
# geometry either bug lives in, so `violated`/`min_clearance` never get
# computed and success_rate depends only on real env.step()/task_oracle,
# neither of which we ever modified. The last real run's number (tuning
# cohort: success_rate=0.930, avg_seq_len=4.65/5, n=100 -- see
# RADII_TO_SWEEP's own comment above) is still fully valid; no need to
# burn compute re-running it just because the radius list changed.
#
# General rule (also applies when resuming a killed/interrupted sweep):
# trim RADII_TO_SWEEP above to just the radii not yet completed (check
# the previous run's run.log) and leave this False -- "without obstacle"
# never depends on which radii are being swept. Set back to True only
# for a genuine from-scratch run (new checkpoint/dataset/REPLAN_STEPS).
INCLUDE_WITHOUT_OBSTACLE = False

_COHORT_TAG = "tuning" if TUNING_MODE else "eval"
RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "calvin_unshielded_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}_{_COHORT_TAG}"
VIS_OUTPUT_DIR = RUN_OUTPUT_DIR / "videos"

_LOG_FILE = None  # opened at the top of main(), see _log()


def _log(msg):
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def _clearance_stats(sequence_results):
    """Same numbers _print_clearance_debug prints, as a plain dict for
    results.json -- None if no attempt in this sweep had an obstacle to
    measure clearance against at all."""
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


def _violated_steps_taken_percentiles(sequence_results):
    """steps_taken (see run_calvin_unshielded_subtask's own docstring)
    over VIOLATED attempts only -- answers "did the arm get a real
    chance to move before hitting this obstacle": a small value means the
    obstacle's own capture radius (obstacle.radius + whichever capsule
    primitive is closest) is large relative to how far this chunk's
    motion covers per replan window, not that the obstacle was placed
    somewhere stale/wrong (obstacle_fn always samples from THIS subtask's
    own real, current joint_angles/first candidate -- see
    run_calvin_unshielded_subtask). None if nothing violated in this
    sweep."""
    steps = [a["steps_taken"] for attempts in sequence_results for a in attempts if a["violated"]]
    if not steps:
        return None
    values = np.asarray(steps)
    return {
        "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
        "min": int(values.min()), "max": int(values.max()),
    }


def _log_violated_steps_taken_debug(label, stats):
    if stats is None:
        _log(f"  [debug] {label}: nothing violated in this sweep")
        return
    _log(
        f"  [debug] {label}: steps_taken over {stats['n']} VIOLATED attempts (does the arm get a "
        f"real chance to move first?) -- mean={stats['mean']:.2f}  median={stats['median']:.2f}  "
        f"p10={stats['p10']:.2f}  p90={stats['p90']:.2f}  min={stats['min']}  max={stats['max']}"
    )


def _rank_violating_sequence_idxs_by_length(sequence_results, top_k):
    """Sequence indices with at least one violated attempt, longest-
    running (most subtasks attempted before stopping) first -- picks
    videos for _save_debug_videos that are actually informative: a
    sequence that reaches several subtasks before the obstacle finally
    stops it shows far more than one that violates on subtask 1, which
    is otherwise just luck of which sequence a fixed idx happened to be.
    Returns up to `top_k` indices, fewer if fewer than `top_k` sequences
    violated at all at this radius -- no fallback to non-violating
    sequences pads the list out, since the whole point is showing the
    obstacle actually doing something."""
    violating = [idx for idx, attempts in enumerate(sequence_results) if any(a["violated"] for a in attempts)]
    violating.sort(key=lambda idx: len(sequence_results[idx]), reverse=True)
    return violating[:top_k]


def _save_debug_videos(
    sequence_idx, radius, obstacle_fn, env, policy, task_oracle, lang_embeddings, val_annotations,
    get_env_state_for_initial_condition, cfg, eval_sequences, sequence_seed_base,
):
    """Re-runs sequence `sequence_idx` alone (same per-sequence seed the
    main sweep already used for it -- reseeding is per-idx for *every*
    sequence in that sweep, not just this one, so this reproduces exactly
    the attempt that already contributed to the printed metrics/stats,
    whichever idx is picked) with trajectory recording on, and merges
    every subtask attempt of that sequence into ONE mp4 under
    VIS_OUTPUT_DIR -- matching how the real CALVIN eval pipeline records
    one continuous video per sequence, not a separate file per subtask.
    Returns the (1-element) list of video path(s) written, kept as a
    list for results.json shape compatibility."""
    initial_state, eval_sequence = eval_sequences[sequence_idx]
    seed_everything(sequence_seed_base + sequence_idx, workers=True)
    attempts = run_calvin_unshielded_sequence(
        env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
        get_env_state_for_initial_condition, ep_len=cfg.ep_len, replan_steps=REPLAN_STEPS,
        obstacle_fn=obstacle_fn, record_trajectory=True, record_camera_frames=True,
    )
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_radius = str(radius).replace(".", "p")

    subtask_records = []
    for subtask, attempt in zip(eval_sequence, attempts):
        if attempt["violated"]:
            outcome = "violated"
        elif attempt["reached"]:
            outcome = "reached"
        else:
            outcome = "failed"  # ran out of ep_len without reaching or violating
        subtask_records.append({
            "subtask": subtask,
            "frames": attempt["camera_frames"],
            "depth_frames": attempt["depth_frames"],
            "obstacle": attempt["obstacle"],
            "outcome": outcome,
        })

    static_camera = next(cam for cam in env.env.cameras if cam.name == "static")
    out_path = VIS_OUTPUT_DIR / f"seq{sequence_idx}_r{safe_radius}.mp4"
    save_sequence_video(
        subtask_records, static_camera, str(out_path),
        base_position=env.env.robot.base_position, base_orientation=env.env.robot.base_orientation,
    )
    return [str(out_path.relative_to(RUN_OUTPUT_DIR))]


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    global _LOG_FILE
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(RUN_OUTPUT_DIR / "run.log", "w", encoding="utf-8")
    _log(f"[run] writing log + results.json (+ videos/ if any) to: {RUN_OUTPUT_DIR}")
    _log(f"[run] cohort: {'TUNING (idx 0..N-1)' if TUNING_MODE else 'EVAL (idx N..2N-1)'} "
         f"-- pass --tuning to switch (see docs/TUNING_WORKFLOW.md muc 0)")
    _log(f"[run] config: num_sequences={cfg.num_sequences} ep_len={cfg.ep_len} replan_steps={REPLAN_STEPS} "
         f"sampler_type={cfg.sampler_type} num_sampling_steps={cfg.num_sampling_steps} debug={cfg.debug}")

    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    # NOT model.multistep = cfg.multistep: that attribute only gates
    # MDTVAgent.step()'s own internal replan counter, which this harness
    # never calls (ForwardOnlyPolicy calls forward() directly) -- setting
    # it would be dead code and could mislead a future reader into
    # thinking it controls this harness's own REPLAN_STEPS above.
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler
    model.eval()

    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    policy = ForwardOnlyPolicy(model, n_candidates=1)

    # Tuning cohort = idx 0..N-1, eval cohort = idx N..2N-1, disjoint --
    # see module docstring / docs/TUNING_WORKFLOW.md muc 0. COHORT_OFFSET
    # shifts both the sequence slice and the seed range so the two cohorts
    # never reuse the same (sequence, seed) pair.
    N = cfg.num_sequences
    COHORT_OFFSET = 0 if TUNING_MODE else N
    eval_sequences = get_sequences(2 * N)[COHORT_OFFSET:COHORT_OFFSET + N]

    # Reseeded identically per sequence index in *both* branches below (not
    # just once at the top of main()) so that, for a given sequence, the
    # "with obstacle" run draws the exact same diffusion-policy noise as
    # the "without obstacle" run up to the point the obstacle is hit --
    # otherwise the two branches are just two independent stochastic
    # rollouts and "with <= without" would only hold statistically over
    # many sequences, not per-sequence as the comparison is meant to show.
    # Offset by COHORT_OFFSET too so eval's seeds (1100..1199) never
    # collide with tuning's (1000..1099), even though both loops below
    # count their own local idx from 0.
    SEQUENCE_SEED_BASE = 1000 + COHORT_OFFSET

    labels_and_radii = ([("without obstacle", None)] if INCLUDE_WITHOUT_OBSTACLE else []) + [
        (f"with obstacle r={r}", r) for r in RADII_TO_SWEEP
    ]

    results_path = RUN_OUTPUT_DIR / "results.json"

    def _write_progress(results):
        # Re-written after EVERY radius/obstacle-config entry, not just
        # once at the end -- see scripts/run_calvin_shielded.py's
        # identical helper for why (a radius sweep is exactly this kind
        # of slow, multi-entry run).
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "tuning_mode": TUNING_MODE,
                "cohort_sequence_idx_range": [COHORT_OFFSET, COHORT_OFFSET + N],
                "radii_to_sweep": RADII_TO_SWEEP,
                "results": results,
            }, f, indent=2)

    results = []
    total_videos_saved = 0
    for label, radius in labels_and_radii:
        sequence_results = []
        for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
            seed_everything(SEQUENCE_SEED_BASE + idx, workers=True)
            if radius is None:
                obstacle_fn = None
            else:
                # See scripts/run_calvin_shielded.py's identical block for
                # why this is a fresh, caller-seeded, non-global rng per
                # sequence.
                obstacle_rng = np.random.default_rng(SEQUENCE_SEED_BASE + idx)
                obstacle_fn = lambda joint_angles, chunk, r=radius, rng=obstacle_rng: sample_obstacle_from_reference_chunk(  # noqa: E731,E501
                    joint_angles, chunk, radius=r, rng=rng,
                )
            attempts = run_calvin_unshielded_sequence(
                env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
                get_env_state_for_initial_condition, ep_len=cfg.ep_len, replan_steps=REPLAN_STEPS,
                obstacle_fn=obstacle_fn,
            )
            sequence_results.append(attempts)

        slots = build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5)
        violation_rate, success_rate = fixed_cohort_rates(slots)
        _log(f"[{label}] violation_rate={violation_rate:.3f}  success_rate={success_rate:.3f}"
             f"  (avg_seq_len={success_rate * 5:.2f}/5, n_sequences={cfg.num_sequences})")

        entry = {
            "label": label,
            "radius": radius,
            "violation_rate": violation_rate,
            "success_rate": success_rate,
            "avg_seq_len": success_rate * 5,
            "n_sequences": cfg.num_sequences,
            "clearance_stats": None,
            "violated_steps_taken_stats": None,
            "video_paths": None,
            "video_skip_reason": None,
            # Full per-sequence, per-subtask-attempt records ('violated'/
            # 'reached'/'min_clearance') -- lets shortstop.calvin_metrics.
            # conservatism_cost pair this run against a shielded run's own
            # sequence_results later, without a re-run (see
            # scripts/run_calvin_shielded.py's identical field).
            "sequence_results": sequence_results,
        }

        if cfg.debug:
            clearance_stats = _clearance_stats(sequence_results)
            entry["clearance_stats"] = clearance_stats
            _log_clearance_debug(label, clearance_stats)
            steps_taken_stats = _violated_steps_taken_percentiles(sequence_results)
            entry["violated_steps_taken_stats"] = steps_taken_stats
            _log_violated_steps_taken_debug(label, steps_taken_stats)
            if radius is not None:
                vis_idxs = _rank_violating_sequence_idxs_by_length(sequence_results, cfg.num_videos)
                if not vis_idxs:
                    reason = ("no sequence violated at this radius -- radius is likely too small "
                              "to show anything at this checkpoint's paths; skipping video")
                    _log(f"  [debug] {label}: {reason}")
                    entry["video_skip_reason"] = reason
                else:
                    video_paths = []
                    for vis_idx in vis_idxs:
                        # Rebuilt fresh here (not reusing the main loop's
                        # own obstacle_fn, whose rng belongs to -- and has
                        # already been consumed by -- the LAST sequence of
                        # the main pass) -- same seed=SEQUENCE_SEED_BASE+
                        # vis_idx as that sequence originally got, so the
                        # video reproduces the EXACT SAME obstacle
                        # placement actually measured.
                        video_obstacle_rng = np.random.default_rng(SEQUENCE_SEED_BASE + vis_idx)
                        video_obstacle_fn = lambda joint_angles, chunk, r=radius, rng=video_obstacle_rng: sample_obstacle_from_reference_chunk(  # noqa: E731,E501
                            joint_angles, chunk, radius=r, rng=rng,
                        )
                        video_paths += _save_debug_videos(
                            vis_idx, radius, video_obstacle_fn, env, policy, task_oracle, lang_embeddings,
                            val_annotations, get_env_state_for_initial_condition, cfg, eval_sequences,
                            SEQUENCE_SEED_BASE,
                        )
                    total_videos_saved += len(video_paths)
                    entry["video_paths"] = video_paths

        results.append(entry)
        _write_progress(results)
        _log(f"  [progress] wrote {len(results)}/{len(labels_and_radii)} entry(ies) so far to: {results_path}")

    if cfg.debug and total_videos_saved > 0:
        _log(f"[debug] saved {total_videos_saved} obstacle-visualization video(s) under: {VIS_OUTPUT_DIR}")

    _log(f"[run] wrote structured results to: {results_path}")
    _log(f"[run] DONE -- zip up {RUN_OUTPUT_DIR} and send it back for tuning analysis")
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
