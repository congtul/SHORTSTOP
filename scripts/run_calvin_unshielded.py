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
    python scripts/run_calvin_unshielded.py

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

# Candidate obstacle radii to sweep (meters) -- edit this list directly
# while tuning; no config knob for it yet since we're still narrowing the
# range by hand, see module docstring.
#
# TUNED (real run, n_sequences=100, checkpoint/dataset per docs/CALVIN_SETUP.md):
# swept [0.02, 0.05, 0.08, 0.12] -> violation_rate 0.076/0.108/0.128/0.148,
# success_rate 0.704/0.586/0.482/0.412 (baseline without obstacle: 0.930).
# No floor effect at 0.02 (violation_rate already meaningfully nonzero), no
# ceiling effect reached by 0.12 (nowhere near 100%) -- full table + the
# min_clearance+radius=const sanity-check finding in
# docs/PARAMETERS_REFERENCE.md muc 1's "radius" entry. CHOSE r=0.08 as the
# default (shortstop.calvin_obstacle.sample_obstacle_from_reference_chunk's
# own default arg) -- balances a meaningful violation_rate against still
# leaving success_rate well above 0 for a shield to visibly improve.
# Re-sweep (edit this list) if the checkpoint/dataset ever changes, or to
# see where ceiling effect kicks in above 0.12 (not yet explored).
RADII_TO_SWEEP = [0.08]

# Resuming a killed/interrupted run: trim RADII_TO_SWEEP above to just the
# radii not yet completed (check the previous run's run.log), and flip
# this to False so this run doesn't redo "without obstacle" too (its own
# result never depends on which radii are being swept, so a previous
# run's logged number for it is still valid -- no need to ever re-run
# it just because the radius list changed). Set back to True for a full
# from-scratch run.
INCLUDE_WITHOUT_OBSTACLE = True

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "calvin_unshielded_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
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


def _find_first_violating_sequence_idx(sequence_results):
    """First sequence index with at least one violated attempt, or None if
    this radius was never hit by any of the swept sequences at all -- the
    latter is itself a useful debug signal (radius likely too small for
    this checkpoint/task set), not just "nothing to visualize"."""
    for idx, attempts in enumerate(sequence_results):
        if any(a["violated"] for a in attempts):
            return idx
    return None


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
        get_env_state_for_initial_condition, ep_len=cfg.ep_len, replan_steps=cfg.multistep,
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
            "obstacle": attempt["obstacle"],
            "outcome": outcome,
        })

    static_camera = next(cam for cam in env.env.cameras if cam.name == "static")
    out_path = VIS_OUTPUT_DIR / f"seq{sequence_idx}_r{safe_radius}.mp4"
    save_sequence_video(subtask_records, static_camera, str(out_path))
    return [str(out_path.relative_to(RUN_OUTPUT_DIR))]


class _ForwardOnlyPolicy:
    """Adapter around the loaded MDTVAgent: `.propose(observation)` calls
    `model(obs, goal)` (forward(), not step() -- see
    shortstop/mdt_policy_client.py's docstring for why) `n_candidates`
    times, returning raw numpy chunks."""

    def __init__(self, model, n_candidates=1):
        self.model = model
        self.n_candidates = n_candidates

    def propose(self, observation):
        goal = observation["goal"]
        return [
            np.asarray(self.model(observation, goal).squeeze(0).detach().cpu())
            for _ in range(self.n_candidates)
        ]


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    global _LOG_FILE
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(RUN_OUTPUT_DIR / "run.log", "w", encoding="utf-8")
    _log(f"[run] writing log + results.json (+ videos/ if any) to: {RUN_OUTPUT_DIR}")
    _log(f"[run] config: num_sequences={cfg.num_sequences} ep_len={cfg.ep_len} multistep={cfg.multistep} "
         f"sampler_type={cfg.sampler_type} num_sampling_steps={cfg.num_sampling_steps} debug={cfg.debug}")

    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.multistep = cfg.multistep
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler
    model.eval()

    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    policy = _ForwardOnlyPolicy(model, n_candidates=1)

    eval_sequences = get_sequences(cfg.num_sequences)

    # Reseeded identically per sequence index in *both* branches below (not
    # just once at the top of main()) so that, for a given sequence, the
    # "with obstacle" run draws the exact same diffusion-policy noise as
    # the "without obstacle" run up to the point the obstacle is hit --
    # otherwise the two branches are just two independent stochastic
    # rollouts and "with <= without" would only hold statistically over
    # many sequences, not per-sequence as the comparison is meant to show.
    SEQUENCE_SEED_BASE = 1000

    labels_and_obstacle_fns = ([("without obstacle", None, None)] if INCLUDE_WITHOUT_OBSTACLE else []) + [
        (
            f"with obstacle r={r}",
            lambda joint_angles, chunk, r=r: sample_obstacle_from_reference_chunk(joint_angles, chunk, radius=r),
            r,
        )
        for r in RADII_TO_SWEEP
    ]

    results = []
    total_videos_saved = 0
    for label, obstacle_fn, radius in labels_and_obstacle_fns:
        sequence_results = []
        for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
            seed_everything(SEQUENCE_SEED_BASE + idx, workers=True)
            attempts = run_calvin_unshielded_sequence(
                env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
                get_env_state_for_initial_condition, ep_len=cfg.ep_len, replan_steps=cfg.multistep,
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
            "video_paths": None,
            "video_skip_reason": None,
        }

        if cfg.debug:
            clearance_stats = _clearance_stats(sequence_results)
            entry["clearance_stats"] = clearance_stats
            _log_clearance_debug(label, clearance_stats)
            if obstacle_fn is not None:
                vis_idx = _find_first_violating_sequence_idx(sequence_results)
                if vis_idx is None:
                    reason = ("no sequence violated at this radius -- radius is likely too small "
                              "to show anything at this checkpoint's paths; skipping video")
                    _log(f"  [debug] {label}: {reason}")
                    entry["video_skip_reason"] = reason
                else:
                    video_paths = _save_debug_videos(
                        vis_idx, radius, obstacle_fn, env, policy, task_oracle, lang_embeddings, val_annotations,
                        get_env_state_for_initial_condition, cfg, eval_sequences, SEQUENCE_SEED_BASE,
                    )
                    total_videos_saved += len(video_paths)
                    entry["video_paths"] = video_paths

        results.append(entry)

    if cfg.debug and total_videos_saved > 0:
        _log(f"[debug] saved {total_videos_saved} obstacle-visualization video(s) under: {VIS_OUTPUT_DIR}")

    results_path = RUN_OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"radii_to_sweep": RADII_TO_SWEEP, "results": results}, f, indent=2)
    _log(f"[run] wrote structured results to: {results_path}")
    _log(f"[run] DONE -- zip up {RUN_OUTPUT_DIR} and send it back for tuning analysis")
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
