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
    trajectory recording on, and saves one GIF per subtask attempt --
    see shortstop/calvin_obstacle_viz.py.

Everything printed is *also* written to a run-specific output directory
(printed at the very start and end so it's easy to find) meant to be
zipped up and handed back for analysis on a machine without this
env/checkpoint/dataset:
  outputs/calvin_unshielded_runs/run_<timestamp>/
    run.log       -- exact copy of everything printed to stdout
    results.json  -- same numbers, structured (radius -> violation_rate,
                      success_rate, clearance stats, which sequence/subtask
                      each GIF came from) -- read this one programmatically
    gifs/*.gif    -- from the debug obstacle visualization above
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
from shortstop.calvin_obstacle_viz import save_subtask_gif  # noqa: E402

# Candidate obstacle radii to sweep (meters) -- edit this list directly
# while tuning; no config knob for it yet since we're still narrowing the
# range by hand, see module docstring.
RADII_TO_SWEEP = [0.02, 0.05, 0.08, 0.12]

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "calvin_unshielded_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
VIS_OUTPUT_DIR = RUN_OUTPUT_DIR / "gifs"

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


def _save_debug_gifs(
    sequence_idx, radius, obstacle_fn, env, policy, task_oracle, lang_embeddings, val_annotations,
    get_env_state_for_initial_condition, cfg, eval_sequences, sequence_seed_base,
):
    """Re-runs sequence `sequence_idx` alone (same per-sequence seed the
    main sweep already used for it -- reseeding is per-idx for *every*
    sequence in that sweep, not just this one, so this reproduces exactly
    the attempt that already contributed to the printed metrics/stats,
    whichever idx is picked) with trajectory recording on, saves one GIF
    per subtask attempt of that sequence to VIS_OUTPUT_DIR. Returns how
    many GIFs were written."""
    initial_state, eval_sequence = eval_sequences[sequence_idx]
    seed_everything(sequence_seed_base + sequence_idx, workers=True)
    attempts = run_calvin_unshielded_sequence(
        env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
        get_env_state_for_initial_condition, ep_len=cfg.ep_len, replan_steps=cfg.multistep,
        obstacle_fn=obstacle_fn, record_trajectory=True,
    )
    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_radius = str(radius).replace(".", "p")
    gif_paths = []
    for i, attempt in enumerate(attempts):
        out_path = VIS_OUTPUT_DIR / f"seq{sequence_idx}_subtask{i}_r{safe_radius}.gif"
        save_subtask_gif(attempt["trajectory"], attempt["obstacle"], str(out_path))
        gif_paths.append(str(out_path.relative_to(RUN_OUTPUT_DIR)))
    return gif_paths


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
    _log(f"[run] writing log + results.json (+ gifs/ if any) to: {RUN_OUTPUT_DIR}")
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

    labels_and_obstacle_fns = [("without obstacle", None, None)] + [
        (
            f"with obstacle r={r}",
            lambda joint_angles, chunk, r=r: sample_obstacle_from_reference_chunk(joint_angles, chunk, radius=r),
            r,
        )
        for r in RADII_TO_SWEEP
    ]

    results = []
    total_gifs_saved = 0
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
            "gif_paths": None,
            "gif_skip_reason": None,
        }

        if cfg.debug:
            clearance_stats = _clearance_stats(sequence_results)
            entry["clearance_stats"] = clearance_stats
            _log_clearance_debug(label, clearance_stats)
            if obstacle_fn is not None:
                vis_idx = _find_first_violating_sequence_idx(sequence_results)
                if vis_idx is None:
                    reason = ("no sequence violated at this radius -- radius is likely too small "
                              "to show anything at this checkpoint's paths; skipping GIF")
                    _log(f"  [debug] {label}: {reason}")
                    entry["gif_skip_reason"] = reason
                else:
                    gif_paths = _save_debug_gifs(
                        vis_idx, radius, obstacle_fn, env, policy, task_oracle, lang_embeddings, val_annotations,
                        get_env_state_for_initial_condition, cfg, eval_sequences, SEQUENCE_SEED_BASE,
                    )
                    total_gifs_saved += len(gif_paths)
                    entry["gif_paths"] = gif_paths

        results.append(entry)

    if cfg.debug and total_gifs_saved > 0:
        _log(f"[debug] saved {total_gifs_saved} obstacle-visualization GIF(s) under: {VIS_OUTPUT_DIR}")

    results_path = RUN_OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"radii_to_sweep": RADII_TO_SWEEP, "results": results}, f, indent=2)
    _log(f"[run] wrote structured results to: {results_path}")
    _log(f"[run] DONE -- zip up {RUN_OUTPUT_DIR} and send it back for tuning analysis")
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
