"""Standalone smoke test: render ONE short CALVIN rollout with a
privileged obstacle, and write a single MP4 showing the real
rgb_static camera frames (same frames results/calvin_eval.gif was made
from) with the obstacle composited on top -- see
shortstop/calvin_obstacle_viz.py.

Purpose: a fast, minimal way to eyeball "does the obstacle actually show
up where expected on the real rendered scene", before trusting the
bigger sweep in run_calvin_unshielded.py. Tries a handful of curated eval
sequences (CANDIDATE_SEQUENCE_IDXS) and picks whichever one runs the most
subtasks (up to N_SUBTASKS) before stopping, since the obstacle sits
right on the path of the chunk about to execute and can cause a
violation as early as subtask 1 -- not a metrics sweep, no
violation_rate/success_rate computed here, just the video and the raw
per-subtask violated/reached/min_clearance numbers.

Run from WSL2, inside the `mdt_env` conda environment (see
docs/CALVIN_SETUP.md, needs a real GPU + the mdt_policy checkpoint +
debug dataset) -- NOT runnable/tested in the dev sandbox this was
written in:

    cd SHORTSTOP
    python scripts/render_obstacle_video.py
"""
import sys
from pathlib import Path

import calvin_env  # noqa: F401 -- import BEFORE the sys.path.insert below, see comment there
import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
# Importing `calvin_env` above, before this insert, matters: mdt_policy/
# calvin_env/ (the submodule's own repo root, no __init__.py) sits *inside*
# MDT_POLICY_ROOT and is itself named "calvin_env" -- once MDT_POLICY_ROOT
# is on sys.path, `import calvin_env` from anywhere (including deep inside
# calvin_env.envs.play_table_env's own internal imports) can resolve to
# that directory as an implicit PEP 420 namespace package instead of the
# real pip-installed `calvin_env` package (mdt_policy/calvin_env/calvin_env/,
# installed via `pip install -e .` per docs/CALVIN_SETUP.md) -- a namespace
# package's __file__ is None, which is exactly the
# `TypeError: expected str, bytes or os.PathLike object, not NoneType`
# crash in PlayTableSimEnv.__init__'s `Path(calvin_env.__file__)` call.
# Importing it first caches the correctly-resolved module in sys.modules
# before MDT_POLICY_ROOT ever shadows it -- same fix already in place in
# scripts/run_calvin_unshielded.py, which is why that script doesn't hit
# this but this one did until this import was added.
sys.path.insert(0, str(MDT_POLICY_ROOT))

from mdt.evaluation.multistep_sequences import get_sequences  # noqa: E402
from mdt.evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition  # noqa: E402
from mdt.utils.utils import get_last_checkpoint  # noqa: E402

from shortstop.calvin_experiment import run_calvin_unshielded_sequence  # noqa: E402
from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk  # noqa: E402
from shortstop.calvin_obstacle_viz import save_sequence_video  # noqa: E402

# How many subtasks of the sequence to run before stopping -- kept below
# 5 since this is a quick visual smoke test, not a full sequence. Raise
# to see more of the sequence in one video.
N_SUBTASKS = 3

OBSTACLE_RADIUS = 0.08  # chosen default, see docs/PARAMETERS_REFERENCE.md muc 1's "radius" sweep table

# Which curated eval sequences (get_sequences()'s own idx, 0-based) to try.
# The obstacle is placed right on the path of the chunk about to be
# executed, so any given sequence can violate as early as its very first
# subtask -- there is no single SEQUENCE_IDX that's guaranteed to run
# N_SUBTASKS' worth of video every time (radius=0.08's own sweep found a
# ~12.8%-per-subtask violation rate, see docs/PARAMETERS_REFERENCE.md, so
# instant-violation on subtask 1 is bad luck, not something a single
# "better" seed avoids for good). Instead of hardcoding one idx and
# hoping, try each of these in turn (same per-idx seeding
# run_calvin_unshielded.py itself uses) and keep whichever one runs the
# most subtasks before stopping -- see the scan in main() below.
CANDIDATE_SEQUENCE_IDXS = [0, 1, 2, 3, 4]
SEQUENCE_SEED_BASE = 1000  # matches run_calvin_unshielded.py's SEQUENCE_SEED_BASE

OUT_PATH = REPO_ROOT / "outputs" / "render_obstacle_video.mp4"


class _ForwardOnlyPolicy:
    """Adapter around the loaded MDTVAgent: `.propose(observation)` calls
    `model(obs, goal)` (forward(), not step() -- see
    shortstop/mdt_policy_client.py's docstring for why) once, returning a
    raw numpy chunk. Same pattern as run_calvin_unshielded.py's own
    `_ForwardOnlyPolicy`, kept as its own small copy here so this
    standalone script has no import-time dependency on that other
    script's private class."""

    def __init__(self, model):
        self.model = model

    def propose(self, observation):
        goal = observation["goal"]
        return [np.asarray(self.model(observation, goal).squeeze(0).detach().cpu())]


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.multistep = cfg.multistep
    model.eval()

    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    policy = _ForwardOnlyPolicy(model)

    eval_sequences = get_sequences(max(CANDIDATE_SEQUENCE_IDXS) + 1)
    obstacle_fn = lambda joint_angles, chunk: sample_obstacle_from_reference_chunk(  # noqa: E731
        joint_angles, chunk, radius=OBSTACLE_RADIUS,
    )

    best_sequence_idx, best_eval_sequence, best_attempts = None, None, None
    for sequence_idx in CANDIDATE_SEQUENCE_IDXS:
        initial_state, eval_sequence = eval_sequences[sequence_idx]
        eval_sequence = eval_sequence[:N_SUBTASKS]

        seed_everything(SEQUENCE_SEED_BASE + sequence_idx, workers=True)
        attempts = run_calvin_unshielded_sequence(
            env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
            get_env_state_for_initial_condition, ep_len=cfg.ep_len, replan_steps=cfg.multistep,
            obstacle_fn=obstacle_fn, record_camera_frames=True,
        )
        # More subtasks attempted before the sequence stopped (violated,
        # failed, or ran out of N_SUBTASKS) makes for a more informative
        # video -- see CANDIDATE_SEQUENCE_IDXS's comment for why no single
        # idx is guaranteed to avoid an instant subtask-1 violation.
        if best_attempts is None or len(attempts) > len(best_attempts):
            best_sequence_idx, best_eval_sequence, best_attempts = sequence_idx, eval_sequence, attempts
        if len(attempts) >= N_SUBTASKS:
            break  # can't do better than showing every requested subtask

    print(f"[render_obstacle_video] picked sequence_idx={best_sequence_idx} "
          f"({len(best_attempts)}/{len(best_eval_sequence)} subtasks attempted before stopping)")
    eval_sequence, attempts = best_eval_sequence, best_attempts

    subtask_records = []
    for subtask, attempt in zip(eval_sequence, attempts):
        outcome = "violated" if attempt["violated"] else "reached" if attempt["reached"] else "failed"
        subtask_records.append({
            "subtask": subtask,
            "frames": attempt["camera_frames"],
            "obstacle": attempt["obstacle"],
            "outcome": outcome,
        })

    static_camera = next(cam for cam in env.env.cameras if cam.name == "static")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_sequence_video(
        subtask_records, static_camera, str(OUT_PATH),
        base_position=env.env.robot.base_position, base_orientation=env.env.robot.base_orientation,
    )

    print(f"[render_obstacle_video] wrote: {OUT_PATH}")
    for subtask, attempt in zip(eval_sequence, attempts):
        print(f"  {subtask}: violated={attempt['violated']} reached={attempt['reached']} "
              f"min_clearance={attempt['min_clearance']}")


if __name__ == "__main__":
    main()
