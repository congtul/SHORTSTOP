"""Standalone smoke test: render ONE short CALVIN rollout with a
privileged obstacle, and write a single MP4 showing the real
rgb_static camera frames (same frames results/calvin_eval.gif was made
from) with the obstacle composited on top -- see
shortstop/calvin_obstacle_viz.py.

Purpose: a fast, minimal way to eyeball "does the obstacle actually show
up where expected on the real rendered scene", before trusting the
bigger sweep in run_calvin_unshielded.py. Runs exactly ONE sequence
(default: just its first subtask, see N_SUBTASKS below), not a metrics
sweep -- no violation_rate/success_rate computed here, just the video
and the raw per-subtask violated/reached/min_clearance numbers.

Run from WSL2, inside the `mdt_env` conda environment (see
docs/CALVIN_SETUP.md, needs a real GPU + the mdt_policy checkpoint +
debug dataset) -- NOT runnable/tested in the dev sandbox this was
written in:

    cd SHORTSTOP
    python scripts/render_obstacle_video.py
"""
import sys
from pathlib import Path

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
from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk  # noqa: E402
from shortstop.calvin_obstacle_viz import save_sequence_video  # noqa: E402

# How many subtasks of the sequence to run before stopping -- default
# just the first one, since this is a quick visual smoke test, not a
# full 5-subtask sequence. Raise to see more of the sequence in one gif.
N_SUBTASKS = 3

OBSTACLE_RADIUS = 0.08  # chosen default, see docs/PARAMETERS_REFERENCE.md muc 1's "radius" sweep table
SEQUENCE_IDX = 0
SEQUENCE_SEED = 1000  # matches run_calvin_unshielded.py's SEQUENCE_SEED_BASE

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

    initial_state, eval_sequence = get_sequences(SEQUENCE_IDX + 1)[SEQUENCE_IDX]
    eval_sequence = eval_sequence[:N_SUBTASKS]

    seed_everything(SEQUENCE_SEED + SEQUENCE_IDX, workers=True)
    attempts = run_calvin_unshielded_sequence(
        env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
        get_env_state_for_initial_condition, ep_len=cfg.ep_len, replan_steps=cfg.multistep,
        obstacle_fn=lambda joint_angles, chunk: sample_obstacle_from_reference_chunk(
            joint_angles, chunk, radius=OBSTACLE_RADIUS,
        ),
        record_camera_frames=True,
    )

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
    save_sequence_video(subtask_records, static_camera, str(OUT_PATH))

    print(f"[render_obstacle_video] wrote: {OUT_PATH}")
    for subtask, attempt in zip(eval_sequence, attempts):
        print(f"  {subtask}: violated={attempt['violated']} reached={attempt['reached']} "
              f"min_clearance={attempt['min_clearance']}")


if __name__ == "__main__":
    main()
