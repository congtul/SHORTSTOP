"""Cross-checks `shortstop.robot_geometry.panda_frames()` (this repo's own
modified-DH forward kinematics) against the REAL simulated Panda inside
CALVIN's own PyBullet sim -- closes the gap flagged in
`shortstop/robot_geometry.py`'s own docstring ("VERIFY AGAINST THE ACTUAL
franka_ros / franka_description URDF ... this was not run against a live
robot") and corrects two stale doc claims (`docs/STAGE7A_ARM_PIPELINE_
DESIGN.md`, `docs/PARAMETERS_REFERENCE.md` muc 8) that assert this was
"already verified" -- that claim only ever compared a handful of PANDA_DH
constants (d/a values) against the URDF's `<origin>` tags by hand, not a
real FK/pose comparison, and both docs reference a stale `SPHERE_RADII`/
`SPHERE_FRAME_INDICES` API this module no longer has (replaced by
`LINK_RADIUS`/`FRAME_RADIUS`/`capsule_segments()`).

PyBullet access path (confirmed by reading calvin_env's own real source,
not guessed -- `mdt_policy/calvin_env/calvin_env/envs/play_table_env.py`,
`mdt_policy/calvin_env/calvin_env/robot/robot.py`, `mdt_policy/calvin_env/
conf/robot/panda.yaml`): the `env` object every CALVIN script already
builds is a `HulcWrapper` (a `gym.Wrapper`), so `env.env` is the raw
`PlayTableSimEnv` -- `env.env.p` (pybullet module/BulletClient),
`env.env.cid` (physics client id), `env.env.robot.robot_uid` (pybullet
body id), `env.env.robot.end_effector_link_id` (=7, the flange link) are
all already-proven-reachable attribute paths this repo's own code
dereferences elsewhere (e.g. `scripts/run_calvin_unshielded.py`'s debug
video saving reads `env.env.robot.base_position`).

`panda_frames(q)`'s 9 entries map to PyBullet link states as:
  - index 0 (base): env.env.p.getBasePositionAndOrientation(robot_uid)[0]
  - index 1..7 (joints 1..7): env.env.p.getLinkState(robot_uid, i)[0] for
    i in range(7) -- pybullet link indices 0..6 = panda.yaml's
    arm_joint_ids, each giving the frame produced by that joint.
  - index 8 (flange): env.env.p.getLinkState(robot_uid,
    env.env.robot.end_effector_link_id)[0] (link index 7, panda_joint8's
    fixed child link).

No shield, no obstacle -- pure geometry comparison. Reuses the unshielded
loop's own real-rollout pattern (Propose -> execute directly) purely to
get a spread of real, reachable joint configurations to compare at, not
to measure violation/success.

Run from WSL2, inside the `mdt_env` conda environment (see
docs/CALVIN_SETUP.md) -- like every other real-data CALVIN script, NOT
runnable in the sandbox this was written in:

    cd SHORTSTOP
    python scripts/verify_robot_geometry_against_pybullet.py

Prints max/mean discrepancy (meters) across every sampled config, for
each of the 9 frame indices separately (so a single bad DH row is
identifiable, not just an aggregate). If everything is within a few mm
(judge against the printed numbers -- no threshold is hardcoded here,
this is a measurement, not a pass/fail gate), update robot_geometry.py's
docstring + the two stale doc claims to cite this real result instead.
"""
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

from shortstop.calvin_experiment import _joint_angles_from_obs, _lang_goal, _to_action_tensor  # noqa: E402
from shortstop.mdt_policy_client import ForwardOnlyPolicy  # noqa: E402
from shortstop.robot_geometry import N_JOINTS, panda_frames  # noqa: E402

N_CANDIDATES = 8
REPLAN_STEPS = 10
N_SEQUENCES_TO_SAMPLE = 10  # a small spread of real configs is enough for a geometry check, not a full sweep

RUN_OUTPUT_DIR = REPO_ROOT / "outputs" / "verify_robot_geometry_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"

_LOG_FILE = None


def _log(msg):
    print(msg)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def _pybullet_frames(env, joint_angles):
    """The same 9 positions panda_frames(joint_angles) predicts, read
    directly from PyBullet's own simulated robot at whatever its REAL
    current joint state happens to be (the caller is responsible for
    having already driven the sim to `joint_angles` via env.step() --
    this function only reads, it doesn't set anything)."""
    p, cid, robot = env.env.p, env.env.cid, env.env.robot
    base = np.array(p.getBasePositionAndOrientation(robot.robot_uid, physicsClientId=cid)[0])
    joints = [
        np.array(p.getLinkState(robot.robot_uid, i, physicsClientId=cid)[0])
        for i in range(N_JOINTS)
    ]
    flange = np.array(p.getLinkState(robot.robot_uid, robot.end_effector_link_id, physicsClientId=cid)[0])
    return np.stack([base] + joints + [flange])


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    global _LOG_FILE
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(RUN_OUTPUT_DIR / "run.log", "w", encoding="utf-8")
    _log(f"[run] writing log to: {RUN_OUTPUT_DIR}")

    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.eval()

    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    policy = ForwardOnlyPolicy(model, n_candidates=N_CANDIDATES)

    eval_sequences = get_sequences(2 * N_SEQUENCES_TO_SAMPLE)[0:N_SEQUENCES_TO_SAMPLE]

    per_frame_errors = [[] for _ in range(N_JOINTS + 2)]
    for idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        seed_everything(1000 + idx, workers=True)
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        for subtask in eval_sequence[:1]:  # one subtask per sequence is enough spread for a geometry check
            obs = env.get_obs()
            goal = _lang_goal(lang_embeddings, val_annotations, subtask)
            for _ in range(REPLAN_STEPS):
                candidates = policy.propose({**obs, "goal": goal})
                chunk = candidates[0]
                for action_row in chunk[:REPLAN_STEPS]:
                    obs, _, _, _ = env.step(_to_action_tensor(action_row))
                    joint_angles = _joint_angles_from_obs(obs)
                    ours = panda_frames(joint_angles)
                    real = _pybullet_frames(env, joint_angles)
                    for i in range(N_JOINTS + 2):
                        per_frame_errors[i].append(float(np.linalg.norm(ours[i] - real[i])))
                break  # one replan cycle per subtask is plenty of samples

    _log("[result] per-frame discrepancy (meters) between panda_frames() and real PyBullet state:")
    for i, errors in enumerate(per_frame_errors):
        values = np.asarray(errors)
        label = "base" if i == 0 else ("flange" if i == N_JOINTS + 1 else f"joint{i}")
        _log(f"  frame {i} ({label}): n={len(values)} mean={values.mean():.5f} max={values.max():.5f}")

    _log(
        "[run] DONE -- judge against the printed numbers (no hardcoded pass/fail threshold here). "
        "If everything is within a few mm, update robot_geometry.py's docstring + docs/"
        "STAGE7A_ARM_PIPELINE_DESIGN.md / docs/PARAMETERS_REFERENCE.md muc 8's stale "
        "'already verified' claims to cite this real result instead. If any frame is off by "
        "more, PANDA_DH/FLANGE_OFFSET need correcting -- a follow-up, not predictable from here."
    )
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
