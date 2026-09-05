import functools

import numpy as np

from shortstop.calvin_progress import calvin_progress_scores
from shortstop.robot_geometry import N_JOINTS, panda_frames, to_world_frame


class _FakeTaskOracle:
    """Minimal stand-in for calvin_env.envs.tasks.Tasks -- only `.tasks`,
    a dict of {subtask_name: functools.partial(base_fn, *config_args)},
    matching Tasks.__init__'s own construction exactly (the base_fn
    itself is irrelevant here, only `.args` -- the config list minus its
    own first element -- is ever read)."""

    def __init__(self, tasks):
        self.tasks = tasks


def _partial(*args):
    return functools.partial(lambda *a, **k: None, *args)


def _scene_info_with(movable_objects=None, doors=None):
    return {
        "scene_info": {
            "movable_objects": movable_objects or {},
            "doors": doors or {},
        }
    }


def test_prefers_the_candidate_whose_predicted_endpoint_is_closer_to_the_target_object():
    q0 = np.zeros(N_JOINTS)
    start = panda_frames(q0)[-1]
    target_pos = start + np.array([0.1, 0.0, 0.0])  # object sits 0.1m ahead along +x

    task_oracle = _FakeTaskOracle({"push_red_block_right": _partial("block_red", 0.1, 0)})
    current_info = _scene_info_with(movable_objects={"block_red": {"current_pos": target_pos.tolist()}})

    toward = np.zeros((4, 7))
    toward[:, 0] = 0.02  # moves +x, toward the object
    away = np.zeros((4, 7))
    away[:, 0] = -0.02  # moves -x, away from the object

    scores = calvin_progress_scores(task_oracle, "push_red_block_right", current_info, q0, [toward, away], replan_steps=4)

    assert scores[0] > scores[1]


def test_corrects_for_the_real_nonzero_robot_base_offset_not_just_identity():
    """Regression test for the base-frame bug (2026-09-05): propagate_arm_
    tube's predicted_flange lives in the robot's own LOCAL base frame,
    but CALVIN reports a real scene object's current_pos in true WORLD
    coordinates -- and the real CALVIN scenes (calvin_scene_A/B/C/D.yaml)
    do NOT place the robot base at the world origin
    (robot_base_position=[-0.34,-0.46,0.24]). `target_local` here is
    IDENTICAL to the first test above's own target (0.1m ahead of the
    start pose along +x, in local coordinates) -- but expressed to
    calvin_progress_scores as CALVIN really would, in world coordinates
    (to_world_frame'd through that same real base offset). Passing the
    correct base_position/base_orientation must recover the exact same
    ranking as the local-frame test (`toward` wins); passing the
    identity default (the pre-fix behavior) on this same world-frame
    target gets the ranking BACKWARDS -- demonstrating this is a real,
    not just cosmetic, correction."""
    q0 = np.zeros(N_JOINTS)
    start_local = panda_frames(q0)[-1]
    target_local = start_local + np.array([0.1, 0.0, 0.0])

    base_position = [-0.34, -0.46, 0.24]  # calvin_scene_D.yaml's real robot_base_position
    base_orientation = [0.0, 0.0, 0.0, 1.0]
    target_world = to_world_frame(target_local, base_position, base_orientation)

    task_oracle = _FakeTaskOracle({"push_red_block_right": _partial("block_red", 0.1, 0)})
    current_info = _scene_info_with(movable_objects={"block_red": {"current_pos": target_world.tolist()}})

    toward = np.zeros((4, 7))
    toward[:, 0] = 0.02
    away = np.zeros((4, 7))
    away[:, 0] = -0.02

    correct = calvin_progress_scores(
        task_oracle, "push_red_block_right", current_info, q0, [toward, away], replan_steps=4,
        base_position=base_position, base_orientation=base_orientation,
    )
    assert correct[0] > correct[1]  # toward wins, matching the pure-local-frame test above

    uncorrected = calvin_progress_scores(
        task_oracle, "push_red_block_right", current_info, q0, [toward, away], replan_steps=4,
    )  # identity default applied to a genuinely world-frame target -- the old, buggy behavior
    assert uncorrected[0] < uncorrected[1]  # ranking flips backwards without the fix


def test_scores_only_the_first_replan_steps_rows_not_the_full_chunk():
    """Regression test for the replan_steps fix: `toward_then_away` is
    closer to the target after 2 rows but peels sharply away over its
    remaining 2; `away_then_toward` starts further but swings back closer
    by the end. Scoring with replan_steps=2 (only what would actually be
    committed before replanning) must prefer `toward_then_away`; scoring
    with replan_steps=4 (the full chunk) must prefer `away_then_toward`
    instead -- the ranking flip proves the score depends on where the
    truncation happens, not silently always the chunk's full length."""
    q0 = np.zeros(N_JOINTS)
    start = panda_frames(q0)[-1]
    target_pos = start + np.array([0.04, 0.0, 0.0])

    task_oracle = _FakeTaskOracle({"push_red_block_right": _partial("block_red", 0.1, 0)})
    current_info = _scene_info_with(movable_objects={"block_red": {"current_pos": target_pos.tolist()}})

    toward_then_away = np.zeros((4, 7))
    toward_then_away[:2, 0] = 0.02   # first 2 rows: net +0.04, right at the target
    toward_then_away[2:, 0] = -0.25  # last 2 rows: peels sharply away

    away_then_toward = np.zeros((4, 7))
    away_then_toward[:2, 0] = -0.02  # first 2 rows: net -0.04, away from the target
    away_then_toward[2:, 0] = 0.06   # last 2 rows: swings back, ends up close

    candidates = [toward_then_away, away_then_toward]
    prefix_scores = calvin_progress_scores(
        task_oracle, "push_red_block_right", current_info, q0, candidates, replan_steps=2,
    )
    full_scores = calvin_progress_scores(
        task_oracle, "push_red_block_right", current_info, q0, candidates, replan_steps=4,
    )

    assert prefix_scores[0] > prefix_scores[1]  # toward_then_away wins if only the first 2 rows count
    assert full_scores[0] < full_scores[1]  # away_then_toward wins once the full chunk counts


def test_neutral_zero_when_subtask_is_not_in_task_oracle():
    q0 = np.zeros(N_JOINTS)
    task_oracle = _FakeTaskOracle({})
    current_info = _scene_info_with()
    candidates = [np.zeros((3, 7)), np.ones((3, 7)) * 0.01]

    scores = calvin_progress_scores(task_oracle, "unknown_subtask", current_info, q0, candidates, replan_steps=3)

    assert scores == [0.0, 0.0]


def test_neutral_zero_when_target_object_has_no_cartesian_position():
    """Door/slider/drawer subtasks (move_door_rel) only expose a joint
    angle (scene_info["doors"][name]["current_state"]), never a
    current_pos -- see calvin_env.scene.objects.door.Door.get_info() --
    so this subtask's target can't be resolved to a point, and g(a) must
    stay neutral rather than silently guessing a wrong one."""
    q0 = np.zeros(N_JOINTS)
    task_oracle = _FakeTaskOracle({"open_drawer": _partial("base__drawer", 0.12)})
    current_info = _scene_info_with(doors={"base__drawer": {"current_state": 0.0}})
    candidates = [np.zeros((3, 7)), np.ones((3, 7)) * 0.01]

    scores = calvin_progress_scores(task_oracle, "open_drawer", current_info, q0, candidates, replan_steps=3)

    assert scores == [0.0, 0.0]


def test_neutral_zero_when_target_arg_is_not_a_single_object_name():
    """push_object_into names a *list* of candidate objects, not one --
    args[0] is a list, not a str, so it can't be used as a scene_info
    lookup key."""
    q0 = np.zeros(N_JOINTS)
    task_oracle = _FakeTaskOracle({
        "push_into_drawer": _partial(["block_red", "block_blue", "block_pink"], "table", "base_link"),
    })
    current_info = _scene_info_with(movable_objects={"block_red": {"current_pos": [1.0, 0.0, 0.0]}})
    candidates = [np.zeros((3, 7))]

    scores = calvin_progress_scores(task_oracle, "push_into_drawer", current_info, q0, candidates, replan_steps=3)

    assert scores == [0.0]
