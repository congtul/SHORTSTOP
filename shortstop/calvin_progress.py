"""Task-progress score g(a) for CALVIN (Stage 8).

Every P-R-C-S shield variant's Select step picks whichever certified/
repaired candidate maximizes g(a) (Report_2's Propose-step formula:
`{a^(i)} ~ pi_theta(.|o_t), g(a^(i)) = task-progress score`; used
identically by shortstop.baselines.ConfThreshShield/STLMonitorShield and
shortstop.shield's own ReachOnlyShield/STLShield -- see their `_score()`
methods, all `-goal_distance`, all a pure function of (state, chunk),
never of "chunk vs. its own pre-repair original": STL-Monitor/Conf-Thresh
never repair anything, so a repair-distance definition would be
meaningless for them). shortstop.arm_shield's module docstring flags
this as unresolved for language-conditioned tasks (LIBERO/CALVIN have no
single fixed goal point the way Reach-Avoid-2D does) and has `select()`
take `scores` as a caller-supplied argument -- this module is that
missing piece, for CALVIN specifically.

Design (Tier 1: forward-kinematics only). A physics-rollout alternative
("Tier 2": use PyBullet's saveState()/restoreState() to actually step
each candidate, read the real resulting object displacement, then
rewind) was considered and rejected -- not for cost, but for realism: no
real robot can execute an action, observe the consequence, and undo it
before really committing. Tier 1 only needs a forward *predictive*
model (here, pure kinematics -- the same nominal reach-tube endpoint
already computed for obstacle placement, see calvin_obstacle.
sample_obstacle_from_reference_chunk), matching how every real baseline
already grounded in this project actually works (MPC-Filter/CBF-Shield
predict via a dynamics model f-hat; ARMTD's reachtube is a forward,
sound over-approximation) -- never a privileged rewind.

    g(a) = -||predicted_flange_position(a) - target_object_position||

`predicted_flange_position` is read at step `replan_steps` of the chunk,
NOT the chunk's own full length -- `replan_steps` is how many rows of
the chunk the harness actually commits to env.step() before replanning
(run_calvin_shielded_subtask calls policy.propose() again after that many
steps, discarding whatever is left of the chunk). Scoring the chunk's
full length would rank candidates by a tail segment that never actually
runs whenever replan_steps < the chunk's own length; today CALVIN's
own act_window_size and the harness's default replan_steps both happen
to be 10, so this was previously silently correct by coincidence, not by
construction.

`target_object_position` is read from the CURRENT real scene state
(env.get_info()["scene_info"]), using the subtask -> object-name mapping
already loaded into task_oracle.tasks by calvin_env.envs.tasks.Tasks.
__init__ (`{name: functools.partial(base_fn, *args)}`, built straight
from conf/tasks/new_playtable_tasks.yaml) -- `partial.args[0]` is the
manipulated object's name for most task types (rotate_object/
push_object/lift_object/move_door_rel/toggle_light all name it first).

Known gap (v1, documented rather than guessed around): doors (move_door_
rel -- sliders/drawers) only expose a joint angle (scene_info["doors"]
[name]["current_state"], see calvin_env.scene.objects.door.Door.
get_info()), not a Cartesian position; place_object's arg is a
destination surface, not a graspable target; stack_objects/
unstack_objects take no object argument at all; push_object_into names a
*list* of candidate objects, not one. For any subtask this can't resolve
a real Cartesian target for, g(a) is 0.0 for every candidate -- neutral
(Select then keeps whichever admissible candidate happens to come
first), not a guess dressed up as a signal.

Myopia caveat (raised in conversation, not solved here): this is a
purely geometric proximity-to-object proxy -- a candidate that reaches
toward the object without ever achieving the contact/displacement the
task actually requires scores exactly as well as one that would truly
progress the task. This is the same *category* of approximation the 2D
prototype's own -goal_distance already accepts (Cartesian proximity, not
"how done is the task"), not a new risk introduced here.
"""
import numpy as np

from .arm_reach import propagate_arm_tube
from .robot_geometry import FLANGE_FRAME_INDEX


def _target_object_position(task_oracle, subtask, current_info):
    """The real, current world position (3,) of the object `subtask` is
    configured (conf/tasks/*.yaml) to manipulate, or None if this
    subtask's target isn't a resolvable Cartesian position -- see module
    docstring's Known gap."""
    task_fn = task_oracle.tasks.get(subtask)
    if task_fn is None or not task_fn.args:
        return None

    obj_name = task_fn.args[0]
    if not isinstance(obj_name, str):
        return None  # e.g. push_object_into's list of candidate objects

    movable_objects = current_info["scene_info"]["movable_objects"]
    if obj_name in movable_objects:
        return np.asarray(movable_objects[obj_name]["current_pos"], dtype=float)
    return None  # doors/lights/buttons/surfaces: no Cartesian position exposed (see module docstring)


def calvin_progress_scores(task_oracle, subtask, current_info, joint_angles, candidates, replan_steps):
    """g(a) for each of `candidates` (raw chunks from policy.propose()),
    same order/length as `candidates` -- see module docstring. All 0.0
    (neutral, no ranking signal) when `subtask`'s target object has no
    resolvable Cartesian position.

    `replan_steps`: how many rows of each chunk the caller will actually
    commit before replanning (see module docstring) -- required, not
    defaulted, so this can't silently drift back to scoring the chunk's
    full length. Each chunk is truncated to its first `replan_steps` rows
    before propagation; `replan_steps` >= a chunk's own length is fine
    (propagate_arm_tube then just sees the whole chunk).
    """
    target = _target_object_position(task_oracle, subtask, current_info)
    if target is None:
        return [0.0] * len(candidates)

    scores = []
    for chunk in candidates:
        tube = propagate_arm_tube(joint_angles, chunk[:replan_steps], w_bar=0.0, model_error=0.0)
        predicted_flange = tube[-1][FLANGE_FRAME_INDEX].center()
        scores.append(-float(np.linalg.norm(predicted_flange - target)))
    return scores
