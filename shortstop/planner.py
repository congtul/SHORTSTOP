"""Two sources of a joint-config path for Stage 7c's keypose shields
(shortstop/keypose_shield.py), sharing one interface:
`planner_fn(joint_angles, target_position) -> (N, 7) path_points array`.

`real_get_path` wraps PyRep's `Arm.get_path()`/`get_nonlinear_path()` (the
actual mechanism behind RLBench's keypose execution) -- NOT usable in this
environment (no PyRep/CoppeliaSim installed), included so the shield code
has one real, confirmed-against-the-actual-API integration point to swap
in later rather than a guessed one. `mock_get_path` is a synthetic stand-
in (damped-least-squares IK + joint-space linear interpolation -- the same
math shortstop/keypose_reach.py's v1 used for its own approximate Reach
step, now scoped to "fake planner for structural testing" instead) so the
shield pipeline can be exercised without either.
"""
import numpy as np

from .keypose_reach import inverse_kinematics_position


def real_get_path(arm, target_position, target_quaternion=None):
    """Wraps PyRep's Arm.get_path(position=..., quaternion=...) and
    returns its `_path_points` reshaped to (N, 7). `arm` is a PyRep
    `Arm` instance (e.g. `Panda`) from a live CoppeliaSim scene -- not
    constructible or testable here. See docs/RLBENCH_SETUP.md.
    """
    try:
        import pyrep  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "real_get_path needs PyRep + a live CoppeliaSim scene -- not available "
            "in this environment. Use mock_get_path for structural testing."
        ) from e
    kwargs = {"position": target_position}
    if target_quaternion is not None:
        kwargs["quaternion"] = target_quaternion
    path = arm.get_path(**kwargs)
    n_joints = len(arm.joints)
    return np.asarray(path._path_points, dtype=float).reshape(-1, n_joints)


def mock_get_path(joint_angles, target_position, n_waypoints=6):
    """Synthetic stand-in matching real_get_path's (N, 7) output shape:
    damped-least-squares IK to a joint config achieving `target_position`
    (position only, no orientation -- see
    shortstop.keypose_reach.inverse_kinematics_position), then linear
    interpolation in joint-space. A real sampling-based planner is not
    obligated to move anywhere near this path -- this exists purely to
    give the Stage 7c shield pipeline *something* path-shaped to certify
    against without PyRep installed.
    """
    q0 = np.asarray(joint_angles, dtype=float)
    q_target = inverse_kinematics_position(target_position, q0)
    alphas = np.linspace(0.0, 1.0, n_waypoints)
    return np.array([q0 + alpha * (q_target - q0) for alpha in alphas])
