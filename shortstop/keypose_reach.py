"""Reach step for the RLBench keypose interface (Stage 7c) -- v2.

Revised design: Certify now runs on the *real* planner's output path, not
a guessed interpolation. PyRep's `Arm.get_path()`/`get_nonlinear_path()`
(the actual mechanism behind RLBench's "policy predicts a keypose, RLBench
moves the arm there" loop) returns an `ArmConfigurationPath` object holding
a dense sequence of joint configurations (`_path_points`, radians, shape
roughly (n_waypoints, 7)) -- treating "policy + planner" as one black box,
*that* dense joint-config sequence is its output, in the same spirit as
shortstop.arm_reach's dense per-step task-space chunk for LIBERO. This
module's Reach step takes that sequence as given (from whatever produced
it -- see `planner.py` below for the two sources this repo has) rather
than approximating a path itself the way v1 did (linear joint-space
interpolation from a guessed IK solution) -- that approximation is gone;
model_error can shrink back down to shortstop.arm_reach's baseline (0.02)
since "the planner might not move in a straight line" is no longer a
separate, unquantified error source once the real path is in hand.

Important asymmetry this creates for Select/Repair (see
shortstop/keypose_shield.py): PyRep's ArmConfigurationPath has no public
API to edit waypoints in place (`_path_points` is private, `__getitem__`
only slices) -- Repair therefore cannot nudge the path directly the way
shortstop.shield.RepairShield nudges a chunk's array. It must nudge the
*target* keypose and ask the planner for an entirely new path, then
re-certify that. This is architecturally different from every other
shield in this repo, not a simplification -- documented here so it isn't
mistaken for one later.
"""
import numpy as np

from .reach import Box
from .robot_geometry import SPHERE_NAMES, end_effector_jacobian, panda_frames, sphere_centers
from .stl import step_robustness


def _signed_distance(box, obstacle):
    closest = box.closest_point(obstacle.center)
    return float(np.linalg.norm(closest - obstacle.center) - obstacle.radius)


def inverse_kinematics_position(target_position, initial_joint_angles, max_iters=50, damping=0.05, tol=1e-4):
    """Damped-least-squares numerical IK for end-effector *position* only
    (3D) -- orientation is not solved for, same simplification
    shortstop/arm_reach.py's Reach step makes. Used by the mock planner
    (see planner.py) and by anything that needs a rough target joint
    config; NOT used by Certify itself any more (v1 was; see module
    docstring).
    """
    q = np.asarray(initial_joint_angles, dtype=float).copy()
    target_position = np.asarray(target_position, dtype=float)
    for _ in range(max_iters):
        current = panda_frames(q)[-1]
        error = target_position - current
        if np.linalg.norm(error) < tol:
            break
        J = end_effector_jacobian(q)
        A = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, error)
        q = q + dq
    return q


def propagate_path_tube(path_points, w_bar, model_error=0.02):
    """Build a per-sphere reachtube directly from a *given* dense joint-
    config path (real PyRep ArmConfigurationPath waypoints, or the mock
    planner's stand-in -- see planner.py). `path_points`: (N, 7) array,
    path_points[0] is the current, exactly-known configuration.

    model_error defaults to shortstop.arm_reach's baseline (0.02), not v1's
    inflated 0.05 -- the "planner might not move in a straight line"
    error source is gone now that the real path is used directly.
    """
    path_points = np.asarray(path_points, dtype=float)
    r = model_error + w_bar

    tube = [{name: Box.point(c) for name, c in zip(SPHERE_NAMES, sphere_centers(path_points[0]))}]
    for q in path_points[1:]:
        centers = sphere_centers(q)
        tube.append({name: Box.point(c).inflate(r) for name, c in zip(SPHERE_NAMES, centers)})
    return tube


def path_robustness_to_go(tube, obstacles):
    """Formula (2) generalized: min over every waypoint (k=1..N-1) and
    every sphere -- the path-tube analogue of
    shortstop.arm_reach.arm_robustness_to_go."""
    return min(
        step_robustness(box, obstacles)
        for spheres in tube[1:]
        for box in spheres.values()
    )


def path_find_counterexample(tube, obstacles):
    """Formula (3) generalized: worst (waypoint, sphere, obstacle) triple."""
    best = None
    for k in range(1, len(tube)):
        for name, box in tube[k].items():
            for obstacle in obstacles:
                value = _signed_distance(box, obstacle)
                if best is None or value < best["robustness"]:
                    best = {
                        "step": k,
                        "sphere": name,
                        "obstacle": obstacle,
                        "witness": box.closest_point(obstacle.center),
                        "robustness": value,
                    }
    return best
