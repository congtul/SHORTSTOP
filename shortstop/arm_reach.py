"""Reach step for the Panda sphere-chain safety geometry (Stage 7a design).

The trained VLA policy (pi0.5 served by openpi) proposes a *task-space*
action chunk: a 6D end-effector pose delta + 1D gripper, per step (see
docs/LIBERO_SETUP.md's I/O contract). The safety geometry
(shortstop/robot_geometry.py) is a chain of 4 spheres positioned by
*joint*-space forward kinematics. To propagate the sphere chain over a
chunk's horizon we need a joint-space trajectory but only have a task-space
action -- so each step, the end-effector's task-space position delta is
turned into an approximate joint delta via the end-effector Jacobian's
pseudo-inverse (standard resolved-rate / differential IK), then every
sphere's position is read off by forward kinematics at the resulting joint
config.

This is an approximation, not a sound reachability bound the way reach.py's
Box propagation is for the 2D point-mass (there, f_hat == f exactly, so
Assumption 1's soundness holds by construction with model_error=0). Two
separate, real sources of error get folded into one inflation term per
sphere here, neither formally bounded:
  - the linearized (Jacobian pseudo-inverse) task-space -> joint-space map
    is only exact at one operating point -- the same caveat
    shortstop.baselines.MPCFilterShield's tangent-plane linearization
    documents;
  - only the position part of the task-space action is fed back into the
    sphere-chain geometry -- rotation and gripper columns are ignored.
Treat this as "enough to exercise the P-R-C-S pipeline shape end-to-end",
not a certified reachtube. Tightening it (an actual Lipschitz bound on the
pseudo-inverse step, or accounting for orientation's effect on sphere
position) is real follow-up work, not attempted here.
"""
import numpy as np

from .reach import Box
from .robot_geometry import SPHERE_NAMES, end_effector_jacobian, sphere_centers
from .stl import step_robustness


def _signed_distance(box, obstacle):
    """Same formula as stl._signed_distance_to_obstacle, kept local rather
    than importing that module's private helper across a module boundary
    for a two-line formula."""
    closest = box.closest_point(obstacle.center)
    return float(np.linalg.norm(closest - obstacle.center) - obstacle.radius)


def _step_joint_config(joint_angles, task_space_delta_pos):
    J = end_effector_jacobian(joint_angles)
    dq = np.linalg.pinv(J) @ task_space_delta_pos
    return joint_angles + dq


def propagate_arm_tube(joint_angles, task_chunk, w_bar, model_error=0.02):
    """Propagate a per-sphere reachtube over a task-space action chunk.

    `task_chunk`: (H, >=3) array; columns 0:3 are the end-effector position
    delta per step (any further columns -- rotation, gripper -- are
    ignored, see module docstring). Returns a list of length H+1: tube[0]
    is the current, exactly-known configuration (zero-width boxes);
    tube[k] (k=1..H) is a dict {sphere_name: Box}.

    `model_error` defaults *nonzero* here, unlike reach.py's Phase-1
    default of 0 -- that 0 was only valid because the 2D point-mass's
    f_hat was the exact true dynamics; the Jacobian pseudo-inverse step
    here is never exact, so claiming model_error=0 would be unearned.
    """
    task_chunk = np.asarray(task_chunk, dtype=float)
    q = np.asarray(joint_angles, dtype=float).copy()
    r = model_error + w_bar

    tube = [{name: Box.point(c) for name, c in zip(SPHERE_NAMES, sphere_centers(q))}]
    for step in task_chunk:
        q = _step_joint_config(q, step[:3])
        centers = sphere_centers(q)
        tube.append({name: Box.point(c).inflate(r) for name, c in zip(SPHERE_NAMES, centers)})
    return tube


def arm_step_robustness(spheres_at_step, obstacles):
    """min over every sphere's own robustness (stl.step_robustness) at one
    timestep -- the arm-geometry generalization of stl.step_robustness."""
    return min(step_robustness(box, obstacles) for box in spheres_at_step.values())


def arm_robustness_to_go(tube, obstacles):
    """Formula (2) generalized over spheres: min over k=1..H and over every
    sphere of inf_{x in sphere_k} dist(x, X_u)."""
    return min(arm_step_robustness(spheres, obstacles) for spheres in tube[1:])


def arm_find_counterexample(tube, obstacles):
    """Formula (3) generalized: worst (step, sphere, obstacle) triple, in
    the same shape as stl.find_counterexample's return dict plus a
    `sphere` key naming which sphere in the chain is the violator."""
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
