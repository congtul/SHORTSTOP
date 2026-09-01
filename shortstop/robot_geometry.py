"""Franka Emika Panda forward kinematics + a sphere-chain safety geometry.

Stage 7a design note: the 2D prototype's "point + circle" abstraction does
not match how real manipulator safety literature represents a robot arm.
ARMTD (Autonomous Reachability-based Manipulator Trajectory Design) and the
"Spherical Forward Occupancy" line of work model the *whole arm* as a chain
of spheres along the links (elbow/forearm/wrist all included, not just the
end-effector) -- because a candidate action can be safe at the gripper while
still swinging the elbow into an obstacle. Sphere/capsule primitives are
popular for the same reason circles were chosen for the 2D prototype:
sphere-to-sphere and sphere-to-point distance are cheap closed-form
calculations, which is what makes a real-time reachability certify step
possible at all.

This module gives just the geometry: modified-DH forward kinematics for the
7 joints, and the fixed set of "key" frames used as sphere centers. It does
not touch the dynamics/action-space question (see shortstop/arm_reach.py for
how a task-space action gets turned into a joint-config update).

DH table: the standard modified-DH (Craig convention) parameters published
for the Panda (matches Franka's own documentation and the values used in
robotics-toolbox-python's Panda model; independently spot-checked via web
search: d1=0.333, d3=0.316, a4=0.0825, a5=-0.0825 agree). VERIFY AGAINST THE
ACTUAL franka_ros / franka_description URDF before relying on this for real
hardware or a real simulator -- this was not run against a live robot.
"""
import numpy as np

# (alpha_{i-1}, a_{i-1}, d_i) for i = 1..7, modified DH (theta_i = joint angle q_i)
PANDA_DH = [
    (0.0, 0.0, 0.333),
    (-np.pi / 2, 0.0, 0.0),
    (np.pi / 2, 0.0, 0.316),
    (np.pi / 2, 0.0825, 0.0),
    (-np.pi / 2, -0.0825, 0.384),
    (np.pi / 2, 0.0, 0.0),
    (np.pi / 2, 0.088, 0.0),
]
FLANGE_OFFSET = (0.0, 0.0, 0.107)  # (alpha, a, d) from joint 7 frame to flange, theta=0

N_JOINTS = 7

# Which frames' origins double as sphere centers for the safety geometry, and
# how big each sphere is (meters). Indices are into the *frame* list returned
# by panda_frames() (0 = base, 1..7 = joint frames, 8 = flange/end-effector).
# 4 spheres along elbow -> forearm -> wrist -> gripper: coarser than a real
# ARMTD-style multi-sphere-per-link model, but each sphere is (loosely) at a
# frame the arm's own state is available at with no extra modeling.
SPHERE_FRAME_INDICES = [3, 5, 7, 8]
SPHERE_RADII = [0.08, 0.08, 0.08, 0.06]  # coarse per-link radius, see caveat below
SPHERE_NAMES = ["elbow", "forearm", "wrist", "gripper"]

# These radii are a placeholder, not measured from the Panda's actual link
# meshes -- real ARMTD-style work fits spheres to the link collision mesh
# (often several per link). Treat this as "good enough to demonstrate the
# pipeline shape", not a validated safety radius; tighten against the real
# URDF's collision geometry before trusting a real run.


def _mdh_transform(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def panda_frames(joint_angles):
    """Return the 9 frame-origin positions (base, 7 joints, flange) as a
    (9, 3) array, given a length-7 array of joint angles (radians).
    """
    joint_angles = np.asarray(joint_angles, dtype=float)
    if joint_angles.shape != (N_JOINTS,):
        raise ValueError(f"expected {N_JOINTS} joint angles, got shape {joint_angles.shape}")

    T = np.eye(4)
    positions = [T[:3, 3].copy()]
    for (alpha, a, d), theta in zip(PANDA_DH, joint_angles):
        T = T @ _mdh_transform(alpha, a, d, theta)
        positions.append(T[:3, 3].copy())
    T = T @ _mdh_transform(*FLANGE_OFFSET, 0.0)
    positions.append(T[:3, 3].copy())
    return np.array(positions)


def sphere_centers(joint_angles):
    """Positions of the safety-geometry sphere centers (4, 3), in the order
    of SPHERE_NAMES."""
    frames = panda_frames(joint_angles)
    return frames[SPHERE_FRAME_INDICES]


def numerical_jacobian(joint_angles, frame_index, eps=1e-6):
    """d(position of frame `frame_index`)/d(joint_angles), a (3, 7) matrix.

    Finite-difference rather than the closed-form geometric Jacobian -- more
    code to get an analytical Jacobian right for a 7-DOF chain than to
    perturb each joint and re-run panda_frames(), and this is not on a
    real-time control loop's critical path for this design pass.
    """
    joint_angles = np.asarray(joint_angles, dtype=float)
    base = panda_frames(joint_angles)[frame_index]
    J = np.zeros((3, N_JOINTS))
    for i in range(N_JOINTS):
        perturbed = joint_angles.copy()
        perturbed[i] += eps
        J[:, i] = (panda_frames(perturbed)[frame_index] - base) / eps
    return J


def end_effector_jacobian(joint_angles):
    """d(flange position)/d(joint_angles), a (3, 7) matrix."""
    return numerical_jacobian(joint_angles, frame_index=len(PANDA_DH) + 1)
