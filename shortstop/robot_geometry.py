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
7 joints (panda_frames(), 9 positions: base, 7 joints, flange), and the
capsule-chain safety geometry built from those frames (capsule_segments(),
one capsule per link, covering the whole arm -- no coarser named subset of
frames is exposed here anymore; both the ground-truth check
(calvin_experiment._clearance) and the shield's own reachtube
(arm_reach.propagate_arm_tube) check the full chain). Does not touch the
dynamics/action-space question (see shortstop/arm_reach.py for how a
task-space action gets turned into a joint-config update).

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

# FLANGE_FRAME_INDEX names the last entry of panda_frames() (index 8 =
# flange/end-effector) -- used wherever code needs "the tip of the chain"
# without a magic number (e.g. calvin_obstacle.sample_obstacle_from_
# reference_chunk's default placement, or a test's "where does the
# gripper end up" check).
FLANGE_FRAME_INDEX = N_JOINTS + 1

# Per-link (link0..link7) capsule radius, measured from the real collision
# meshes shipped with mdt_policy's calvin_env
# (mdt_policy/calvin_env/data/franka_panda/meshes/collision/link{0..7}.obj),
# confirmed to be the right mesh for each link by cross-checking every
# PANDA_DH row against the URDF's own joint <origin> values (both agree
# exactly, e.g. d=0.333/a=0.0825/d=0.384/a=0.088/flange 0.107) -- panda_
# frames()'s positions[i] are exactly panda_link{i}'s own origin, so each
# link's collision mesh is already expressed in a local frame centered
# exactly on that frame's point.
#
# Method: loaded each mesh's vertices, found the principal (longest) axis
# via SVD (a link's mesh is long-and-thin along the arm, not spherical),
# took the *perpendicular* distance from that axis for every vertex (this
# is what separates true cross-sectional thickness from the link's length
# -- naively using raw distance from the origin point conflates the two
# and wildly overestimates, e.g. forearm's raw max vertex distance is
# 0.27m, almost all of which is the link's 0.35m length, not its girth),
# then took the max perpendicular distance (safety margin: cover the
# mesh's full real cross-section, not just a typical/median slice)
# rounded up to the nearest cm:
#   link0 0.1206->0.13  link1 0.1002->0.11  link2 0.0995->0.10
#   link3 0.0854->0.09  link4 0.0862->0.09  link5 0.0930->0.10
#   link6 0.1082->0.11  link7 0.0538->0.06
#
# A capsule from panda_frames()[i] to panda_frames()[i+1] with
# LINK_RADIUS[i] covers link i's *entire* physical span, not just one
# end -- a single point-sphere at each endpoint alone would miss a
# collision happening partway along a link (elbow/forearm/wrist links are
# 0.14-0.35m long, far longer than their own radius). See
# capsule_segments() below. Does NOT cover panda_hand's own bulk beyond
# the flange (frame 8) -- that's what GRIPPER_TIP_OFFSET/RADIUS below are
# for, layered on top of this chain by calvin_experiment._clearance, not
# a capsule endpoint here.
LINK_RADIUS = [0.13, 0.11, 0.10, 0.09, 0.09, 0.10, 0.11, 0.06]

# panda.urdf's fixed joint chain past the flange: panda_hand_joint
# (rotation only, zero translation -- panda_hand's origin coincides
# exactly with the flange) then tcp_joint (translate (0,0,0.1) along
# panda_hand's own z-axis). A translation purely along an axis is
# unaffected by any rotation *about* that same axis, so the yaw values
# in both fixed joints drop out of the *position* math entirely -- the
# TCP sits GRIPPER_TIP_OFFSET further out from the flange, along
# whichever direction the flange's own z-axis already points. See
# gripper_tip_position().
GRIPPER_TIP_OFFSET = 0.1

# Capsule radius for the flange->TCP segment (fingers + the part of
# panda_hand beyond the flange -- see gripper_tip_position()'s docstring
# for why this region needs its own check, not just a point sphere at
# the flange itself). finger.obj's own measured cross-section (same
# method as LINK_RADIUS) is a mere 0.019m -- but the two fingers *also*
# spread apart laterally up to 0.04m each side when open (panda_finger_
# joint1/2's prismatic limit in panda.urdf), and this capsule doesn't
# model gripper open/close state at all (a single fixed radius along the
# centerline instead of two separate finger capsules that move with the
# gripper's actual width). 0.04 (max one-sided finger spread) + 0.02
# (finger's own thickness, rounded up from 0.019) = 0.06, a deliberately
# conservative fixed bound for "gripper anywhere from fully closed to
# fully open", not a precise measurement the way LINK_RADIUS is.
GRIPPER_TIP_RADIUS = 0.06

# Conservative per-*frame* radius (index 0..8, matching panda_frames()),
# used by arm_reach.propagate_arm_tube to inflate a reachtube box at
# *every* frame of the chain -- replaces the coarser 4-named-point model
# (elbow/forearm/wrist/gripper) this module used to expose, so the
# shield's own Certify step now covers the same 9-frame chain as
# calvin_experiment._clearance's ground-truth capsule check, instead of
# a 4-point subset of it.
#
# A frame sits at the junction of up to two links, so its own physical
# extent is bounded by whichever adjacent link is thicker: frame 0
# (base) only touches link0; frame i (1<=i<=7) touches link(i-1) and
# link(i); frame 8 (flange) touches link7 *and* must also cover the
# fingers reaching past it (GRIPPER_TIP_OFFSET + GRIPPER_TIP_RADIUS) --
# propagate_arm_tube has no orientation tracking (see its module
# docstring), so it cannot propagate the fingertip as its own point the
# way calvin_experiment._clearance does with gripper_tip_position();
# this folds the entire flange->fingertip reach into one conservative
# sphere centered on the flange point instead.
FRAME_RADIUS = (
    [LINK_RADIUS[0]]
    + [max(LINK_RADIUS[i - 1], LINK_RADIUS[i]) for i in range(1, len(LINK_RADIUS))]
    + [GRIPPER_TIP_OFFSET + GRIPPER_TIP_RADIUS]
)


def _mdh_transform(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _panda_transforms(joint_angles):
    """Same 9 frames as panda_frames() (base, 7 joints, flange), but as
    full 4x4 homogeneous transforms (position *and* orientation) instead
    of just positions -- panda_frames() itself only ever needed the
    position column, but gripper_tip_position() needs the flange's
    orientation too, to project the TCP offset in the right direction.
    """
    joint_angles = np.asarray(joint_angles, dtype=float)
    if joint_angles.shape != (N_JOINTS,):
        raise ValueError(f"expected {N_JOINTS} joint angles, got shape {joint_angles.shape}")

    T = np.eye(4)
    transforms = [T.copy()]
    for (alpha, a, d), theta in zip(PANDA_DH, joint_angles):
        T = T @ _mdh_transform(alpha, a, d, theta)
        transforms.append(T.copy())
    T = T @ _mdh_transform(*FLANGE_OFFSET, 0.0)
    transforms.append(T.copy())
    return transforms


def panda_frames(joint_angles):
    """Return the 9 frame-origin positions (base, 7 joints, flange) as a
    (9, 3) array, given a length-7 array of joint angles (radians).
    """
    return np.array([T[:3, 3] for T in _panda_transforms(joint_angles)])


def gripper_tip_position(joint_angles):
    """Position of the gripper's TCP (tool-center-point, roughly at the
    fingertips) -- GRIPPER_TIP_OFFSET beyond the flange, along the
    flange's own pointing direction (its local z-axis). See
    GRIPPER_TIP_OFFSET's docstring for why the exact intermediate
    rotation values (panda_hand_joint/tcp_joint's yaw) don't matter here.

    Exists so the safety geometry's gripper coverage can extend past the
    flange (a capsule from panda_frames()[-1] to this point, radius
    GRIPPER_TIP_RADIUS) to the fingers -- without it, a collision at the
    fingertips (the part that actually contacts objects during a grasp)
    could occur well before the flange-centered point alone would detect
    it, since the fingers point *ahead* of the flange in the direction
    of travel, not at it.
    """
    flange_T = _panda_transforms(joint_angles)[-1]
    flange_position = flange_T[:3, 3]
    flange_rotation = flange_T[:3, :3]
    return flange_position + flange_rotation @ np.array([0.0, 0.0, GRIPPER_TIP_OFFSET])


def capsule_segments(joint_angles):
    """8 capsules covering the *entire* arm (link0..link7), each as
    (point_a, point_b, radius): point_a/point_b are consecutive
    panda_frames() positions (frame i, frame i+1), radius is that link's
    own LINK_RADIUS[i] -- this is the check to use whenever a collision
    could occur anywhere along a link's length, not just at one of its
    two endpoints.
    """
    frames = panda_frames(joint_angles)
    return [(frames[i], frames[i + 1], LINK_RADIUS[i]) for i in range(len(LINK_RADIUS))]


def point_to_segment_distance(point, a, b):
    """Closest distance from `point` to the line segment a-b -- the
    standard closed-form point-to-capsule-axis primitive (project onto
    the segment, clamp to its ends), same cost class as point-to-sphere.
    Subtract the capsule's own radius (and, for an obstacle, its radius
    too) from the result the same way a plain point-to-sphere-center
    distance already gets radii subtracted elsewhere in this codebase --
    this function returns the raw center-line distance, not a clearance.
    """
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq < 1e-12:  # degenerate zero-length segment -> point-to-point
        return float(np.linalg.norm(point - a))
    t = np.clip(float((point - a) @ ab) / length_sq, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


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
