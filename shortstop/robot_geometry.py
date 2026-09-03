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
SPHERE_NAMES = ["elbow", "forearm", "wrist", "gripper"]

# Measured from the real collision meshes shipped with mdt_policy's
# calvin_env (mdt_policy/calvin_env/data/franka_panda/meshes/collision/
# {link3,link5,link7,hand}.obj), confirmed to be the right mesh for each
# name by cross-checking every PANDA_DH row against the URDF's own joint
# <origin> values (both agree exactly, e.g. d=0.333/a=0.0825/d=0.384/
# a=0.088/flange 0.107) -- panda_frames()'s positions[3]/[5]/[7] are
# exactly panda_link{3,5,7}'s own origin, and positions[8] (flange)
# coincides in position (not just approximately) with panda_hand's
# origin (panda_hand_joint's <origin> has zero xyz translation, only a
# yaw rotation), so each link's collision mesh is already expressed in
# a local frame centered exactly on our sphere point.
#
# Method: loaded each mesh's vertices, found the principal (longest)
# axis via SVD (a link's mesh is long-and-thin along the arm, not
# spherical), took the *perpendicular* distance from that axis for every
# vertex (this is what separates true cross-sectional thickness from the
# link's length -- naively using raw distance from the origin point
# conflates the two and wildly overestimates, e.g. forearm's raw max
# vertex distance is 0.27m, almost all of which is the link's 0.35m
# length, not its girth), then took the max perpendicular distance
# (safety margin: cover the mesh's full real cross-section, not just a
# typical/median slice) rounded up to the nearest cm:
#   elbow (link3):   max perp radius 0.085m -> 0.09m
#   forearm (link5): max perp radius 0.093m -> 0.10m
#   wrist (link7):   max perp radius 0.054m -> 0.06m
#   gripper (hand):  max perp radius 0.048m -> 0.05m
# Still a real simplification even with a measured number: one sphere
# per link cannot capture a link's full along-axis extent (elbow/
# forearm/wrist links span 0.14-0.35m -- see the perpendicular-radius
# script this was derived from), and links 1/2/4/6 have no sphere of
# their own at all (coarser than a real ARMTD-style multi-sphere-per-link
# model, by design -- see module docstring). Re-derive by rerunning the
# same mesh analysis if SPHERE_FRAME_INDICES/PANDA_DH ever change, or if
# a tighter (multi-sphere-per-link) model is worth the added complexity.
SPHERE_RADII = [0.09, 0.10, 0.06, 0.05]

# name -> physical radius lookup, used by arm_reach.propagate_arm_tube and
# calvin_experiment._clearance to inflate the safety margin by the arm's own
# thickness, not just disturbance/model-error (a sphere is a physical volume,
# not a point -- collision is center-distance <= obstacle.radius + this).
SPHERE_RADIUS = dict(zip(SPHERE_NAMES, SPHERE_RADII))

# Per-link (link0..link7) capsule radius -- same "max perpendicular-to-
# principal-axis" measurement as SPHERE_RADII above (see its docstring),
# just applied to *every* one of the 8 real collision meshes
# (mdt_policy/calvin_env/data/franka_panda/meshes/collision/link{0..7}.obj)
# instead of only the 4 already picked as named sphere points:
#   link0 0.1206->0.13  link1 0.1002->0.11  link2 0.0995->0.10
#   link3 0.0854->0.09  link4 0.0862->0.09  link5 0.0930->0.10
#   link6 0.1082->0.11  link7 0.0538->0.06
# (link3/5/7 match SPHERE_RADII's elbow/forearm/wrist exactly -- same
# source mesh, same method, a sanity check that both measurements agree.)
#
# A capsule from panda_frames()[i] to panda_frames()[i+1] with
# LINK_RADIUS[i] covers link i's *entire* physical span, not just one
# end -- fixing the 4-named-sphere model's blind spot: a collision
# happening partway along a link (elbow/forearm/wrist links are
# 0.14-0.35m long, far longer than their own sphere radius) or on a link
# with no named sphere at all (link0/1/2/4/6) was invisible to
# sphere_centers()-based checks alone. See capsule_segments() below.
# Does NOT cover panda_hand's own bulk beyond the flange (frame 8) --
# that's what SPHERE_RADIUS["gripper"] and GRIPPER_TIP_OFFSET/RADIUS
# below are for, layered on top of this chain by
# calvin_experiment._clearance, not a capsule endpoint here.
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
# for why this region needs its own check, not just SPHERE_RADIUS
# ["gripper"]'s point sphere at the flange). finger.obj's own measured
# cross-section (same method as LINK_RADIUS) is a mere 0.019m -- but the
# two fingers *also* spread apart laterally up to 0.04m each side when
# open (panda_finger_joint1/2's prismatic limit in panda.urdf), and this
# capsule doesn't model gripper open/close state at all (a single fixed
# radius along the centerline instead of two separate finger capsules
# that move with the gripper's actual width). 0.04 (max one-sided finger
# spread) + 0.02 (finger's own thickness, rounded up from 0.019) = 0.06,
# a deliberately conservative fixed bound for "gripper anywhere from
# fully closed to fully open", not a precise measurement the way
# LINK_RADIUS/SPHERE_RADII are.
GRIPPER_TIP_RADIUS = 0.06


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


def sphere_centers(joint_angles):
    """Positions of the safety-geometry sphere centers (4, 3), in the order
    of SPHERE_NAMES."""
    frames = panda_frames(joint_angles)
    return frames[SPHERE_FRAME_INDICES]


def capsule_segments(joint_angles):
    """8 capsules covering the *entire* arm (link0..link7), each as
    (point_a, point_b, radius): point_a/point_b are consecutive
    panda_frames() positions (frame i, frame i+1), radius is that link's
    own LINK_RADIUS[i] -- see LINK_RADIUS's docstring for why this (not
    sphere_centers()'s 4 named points) is the check to use whenever a
    collision could occur anywhere along a link's length, not just at
    one of its two endpoints, or on a link with no named sphere at all.
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
