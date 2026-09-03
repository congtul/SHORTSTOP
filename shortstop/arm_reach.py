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

Known geometric caveat from inflating boxes by SPHERE_RADIUS (each
sphere's own physical radius, not just disturbance/model_error -- see
propagate_arm_tube's docstring): Box.inflate() grows a box by the same
amount on every axis (an axis-aligned-cube Minkowski sum), not a true
spherical one. If a chunk's per-step displacement is *smaller* than the
total inflation (sphere_radius + w_bar + model_error), an obstacle placed
near one step's position can end up geometrically "inside" the *previous*
step's inflated box too (its closest_point() call just returns the query
point unchanged), making adjacent tube steps indistinguishable in
arm_find_counterexample/arm_step_robustness -- discovered via
tests/test_arm_reach.py needing a larger (0.3, not 0.05) per-step
displacement to keep two steps separable once SPHERE_RADIUS inflation was
added. Not a correctness bug in the sense of missing a real violation
(the box only ever over-approximates, never under-approximates, so a
degenerate match still correctly flags danger) -- but it does mean the
reported *step*/*sphere* of a counterexample can be misleading (an
earlier, uninvolved step) whenever per-step motion is small relative to
SPHERE_RADII + the obstacle's own radius. Keep replan_steps/chunk
granularity coarse enough relative to typical obstacle+sphere radii, or
tighten this to a real spherical Minkowski sum, if this ever matters for
a real run's diagnostics.

KNOWN MISMATCH, not yet fixed: this module (and arm_shield.py's Certify
step built on top of it) still only checks the 4 named sphere_centers()
points (elbow/forearm/wrist/gripper) -- unlike
shortstop.calvin_experiment._clearance's ground-truth check, which was
upgraded to check the full 8-capsule chain (robot_geometry.capsule_segments,
one capsule per link, covering every link's whole length plus the ones
with no named sphere at all: link0/1/2/4/6). Until propagate_arm_tube is
similarly upgraded to propagate capsule (or per-link) reachtubes instead
of 4 point boxes, the shield would certify against a coarser geometry
than what the ground-truth check (and therefore the measured
violation_rate) actually holds it to -- fix this before wiring
ArmRepairShield into a real CALVIN/LIBERO run and trusting a
shielded-vs-unshielded comparison.
"""
import numpy as np

from .reach import Box
from .robot_geometry import SPHERE_NAMES, SPHERE_RADIUS, end_effector_jacobian, sphere_centers
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
    is the current, exactly-known configuration (zero-width boxes, still
    a point -- see below for why that's fine here);
    tube[k] (k=1..H) is a dict {sphere_name: Box}.

    `model_error` defaults *nonzero* here, unlike reach.py's Phase-1
    default of 0 -- that 0 was only valid because the 2D point-mass's
    f_hat was the exact true dynamics; the Jacobian pseudo-inverse step
    here is never exact, so claiming model_error=0 would be unearned.

    Each predicted sphere's box is inflated by `model_error + w_bar +
    SPHERE_RADIUS[name]` -- not just the disturbance/model-error bound.
    A sphere is a physical volume (radius SPHERE_RADIUS[name]), not a
    point: real collision happens at center-distance <= obstacle.radius +
    sphere_radius, so the sphere's own thickness has to inflate the box
    the same way disturbance/model-error already did, or every downstream
    margin/distance check (arm_step_robustness, arm_find_counterexample,
    shortstop.calvin_experiment._clearance) silently treats the arm as a
    zero-thickness skeleton and under-counts real risk. tube[0] is left
    as an exact point (not inflated at all, physical radius included) --
    arm_robustness_to_go/arm_find_counterexample only ever read tube[1:],
    so this is a don't-care, but flagged here so it's not mistaken for an
    oversight if something else ever reads tube[0].
    """
    task_chunk = np.asarray(task_chunk, dtype=float)
    q = np.asarray(joint_angles, dtype=float).copy()
    disturbance_r = model_error + w_bar

    tube = [{name: Box.point(c) for name, c in zip(SPHERE_NAMES, sphere_centers(q))}]
    for step in task_chunk:
        q = _step_joint_config(q, step[:3])
        centers = sphere_centers(q)
        tube.append({
            name: Box.point(c).inflate(disturbance_r + SPHERE_RADIUS[name])
            for name, c in zip(SPHERE_NAMES, centers)
        })
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
