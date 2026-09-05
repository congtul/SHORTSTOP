"""Reach step for the Panda sphere-chain safety geometry (Stage 7a design).

The trained VLA policy (pi0.5 served by openpi) proposes a *task-space*
action chunk: a 6D end-effector pose delta + 1D gripper, per step (see
docs/LIBERO_SETUP.md's I/O contract). The safety geometry
(shortstop/robot_geometry.py) is a chain of capsules positioned by
*joint*-space forward kinematics. To propagate the chain over a chunk's
horizon we need a joint-space trajectory but only have a task-space
action -- so each step, the end-effector's task-space position delta is
turned into an approximate joint delta via the end-effector Jacobian's
pseudo-inverse (standard resolved-rate / differential IK), then every
frame's position is read off by forward kinematics at the resulting joint
config.

This is an approximation, not a sound reachability bound the way reach.py's
Box propagation is for the 2D point-mass (there, f_hat == f exactly, so
Assumption 1's soundness holds by construction with model_error=0). Two
separate, real sources of error get folded into one inflation term per
frame here, neither formally bounded:
  - the linearized (Jacobian pseudo-inverse) task-space -> joint-space map
    is only exact at one operating point -- the same caveat
    shortstop.baselines.MPCFilterShield's tangent-plane linearization
    documents;
  - only the position part of the task-space action is fed back into the
    chain's geometry -- rotation and gripper columns are ignored.
Treat this as "enough to exercise the P-R-C-S pipeline shape end-to-end",
not a certified reachtube. Tightening it (an actual Lipschitz bound on the
pseudo-inverse step, or accounting for orientation's effect on frame
position) is real follow-up work, not attempted here.

RESOLVED (was "KNOWN MISMATCH" in earlier revisions of this module): this
module (and arm_shield.py's Certify step built on top of it) used to check
only the 4 named sphere_centers() points (elbow/forearm/wrist/gripper),
coarser than shortstop.calvin_experiment._clearance's ground-truth check
(the full capsule_segments() chain, one capsule per link, covering every
link's whole length). propagate_arm_tube now certifies every one of
panda_frames()'s 9 points (base through flange) AND every one of
capsule_segments()'s 8 links -- the same chain the ground-truth check
connects.

RESOLVED (2026-09-05, was "Box.inflate() grows a box, not a sphere" /
"per-point-box, not capsule-vs-capsule"): this module used to represent
every predicted frame/link as an axis-aligned `Box` (shortstop.reach.Box),
inflated by `disturbance_r + physical_radius` on every axis -- a cube
Minkowski sum, not the spherical one the math actually calls for, AND (for
links) only a coarse bounding box of the two endpoint boxes, not an exact
capsule. Both replaced by a single `Capsule` primitive (below): an exact
line segment (a==b for a point/frame primitive) plus a scalar radius,
with distance computed via robot_geometry.point_to_segment_distance/
closest_point_on_segment -- the SAME exact formula shortstop.calvin_
experiment._clearance's ground-truth check already uses. This is no
longer an over-approximation at all for the *shape* of the uncertainty
region (a true sphere/capsule, not a cube or its bounding box) -- the only
remaining approximation is the Jacobian-linearization source documented
above (a real, different, and still-open issue), not the geometry's own
shape.

RESOLVED (2026-09-05, a real, previously-undiscovered scale-mismatch bug,
found via scripts/calibrate_arm_model_error.py's first real run --
residuals averaging 0.32m, p99=0.74m, absurd for a single real step):
`_step_joint_config` used to treat `task_space_delta_pos` (a raw task
chunk's position columns, exactly as CALVIN's own policy/harness produces
them) as already being a real Cartesian delta in meters. It is NOT, for
CALVIN -- confirmed from the real source (`mdt_policy/calvin_env/
calvin_env/robot/robot.py::Robot.relative_to_absolute`): the real env
scales a raw action's position columns by `max_rel_pos` (0.02, the
constructor default -- confirmed NOT overridden by `mdt_policy/calvin_env/
conf/robot/panda.yaml`, which only sets `magic_scaling_factor_pos: 1`)
before applying it. Every caller in this module was therefore predicting
the arm moves roughly 1/0.02 = 50x farther per step than it actually does
-- exactly the residual magnitude measured. Fixed via `CALVIN_ACTION_SCALE`
below, applied inside `_step_joint_config` (the one place every other
function in this module -- `propagate_arm_tube`, `nominal_joint_trajectory`,
`step_prediction_residual` -- funnels through).

CAVEAT this fix introduces: `CALVIN_ACTION_SCALE` is CALVIN-specific (this
module's own original docstring above describes a *generic*, benchmark-
agnostic design shared with a future LIBERO/pi0.5 integration -- see
docs/LIBERO_SETUP.md's I/O contract). Baking a CALVIN-only constant in
here trades that generality for a single, non-scattered fix location (a
deliberate choice, not an oversight -- the alternative was threading an
`action_scale` parameter through every call site across calvin_obstacle.py/
calvin_progress.py/arm_shield.py/calvin_experiment.py, judged more error-
prone to miss a spot than centralizing it here). Before ever wiring
LIBERO/pi0.5 into this same module, VERIFY whether pi0.5's own served
action already IS a real Cartesian delta (no analogous scaling step) --
if so, this constant must become a real parameter (defaulting differently
per benchmark), not silently reused as-is.
"""
import numpy as np

from .robot_geometry import (
    FRAME_RADIUS, LINK_RADIUS, closest_point_on_segment, end_effector_jacobian, panda_frames,
)

# CALVIN's own raw-action -> real-Cartesian-delta scale (see the module
# docstring's "RESOLVED (2026-09-05, a real ... scale-mismatch bug)" entry
# for the full derivation) -- confirmed from `mdt_policy/calvin_env/
# calvin_env/robot/robot.py::Robot.__init__`'s own `max_rel_pos=0.02`
# constructor default, times `magic_scaling_factor_pos` (confirmed = 1,
# `mdt_policy/calvin_env/conf/robot/panda.yaml`, not overridden anywhere
# else in this repo's own config/patch).
CALVIN_ACTION_SCALE = 0.02


class Capsule:
    """A 3D line segment (a, b) with a scalar radius -- this module's own
    reachtube primitive, replacing shortstop.reach.Box (see module
    docstring for why). `a == b` represents a point/frame primitive
    (matching robot_geometry.point_to_segment_distance's own degenerate
    handling); `a != b` represents a link/capsule primitive. `radius`
    already includes every source of inflation (physical radius +
    disturbance/model_error) added together -- a single scalar subtracted
    at distance-computation time, never baked into the segment's own
    shape, so the distance math is exact sphere/capsule geometry rather
    than an axis-aligned-box approximation of it."""
    __slots__ = ("a", "b", "radius")

    def __init__(self, a, b, radius):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.radius = radius

    def center(self):
        return (self.a + self.b) / 2.0

    def closest_point(self, p):
        return closest_point_on_segment(p, self.a, self.b)


def _signed_distance(capsule, obstacle):
    """Exact capsule-vs-sphere signed distance: closest point on the
    capsule's own axis to the obstacle center, minus BOTH radii (the
    capsule's own physical+disturbance radius, and the obstacle's)."""
    closest = capsule.closest_point(obstacle.center)
    return float(np.linalg.norm(closest - obstacle.center) - capsule.radius - obstacle.radius)


def _step_joint_config(joint_angles, task_space_delta_pos):
    J = end_effector_jacobian(joint_angles)
    dq = np.linalg.pinv(J) @ (task_space_delta_pos * CALVIN_ACTION_SCALE)
    return joint_angles + dq


def step_prediction_residual(joint_angles, action, next_joint_angles):
    """Cartesian (task-space) residual between the REAL next joint config
    (from one actually-executed step) and the NOMINAL one
    _step_joint_config would have predicted for the same action -- the
    raw ingredient for calibrating `model_error` (see scripts/
    calibrate_arm_model_error.py): max per-frame Euclidean distance
    between panda_frames(next_joint_angles) and panda_frames(predicted),
    dimensionally consistent with how model_error is actually used (a
    Cartesian capsule-inflation radius in meters -- Capsule's own
    `radius`, see propagate_arm_tube -- not a joint-space quantity).

    `action`: the actual task-space position delta (any array whose first
    3 columns are the end-effector position delta, e.g. one row of a task
    chunk) that produced `next_joint_angles` from `joint_angles` in a real
    rollout. `model_error` should be set to a high quantile (e.g. 0.99) of
    this residual over many real steps, times a safety factor -- the same
    recipe as shortstop.calibration.calibrate_w_bar, just measuring a
    different (kinematic-linearization, not disturbance) error source."""
    action = np.asarray(action, dtype=float)
    predicted = _step_joint_config(np.asarray(joint_angles, dtype=float), action[:3])
    predicted_frames = panda_frames(predicted)
    real_frames = panda_frames(np.asarray(next_joint_angles, dtype=float))
    return float(np.max(np.linalg.norm(real_frames - predicted_frames, axis=1)))


def nominal_joint_trajectory(joint_angles, task_chunk):
    """The nominal (undisturbed) joint-config trajectory _step_joint_config
    predicts for a task-space action chunk -- the same Jacobian-pseudo-
    inverse stepping propagate_arm_tube uses internally, exposed
    separately for callers that only need joint angles (e.g. a
    joint-limit check) without building a full reachtube. Returns a list
    of length H+1: trajectory[0] is the current config, trajectory[k]
    (k=1..H) is the nominal config after k steps."""
    task_chunk = np.asarray(task_chunk, dtype=float)
    q = np.asarray(joint_angles, dtype=float).copy()
    trajectory = [q.copy()]
    for step in task_chunk:
        q = _step_joint_config(q, step[:3])
        trajectory.append(q.copy())
    return trajectory


def propagate_arm_tube(joint_angles, task_chunk, w_bar, model_error=0.02):
    """Propagate a per-frame + per-link reachtube over a task-space action
    chunk.

    `task_chunk`: (H, >=3) array; columns 0:3 are the end-effector position
    delta per step (any further columns -- rotation, gripper -- are
    ignored, see module docstring). Returns a list of length H+1: tube[0]
    is the current, exactly-known configuration (zero-radius Capsules,
    still exact points -- see below for why that's fine here);
    tube[k] (k=1..H) is a dict keyed by BOTH:
      - frame_index (int, 0..8, matching robot_geometry.panda_frames()'s
        order: base, 7 joints, flange) -> that frame's own point Capsule
        (a==b), and
      - ("link", i) for i in 0..7 (matching robot_geometry.
        capsule_segments()'s link ordering) -> that link's own exact
        Capsule between its two endpoint frames.
    Consumers that just iterate `.values()`/`.items()` (arm_step_robustness,
    arm_find_counterexample) automatically cover both without change.

    `model_error` defaults *nonzero* here, unlike reach.py's Phase-1
    default of 0 -- that 0 was only valid because the 2D point-mass's
    f_hat was the exact true dynamics; the Jacobian pseudo-inverse step
    here is never exact, so claiming model_error=0 would be unearned.

    Each predicted frame's Capsule radius is `model_error + w_bar +
    FRAME_RADIUS[i]` -- not just the disturbance/model-error bound. A
    frame's own physical extent (FRAME_RADIUS[i], see its docstring) is
    not a point: real collision happens at center-distance <=
    obstacle.radius + frame_radius, so this thickness has to inflate the
    radius the same way disturbance/model-error already did, or every
    downstream margin/distance check (arm_step_robustness,
    arm_find_counterexample, shortstop.calvin_experiment._clearance)
    silently treats the arm as a zero-thickness skeleton and under-counts
    real risk. Each link's Capsule radius is `model_error + w_bar +
    LINK_RADIUS[i]` (that link's own true physical radius, not
    FRAME_RADIUS[i]'s max-of-neighbors heuristic -- this Capsule
    represents the whole link's exact swept segment, not a junction
    point). tube[0] is left at radius 0 (not inflated at all, no link
    Capsules either) -- arm_robustness_to_go/arm_find_counterexample only
    ever read tube[1:], so this is a don't-care, but flagged here so it's
    not mistaken for an oversight if something else ever reads tube[0].
    """
    task_chunk = np.asarray(task_chunk, dtype=float)
    q = np.asarray(joint_angles, dtype=float).copy()
    disturbance_r = model_error + w_bar

    tube = [{i: Capsule(c, c, 0.0) for i, c in enumerate(panda_frames(q))}]
    for step in task_chunk:
        q = _step_joint_config(q, step[:3])
        frames = panda_frames(q)
        step_capsules = {
            i: Capsule(c, c, disturbance_r + FRAME_RADIUS[i])
            for i, c in enumerate(frames)
        }
        step_capsules.update({
            ("link", i): Capsule(frames[i], frames[i + 1], disturbance_r + LINK_RADIUS[i])
            for i in range(len(LINK_RADIUS))
        })
        tube.append(step_capsules)
    return tube


def arm_step_robustness(frames_at_step, obstacles):
    """min over every Capsule's own robustness at one timestep -- the
    arm-geometry generalization of stl.step_robustness (not reused
    directly: that one is Box-specific, see module docstring on why this
    module now has its own exact Capsule-vs-sphere primitive instead).
    `frames_at_step` is one tube[k] dict, so this covers both the 9
    per-frame Capsules AND the 8 per-link Capsules (see
    propagate_arm_tube), just by iterating every value regardless of key
    type."""
    return min(_signed_distance(capsule, o) for capsule in frames_at_step.values() for o in obstacles)


def arm_robustness_to_go(tube, obstacles):
    """Formula (2) generalized over frames+links: min over k=1..H and over
    every frame/link Capsule of inf_{x in capsule} dist(x, X_u)."""
    return min(arm_step_robustness(frames, obstacles) for frames in tube[1:])


def arm_find_counterexample(tube, obstacles):
    """Formula (3) generalized: worst (step, frame-or-link, obstacle)
    triple, in the same shape as stl.find_counterexample's return dict
    plus a `frame` key naming which primitive is the violator -- either a
    plain int (0..8, a panda_frames() frame index) or a `("link", i)`
    tuple (0..7, a capsule_segments() link, see propagate_arm_tube)."""
    best = None
    for k in range(1, len(tube)):
        for frame_index, capsule in tube[k].items():
            for obstacle in obstacles:
                value = _signed_distance(capsule, obstacle)
                if best is None or value < best["robustness"]:
                    best = {
                        "step": k,
                        "frame": frame_index,
                        "obstacle": obstacle,
                        "witness": capsule.closest_point(obstacle.center),
                        "robustness": value,
                    }
    return best
