"""Privileged, virtual obstacle placement for the CALVIN pipeline (Stage 7b).

The obstacle exists only as a geometric (center, radius) pair checked by
our own capsule-chain code (shortstop.arm_reach/robot_geometry) -- it is
never spawned in the PyBullet scene and never rendered into the camera
images MDT's vision encoder sees. This matches the paper's own premise
("Policy không cần biết safety mechanism tồn tại ở phía sau" --
report/ShortStop_Report_1.tex) and avoids a vision-domain-shift confound
(see docs/STAGE7B_CALVIN_PIPELINE_DESIGN.md's "Quyết định thiết kế cho
X_u").

Placement: sampled from a reference candidate chunk's own (nominal,
noise-free) reach-tube -- a point the arm actually would sweep through for
this episode's current joint configuration -- rather than a fixed world
position. This needs no knowledge of CALVIN's robot-base-to-world
transform: the obstacle lives entirely in the same robot-base frame
robot_geometry.panda_frames() already uses, since both the reference
chunk's reach-tube and the real per-step joint angles are expressed in
that same frame.
"""
import numpy as np

from .arm_reach import propagate_arm_tube
from .env import Obstacle
from .robot_geometry import FLANGE_FRAME_INDEX


def _perpendicular_basis(direction_unit):
    """Two unit vectors spanning the plane perpendicular to
    `direction_unit` -- used to draw a random offset that's guaranteed
    orthogonal to the direction of travel, not just "some other
    direction". Picks a reference axis not near-parallel to
    `direction_unit` (world Z, falling back to world X) so the cross
    product is never near-degenerate."""
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction_unit, reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    u = np.cross(direction_unit, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(direction_unit, u)
    v = v / np.linalg.norm(v)
    return u, v


def sample_obstacle_from_reference_chunk(
    joint_angles, reference_chunk, radius=0.06, frame_index=None,
    horizon_multiplier=2, offset_max=0.6, rng=None,
):
    """Place an obstacle near (not exactly at) the endpoint of
    `reference_chunk`'s own nominal (w_bar=0, model_error=0) reach-tube.

    FIXED 2026-09-05 -- the original version of this function (exact
    endpoint, no offset, H = len(reference_chunk) = 1 replan window) was
    found, via a real re-sweep after the CALVIN_ACTION_SCALE fix, to
    place the obstacle EXACTLY on the deterministic centerline the arm's
    own primitives (finger capsules especially, which extend
    GRIPPER_TIP_OFFSET past the flange along the direction of travel)
    are predicted to sweep through -- a real run showed 100% of subtask-1
    attempts violating within 1-8 real steps, IDENTICALLY at radius=0.0
    and radius=0.02 (obstacle.radius was irrelevant; the arm's own
    capsule geometry alone did all the "capturing"). Two independent,
    deliberately-separate mechanisms fix this (see docs/PARAMETERS_
    REFERENCE.md's "radius" entry for the full writeup):

    1. `horizon_multiplier` (default 2, still an unverified placeholder --
       not yet independently swept, see "How to apply" in the
       calvin_obstacle_offset_floor_too_high memory): the reachtube is
       propagated over `reference_chunk` TILED this many times
       (`np.tile`), not the raw chunk alone -- extends how far along the
       SAME general direction the target sits, giving more real steps of
       runway before the arm could possibly reach it (directly answers
       "does the arm get a real chance to move" -- see
       calvin_experiment.run_calvin_unshielded_subtask's own
       `steps_taken` diagnostic). Reuses propagate_arm_tube's own
       per-step re-linearization unchanged (each tiled repetition still
       re-computes the Jacobian from the updated joint config, not a
       stale one) -- self-scales to whatever this chunk's own real pace
       is, rather than guessing an absolute extra distance.
    2. `offset_max` + `rng` (default `offset_max=0.6` -- CONFIRMED
       2026-09-06 by a real sweep, see docs/PARAMETERS_REFERENCE.md's
       "radius" entry: this value made the r=0.0/point-obstacle floor
       drop from 0.522 to 0.269 attempted-subtask violation fraction,
       within 1pp of the value simple scaling predicts, confirming this
       is a clean, well-behaved lever, not noise; `rng=None` skips this
       entirely, keeping the OLD exact-endpoint behavior for any caller
       not yet passing an `rng`): a random point offset from the
       endpoint, in the plane PERPENDICULAR to the direction of travel,
       magnitude drawn Uniform(0, offset_max) -- makes `radius` a
       meaningful difficulty knob again (a point placed exactly on the
       centerline is swept regardless of `radius`'s own value, since the
       arm's own primitive thickness alone already exceeds most tested
       radii; a random perpendicular miss distance is what `radius` can
       then meaningfully compete against). Deliberately NOT sourced from
       `model_error`/`w_bar` -- those are the shield's own calibrated,
       deliberately-tight uncertainty budget, not a benchmark-difficulty
       generator; conflating them would mean a future recalibration
       silently changes this benchmark's own difficulty. `rng` must be an
       explicit `numpy.random.Generator` the CALLER seeds reproducibly
       (e.g. per-sequence, so a radius sweep compares the same random
       placement across every radius) -- deliberately NOT drawn from
       numpy's global RNG state, which `pytorch_lightning.seed_everything`
       already manages for the policy's own diffusion noise; consuming
       global-RNG draws here would desync that stream and break the
       "with vs without obstacle follows the identical trajectory"
       property run_calvin_unshielded_subtask's own docstring relies on.

    **IMPORTANT -- every CALVIN driver script must pass the SAME
    `radius`/`offset_max` explicitly** (each already declares its own
    `OBSTACLE_RADIUS`/`OBSTACLE_OFFSET_MAX` module constant rather than
    relying on these function defaults) -- a script that silently relies
    on this function's own default while another script overrides it is
    comparing two DIFFERENT obstacle difficulties, not the same benchmark
    under a different shield (found 2026-09-06: run_calvin_unshielded.py
    had already moved to offset_max=0.6 while every shielded baseline
    script was still silently getting 0.3 from this default -- fixed by
    adding the same explicit constant everywhere).

    `radius` defaults to 0.06 -- REVISED 2026-09-06 (same day) from an
    initial 0.08, once ArmRepairShield's own single-shot trust_region=
    0.05m escape budget made 0.08's capture zone look too large to
    reliably repair around -- see docs/PARAMETERS_REFERENCE.md muc 1's
    "radius" entry for the full sweep table and the trust_region
    reasoning.

    `frame_index`: which panda_frames() point (0..8) to sample from --
    defaults to the flange (FLANGE_FRAME_INDEX), i.e. the end of the
    chain.
    """
    if frame_index is None:
        frame_index = FLANGE_FRAME_INDEX
    reference_chunk = np.asarray(reference_chunk, dtype=float)
    extended_chunk = np.tile(reference_chunk, (horizon_multiplier, 1))
    tube = propagate_arm_tube(joint_angles, extended_chunk, w_bar=0.0, model_error=0.0)
    start = tube[0][frame_index].center()
    end = tube[-1][frame_index].center()

    center = end
    if rng is not None:
        direction = end - start
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-9:
            direction_unit = direction / direction_norm
            u, v = _perpendicular_basis(direction_unit)
            theta = rng.uniform(0.0, 2 * np.pi)
            magnitude = rng.uniform(0.0, offset_max)
            center = end + magnitude * (np.cos(theta) * u + np.sin(theta) * v)

    return Obstacle(center=center, radius=radius)
