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
from .arm_reach import propagate_arm_tube
from .env import Obstacle
from .robot_geometry import FLANGE_FRAME_INDEX


def sample_obstacle_from_reference_chunk(joint_angles, reference_chunk, radius=0.08, frame_index=None):
    """Place an obstacle at the endpoint of `reference_chunk`'s own
    nominal (w_bar=0, model_error=0) reach-tube -- the same "obstacle at
    wherever a candidate actually goes" pattern already used in
    tests/test_calvin_pipeline_integration.py, generalized into a
    reusable helper for the real eval harness.

    `radius` defaults to 0.08 -- the value chosen after a real radius
    sweep (0.02/0.05/0.08/0.12) on CALVIN, see docs/PARAMETERS_REFERENCE.md
    muc 1's "radius" entry for the full sweep table and reasoning.

    `frame_index`: which panda_frames() point (0..8) to sample from --
    defaults to the flange (FLANGE_FRAME_INDEX), i.e. the end of the
    chain.
    """
    if frame_index is None:
        frame_index = FLANGE_FRAME_INDEX
    tube = propagate_arm_tube(joint_angles, reference_chunk, w_bar=0.0, model_error=0.0)
    center = tube[-1][frame_index].center()
    return Obstacle(center=center, radius=radius)
