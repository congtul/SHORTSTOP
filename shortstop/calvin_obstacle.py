"""Privileged, virtual obstacle placement for the CALVIN pipeline (Stage 7b).

The obstacle exists only as a geometric (center, radius) pair checked by
our own sphere-chain code (shortstop.arm_reach/robot_geometry) -- it is
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
robot_geometry.sphere_centers()/panda_frames() already use, since both
the reference chunk's reach-tube and the real per-step joint angles are
expressed in that same frame.
"""
from .arm_reach import propagate_arm_tube
from .env import Obstacle
from .robot_geometry import SPHERE_NAMES


def sample_obstacle_from_reference_chunk(joint_angles, reference_chunk, radius=0.05, sphere_name=None):
    """Place an obstacle at the endpoint of `reference_chunk`'s own
    nominal (w_bar=0, model_error=0) reach-tube -- the same "obstacle at
    wherever a candidate actually goes" pattern already used in
    tests/test_calvin_pipeline_integration.py, generalized into a
    reusable helper for the real eval harness.

    `sphere_name`: which link's tube to sample from -- defaults to the
    last chain link (gripper).
    """
    if sphere_name is None:
        sphere_name = SPHERE_NAMES[-1]
    tube = propagate_arm_tube(joint_angles, reference_chunk, w_bar=0.0, model_error=0.0)
    center = tube[-1][sphere_name].center()
    return Obstacle(center=center, radius=radius)
