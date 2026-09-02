import numpy as np

from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk
from shortstop.robot_geometry import N_JOINTS, SPHERE_NAMES, panda_frames


def test_obstacle_sits_at_the_reference_chunk_gripper_endpoint():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    chunk = np.zeros((4, 7))
    chunk[:, 0] = 0.02  # steady +x end-effector position delta each step

    obstacle = sample_obstacle_from_reference_chunk(q0, chunk, radius=0.05)

    # endpoint should have moved away from the start along +x
    assert obstacle.center[0] > current[0]
    assert obstacle.radius == 0.05


def test_obstacle_frame_matches_sphere_centers_frame():
    """The obstacle must live in the same frame sphere_centers() reports,
    since calvin_experiment checks real per-step joint angles against it
    directly with no extra transform -- a chunk that keeps the arm still
    should place the obstacle essentially at the arm's own current pose.
    """
    q0 = np.zeros(N_JOINTS)
    still_chunk = np.zeros((3, 7))
    obstacle = sample_obstacle_from_reference_chunk(q0, still_chunk, radius=0.05, sphere_name=SPHERE_NAMES[-1])
    current_gripper = panda_frames(q0)[-1]
    assert np.allclose(obstacle.center, current_gripper)
