import numpy as np

from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk
from shortstop.robot_geometry import FLANGE_FRAME_INDEX, N_JOINTS, panda_frames


def test_obstacle_sits_at_the_reference_chunk_gripper_endpoint():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    chunk = np.zeros((4, 7))
    chunk[:, 0] = 0.02  # steady +x end-effector position delta each step

    obstacle = sample_obstacle_from_reference_chunk(q0, chunk, radius=0.05)

    # endpoint should have moved away from the start along +x
    assert obstacle.center[0] > current[0]
    assert obstacle.radius == 0.05


def test_obstacle_frame_matches_panda_frames_frame():
    """The obstacle must live in the same frame panda_frames() reports,
    since calvin_experiment checks real per-step joint angles against it
    directly with no extra transform -- a chunk that keeps the arm still
    should place the obstacle essentially at the arm's own current pose.
    """
    q0 = np.zeros(N_JOINTS)
    still_chunk = np.zeros((3, 7))
    obstacle = sample_obstacle_from_reference_chunk(q0, still_chunk, radius=0.05, frame_index=FLANGE_FRAME_INDEX)
    current_gripper = panda_frames(q0)[-1]
    assert np.allclose(obstacle.center, current_gripper)


def test_horizon_multiplier_extends_the_endpoint_further_along_the_same_direction():
    """Regression test for the 2026-09-05 fix (see this function's own
    docstring): a real re-sweep found the OLD exact-endpoint-of-1-chunk
    placement gave the arm no real chance to move before violating (100%
    of subtask-1 attempts violated within 1-8 real steps, out of a
    10-step replan window). `horizon_multiplier` tiles the reference
    chunk to extend how far along the same direction the target sits --
    a bigger multiplier must move the endpoint strictly farther (same
    straight-line +x chunk as test_obstacle_sits_at_the_reference_chunk_
    gripper_endpoint, tiled 1x vs 3x)."""
    q0 = np.zeros(N_JOINTS)
    chunk = np.zeros((4, 7))
    chunk[:, 0] = 0.02  # steady +x end-effector position delta each step

    near = sample_obstacle_from_reference_chunk(q0, chunk, radius=0.05, horizon_multiplier=1)
    far = sample_obstacle_from_reference_chunk(q0, chunk, radius=0.05, horizon_multiplier=3)

    current = panda_frames(q0)[-1]
    assert far.center[0] > near.center[0] > current[0]


def test_rng_offsets_the_obstacle_perpendicular_to_the_direction_of_travel():
    """Regression test for the 2026-09-05 fix's second mechanism: without
    `rng`, the obstacle sits exactly on the deterministic centerline of
    the arm's own predicted path -- the same real run found this makes
    `radius` irrelevant (the arm's own capsule thickness alone already
    exceeds most tested radii, so a point ON the centerline is swept
    regardless of `radius`'s value). With `rng`, the offset must be (a)
    nonzero, (b) within `offset_max`, and (c) perpendicular to the
    direction of travel (a component ALONG the direction of travel would
    just be extending/shortening the horizon again, not creating a real
    "near miss" -- redundant with `horizon_multiplier` and wouldn't fix
    the radius-irrelevance problem)."""
    q0 = np.zeros(N_JOINTS)
    chunk = np.zeros((4, 7))
    chunk[:, 0] = 0.02  # steady +x -- direction of travel is +x

    deterministic = sample_obstacle_from_reference_chunk(q0, chunk, radius=0.05, rng=None)
    rng = np.random.default_rng(0)
    offset_max = 0.3
    randomized = sample_obstacle_from_reference_chunk(q0, chunk, radius=0.05, rng=rng, offset_max=offset_max)

    offset = randomized.center - deterministic.center
    assert not np.allclose(offset, 0.0)
    assert np.linalg.norm(offset) <= offset_max + 1e-9
    # pure +x chunk -> travel is APPROXIMATELY +x (the Jacobian-pinv step
    # then forward-kinematics chain isn't perfectly linear, so a tiny
    # along-travel residual is expected -- assert it's negligible next to
    # the real y/z offset components (which are ~offset_max scale), not
    # exactly zero to machine precision).
    direction_unit = np.array([1.0, 0.0, 0.0])
    assert abs(np.dot(offset, direction_unit)) < 1e-3


def test_rng_is_reproducible_given_the_same_seed():
    """Same seed -> same draw -- required for a radius sweep to compare
    the same random placement across every radius (see this function's
    own docstring on why `rng` must be caller-seeded, not global)."""
    q0 = np.zeros(N_JOINTS)
    chunk = np.zeros((4, 7))
    chunk[:, 0] = 0.02

    a = sample_obstacle_from_reference_chunk(q0, chunk, radius=0.05, rng=np.random.default_rng(42))
    b = sample_obstacle_from_reference_chunk(q0, chunk, radius=0.05, rng=np.random.default_rng(42))

    assert np.allclose(a.center, b.center)
