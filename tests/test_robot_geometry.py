import numpy as np

from shortstop.robot_geometry import (
    N_JOINTS,
    SPHERE_NAMES,
    end_effector_jacobian,
    numerical_jacobian,
    panda_frames,
    sphere_centers,
)


def test_panda_frames_returns_9_positions_and_flange_is_above_base():
    q = np.zeros(N_JOINTS)
    frames = panda_frames(q)
    assert frames.shape == (9, 3)
    assert np.allclose(frames[0], [0.0, 0.0, 0.0])  # base frame at origin
    # flange should be a plausible distance from the base -- not a precise
    # spec check (see robot_geometry.py's DH-table caveat), just "not
    # degenerate" (zero, or absurdly far)
    reach = np.linalg.norm(frames[-1])
    assert 0.3 < reach < 1.5


def test_sphere_centers_matches_selected_frames():
    q = np.random.default_rng(0).uniform(-0.5, 0.5, size=N_JOINTS)
    frames = panda_frames(q)
    centers = sphere_centers(q)
    assert centers.shape == (len(SPHERE_NAMES), 3)
    assert np.allclose(centers[-1], frames[-1])  # last sphere == flange


def test_only_upstream_joints_affect_a_given_frame():
    """Frame i's *origin* only depends on joint angles 0..i-2, one step
    earlier than one might naively guess: theta_i rotates frame i about its
    own z-axis relative to frame i-1, which does not move frame i's own
    origin (only how frame i's axes are oriented, which matters for frame
    i+1's origin) -- a basic structural property of the DH chain, and
    exactly why the Panda's joint 7 (a wrist roll whose axis passes through
    the flange) does not move the flange position, only its orientation."""
    rng = np.random.default_rng(1)
    q = rng.uniform(-0.3, 0.3, size=N_JOINTS)
    frames_before = panda_frames(q)

    q_perturbed = q.copy()
    q_perturbed[3] += 0.2  # perturb joint 4 (index 3)
    frames_after = panda_frames(q_perturbed)

    # frames 0..4 (base through joint 4's own origin) must be unchanged;
    # frame 5 onward (downstream of joint 4's rotation) must move
    assert np.allclose(frames_before[:5], frames_after[:5])
    assert not np.allclose(frames_before[5], frames_after[5])


def test_numerical_jacobian_matches_finite_difference_of_fk():
    """Self-consistency check: the Jacobian's own linear prediction of a
    small joint perturbation's effect on position should match panda_frames
    evaluated at that perturbed config."""
    rng = np.random.default_rng(2)
    q = rng.uniform(-0.4, 0.4, size=N_JOINTS)
    J = end_effector_jacobian(q)

    dq = rng.uniform(-1e-4, 1e-4, size=N_JOINTS)
    predicted = panda_frames(q)[-1] + J @ dq
    actual = panda_frames(q + dq)[-1]
    assert np.allclose(predicted, actual, atol=1e-6)


def test_numerical_jacobian_for_a_middle_sphere_frame_is_nonzero_and_shaped():
    q = np.random.default_rng(3).uniform(-0.4, 0.4, size=N_JOINTS)
    J = numerical_jacobian(q, frame_index=3)
    assert J.shape == (3, N_JOINTS)
    assert np.linalg.norm(J) > 0
