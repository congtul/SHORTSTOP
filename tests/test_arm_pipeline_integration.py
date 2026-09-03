"""End-to-end structural check of the Stage 7a P-R-C-S loop on synthetic
data: MockPi05PolicyClient.propose() -> ArmRepairShield.select(). No real
LIBERO/pi0.5 session involved -- this only confirms the pieces are wired
together correctly (shapes, that a genuinely-unsafe candidate gets rejected
or repaired), not that the numbers mean anything on a real robot.
"""
import numpy as np

from shortstop.arm_reach import propagate_arm_tube
from shortstop.arm_shield import ArmRepairShield
from shortstop.env import Obstacle
from shortstop.pi_policy_client import MockPi05PolicyClient
from shortstop.robot_geometry import FLANGE_FRAME_INDEX, N_JOINTS


def test_full_propose_reach_certify_repair_loop_on_synthetic_candidates():
    rng = np.random.default_rng(0)
    q = np.zeros(N_JOINTS)
    policy = MockPi05PolicyClient(horizon=4, action_dim=7, n_candidates=6, noise_std=0.05, rng=rng)

    candidates = policy.propose(observation={})  # Propose
    assert len(candidates) == 6

    # place an obstacle exactly at the gripper endpoint of whichever
    # candidate moves it furthest in +x, guaranteeing at least one genuine
    # violation for the shield to react to
    endpoints = []
    for c in candidates:
        tube = propagate_arm_tube(q, c, w_bar=0.0, model_error=0.0)  # Reach
        endpoints.append(tube[-1][FLANGE_FRAME_INDEX].center())
    target = max(endpoints, key=lambda p: p[0])
    obstacle = Obstacle(center=target, radius=0.05)

    shield = ArmRepairShield(
        obstacles=[obstacle], w_bar=0.01, model_error=0.02, epsilon=0.02,
        trust_region=0.3, step_size=0.05, max_repair_iters=3,
    )
    scores = list(range(len(candidates)))  # arbitrary task-progress stand-in
    action, info = shield.select(q, candidates, scores)  # Certify -> Select/Repair

    assert action.shape == (4, 7)
    assert "admissible_mask" in info
    assert len(info["admissible_mask"]) == len(candidates)
    # the certify step must have found *something* to react to: either a
    # rejection, a repair, or (least likely given the setup) a fallback
    assert (not all(info["admissible_mask"])) or info["repair_attempted"] or info["fallback"]
