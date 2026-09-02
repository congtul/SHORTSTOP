"""End-to-end structural check of the Stage 7b P-R-C-S loop on synthetic
data: MockMDTPolicyClient.propose() -> ArmRepairShield.select(). No real
CALVIN/MDT session involved -- this only confirms the pieces are wired
together correctly, reusing Stage 7a's arm_reach/arm_shield unmodified
since CALVIN shares LIBERO's 7D relative-chunk convention and Panda robot
(see docs/STAGE7B_CALVIN_PIPELINE_DESIGN.md).
"""
import numpy as np

from shortstop.arm_reach import propagate_arm_tube
from shortstop.arm_shield import ArmRepairShield
from shortstop.env import Obstacle
from shortstop.mdt_policy_client import MockMDTPolicyClient
from shortstop.robot_geometry import N_JOINTS, SPHERE_NAMES


def test_full_propose_reach_certify_repair_loop_on_synthetic_candidates():
    rng = np.random.default_rng(0)
    q = np.zeros(N_JOINTS)
    policy = MockMDTPolicyClient(horizon=10, action_dim=7, n_candidates=6, noise_std=0.05, rng=rng)

    candidates = policy.propose(observation={})  # Propose
    assert len(candidates) == 6

    endpoints = []
    for c in candidates:
        tube = propagate_arm_tube(q, c, w_bar=0.0, model_error=0.0)  # Reach
        endpoints.append(tube[-1][SPHERE_NAMES[-1]].center())
    target = max(endpoints, key=lambda p: p[0])
    obstacle = Obstacle(center=target, radius=0.05)

    shield = ArmRepairShield(
        obstacles=[obstacle], w_bar=0.01, model_error=0.02, epsilon=0.02,
        trust_region=0.3, step_size=0.05, max_repair_iters=3,
    )
    scores = list(range(len(candidates)))
    action, info = shield.select(q, candidates, scores)  # Certify -> Select/Repair

    assert action.shape == (10, 7)
    assert len(info["admissible_mask"]) == len(candidates)
    assert (not all(info["admissible_mask"])) or info["repair_attempted"] or info["fallback"]
