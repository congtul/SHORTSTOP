"""End-to-end structural check of the Stage 7c P-R-C-S loop (v2) on
synthetic data: MockMiniDiffuserClient.propose() -> planner.mock_get_path
-> KeyposeRepairShield.select(). No real RLBench/Mini-Diffuser/PyRep
session involved.
"""
import numpy as np

from shortstop.env import Obstacle
from shortstop.keypose_reach import propagate_path_tube
from shortstop.keypose_shield import KeyposeRepairShield
from shortstop.mini_diffuser_client import MockMiniDiffuserClient
from shortstop.planner import mock_get_path
from shortstop.robot_geometry import N_JOINTS, SPHERE_NAMES


def test_full_propose_plan_certify_repair_loop_on_synthetic_keyposes():
    rng = np.random.default_rng(0)
    q0 = np.zeros(N_JOINTS)
    policy = MockMiniDiffuserClient(n_candidates=6, position_noise_std=0.05, rng=rng)

    candidates = policy.propose(observation={})  # Propose
    assert len(candidates) == 6

    endpoints = []
    for c in candidates:
        path_points = mock_get_path(q0, c[:3])  # "Policy + Planner" black box
        tube = propagate_path_tube(path_points, w_bar=0.0, model_error=0.0)  # Reach
        endpoints.append(tube[-1][SPHERE_NAMES[-1]].center())
    target = max(endpoints, key=lambda p: p[0])
    obstacle = Obstacle(center=target, radius=0.05)

    shield = KeyposeRepairShield(
        obstacles=[obstacle], w_bar=0.01, planner_fn=mock_get_path, model_error=0.02, epsilon=0.02,
        trust_region=0.3, step_size=0.05, max_repair_iters=3,
    )
    scores = list(range(len(candidates)))
    action, info = shield.select(q0, candidates, scores)  # Certify -> Select/Repair

    assert action.shape == (8,)
    assert len(info["admissible_mask"]) == len(candidates)
    assert (not all(info["admissible_mask"])) or info["repair_attempted"] or info["fallback"]
