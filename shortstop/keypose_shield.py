"""Stage 7c shields for the RLBench keypose interface -- v2, built on
shortstop/keypose_reach.py's revised (real-path) Reach step and
shortstop/planner.py's `planner_fn(joint_angles, target_position) ->
path_points` interface.

Every shield here takes `planner_fn` in its constructor (dependency
injection: pass `shortstop.planner.mock_get_path` for structural testing,
or a closure around `shortstop.planner.real_get_path` bound to a live
PyRep `Arm` for a real run) and calls it once per candidate to get that
candidate's actual planned path before certifying it -- Reach/Certify
operate on that real (or mocked) path, not a guessed one (see
keypose_reach.py's module docstring for why v1's own interpolation guess
was replaced).

Repair is structurally different from every other shield in this repo
(2D's RepairShield, Stage 7a's ArmRepairShield): PyRep's
ArmConfigurationPath has no public API to edit waypoints in place, so a
rejected candidate cannot be "nudged" the way a chunk's numpy array can.
Repair here nudges the *target keypose's position* and calls `planner_fn`
again for an entirely new path, re-certifying that -- repeated up to
max_repair_iters times. This is a real, not cosmetic, consequence of
RLBench's execution model (see module docstring above and
docs/STAGE7C_ARM_PIPELINE_DESIGN.md).

Same merge rationale as shortstop.arm_shield: Stage 3 (CE search) is
folded into Repair as an internal step, not its own ablation row.
"""
import numpy as np

from .keypose_reach import path_find_counterexample, path_robustness_to_go, propagate_path_tube


class KeyposeReachOnlyShield:
    """Stage 1 equivalent: binary reject if the *actual planned path* to
    the target keypose intersects any obstacle."""

    def __init__(self, obstacles, w_bar, planner_fn, model_error=0.02):
        self.obstacles = obstacles
        self.w_bar = w_bar
        self.planner_fn = planner_fn
        self.model_error = model_error

    def _admissible(self, joint_angles, keypose):
        path_points = self.planner_fn(joint_angles, keypose[:3])
        tube = propagate_path_tube(path_points, self.w_bar, self.model_error)
        return path_robustness_to_go(tube, self.obstacles) >= 0.0

    def select(self, joint_angles, candidates, scores):
        mask = [self._admissible(joint_angles, c) for c in candidates]
        admissible_idx = [i for i, ok in enumerate(mask) if ok]
        if not admissible_idx:
            fallback = np.zeros_like(candidates[0])
            return fallback, {"fallback": True, "n_admissible": 0, "admissible_mask": mask}
        best_i = max(admissible_idx, key=lambda i: scores[i])
        return candidates[best_i], {
            "fallback": False, "n_admissible": len(admissible_idx), "admissible_mask": mask,
        }


class KeyposeSTLShield(KeyposeReachOnlyShield):
    """Stage 2 equivalent: STL robustness-to-go margin on the real planned
    path instead of a binary intersection test."""

    def __init__(self, obstacles, w_bar, planner_fn, model_error=0.02, epsilon=0.02):
        super().__init__(obstacles, w_bar, planner_fn, model_error)
        self.epsilon = epsilon

    def _admissible(self, joint_angles, keypose):
        path_points = self.planner_fn(joint_angles, keypose[:3])
        tube = propagate_path_tube(path_points, self.w_bar, self.model_error)
        return path_robustness_to_go(tube, self.obstacles) >= self.epsilon


class KeyposeRepairShield(KeyposeSTLShield):
    """Stage 3+4 merged into one ablation row. Repair = nudge the target
    keypose's position, call `planner_fn` again for a brand new path,
    re-certify -- NOT an in-place edit of the previous path (see module
    docstring for why: PyRep's ArmConfigurationPath doesn't support one).
    """

    def __init__(
        self, obstacles, w_bar, planner_fn, model_error=0.02, epsilon=0.02,
        trust_region=0.05, step_size=0.02, max_repair_iters=1,
    ):
        super().__init__(obstacles, w_bar, planner_fn, model_error, epsilon)
        self.trust_region = trust_region
        self.step_size = step_size
        self.max_repair_iters = max_repair_iters

    def _repair_direction(self, counterexample):
        direction = counterexample["witness"] - counterexample["obstacle"].center
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return direction / norm

    def _repair(self, joint_angles, keypose, counterexample):
        original_pos = keypose[:3].copy()
        pos = original_pos.copy()
        ce = counterexample
        for _ in range(self.max_repair_iters):
            direction = self._repair_direction(ce)
            pos = pos + self.step_size * direction
            delta = pos - original_pos
            norm = np.linalg.norm(delta)
            if norm > self.trust_region:
                pos = original_pos + delta * (self.trust_region / norm)

            # re-plan from scratch to the nudged target -- cannot just
            # re-check the old path, the old path is no longer relevant
            path_points = self.planner_fn(joint_angles, pos)
            tube = propagate_path_tube(path_points, self.w_bar, self.model_error)
            if path_robustness_to_go(tube, self.obstacles) >= self.epsilon:
                repaired = keypose.copy()
                repaired[:3] = pos
                return repaired, True
            ce = path_find_counterexample(tube, self.obstacles)
        repaired = keypose.copy()
        repaired[:3] = pos
        return repaired, False

    def select(self, joint_angles, candidates, scores):
        mask = []
        repaired = list(candidates)
        repair_attempted = False
        repair_succeeded = False

        for i, keypose in enumerate(candidates):
            path_points = self.planner_fn(joint_angles, keypose[:3])
            tube = propagate_path_tube(path_points, self.w_bar, self.model_error)
            if path_robustness_to_go(tube, self.obstacles) >= self.epsilon:
                mask.append(True)
                continue
            repair_attempted = True
            ce = path_find_counterexample(tube, self.obstacles)
            fixed_keypose, success = self._repair(joint_angles, keypose, ce)
            repaired[i] = fixed_keypose
            mask.append(success)
            repair_succeeded = repair_succeeded or success

        admissible_idx = [i for i, ok in enumerate(mask) if ok]
        if not admissible_idx:
            fallback = np.zeros_like(candidates[0])
            return fallback, {
                "fallback": True, "n_admissible": 0, "admissible_mask": mask,
                "repair_attempted": repair_attempted, "repair_succeeded": repair_succeeded,
            }
        best_i = max(admissible_idx, key=lambda i: scores[i])
        return repaired[best_i], {
            "fallback": False, "n_admissible": len(admissible_idx), "admissible_mask": mask,
            "repair_attempted": repair_attempted, "repair_succeeded": repair_succeeded,
        }
