"""Stage 7a shields for the Panda sphere-chain arm (shortstop/arm_reach.py),
mirroring shortstop/shield.py's Stage 1/2/4 structure -- but with Stage 3
(CE search) folded into Stage 4 as an internal step, not its own ablation
row.

Why merge: the 2D prototype's own ablation (shortstop/shield.py's CEShield
docstring; Report 2 Table 1) already shows Stage 2 (STLShield) and Stage 3
(CEShield) produce *identical* accept/reject decisions and metrics --
CEShield only adds counterexample diagnostics on top of Stage 2's test, it
never changes which chunks get through. Reporting Stage 3 as its own
ablation row for the arm would repeat that same no-op finding for no new
information. The counterexample search itself is still here (ArmRepairShield
needs it to know what to repair) -- it's just not surfaced as a separate
comparison row the way shortstop.shield.CEShield is for the 2D prototype.

Open design question this module does NOT resolve: g(a), the task-progress
score from Eq. (5), has no obvious analogue here. The 2D prototype's g(a) =
-goal_distance only exists because Reach-Avoid-2D has one fixed goal point;
a LIBERO manipulation task (language-conditioned, no single target
coordinate) doesn't. `select()` below takes `scores` as a caller-supplied
argument rather than assuming one -- see docs/LIBERO_SETUP.md's follow-up
notes for candidate choices (e.g. "prefer the least-repaired candidate", or
whatever score pi0.5's own serving stack exposes, if any).

Note: none of this has been run against a real LIBERO/pi0.5 session -- see
shortstop/arm_reach.py's docstring for the approximations in the Reach step
itself, and shortstop/robot_geometry.py's for the sphere-chain geometry.
"""
import numpy as np

from .arm_reach import arm_find_counterexample, arm_robustness_to_go, propagate_arm_tube


class ArmReachOnlyShield:
    """Stage 1 equivalent: binary reject on any sphere/obstacle
    intersection anywhere in the tube -- no STL margin, no repair."""

    def __init__(self, obstacles, w_bar, model_error=0.02):
        self.obstacles = obstacles
        self.w_bar = w_bar
        self.model_error = model_error

    def _admissible(self, joint_angles, task_chunk):
        tube = propagate_arm_tube(joint_angles, task_chunk, self.w_bar, self.model_error)
        return arm_robustness_to_go(tube, self.obstacles) >= 0.0

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


class ArmSTLShield(ArmReachOnlyShield):
    """Stage 2 equivalent: STL robustness-to-go margin (Eq. 2) instead of a
    binary intersection test."""

    def __init__(self, obstacles, w_bar, model_error=0.02, epsilon=0.02):
        super().__init__(obstacles, w_bar, model_error)
        self.epsilon = epsilon

    def _admissible(self, joint_angles, task_chunk):
        tube = propagate_arm_tube(joint_angles, task_chunk, self.w_bar, self.model_error)
        return arm_robustness_to_go(tube, self.obstacles) >= self.epsilon


class ArmRepairShield(ArmSTLShield):
    """Stage 3+4 merged into one ablation row: counterexample-guided repair
    (Eq. 3-4) -- see module docstring for why Stage 3 isn't its own row
    here. max_repair_iters=1 matches Algorithm 1 (one gradient step, one
    re-certification, no retry); see shortstop.shield.RepairShield's
    docstring for why >1 is a CEGIS-style extension beyond the paper.
    """

    def __init__(
        self, obstacles, w_bar, model_error=0.02, epsilon=0.02,
        trust_region=0.05, step_size=0.02, max_repair_iters=1,
    ):
        super().__init__(obstacles, w_bar, model_error, epsilon)
        self.trust_region = trust_region
        self.step_size = step_size
        self.max_repair_iters = max_repair_iters

    def _repair_direction(self, counterexample):
        direction = counterexample["witness"] - counterexample["obstacle"].center
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0])  # degenerate: push "up", same spirit as
            # RepairShield's nominal-point fallback, but no single nominal
            # rollout point exists here (per-sphere, not per-state) to fall
            # back to -- an arbitrary but well-defined direction instead.
        return direction / norm

    def _repair(self, joint_angles, task_chunk, counterexample):
        original = task_chunk.copy()
        chunk = task_chunk.copy()
        ce = counterexample
        for _ in range(self.max_repair_iters):
            k_star = ce["step"]
            direction = self._repair_direction(ce)
            # Eq. 4's structure: nudge every position-delta column up to the
            # violating step, along the counterexample's push-away
            # direction (task-space position columns only, see
            # propagate_arm_tube's docstring on why rotation/gripper are
            # untouched by the Reach step).
            chunk[:k_star, :3] = chunk[:k_star, :3] + self.step_size * direction
            delta = chunk - original
            norm = np.linalg.norm(delta)
            if norm > self.trust_region:
                chunk = original + delta * (self.trust_region / norm)

            tube = propagate_arm_tube(joint_angles, chunk, self.w_bar, self.model_error)
            if arm_robustness_to_go(tube, self.obstacles) >= self.epsilon:
                return chunk, True
            ce = arm_find_counterexample(tube, self.obstacles)
        return chunk, False

    def select(self, joint_angles, candidates, scores):
        mask = []
        repaired = list(candidates)
        repair_attempted = False
        repair_succeeded = False

        for i, chunk in enumerate(candidates):
            tube = propagate_arm_tube(joint_angles, chunk, self.w_bar, self.model_error)
            if arm_robustness_to_go(tube, self.obstacles) >= self.epsilon:
                mask.append(True)
                continue
            repair_attempted = True
            ce = arm_find_counterexample(tube, self.obstacles)
            fixed_chunk, success = self._repair(joint_angles, chunk, ce)
            repaired[i] = fixed_chunk
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
