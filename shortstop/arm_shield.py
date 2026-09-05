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
import cvxpy as cp
import numpy as np

from .arm_reach import (
    arm_find_counterexample, arm_robustness_to_go, nominal_joint_trajectory, propagate_arm_tube,
)
from .robot_geometry import (
    FLANGE_FRAME_INDEX, FRAME_RADIUS, JOINT_LIMITS, LINK_RADIUS, N_JOINTS,
    end_effector_jacobian, numerical_jacobian, panda_frames, within_joint_limits,
)


def _closest_point_t(point, a, b):
    """Same clamped-projection math as robot_geometry.closest_point_on_
    segment, but returning the scalar `t` in [0,1] instead of the point
    itself -- ArmMPCFilterShield needs `t` separately, to blend the two
    endpoint frames' own Jacobians by the same weight (a link's closest
    point is an affine combination of its two endpoint frames, so its
    Jacobian w.r.t. joint angles is the same affine combination of theirs
    -- no separate finite-difference pass needed for link primitives)."""
    ab = b - a
    length_sq = float(ab @ ab)
    if length_sq < 1e-12:
        return 0.0
    return float(np.clip((point - a) @ ab / length_sq, 0.0, 1.0))


class ArmReachOnlyShield:
    """Stage 1 equivalent: binary reject on any sphere/obstacle
    intersection anywhere in the tube -- no STL margin, no repair."""

    def __init__(self, obstacles, w_bar, model_error=0.02):
        self.obstacles = obstacles
        self.w_bar = w_bar
        self.model_error = model_error

    def _trajectory_within_joint_limits(self, joint_angles, task_chunk):
        """A candidate whose nominal joint trajectory ever exits
        robot_geometry.JOINT_LIMITS isn't physically executable by the
        real robot's own safety controller, regardless of obstacle
        clearance -- checked separately from (and before) the obstacle
        certify step, same cost class (one more Jacobian-stepping pass,
        no reachtube/obstacle math needed)."""
        trajectory = nominal_joint_trajectory(joint_angles, task_chunk)
        return all(within_joint_limits(q) for q in trajectory[1:])

    def _admissible(self, joint_angles, task_chunk):
        if not self._trajectory_within_joint_limits(joint_angles, task_chunk):
            return False
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

    def recertify(self, joint_angles, remaining_chunk):
        """Cheap per-step re-check of an already-selected chunk's
        remaining tail against the REAL current state -- lets the harness
        (shortstop.calvin_experiment.run_calvin_shielded_subtask) refresh
        certification every real env-step even though Propose (K new
        diffusion samples) only runs every `replan_steps` steps. See
        docs/PARAMETERS_REFERENCE.md's "tach tan suat filter khoi policy"
        entry: feasible here (and for every ArmReachOnlyShield/ArmSTLShield
        subclass -- ArmSTLMonitorShield, ArmRepairShield) because Certify
        only needs propagate_arm_tube on the current real state, no fresh
        K-candidate sample -- unlike ArmConfThreshShield, whose
        disagreement is tied to one specific K-sample and has no
        `recertify` at all (see its own docstring). Reuses the exact same
        admissibility test `select()` applies at Propose time, just
        against a shorter suffix from a possibly-drifted real state
        instead of the nominal one assumed when this chunk was chosen."""
        return self._admissible(joint_angles, remaining_chunk)


class ArmSTLShield(ArmReachOnlyShield):
    """Stage 2 equivalent: STL robustness-to-go margin (Eq. 2) instead of a
    binary intersection test."""

    def __init__(self, obstacles, w_bar, model_error=0.02, epsilon=0.02):
        super().__init__(obstacles, w_bar, model_error)
        self.epsilon = epsilon

    def _admissible(self, joint_angles, task_chunk):
        if not self._trajectory_within_joint_limits(joint_angles, task_chunk):
            return False
        tube = propagate_arm_tube(joint_angles, task_chunk, self.w_bar, self.model_error)
        return arm_robustness_to_go(tube, self.obstacles) >= self.epsilon


class ArmSTLMonitorShield(ArmSTLShield):
    """Table II's STL-Monitor baseline (docs/main (3).txt Sec. V.D):
    nominal STL robustness on the f-hat rollout, no reachtube, no
    counterexample search. Exactly ArmSTLShield with the disturbance/
    model-error bound zeroed: w_bar=0/model_error=0 (the tube still
    inflates by each frame's own physical radius regardless, since that's
    real geometry, not an uncertainty bound). No repair -- unlike
    ArmRepairShield below, a rejected chunk is simply dropped from the
    admissible set, never nudged back toward safety.

    `epsilon` is a required, explicit argument, NOT hardcoded -- the
    paper's own text is ambiguous about its value here. Sec. IV says
    STL-Monitor "rejects if negative" (literal reading: epsilon=0.0), but
    the very next sentence says "All model-based baselines use the
    identical f-hat, epsilon and fallback for a fair comparison" (shared
    reading: epsilon = ShortStop's own calibrated margin, 0.02 in this
    codebase's default budgets) -- these two readings disagree, and which
    one the paper means can't be resolved from the text alone (likely a
    two-column PDF extraction artifact). Rather than silently picking one,
    this class requires the caller to say which -- see scripts/
    run_calvin_stl_monitor.py's own sweep over both readings (and points
    in between) to resolve this empirically instead."""

    def __init__(self, obstacles, epsilon):
        super().__init__(obstacles, w_bar=0.0, model_error=0.0, epsilon=epsilon)


class ArmRepairShield(ArmSTLShield):
    """Stage 3+4 merged into one ablation row: counterexample-guided repair
    (Eq. 3-4) -- see module docstring for why Stage 3 isn't its own row
    here. max_repair_iters=1 matches Algorithm 1 (one gradient step, one
    re-certification, no retry); see shortstop.shield.RepairShield's
    docstring for why >1 is a CEGIS-style extension beyond the paper.

    Defines its own `resolve(joint_angles, remaining_chunk)` (2026-09-05,
    same reasoning as ArmMPCFilterShield's own `resolve` -- see its
    docstring): this shield already HAS a real correction mechanism
    (`_repair`), so relying on the inherited `ArmReachOnlyShield.recertify`
    (a pure pass/fail check) at per-step drift time would waste exactly
    the capability that distinguishes ShortStop from STL-Monitor --
    detecting real-world drift only to give up and re-propose, instead of
    trying to repair around it first. `run_calvin_shielded_subtask`
    prefers `resolve` over `recertify` whenever a shield defines both (see
    that harness's own docstring), so this takes effect automatically.
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
            # A repaired chunk must still respect JOINT_LIMITS, not just
            # clear the obstacle -- pushing away from an obstacle in
            # task-space can drive the Jacobian-IK joint solution past a
            # real physical limit even when the obstacle margin looks fine.
            if (
                self._trajectory_within_joint_limits(joint_angles, chunk)
                and arm_robustness_to_go(tube, self.obstacles) >= self.epsilon
            ):
                return chunk, True
            ce = arm_find_counterexample(tube, self.obstacles)
        return chunk, False

    def select(self, joint_angles, candidates, scores):
        mask = []
        repaired = list(candidates)
        repair_attempted = False
        repair_succeeded = False

        for i, chunk in enumerate(candidates):
            joint_limits_ok = self._trajectory_within_joint_limits(joint_angles, chunk)
            tube = propagate_arm_tube(joint_angles, chunk, self.w_bar, self.model_error)
            if joint_limits_ok and arm_robustness_to_go(tube, self.obstacles) >= self.epsilon:
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

    def resolve(self, joint_angles, remaining_chunk):
        """Re-attempts REPAIR from the REAL current state every real
        env-step, not just a binary pass/fail -- see class docstring for
        why. Mirrors select()'s own per-candidate logic (already
        admissible -> keep as-is; not admissible -> counterexample-guided
        repair -> success or give up), applied to the real current
        state's remaining tail instead of one of K nominal candidates.
        Returns the (possibly repaired) `remaining_chunk`, or `None` if
        repair fails -- the harness treats `None` exactly like
        `recertify()` returning `False` (abandon, re-propose)."""
        joint_limits_ok = self._trajectory_within_joint_limits(joint_angles, remaining_chunk)
        tube = propagate_arm_tube(joint_angles, remaining_chunk, self.w_bar, self.model_error)
        if joint_limits_ok and arm_robustness_to_go(tube, self.obstacles) >= self.epsilon:
            return remaining_chunk
        ce = arm_find_counterexample(tube, self.obstacles)
        fixed_chunk, success = self._repair(joint_angles, remaining_chunk, ce)
        return fixed_chunk if success else None


class ArmMPCFilterShield(ArmReachOnlyShield):
    """CALVIN/arm analogue of shortstop.baselines.MPCFilterShield -- an
    H-step predictive safety filter (Wabersich & Zeilinger, "A predictive
    safety filter for learning-based control of constrained nonlinear
    dynamical systems," Automatica 2021, paper's ref [33]). Unlike every
    other shield in this module, this does not pick among K candidates --
    it takes the policy's OWN top candidate (`candidates[0]`) and solves a
    QP that corrects it minimally to satisfy every obstacle/joint-limit
    constraint over the whole chunk, exactly mirroring
    `baselines.MPCFilterShield`'s own scope (that class's docstring
    explains why an earlier *1-step*-lookahead version was found
    materially weaker than a real predictive safety filter -- this reuses
    its CURRENT, full-H-step design, not the discarded one).

    Why this needs its own linearization scheme (2D's is exact, this
    isn't): 2D's dynamics `x_next = x + a*dt` is EXACTLY linear in the
    action, so a single QP with hyperplane constraints is not an
    approximation of the dynamics itself, only of the (non-convex) circle
    constraint. The arm's task-space-action -> joint-space -> Cartesian
    chain (arm_reach.py's Jacobian pseudo-inverse stepping) is genuinely
    nonlinear -- so this ALSO linearizes the dynamics, once, around the
    policy's own nominal chunk (a discrete-time linear time-varying
    system), then solves one QP for the whole H-step chunk:

      1. `q_nominal = nominal_joint_trajectory(joint_angles, nominal_chunk)`
         -- the UNCORRECTED trajectory the policy's own chunk would
         produce (arm_reach.py, same Jacobian-pinv stepping
         propagate_arm_tube uses internally).
      2. Per step k=1..H, the sensitivity of that step's joint delta to a
         perturbation of THAT step's action is `pinv(J_ee(q_nominal[k-1]))`
         -- the same flange Jacobian `_step_joint_config` itself uses,
         evaluated at (and frozen at) the nominal trajectory's own
         joint config for that step -- a single linearization pass, not
         re-linearized after solving (same documented limitation
         `baselines.MPCFilterShield` already carries for its own
         tangent-plane hyperplanes).
      3. Cumulative joint-delta at step k is the sum of every prior step's
         contribution (a linear/affine cvxpy expression in the decision
         variable `a`, since each step's own contribution is linear).
      4. Every one of the 9 frames' AND 8 links' positions at step k is
         linearized the same way: `nominal_position + frame_jacobian(
         q_nominal[k]) @ cumulative_delta_q_k` (`numerical_jacobian`,
         finite-difference w.r.t. joint angles, frozen at the nominal
         trajectory). A link's own closest point to an obstacle is an
         affine blend of its two endpoint frames (`_closest_point_t`'s
         `t`), so its Jacobian is the same blend of theirs -- no extra
         finite-difference pass needed for links. Constrains the FULL
         17-primitive chain (9 frames + 8 links), matching ShortStop's own
         reachtube scope exactly -- a coarser (e.g. flange-only) version
         would look artificially safer than ShortStop here for the wrong
         reason (missing the same mid-link collisions Category A.3 fixed
         for ShortStop's own reachtube), not because MPC-Filter is
         actually a better filter.
      5. Each primitive-vs-obstacle constraint is a tangent-plane
         hyperplane at the NOMINAL closest point (same construction as
         `baselines.MPCFilterShield`'s circle hyperplane), tightened by
         that primitive's own physical radius (FRAME_RADIUS/LINK_RADIUS)
         plus `w_bar + model_error` -- the same total inflation
         `arm_reach._signed_distance` uses.
      6. JOINT_LIMITS added as direct linear constraints on the same
         cumulative joint-delta expression -- the arm's analogue of 2D's
         `max_action_norm` bound (a physical-validity constraint every
         other arm shield already enforces, so MPC-Filter isn't unfairly
         advantaged by being allowed to violate it).

    No soundness proof, same caveat `baselines.MPCFilterShield`'s own
    docstring states: valid only at the ONE linearization point each solve
    used (a real iterative PSF would re-linearize and re-solve, SQP-style,
    until convergence within a single solve) -- documented as a real
    limitation, not silently presented as certified the way ShortStop's
    own exact capsule-vs-sphere reachtube is. `resolve()` (below) mitigates
    this ACROSS solves (each real step gets its own fresh linearization,
    rather than trusting one linearization for the whole remaining
    horizon) but chaining `select()` then `resolve()` still compounds two
    separate single-pass linearizations -- verified numerically (tests/
    test_arm_shield.py's own resolve tests), the resulting slack loss is
    small (a fraction of a millimeter for a realistic case) but real, not
    exactly zero.

    Defines its own `resolve(joint_angles, remaining_chunk)` (2026-09-05),
    NOT just the inherited `recertify` every other shield in this module
    relies on -- `resolve()` genuinely RE-SOLVES the QP from the REAL
    current state every real env-step (see its own docstring), matching
    what "predictive safety filter" actually means in the literature
    (receding-horizon re-optimization), rather than merely re-checking
    whether the ORIGINAL, now-stale correction is still admissible.
    `run_calvin_shielded_subtask` prefers `resolve` over `recertify` when
    a shield defines both (see that harness's own docstring) -- inherited
    `_admissible`/`_trajectory_within_joint_limits`/`recertify` stay
    reachable (e.g. for a caller that wants the cheap binary check
    directly) but are no longer what the harness itself calls for this
    class.

    `select()`'s returned info dict deliberately marks every OTHER
    candidate (`candidates[1:]`) as trivially admissible regardless of
    whether they were ever evaluated -- MPC-Filter never looks at them at
    all (matches `baselines.MPCFilterShield`'s own `[not intervened] +
    [True] * (K - 1)` convention exactly, for metric-computation
    consistency between the 2D and arm versions of this baseline)."""

    def select(self, joint_angles, candidates, scores):
        del scores  # MPC-Filter corrects candidates[0] directly, never ranks -- see class docstring
        nominal_chunk = np.asarray(candidates[0], dtype=float).copy()
        corrected_position = self._solve_qp(joint_angles, nominal_chunk)

        if corrected_position is None:
            mask = [False] + [True] * (len(candidates) - 1)
            return np.zeros_like(nominal_chunk), {
                "fallback": True, "n_admissible": 0, "admissible_mask": mask,
                "repair_attempted": True, "repair_succeeded": False,
            }

        chunk = nominal_chunk.copy()
        chunk[:, :3] = corrected_position
        intervened = not np.allclose(corrected_position, nominal_chunk[:, :3], atol=1e-6)
        mask = [not intervened] + [True] * (len(candidates) - 1)
        n_admissible = len(candidates) - (1 if intervened else 0)
        return chunk, {
            "fallback": False, "n_admissible": n_admissible, "admissible_mask": mask,
            "intervened": intervened, "repair_attempted": intervened, "repair_succeeded": intervened,
        }

    def resolve(self, joint_angles, remaining_chunk):
        """Re-solves the SAME QP `select()` uses, but from the REAL
        current state (`joint_angles`, as observed after the real env has
        just executed a row) instead of the state assumed when the
        remaining chunk was last chosen -- and using `remaining_chunk`
        itself (not a fresh policy proposal) as the nominal reference to
        stay close to. This is what makes ArmMPCFilterShield a genuine
        receding-horizon predictive safety filter rather than merely a
        one-shot correction re-checked for staleness: the paper's own
        Thm. 1 proof describes exactly this pattern ("only the first
        action of a* is committed before re-deciding"), and a real PSF
        (Wabersich & Zeilinger) re-solves at every control step, not just
        at Propose's own cadence -- unlike `ArmReachOnlyShield.recertify`
        (a cheap binary re-check reused unchanged by every OTHER shield
        in this module, including this one's own inherited version, which
        `run_calvin_shielded_subtask` only falls back to for shields
        without a `resolve` method -- see that harness's own docstring),
        this ACTUALLY re-optimizes, so the harness swaps in its result
        rather than merely gating on it.

        Returns a corrected `remaining_chunk`-shaped array (columns 3:
        unchanged, matching `select()`'s own convention), or `None` if
        this step's QP is infeasible (the harness treats `None` exactly
        like `recertify()` returning `False`: abandon the rest of this
        chunk, re-propose immediately)."""
        corrected_position = self._solve_qp(joint_angles, remaining_chunk)
        if corrected_position is None:
            return None
        resolved = np.asarray(remaining_chunk, dtype=float).copy()
        resolved[:, :3] = corrected_position
        return resolved

    def _solve_qp(self, joint_angles, nominal_chunk):
        """Shared QP core for both `select()` (nominal = the policy's own
        top candidate) and `resolve()` (nominal = the previously-selected
        chunk's remaining tail, re-linearized from a NEW real state) --
        see each caller's own docstring for what differs between them.
        Returns the corrected position columns (`nominal_chunk[:, :3]`'s
        shape), or `None` if infeasible."""
        joint_angles = np.asarray(joint_angles, dtype=float)
        nominal_chunk = np.asarray(nominal_chunk, dtype=float)
        horizon = len(nominal_chunk)

        q_nominal = nominal_joint_trajectory(joint_angles, nominal_chunk)  # length horizon+1
        ee_pinv = [np.linalg.pinv(end_effector_jacobian(q_nominal[k])) for k in range(horizon)]
        frames_nominal = [panda_frames(q) for q in q_nominal]  # length horizon+1, each (9,3)

        a = cp.Variable((horizon, 3))
        delta_a = a - nominal_chunk[:, :3]
        delta_q_cumulative = []
        running = 0
        for j in range(horizon):
            running = running + ee_pinv[j] @ delta_a[j]
            delta_q_cumulative.append(running)

        constraints = []
        for k in range(1, horizon + 1):
            dq_k = delta_q_cumulative[k - 1]
            constraints.append(q_nominal[k] + dq_k >= JOINT_LIMITS[:, 0])
            constraints.append(q_nominal[k] + dq_k <= JOINT_LIMITS[:, 1])

            frame_points = frames_nominal[k]
            frame_jacobians = [numerical_jacobian(q_nominal[k], i) for i in range(len(frame_points))]

            for obstacle in self.obstacles:
                for i, nominal_point in enumerate(frame_points):
                    self._add_tangent_constraint(
                        constraints, nominal_point, frame_jacobians[i], dq_k, obstacle, FRAME_RADIUS[i],
                    )
                for i in range(len(LINK_RADIUS)):
                    a_pt, b_pt = frame_points[i], frame_points[i + 1]
                    t = _closest_point_t(obstacle.center, a_pt, b_pt)
                    link_point = a_pt + t * (b_pt - a_pt)
                    link_jacobian = (1.0 - t) * frame_jacobians[i] + t * frame_jacobians[i + 1]
                    self._add_tangent_constraint(
                        constraints, link_point, link_jacobian, dq_k, obstacle, LINK_RADIUS[i],
                    )

        problem = cp.Problem(cp.Minimize(cp.sum_squares(delta_a)), constraints)
        try:
            problem.solve()
        except cp.error.SolverError:
            a.value = None

        return None if a.value is None else np.asarray(a.value)

    def _add_tangent_constraint(self, constraints, nominal_point, jacobian, dq_k, obstacle, physical_radius):
        direction = nominal_point - obstacle.center
        norm = np.linalg.norm(direction)
        n = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        radius = obstacle.radius + physical_radius + self.w_bar + self.model_error
        corrected_point = nominal_point + jacobian @ dq_k
        constraints.append(n @ (corrected_point - obstacle.center) >= radius)


class ArmConfThreshShield:
    """CALVIN/LIBERO-analogue of shortstop.baselines.ConfThreshShield --
    rejects candidates whose predicted flange endpoint disagrees too much
    with the K-candidate centroid (sampler-ensemble disagreement, a proxy
    for confidence -- see docs/main (3).txt Sec. V.D: "Conf-Thresh, which
    rejects chunks whose sampler-ensemble disagreement ... exceeds a tuned
    threshold"). No obstacle/dynamics-model knowledge at all -- unlike
    ArmReachOnlyShield/ArmSTLShield/ArmRepairShield above, this class takes
    no `obstacles`/`w_bar`/`model_error`, since the 2D ConfThreshShield
    stores an `obstacles` arg but never actually reads it either (the
    filter is disagreement-only by the paper's own definition).

    Among the survivors, select() picks the highest caller-supplied
    score(a) -- see shortstop.calvin_progress for CALVIN's g(a) (a
    kinematic goal-distance proxy for the paper's task-progress slot, not
    a literal measure of task completion).

    `replan_steps` (required, not defaulted -- same reasoning as
    shortstop.calvin_progress.calvin_progress_scores's own `replan_steps`
    arg): how many rows of a chunk the caller will actually commit to
    env.step() before replanning (the harness's own `replan_steps`
    argument -- see shortstop.calvin_experiment.
    run_calvin_shielded_subtask). Disagreement is measured at the
    predicted endpoint of each candidate's first `replan_steps` rows, NOT
    its full length -- the paper's own Alg. 1 only ever commits the first
    action of a chunk before re-deciding (Thm. 1's proof: "Only the first
    action of a* is committed before re-deciding"), so a candidate's
    disagreement should reflect what actually gets executed before the
    next decision, not a tail segment that gets thrown away regardless of
    which candidate is chosen. `replan_steps` >= a chunk's own length is
    fine (propagate_arm_tube then just sees the whole chunk). Note this
    is an empirical filter, not a certified one either way -- Conf-Thresh
    carries no soundness guarantee (see docs/main (3).txt Table II: 11.2%
    violation, 0.43 precision, unlike ShortStop's own reachtube+STL
    certificate).
    """

    def __init__(self, disagreement_threshold, replan_steps):
        self.disagreement_threshold = disagreement_threshold
        self.replan_steps = replan_steps

    def _endpoint(self, joint_angles, task_chunk):
        tube = propagate_arm_tube(joint_angles, task_chunk[:self.replan_steps], w_bar=0.0, model_error=0.0)
        return tube[-1][FLANGE_FRAME_INDEX].center()

    def select(self, joint_angles, candidates, scores):
        endpoints = [self._endpoint(joint_angles, c) for c in candidates]
        centroid = np.mean(endpoints, axis=0)
        disagreement = [float(np.linalg.norm(e - centroid)) for e in endpoints]
        mask = [d <= self.disagreement_threshold for d in disagreement]
        admissible_idx = [i for i, ok in enumerate(mask) if ok]
        if not admissible_idx:
            fallback = np.zeros_like(candidates[0])
            return fallback, {
                "fallback": True, "n_admissible": 0,
                "admissible_mask": mask, "disagreement": disagreement,
            }
        best_i = max(admissible_idx, key=lambda i: scores[i])
        return candidates[best_i], {
            "fallback": False, "n_admissible": len(admissible_idx),
            "admissible_mask": mask, "disagreement": disagreement,
        }
