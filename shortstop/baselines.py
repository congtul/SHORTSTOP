"""Table II's five comparison baselines (Sec. VI-A): Conf-Thresh, MPC-Filter,
CBF-Shield, STL-Monitor. (Unshielded is just `None` as a shield_factory,
already handled by shortstop.experiment.run_episode.)

Every class here implements the same `.select(state, candidates) ->
(action_chunk, info)` interface as shortstop.shield's classes, so they drop
straight into the existing run_episode()/run_ablation.py harness -- add one
line to STAGES in scripts/run_ablation.py and it's in the comparison table.

Three different shapes, matching how the paper itself describes them --
note K (the number of candidate chunks proposed per step) is *ShortStop's
own* Algorithm 1 concept ("Propose K candidate chunks"), not a generic
interface every method is entitled to; giving a baseline only 1 of the K
candidates shortstop.experiment.run_episode happens to generate (a sharing
convenience of this harness, not a paper mechanism) is not shortchanging
it -- it is what the paper's own wording describes:

- STL-Monitor: paper explicitly frames it as isolating "the value of
  [reachtube-based counterexample guidance]" -- an *ablation of ShortStop's
  own pipeline*, so it deliberately keeps ShortStop's K-candidate/best-of-K
  selection fixed and swaps out only the certification step. See its
  docstring.
- Conf-Thresh: paper's wording is plural ("rejects **chunks** whose...
  disagreement... exceeds a threshold"), consistent with a multi-candidate
  mechanism, so it also searches among K like ShortStop's own shields.
- MPC-Filter, CBF-Shield: independent algorithms from other papers ([33] and
  [37]-[39] respectively), and the paper describes each in the singular --
  "minimally corrects **the first action**", "**a** hand-designed distance
  barrier" -- consistent with "Unshielded (raw diffusion policy)" also being
  a single-sample execution. They act on candidates[0] only (candidates[1:]
  are provided but ignored, since with a single corrected input there is
  nothing to select among); admissible_mask conveys whether the correction
  actually changed the action, so shortstop.experiment.run_episode's
  existing activation/precision bookkeeping (which loops per-candidate)
  still works unmodified on a one-relevant-candidate mask.
"""
import cvxpy as cp
import numpy as np

from .reach import Box, nominal_rollout
from .stl import robustness_to_go


class ConfThreshShield:
    """"Conf-Thresh": rejects chunks whose sampler-ensemble disagreement (a
    proxy for confidence) exceeds a tuned threshold -- no dynamics model, no
    reachtube, no obstacle geometry at all (Table II: 0.43 precision,
    "disagreement is a poor safety proxy").

    This prototype has no literal model ensemble (shortstop.policy's
    GaussianChunkPolicy is a single Gaussian sampler, not several
    independently-trained models), so ensemble disagreement is approximated
    by how far each candidate's endpoint sits from the centroid of all K
    candidates' endpoints -- the same "spread across i.i.d. samples as an
    epistemic-uncertainty proxy" idea real ensembles rely on, just without
    literal separate networks. Document this as an approximation, the same
    way CEShield's closed-form search stands in for the paper's numeric
    M-step search.

    Measured limitation: a threshold sweep (0.5 down to 0.06, i.e. rejecting
    almost every candidate) left violation_rate on run_ablation.py's
    scenario completely flat (~0.81, same as Unshielded at every threshold).
    This proxy is *exactly* uncorrelated with real danger here, not just
    "poor" the way the paper's real diffusion-ensemble Conf-Thresh is
    (Table II: 11.2% violation, a real if weak improvement over 18.7%
    unshielded) -- because GaussianChunkPolicy samples i.i.d. Gaussian noise
    around one fixed goal-reference direction with no coupling to obstacle
    placement at all, so "how far a candidate strays from the K-candidate
    centroid" carries literally zero information about whether the shared
    reference direction happens to point at an obstacle. A real trained
    ensemble's disagreement can pick up *some* signal from out-of-
    distribution states; this synthetic stand-in structurally cannot. Keep
    this in mind before citing this baseline's numbers as a faithful
    reproduction of the paper's Conf-Thresh row.
    """

    def __init__(self, goal, obstacles, dt, w_bar, model_error=0.0, disagreement_threshold=0.15):
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = obstacles
        self.dt = dt
        self.disagreement_threshold = disagreement_threshold

    def _endpoint(self, state, chunk):
        return nominal_rollout(state, chunk, self.dt)[-1]

    def _score(self, state, chunk):
        return -np.linalg.norm(self._endpoint(state, chunk) - self.goal)

    def select(self, state, candidates):
        endpoints = [self._endpoint(state, c) for c in candidates]
        centroid = np.mean(endpoints, axis=0)
        disagreement = [np.linalg.norm(e - centroid) for e in endpoints]
        mask = [d <= self.disagreement_threshold for d in disagreement]

        admissible = [c for c, ok in zip(candidates, mask) if ok]
        if not admissible:
            fallback = np.zeros_like(candidates[0])
            return fallback, {"fallback": True, "n_admissible": 0, "admissible_mask": mask}
        best = max(admissible, key=lambda c: self._score(state, c))
        return best, {"fallback": False, "n_admissible": len(admissible), "admissible_mask": mask}


class STLMonitorShield:
    """"STL-Monitor": STL robustness on the *nominal* rollout only -- no
    reachtube, no counterexample search (Sec. III-B: "our STL-monitor
    baseline (nominal robustness, no counterexample search) isolates the
    value of [reachtube-based counterexample guidance]"). Rejects a
    candidate iff its nominal robustness is negative (paper: "rejects if
    negative") -- a plain threshold of 0, not ShortStop's own calibrated
    epsilon margin.

    Implemented by reusing stl.robustness_to_go on a "tube" of zero-width
    boxes (Box.point at every nominal-rollout position) -- the same formula,
    just without the reachtube's disturbance/model-error inflation that
    shortstop.shield.STLShield adds.

    Unlike MPC-Filter/CBF-Shield (independent algorithms from other papers,
    which the paper's text describes acting on a single proposed input --
    see their docstrings), this class still searches for the best-scoring
    candidate among all K, exactly like ShortStop's own STLShield. That is
    intentional, not an oversight: the paper explicitly frames STL-Monitor
    as isolating "the value of [reachtube-based counterexample guidance]" --
    i.e. an ablation of ShortStop's own pipeline with only the certification
    step swapped out (reachtube+epsilon -> nominal rollout+zero threshold),
    everything else (K candidates, best-of-K selection, fallback) held
    fixed, precisely so the comparison isolates that one difference.
    """

    def __init__(self, goal, obstacles, dt, w_bar, model_error=0.0):
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = obstacles
        self.dt = dt

    def _nominal_robustness(self, state, chunk):
        path = nominal_rollout(state, chunk, self.dt)
        tube = [Box.point(p) for p in path]
        return robustness_to_go(tube, self.obstacles)

    def _score(self, state, chunk):
        final = nominal_rollout(state, chunk, self.dt)[-1]
        return -np.linalg.norm(final - self.goal)

    def select(self, state, candidates):
        mask = [self._nominal_robustness(state, c) >= 0.0 for c in candidates]

        admissible = [c for c, ok in zip(candidates, mask) if ok]
        if not admissible:
            fallback = np.zeros_like(candidates[0])
            return fallback, {"fallback": True, "n_admissible": 0, "admissible_mask": mask}
        best = max(admissible, key=lambda c: self._score(state, c))
        return best, {"fallback": False, "n_admissible": len(admissible), "admissible_mask": mask}


class _QPCorrectionShield:
    """Single-step QP correction machinery, used by CBFShield only.

    CBF-QP is *pointwise by construction* (Ames et al.'s formulation --
    Sec. III-D of the paper explicitly contrasts it with predictive
    filters: "Control barrier functions enforce forward invariance ... via
    a pointwise QP"), so a single-action, single-constraint QP each step is
    the faithful implementation, unlike MPCFilterShield (see its docstring
    for why that one needs the full horizon instead).

    Corrects only the first action of candidates[0] via a small QP, rather
    than selecting among the K candidates -- see module docstring for why.
    Subclasses provide `_build_constraints(state, a, a_nominal) -> list of
    cvxpy constraints`; everything else (objective, action bound, solving,
    reporting whether an intervention happened) is shared.
    """

    def __init__(self, goal, obstacles, dt, w_bar, max_action_norm=1.0, model_error=0.0, action_dim=2):
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = obstacles
        self.dt = dt
        self.w_bar = w_bar
        self.max_action_norm = max_action_norm
        self.action_dim = action_dim

    def _build_constraints(self, state, a, a_nominal):
        raise NotImplementedError

    def select(self, state, candidates):
        state = np.asarray(state, dtype=float)
        chunk = candidates[0].copy()
        a_nominal = chunk[0].copy()

        a = cp.Variable(self.action_dim)
        constraints = [cp.norm(a, "inf") <= self.max_action_norm]
        constraints += self._build_constraints(state, a, a_nominal)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(a - a_nominal)), constraints)
        try:
            problem.solve()
        except cp.error.SolverError:
            a.value = None

        if a.value is None:
            fallback = np.zeros_like(chunk)
            mask = [False] + [True] * (len(candidates) - 1)
            return fallback, {
                "fallback": True,
                "n_admissible": 0,
                "admissible_mask": mask,
                # QP infeasible: the correction was needed but failed --
                # counts toward recovery_rate the same way a failed
                # RepairShield repair does (shortstop.metrics.aggregate).
                "repair_attempted": True,
                "repair_succeeded": False,
            }

        a_corrected = np.clip(a.value, -self.max_action_norm, self.max_action_norm)
        chunk[0] = a_corrected
        intervened = not np.allclose(a_corrected, a_nominal, atol=1e-6)
        mask = [not intervened] + [True] * (len(candidates) - 1)
        n_admissible = len(candidates) - (1 if intervened else 0)
        return chunk, {
            "fallback": False,
            "n_admissible": n_admissible,
            "admissible_mask": mask,
            "intervened": intervened,
            # A feasible QP solve always means "the correction worked" by
            # construction -- there's no separate re-certification step
            # like RepairShield's, so attempted == succeeded whenever an
            # intervention happened at all.
            "repair_attempted": intervened,
            "repair_succeeded": intervened,
        }


class MPCFilterShield:
    """"MPC-Filter": a predictive safety filter [Wabersich & Zeilinger,
    "A predictive safety filter for learning-based control of constrained
    nonlinear dynamical systems," Automatica 2021 -- the paper's ref [33]].

    That reference solves a constrained optimal-control problem over the
    *entire* prediction horizon (state constraints at every step, a
    terminal safe-set constraint), then -- receding-horizon style -- only
    *executes* the first corrected action before re-solving next step. This
    is the same execution pattern every shield in this repo already uses
    (only chunk[0] is ever applied). An earlier version of this class
    collapsed the whole thing to a 1-step lookahead (constrain only the very
    next position, ignore steps 2..H of the same chunk) -- that is
    materially weaker than a real PSF: it made this baseline look almost as
    safe as ShortStop while keeping far higher success, because a myopic
    filter on this prototype's slow-moving, well-separated circular
    obstacles rarely needs to sacrifice any progress. This version instead
    optimizes the *whole* H-step chunk at once, with an obstacle constraint
    at every predicted step -- matching ShortStop's own H-step reachtube
    certification in scope, if not in soundness (see below).

    Obstacle avoidance ("stay outside this circle") is non-convex, so each
    step's constraint is linearized as a supporting hyperplane of the circle
    at that step's *nominal* (uncorrected) predicted point -- the same
    single-linearization-pass simplification the 1-step version used,
    just applied at every k=1..H instead of only k=1. Every hyperplane is
    tightened by `w_bar` for the same reason as before: an equality-tight
    correction has roughly even odds of being undone by the next
    disturbance draw.

    Unlike ShortStop, there is no soundness proof here: the linearization is
    exact only at the one nominal point per step, so a large correction at
    an early step can invalidate the picture the later steps were
    linearized around. A real iterative PSF would re-linearize and re-solve
    (SQP-style) until convergence; this is a single linearization pass,
    documented as such rather than silently presented as equivalent.
    """

    def __init__(self, goal, obstacles, dt, w_bar, max_action_norm=1.0, model_error=0.0, action_dim=2):
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = obstacles
        self.dt = dt
        self.w_bar = w_bar
        self.max_action_norm = max_action_norm
        self.action_dim = action_dim

    def select(self, state, candidates):
        state = np.asarray(state, dtype=float)
        nominal_chunk = candidates[0]
        horizon = len(nominal_chunk)
        nominal_path = nominal_rollout(state, nominal_chunk, self.dt)  # length horizon+1

        a = cp.Variable((horizon, self.action_dim))
        constraints = [cp.norm(a, "inf", axis=1) <= self.max_action_norm]

        fallback_direction = np.eye(self.action_dim)[0]
        x = state
        for k in range(horizon):
            x_next = x + a[k] * self.dt
            p_nominal = nominal_path[k + 1]
            for o in self.obstacles:
                direction = p_nominal - o.center
                norm = np.linalg.norm(direction)
                n = direction / norm if norm > 1e-9 else fallback_direction
                constraints.append(n @ (x_next - o.center) >= o.radius + self.w_bar)
            x = x_next

        problem = cp.Problem(cp.Minimize(cp.sum_squares(a - nominal_chunk)), constraints)
        try:
            problem.solve()
        except cp.error.SolverError:
            a.value = None

        if a.value is None:
            mask = [False] + [True] * (len(candidates) - 1)
            return np.zeros_like(nominal_chunk), {
                "fallback": True,
                "n_admissible": 0,
                "admissible_mask": mask,
                "repair_attempted": True,
                "repair_succeeded": False,
            }

        chunk = np.clip(a.value, -self.max_action_norm, self.max_action_norm)
        intervened = not np.allclose(chunk, nominal_chunk, atol=1e-6)
        mask = [not intervened] + [True] * (len(candidates) - 1)
        n_admissible = len(candidates) - (1 if intervened else 0)
        return chunk, {
            "fallback": False,
            "n_admissible": n_admissible,
            "admissible_mask": mask,
            "intervened": intervened,
            "repair_attempted": intervened,
            "repair_succeeded": intervened,
        }


class CBFShield(_QPCorrectionShield):
    """"CBF-Shield": a control-barrier-function QP with a hand-designed
    distance barrier h(x) = ||x - center||^2 - radius^2 (safe iff h >= 0),
    enforcing the standard CBF derivative condition
    grad_h(x) . a + alpha*h(x) >= 0 for a tunable class-K gain `alpha`
    (Ames et al., refs [37]-[39]) -- linear in `a` since h is quadratic in
    state alone, so this is an exact constraint for this dynamics, not an
    approximation (unlike MPC-Filter's tangent-plane linearization).
    """

    def __init__(self, goal, obstacles, dt, w_bar, max_action_norm=1.0, alpha=1.0, model_error=0.0, action_dim=2):
        super().__init__(goal, obstacles, dt, w_bar, max_action_norm, model_error, action_dim)
        self.alpha = alpha

    def _build_constraints(self, state, a, a_nominal):
        constraints = []
        for o in self.obstacles:
            diff = state - o.center
            h = float(diff @ diff - o.radius ** 2)
            grad_h = 2.0 * diff  # d/dx [(x-c).(x-c) - r^2]
            constraints.append(grad_h @ a >= -self.alpha * h)
        return constraints
