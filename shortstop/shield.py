import numpy as np

from .reach import nominal_rollout, propagate_tube
from .stl import find_counterexample, robustness_to_go


class ReachOnlyShield:
    """Stage 1 shield: Propose -> Reach -> reject on tube/obstacle intersection.

    Binary admissibility only -- no STL robustness-to-go (Stage 2, see
    STLShield below), no counterexample localization (Stage 3, CEShield), no
    repair (Stage 4, RepairShield).
    Selection among admissible chunks uses a progress-to-goal score as a
    stand-in for g(a) in Eq. (5); falls back to braking (zero action) if no
    candidate chunk is admissible.
    """

    def __init__(self, goal, obstacles, dt, w_bar, model_error=0.0):
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = obstacles
        self.dt = dt
        self.w_bar = w_bar
        self.model_error = model_error

    def _admissible(self, state, chunk):
        tube = propagate_tube(state, chunk, self.dt, self.w_bar, self.model_error)
        return not any(box.intersects_any(self.obstacles) for box in tube[1:])

    def _score(self, state, chunk):
        final = nominal_rollout(state, chunk, self.dt)[-1]
        return -np.linalg.norm(final - self.goal)

    def select(self, state, candidates):
        mask = [self._admissible(state, c) for c in candidates]
        admissible = [c for c, ok in zip(candidates, mask) if ok]
        if not admissible:
            fallback = np.zeros_like(candidates[0])
            return fallback, {"fallback": True, "n_admissible": 0, "admissible_mask": mask}
        best = max(admissible, key=lambda c: self._score(state, c))
        return best, {"fallback": False, "n_admissible": len(admissible), "admissible_mask": mask}


class STLShield(ReachOnlyShield):
    """Stage 2 shield: replace the binary tube/obstacle intersection test
    with a continuous STL robustness-to-go margin (Eq. 2),
    rho(phi, R) >= epsilon.

    epsilon > 0 keeps a calibrated safety margin instead of accepting chunks
    that graze an obstacle boundary exactly -- this is the "epsilon
    calibration" knob flagged as an open challenge in Part 6 of the report.
    Selection logic (progress score, fallback) is unchanged from Stage 1.
    """

    def __init__(self, goal, obstacles, dt, w_bar, model_error=0.0, epsilon=0.05):
        super().__init__(goal, obstacles, dt, w_bar, model_error)
        self.epsilon = epsilon

    def _robustness(self, state, chunk):
        tube = propagate_tube(state, chunk, self.dt, self.w_bar, self.model_error)
        return robustness_to_go(tube, self.obstacles)

    def _admissible(self, state, chunk):
        return self._robustness(state, chunk) >= self.epsilon


class CEShield(STLShield):
    """Stage 3 shield: same STL-to-go admissibility test as Stage 2, but also
    localizes a concrete counterexample (Eq. 3) for every rejected candidate.

    This does not change which chunks get accepted -- finding *where* a
    candidate fails is a prerequisite for Stage 4's repair, not yet a fix by
    itself. `info["counterexamples"]` is aligned with the candidate list;
    `None` for admissible chunks.

    `_diagnose` is factored out (rather than inlined in `select`) so Stage 4
    (RepairShield) can inherit from this class and reuse the exact same
    admissibility + counterexample pass as its starting point, instead of
    re-deriving it independently.
    """

    def _diagnose(self, state, candidates):
        mask = []
        counterexamples = []
        for chunk in candidates:
            tube = propagate_tube(state, chunk, self.dt, self.w_bar, self.model_error)
            ok = robustness_to_go(tube, self.obstacles) >= self.epsilon
            mask.append(ok)
            counterexamples.append(None if ok else find_counterexample(tube, self.obstacles))
        return mask, counterexamples

    def select(self, state, candidates):
        mask, counterexamples = self._diagnose(state, candidates)
        admissible = [c for c, ok in zip(candidates, mask) if ok]
        if not admissible:
            fallback = np.zeros_like(candidates[0])
            return fallback, {
                "fallback": True,
                "n_admissible": 0,
                "admissible_mask": mask,
                "counterexamples": counterexamples,
            }
        best = max(admissible, key=lambda c: self._score(state, c))
        return best, {
            "fallback": False,
            "n_admissible": len(admissible),
            "admissible_mask": mask,
            "counterexamples": counterexamples,
        }


class RepairShield(CEShield):
    """Stage 4 shield: full Algorithm 1 Select/Repair, built directly on top
    of Stage 3's counterexample search (CEShield._diagnose) rather than
    re-deriving it.

    Eq. 4 has *two* separate knobs that the paper's Table VII also lists
    separately (step size eta=0.05, trust region delta=0.1):
    a' = Proj_{A,delta}(a + eta * d) -- eta controls how big one gradient
    step is, delta caps how far the *cumulative* repair is allowed to drift
    from the original candidate. Collapsing them into a single
    "trust_region" (as an earlier version of this class did, using it
    directly as the step length) is not faithful to Eq. 4: it means every
    repair moves by exactly the trust-region radius instead of taking a
    small step within it.

    With max_repair_iters=1 (the default), this matches Algorithm 1 lines
    7-13 exactly: a rejected candidate gets *one* counterexample-guided
    gradient step (Eq. 4), projected into a trust region
    ||a' - a_original|| <= trust_region, then *one* re-certification. If
    that's not enough, the paper simply drops the candidate -- no retry.

    max_repair_iters > 1 goes *beyond* the paper: it repeats the
    counterexample-search-then-repair cycle (in the spirit of classic CEGIS,
    which iterates until no counterexample remains), re-localizing a new
    counterexample after each unsuccessful step instead of giving up after
    one attempt, with every step still projected back to the same trust
    region around the *original* candidate. Algorithm 1 does not specify
    this -- treat it as an experimental knob, not the paper-faithful
    default.

    A latency footgun this class exposes: `shield_activation_rate` here
    reflects usability *after* repair, so a candidate that Stage 3 would
    have flagged as rejected can show up here as "no activation" once
    repaired -- for the exact same underlying obstacle difficulty. Because
    shortstop.metrics.aggregate()'s `latency_ms_median` is a median over
    *every* decision step, if that lowered activation rate pushes the
    fraction of "had to diagnose/repair" steps below 50%, the median stops
    reflecting the expensive branch at all, even though the mean/p95 still
    do. This is exactly what made an earlier (now-fixed) version of this
    pipeline look like "Stage 4 is faster than Stage 3" -- it wasn't; a
    redundant reachtube recomputation in CEShield.select() was inflating
    Stage 3's own cost on top of the median-hiding-tail effect. Always read
    latency_ms_mean/latency_ms_p95 alongside the median when comparing
    stages with different activation rates.

    Note: `info["admissible_mask"]` here reflects final usability *after*
    repair, not the original candidate -- a chunk that started unsafe but was
    fixed shows up as admissible (it does get executed, just not verbatim).
    """

    def __init__(
        self,
        goal,
        obstacles,
        dt,
        w_bar,
        model_error=0.0,
        epsilon=0.05,
        trust_region=0.3,
        step_size=0.05,
        max_repair_iters=1,
        max_action_norm=1.0,
    ):
        super().__init__(goal, obstacles, dt, w_bar, model_error, epsilon)
        self.trust_region = trust_region
        self.step_size = step_size
        self.max_repair_iters = max_repair_iters
        self.max_action_norm = max_action_norm

    def _repair_direction(self, state, chunk, counterexample):
        k_star = counterexample["step"]
        obstacle = counterexample["obstacle"]

        direction = counterexample["witness"] - obstacle.center
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            # Degenerate: witness sits on the obstacle center (deep inside)
            # -- push away from the chunk's own nominal path point instead of
            # an undefined direction. (box.center() == nominal_rollout point,
            # since inflate() doesn't move the box center.)
            nominal_point = nominal_rollout(state, chunk, self.dt)[k_star]
            direction = nominal_point - obstacle.center
            norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])
        return k_star, direction

    def _project_to_trust_region(self, chunk, original):
        """Pi_{A,delta} in Eq. 4: cap the total deviation from the original
        candidate to an L2 ball of radius trust_region, then clip to the
        action-magnitude bound. Measured over the whole chunk (rather than
        only the entries touched so far) since a later iteration may move a
        different prefix (a new k_star) than an earlier one did."""
        delta = chunk - original
        norm = np.linalg.norm(delta)
        if norm > self.trust_region:
            delta = delta * (self.trust_region / norm)
        return np.clip(original + delta, -self.max_action_norm, self.max_action_norm)

    def _repair(self, state, chunk, counterexample):
        """Counterexample-guided repair, starting from Stage 3's
        counterexample. With max_repair_iters=1 (default) this is exactly
        Algorithm 1's single gradient step (size eta) + single
        re-certification, projected into the trust region; with
        max_repair_iters > 1 it re-localizes a new counterexample after each
        failed step and tries again (CEGIS-style extension, see class
        docstring). Returns (repaired chunk, success)."""
        original = chunk.copy()
        chunk = chunk.copy()
        ce = counterexample
        for _ in range(self.max_repair_iters):
            k_star, direction = self._repair_direction(state, chunk, ce)

            # d(x_k)/d(a_i) = dt * I for every i < k_star (Eq. 4): the
            # gradient nudges every action up to the violating step equally.
            chunk[:k_star] = chunk[:k_star] + self.step_size * direction
            chunk = self._project_to_trust_region(chunk, original)

            tube = propagate_tube(state, chunk, self.dt, self.w_bar, self.model_error)
            if robustness_to_go(tube, self.obstacles) >= self.epsilon:
                return chunk, True
            ce = find_counterexample(tube, self.obstacles)

        return chunk, False

    def select(self, state, candidates):
        mask, counterexamples = self._diagnose(state, candidates)
        repaired_candidates = list(candidates)
        repair_attempted = False
        repair_succeeded = False

        for i, (chunk, ok, ce) in enumerate(zip(candidates, mask, counterexamples)):
            if ok:
                continue
            repair_attempted = True
            fixed_chunk, success = self._repair(state, chunk, ce)
            if success:
                repaired_candidates[i] = fixed_chunk
                mask[i] = True
                repair_succeeded = True
            # else: repair failed after max_repair_iters attempts -- with the
            # default of 1, this matches Algorithm 1 (no retry; a failed
            # repair is simply dropped).

        admissible = [c for c, ok in zip(repaired_candidates, mask) if ok]
        if not admissible:
            fallback = np.zeros_like(candidates[0])
            return fallback, {
                "fallback": True,
                "n_admissible": 0,
                "admissible_mask": mask,
                "repair_attempted": repair_attempted,
                "repair_succeeded": repair_succeeded,
            }
        best = max(admissible, key=lambda c: self._score(state, c))
        return best, {
            "fallback": False,
            "n_admissible": len(admissible),
            "admissible_mask": mask,
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
        }
