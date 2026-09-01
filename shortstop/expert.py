"""Scripted expert for Reach-Avoid-2D (Stage 6a dataset generation).

Perpendicular-offset circumnavigation planner: threads a polyline from start
to goal that stays outside every obstacle's safety disk (radius + margin).
When the straight shot to the goal is blocked by an obstacle, two detour
waypoints ("entry", "exit") are inserted that skirt around it, offset
perpendicular to the current travel direction. Choosing the "upper" vs
"lower" perpendicular consistently at every contested obstacle gives two
globally-distinct bypass variants per scenario -- the source of
multimodality for the behavior-cloning dataset (GaussianChunkPolicy has no
real behavioral diversity; this does).

Two waypoints per detour, not one: a single point placed directly beside the
obstacle is not enough when the obstacle sits close to dead-ahead on the
straight line -- the leg from that single point back toward a *distant*
goal can cut back through the same disk. Offsetting the entry/exit pair
*along* the travel direction as well as across it keeps the straight
entry->exit segment at constant perpendicular clearance the whole way past
the obstacle (verified in tests/test_expert.py; an earlier single-tangent-
point and single-center-offset version both failed on exactly this case).

A blocking obstacle can still shift the geometry enough that some *other*
leg of the resulting path clips a different obstacle (rare, since the 3
obstacles per scenario are sparse) -- `plan_waypoints` explicitly re-checks
every leg of the finished polyline against every obstacle before returning,
and raises if any leg is unsafe, rather than silently emitting a bad path.
Scenarios where a mode fails this check are simply dropped from the dataset
(see generate_demo_pair) -- this only needs a high success rate, not 100%.

The polyline is then driven through the *real* ReachAvoid2D via a reactive
pure-pursuit controller, so recorded (state, action) pairs already include
the env's disturbance noise -- avoids the train/test distribution mismatch
an open-loop recorded action sequence would have.
"""
import numpy as np

from .env import ReachAvoid2D


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def _perp_unit(v):
    d = _unit(v)
    return np.array([-d[1], d[0]])


def _segment_blocked(p, q, center, radius):
    """Does segment p->q pass within `radius` of `center`?

    Returns (blocked, t) where t in [0, 1] is the projection parameter of
    the closest approach along the segment (clamped to the segment).
    """
    p, q, center = np.asarray(p, float), np.asarray(q, float), np.asarray(center, float)
    seg = q - p
    seg_len2 = seg @ seg
    t = 0.0 if seg_len2 < 1e-12 else float(np.clip((center - p) @ seg / seg_len2, 0.0, 1.0))
    closest = p + t * seg
    blocked = np.linalg.norm(closest - center) < radius
    return blocked, t


def _choose_perp(travel_direction, mode):
    perp = _perp_unit(travel_direction)
    upper, lower = (perp, -perp) if perp[1] >= -perp[1] else (-perp, perp)
    return upper if mode == "upper" else lower


def _bypass_waypoints(travel_direction, obstacle, margin, mode, lookahead_factor=0.3):
    """Two detour points that skirt `obstacle` at constant perpendicular
    clearance (radius + margin) from its center: `entry` (offset *before*
    the obstacle along `travel_direction`) and `exit_` (offset *after* it).
    Because both share the same perpendicular offset, the straight segment
    entry->exit stays at exactly that clearance the whole way -- a single
    offset waypoint placed only *beside* the obstacle is not enough (the
    leg back toward a distant goal can cut through the disk again, e.g.
    when the obstacle sits dead-ahead on the straight line).
    """
    d = _unit(travel_direction)
    n = _choose_perp(d, mode)
    r_safe = obstacle.radius + margin
    along = r_safe * lookahead_factor
    entry = obstacle.center + n * r_safe - d * along
    exit_ = obstacle.center + n * r_safe + d * along
    return entry, exit_


def _nearest_blocking(p, q, obstacles, margin):
    blocking, blocking_t = None, None
    for obs in obstacles:
        blocked, t = _segment_blocked(p, q, obs.center, obs.radius + margin)
        if blocked and (blocking is None or t < blocking_t):
            blocking, blocking_t = obs, t
    return blocking


def plan_waypoints(start, goal, obstacles, margin=0.1, mode="upper", max_iters=None):
    """Perpendicular-offset circumnavigation planner (see module docstring).

    Walks a stack of pending targets, starting with just `goal`. Whenever
    the segment from the current point to the top-of-stack target is
    blocked, an entry/exit detour pair around the nearest blocker is pushed
    in its place -- so if *that* detour leg is itself blocked by some other
    obstacle (two obstacles close together), the next iteration detours
    around that one too, before ever falling back to the original target.
    Every accepted leg is checked this way, not just the top-level
    start->goal shot, so no separate end-of-path verification is needed.
    `mode` picks "upper" or "lower" at every contested obstacle, giving one
    of the two globally-consistent bypass variants. Raises RuntimeError if
    the stack doesn't empty out within max_iters (obstacles packed too
    tightly for this detour shape to resolve).
    """
    if mode not in ("upper", "lower"):
        raise ValueError("mode must be 'upper' or 'lower'")
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    waypoints = [start]
    current = start
    pending = [goal]
    if max_iters is None:
        max_iters = 6 * len(obstacles) + 4

    for _ in range(max_iters):
        if not pending:
            return waypoints
        target = pending[-1]
        blocking = _nearest_blocking(current, target, obstacles, margin)
        if blocking is None:
            waypoints.append(target)
            current = target
            pending.pop()
            continue
        entry, exit_ = _bypass_waypoints(target - current, blocking, margin, mode)
        pending.append(exit_)
        pending.append(entry)

    raise RuntimeError("planner did not converge within max_iters")


def rollout_expert(env, waypoints, switch_radius=0.15):
    """Drive `env` along `waypoints` with a reactive pure-pursuit controller.

    Always steers at max speed toward the nearest not-yet-reached waypoint,
    advancing once within `switch_radius`. Runs through the real
    `env.step`, so recorded states/actions include the env's actual
    disturbance noise.
    """
    waypoints = [np.asarray(w, dtype=float) for w in waypoints]
    state = env.reset()
    states = [state.copy()]
    actions = []
    target_idx = 1  # waypoints[0] == start

    for _ in range(env.max_steps):
        target = waypoints[target_idx]
        direction = target - state
        if np.linalg.norm(direction) < switch_radius and target_idx < len(waypoints) - 1:
            target_idx += 1
            target = waypoints[target_idx]
            direction = target - state

        action = _unit(direction) * env.max_action_norm
        actions.append(action)
        state, done, info = env.step(action)
        states.append(state.copy())
        if done:
            break

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "reached": bool(info["reached"]),
        "violated": bool(info["violated"]),
    }


def generate_demo_pair(start, goal, obstacles, dt=0.1, w_bar=0.02, margin=0.1, rng=None):
    """Roll out both the 'upper' and 'lower' bypass variants for one scenario.

    Returns a dict keyed by mode, containing only the modes whose plan
    converged and whose rollout reached the goal without a violation
    (should be the vast majority of scenarios; see tests/test_expert.py and
    scripts/verify_expert.py for measured rates).
    """
    rng = rng if rng is not None else np.random.default_rng()
    out = {}
    for mode in ("upper", "lower"):
        try:
            waypoints = plan_waypoints(start, goal, obstacles, margin=margin, mode=mode)
        except RuntimeError:
            continue
        env = ReachAvoid2D(
            start=start, goal=goal, obstacles=obstacles, dt=dt, w_bar=w_bar,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        result = rollout_expert(env, waypoints)
        if result["reached"] and not result["violated"]:
            out[mode] = result
    return out
