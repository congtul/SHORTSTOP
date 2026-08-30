import numpy as np


def _signed_distance_to_obstacle(box, obstacle):
    """Inner term of formula (2), dist(x, X_u), minimized over x in box for
    one circular obstacle.

    dist(x, X_u) for a circular unsafe region is ||x - center|| - radius
    (positive outside, negative inside). This is monotone in ||x - center||,
    so its infimum over the box reduces to the box's closest point to the
    circle center.
    """
    closest = box.closest_point(obstacle.center)
    return float(np.linalg.norm(closest - obstacle.center) - obstacle.radius)


def step_robustness(box, obstacles):
    """inf_{x in box} dist(x, X_u) for X_u = union of obstacles."""
    if not obstacles:
        return float("inf")
    return min(_signed_distance_to_obstacle(box, o) for o in obstacles)


def robustness_to_go(tube, obstacles):
    """Formula (2): rho(phi, R) = min_{1<=k<=H} inf_{x in R_k} dist(x, X_u).

    tube[0] is R_0, the already-realized current state, and is excluded --
    same convention ReachOnlyShield uses (tube[1:]) for the binary check this
    replaces.
    """
    return min(step_robustness(box, obstacles) for box in tube[1:])


def find_counterexample(tube, obstacles):
    """Formula (3): argmin_{k, x in R_k} dist(x, X_u).

    Returns a dict with the violating tube index `step` (1-based, matching
    tube[k]), the offending `obstacle`, the witness point `witness`, and the
    `robustness` value there. Closed-form box-vs-circle search stands in for
    the paper's projected-gradient / vertex-sampling search -- exact here
    because obstacles are circles and the reachtube is axis-aligned boxes.
    """
    best = None
    for k in range(1, len(tube)):
        box = tube[k]
        for obstacle in obstacles:
            value = _signed_distance_to_obstacle(box, obstacle)
            if best is None or value < best["robustness"]:
                best = {
                    "step": k,
                    "obstacle": obstacle,
                    "witness": box.closest_point(obstacle.center),
                    "robustness": value,
                }
    return best
