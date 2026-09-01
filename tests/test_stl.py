import numpy as np

from shortstop.env import Obstacle
from shortstop.reach import propagate_tube
from shortstop.stl import find_counterexample, robustness_to_go, step_robustness


def test_step_robustness_sign():
    obstacle = Obstacle(center=[0.0, 0.0], radius=0.3)
    box = propagate_tube([1.0, 0.0], actions=np.zeros((1, 2)), dt=0.1, w_bar=0.0)[-1]
    assert step_robustness(box, [obstacle]) > 0.0  # far outside

    box_inside = propagate_tube([0.0, 0.0], actions=np.zeros((1, 2)), dt=0.1, w_bar=0.0)[-1]
    assert step_robustness(box_inside, [obstacle]) < 0.0  # sitting on the obstacle center


def test_robustness_to_go_excludes_r0():
    # R_0 = (0,0) sits inside the obstacle (dist = -0.05), but by step k=1 the
    # point mass has already moved past its boundary (dist = 0.1 - 0.05 = 0.05).
    obstacle = Obstacle(center=[0.0, 0.0], radius=0.05)
    tube = propagate_tube([0.0, 0.0], actions=np.tile([1.0, 0.0], (3, 1)), dt=0.1, w_bar=0.0)

    assert step_robustness(tube[0], [obstacle]) < 0.0
    rho = robustness_to_go(tube, [obstacle])
    assert rho > 0.0


def test_find_counterexample_locates_worst_step_and_obstacle():
    far = Obstacle(center=[10.0, 10.0], radius=0.1)
    near = Obstacle(center=[0.5, 0.0], radius=0.3)
    tube = propagate_tube([0.0, 0.0], np.tile([1.0, 0.0], (8, 1)), dt=0.1, w_bar=0.0)

    ce = find_counterexample(tube, [far, near])

    assert ce["obstacle"] is near
    assert ce["step"] == 5  # position 0.1*5 = 0.5 == obstacle center
    assert ce["robustness"] < 0.0
    assert np.allclose(ce["witness"], [0.5, 0.0])
