import numpy as np

from shortstop.env import Obstacle, ReachAvoid2D


def test_reaches_goal_without_obstacles():
    env = ReachAvoid2D(
        start=[0.0, 0.0], goal=[1.0, 0.0], goal_radius=0.2,
        obstacles=[], dt=0.1, w_bar=0.0, rng=np.random.default_rng(0),
    )
    env.reset()
    action = np.array([1.0, 0.0])
    info = {}
    for _ in range(50):
        _, done, info = env.step(action)
        if done:
            break
    assert info["reached"]


def test_detects_obstacle_violation():
    obstacle = Obstacle(center=[0.5, 0.0], radius=0.3)
    env = ReachAvoid2D(
        start=[0.0, 0.0], goal=[5.0, 0.0], obstacles=[obstacle],
        dt=0.1, w_bar=0.0, rng=np.random.default_rng(0),
    )
    env.reset()
    action = np.array([1.0, 0.0])  # max_action_norm=1.0, so this is not clipped
    info = {}
    for _ in range(20):
        _, done, info = env.step(action)
        if done:
            break
    assert info["violated"]
