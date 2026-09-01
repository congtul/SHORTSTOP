import numpy as np

from shortstop.dataset import build_dataset, encode_obstacles, windows_from_demo
from shortstop.env import Obstacle


def test_encode_obstacles_is_sorted_by_center_x_and_fixed_size():
    obstacles = [
        Obstacle(center=[1.0, 2.0], radius=0.5),
        Obstacle(center=[-1.0, 0.0], radius=0.4),
        Obstacle(center=[0.0, -3.0], radius=0.6),
    ]
    vec = encode_obstacles(obstacles)
    assert vec.shape == (9,)
    xs = vec.reshape(3, 3)[:, 0]
    assert np.all(np.diff(xs) >= 0)


def test_windows_from_demo_pairs_state_t_with_actions_from_t():
    demo = {
        "states": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        "actions": np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
    }
    windows = list(windows_from_demo(demo, obstacle_vec=np.zeros(9), horizon=2))
    assert len(windows) == 2  # 3 actions, horizon 2 -> t in {0, 1}
    state0, ov0, chunk0 = windows[0]
    assert np.allclose(state0, [0.0, 0.0])
    assert np.allclose(chunk0, [[1.0, 0.0], [1.0, 0.0]])


def test_build_dataset_reaches_target_and_every_window_shape_is_consistent():
    data = build_dataset(target_demos=20, horizon=8, seed_start=0)

    assert data["n_demos"] >= 20
    n = len(data["states"])
    assert data["obstacle_vecs"].shape == (n, 9)
    assert data["action_chunks"].shape == (n, 8, 2)
    assert data["states"].shape == (n, 2)
