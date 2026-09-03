import numpy as np

from shortstop.calvin_obstacle_viz import save_subtask_gif
from shortstop.env import Obstacle
from shortstop.robot_geometry import N_JOINTS, sphere_centers


def _fake_trajectory(n_steps=4):
    trajectory = []
    for k in range(n_steps):
        q = np.zeros(N_JOINTS)
        q[0] = 0.05 * k
        trajectory.append(sphere_centers(q))
    return trajectory


def test_save_subtask_gif_writes_a_nonempty_file_with_obstacle(tmp_path):
    trajectory = _fake_trajectory()
    obstacle = Obstacle(center=trajectory[-1][-1], radius=0.05)
    out_path = tmp_path / "attempt.gif"

    save_subtask_gif(trajectory, obstacle, str(out_path), fps=5)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_subtask_gif_writes_a_nonempty_file_without_obstacle(tmp_path):
    trajectory = _fake_trajectory()
    out_path = tmp_path / "attempt_no_obstacle.gif"

    save_subtask_gif(trajectory, None, str(out_path), fps=5)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
