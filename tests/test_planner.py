import numpy as np
import pytest

from shortstop.planner import mock_get_path, real_get_path
from shortstop.robot_geometry import N_JOINTS, panda_frames


def test_mock_get_path_shape_and_endpoints():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    target = current + np.array([0.05, 0.0, 0.0])

    path = mock_get_path(q0, target, n_waypoints=5)
    assert path.shape == (5, N_JOINTS)
    assert np.allclose(path[0], q0)

    achieved = panda_frames(path[-1])[-1]
    assert np.linalg.norm(achieved - target) < 1e-3


def test_real_get_path_raises_without_pyrep_installed():
    with pytest.raises(ImportError, match="PyRep"):
        real_get_path(arm=None, target_position=[0.0, 0.0, 0.0])
