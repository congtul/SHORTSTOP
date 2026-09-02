import numpy as np
import pytest

from shortstop.mini_diffuser_client import MiniDiffuserClient, MockMiniDiffuserClient


def test_mock_client_returns_k_keyposes_of_the_right_shape():
    client = MockMiniDiffuserClient(n_candidates=6, rng=np.random.default_rng(0))
    candidates = client.propose(observation={})
    assert len(candidates) == 6
    for keypose in candidates:
        assert keypose.shape == (8,)  # 3 position + 4 quaternion + 1 gripper


def test_real_client_is_explicitly_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="serving interface"):
        MiniDiffuserClient()
