import numpy as np
import pytest

from shortstop.mdt_policy_client import MDTPolicyClient, MockMDTPolicyClient


def test_mock_client_returns_k_chunks_of_the_right_shape():
    client = MockMDTPolicyClient(horizon=10, action_dim=7, n_candidates=8, rng=np.random.default_rng(0))
    candidates = client.propose(observation={})
    assert len(candidates) == 8
    for chunk in candidates:
        assert chunk.shape == (10, 7)


def test_real_client_raises_a_clear_error_without_mdt_policy_installed():
    with pytest.raises(ImportError, match="mdt_policy"):
        MDTPolicyClient(checkpoint_path="unused")
