import numpy as np
import pytest

from shortstop.pi_policy_client import MockPi05PolicyClient, Pi05PolicyClient


def test_mock_client_returns_k_chunks_of_the_right_shape():
    client = MockPi05PolicyClient(horizon=8, action_dim=7, n_candidates=8, rng=np.random.default_rng(0))
    candidates = client.propose(observation={})
    assert len(candidates) == 8
    for chunk in candidates:
        assert chunk.shape == (8, 7)


def test_real_client_raises_a_clear_error_without_openpi_installed():
    with pytest.raises(ImportError, match="openpi_client"):
        Pi05PolicyClient()
