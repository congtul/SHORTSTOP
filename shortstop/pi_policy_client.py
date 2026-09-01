"""Client for the pi0.5 policy server (Stage 7a Propose step).

I/O contract confirmed by reading openpi's own LIBERO eval example
(examples/libero/main.py) rather than guessed:
  - observation sent to the server: two 224x224 uint8 RGB images (main +
    wrist camera), a 7D proprioceptive state (3D end-effector position +
    3D axis-angle rotation + 1D gripper), and a language "prompt" string.
  - action returned: a *chunk* (sequence) of 7D vectors -- 6D end-effector
    pose delta (3D position + 3D rotation) + 1D gripper, executed via
    receding-horizon replanning (only the first `replan_steps` actions of
    each returned chunk are used before requesting a new one). This
    replan pattern is exactly the "propose chunk, execute a prefix,
    recertify" loop shortstop.experiment.run_episode already implements
    for the 2D prototype.
  - transport is a websocket (openpi_client.websocket_client_policy.
    WebsocketClientPolicy(host, port)), not plain HTTP.

Open question, not resolved here: does one `infer()` call return diverse
samples across repeated calls with the same observation? No exposed
temperature/seed parameter was found in openpi's docs. pi0.5 is
flow-matching-based, same family as shortstop.diffusion_policy's DDPM/DDIM
sampler -- which gets its diversity purely from fresh noise at the start of
each sampling call, no explicit "K" or temperature knob either. By the same
mechanism, repeated infer() calls to a flow-matching model should plausibly
also differ (fresh noise each call) -- but this is an expectation reasoned
by analogy, not something confirmed against the real server. Verify this
first thing when real testing starts; if calls turn out deterministic,
Propose's whole "K diverse candidates" premise needs a different mechanism
(e.g. varying the prompt phrasing, or checking for an undocumented seed
argument).
"""
import numpy as np


class Pi05PolicyClient:
    """Thin wrapper matching shortstop.policy's `.propose(state) -> list of
    K action chunks` interface, so it drops into the same Propose slot as
    GaussianChunkPolicy/DiffusionChunkPolicy. Requires openpi's client
    package and a running policy server (see docs/LIBERO_SETUP.md) --
    NOT importable or testable in this environment; see
    MockPi05PolicyClient for structural testing without either.
    """

    def __init__(self, host="localhost", port=8000, n_candidates=8):
        try:
            from openpi_client import websocket_client_policy
        except ImportError as e:
            raise ImportError(
                "Pi05PolicyClient needs the openpi_client package (part of the "
                "openpi repo, see docs/LIBERO_SETUP.md) -- not available in this "
                "environment. Use MockPi05PolicyClient for structural testing."
            ) from e
        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.n_candidates = n_candidates

    def propose(self, observation):
        """`observation`: dict with keys "observation/image",
        "observation/wrist_image" (uint8 HxWx3), "observation/state" (7,),
        "prompt" (str) -- see module docstring. Returns a list of
        `n_candidates` (horizon, 7) action chunks, one `infer()` call per
        candidate (see module docstring's open question about whether
        repeated calls are actually diverse).
        """
        return [np.asarray(self._client.infer(observation)["actions"]) for _ in range(self.n_candidates)]


class MockPi05PolicyClient:
    """Structural stand-in for Pi05PolicyClient -- same `.propose()`
    interface and output shape, but returns synthetic action chunks
    instead of calling a real server. For exercising the rest of the
    Stage 7a pipeline (ArmReachOnlyShield/ArmSTLShield/ArmRepairShield)
    without a live LIBERO/openpi session.
    """

    def __init__(self, horizon=8, action_dim=7, n_candidates=8, noise_std=0.02, rng=None):
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_candidates = n_candidates
        self.noise_std = noise_std
        self.rng = rng if rng is not None else np.random.default_rng()

    def propose(self, observation):
        del observation  # unused: this stand-in ignores the actual observation
        return [
            self.rng.normal(0.0, self.noise_std, size=(self.horizon, self.action_dim))
            for _ in range(self.n_candidates)
        ]
