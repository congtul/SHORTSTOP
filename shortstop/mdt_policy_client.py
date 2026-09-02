"""Client for the MDT (Multimodal Diffusion Transformer) policy on CALVIN
(Stage 7b Propose step).

I/O contract confirmed by reading MDT's own model code
(`mdt/models/mdt_agent.py`, `MDTAgent.step`/`.denoise_actions`) and CALVIN's
action-space docs, rather than guessed:
  - observation: a dict with (at least) `rgb_obs["rgb_static"]` (main camera,
    CALVIN default 200x200 RGB) and a `goal` (language-conditioned goal
    embedding) -- same closed-loop-image requirement as LIBERO, see
    docs/CALVIN_SETUP.md.
  - action **chunk**: MDT's own `step(obs, goal)` samples a fresh chunk of
    shape (batch, act_window_size=10, 7) via diffusion denoising every
    `self.multistep` calls, caches it in `self.pred_action_seq`, and returns
    *one* action per call by indexing into that cache -- i.e. MDT's public
    `step()` already does its own "propose chunk, execute a prefix" loop
    internally and never exposes the raw chunk to the caller.
  - the 7 action columns are the same convention as LIBERO/pi0.5: CALVIN's
    "relative cartesian displacement" action space is 3D end-effector
    position delta + 3D orientation delta (Euler, not axis-angle -- but
    shortstop.arm_reach only reads columns 0:3, so this difference doesn't
    matter) + 1D gripper.

Because MDT's `step()` hides the chunk, ShortStop cannot sit where it does
for LIBERO (intercepting the *client's* replan loop) -- instead this client
must call MDT's own chunk-producing sub-step (the model's forward pass that
`step()` calls internally when `rollout_step_counter % multistep == 0`,
returning `pred_action_seq` before MDT's own indexing/caching), then run
ShortStop's own prefix-execution loop externally, exactly like
shortstop.experiment.run_episode already does for the 2D prototype. The
exact public method name for "give me the chunk, don't cache/execute it
yourself" was not confirmed against a runnable checkout (no GPU/CALVIN in
this environment) -- likely `model(obs, goal)` (`__call__`/`forward`) itself,
per the `denoise_actions` code path, but verify this first against a real
checkout before wiring up Pi05-style serving (see docs/CALVIN_SETUP.md).
"""
import numpy as np


class MDTPolicyClient:
    """Thin wrapper matching shortstop.policy's `.propose(state) -> list of
    K action chunks` interface, so it drops into the same Propose slot as
    GaussianChunkPolicy/DiffusionChunkPolicy/Pi05PolicyClient. Requires the
    `mdt_policy` package plus a loaded checkpoint (see
    docs/CALVIN_SETUP.md) -- NOT importable or testable in this environment;
    see MockMDTPolicyClient for structural testing without either.
    """

    def __init__(self, checkpoint_path, device="cuda", n_candidates=8):
        try:
            from mdt.models.mdt_agent import MDTAgent
        except ImportError as e:
            raise ImportError(
                "MDTPolicyClient needs the mdt_policy package "
                "(github.com/intuitive-robots/mdt_policy, see "
                "docs/CALVIN_SETUP.md) -- not available in this environment. "
                "Use MockMDTPolicyClient for structural testing."
            ) from e
        self._model = MDTAgent.load_from_checkpoint(checkpoint_path).to(device)
        self.n_candidates = n_candidates

    def propose(self, observation):
        """`observation`: dict with `rgb_obs["rgb_static"]` (uint8 HxWx3, and
        optionally `rgb_gripper`), `goal` (language-conditioned goal
        embedding) -- see module docstring. Returns a list of `n_candidates`
        (act_window_size, 7) action chunks, one fresh diffusion sample per
        candidate (see module docstring's open question about which model
        method actually returns the raw, un-cached chunk).
        """
        return [
            np.asarray(self._model(observation, observation["goal"]).squeeze(0).detach().cpu())
            for _ in range(self.n_candidates)
        ]


class MockMDTPolicyClient:
    """Structural stand-in for MDTPolicyClient -- same `.propose()`
    interface and output shape ((10, 7) chunks, CALVIN's `act_window_size`
    default), but returns synthetic action chunks instead of calling a real
    model. For exercising the rest of the Stage 7b pipeline
    (ArmReachOnlyShield/ArmSTLShield/ArmRepairShield, reused unmodified from
    Stage 7a since CALVIN shares LIBERO's 7D relative-chunk convention and
    Panda robot) without a live CALVIN/MDT session.
    """

    def __init__(self, horizon=10, action_dim=7, n_candidates=8, noise_std=0.02, rng=None):
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
