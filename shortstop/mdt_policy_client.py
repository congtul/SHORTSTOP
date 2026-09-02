"""Client for the MDT (Multimodal Diffusion Transformer) policy on CALVIN
(Stage 7b Propose step).

I/O contract confirmed by reading the real `mdt_policy` checkout's model
code (`mdt/models/mdtv_agent.py`, class `MDTVAgent` -- the "V" variant
behind the 6 public `mdtv-*` checkpoints, not the older `MDTAgent` in
`mdt_agent.py`) and CALVIN's action-space docs, not guessed:
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
calls `MDTVAgent.forward(obs, goal)` directly (i.e. `self._model(obs, goal)`)
instead of `.step()`. **Confirmed against the real checkout**:
`forward()` (`mdtv_agent.py:688`) calls `denoise_actions()` and returns its
`act_seq` straight back to the caller -- no caching, no indexing, exactly
one fresh (batch, act_window_size=10, 7) chunk per call, matching `step()`'s
own internal `pred_action_seq = self(obs, goal)` line. ShortStop then runs
its own prefix-execution loop externally on that chunk, exactly like
shortstop.experiment.run_episode already does for the 2D prototype -- this
*replaces* `step()`'s internal `multistep` counter loop, the two are not
used together.

Diversity across repeated calls (needed for Propose's "K candidates") is
also confirmed, not assumed: `denoise_actions()` draws
`x = torch.randn((len(latent_goal), self.act_window_size, 7), ...) *
self.sigma_max` fresh on every call, with no seed reset in between --
so K calls to `propose()` genuinely sample K different chunks. This holds
for every `sampler_type` in `sample_loop()`, including the ones commented
"ODE deterministic" (`ddim`/`euler`/`dpm`/`lms`/...) -- those are only
deterministic *given* a fixed starting noise `x`, and `x` itself is
resampled every call; the "SDE stochastic" samplers (`ancestral`,
`euler_ancestral`) add even more per-step noise on top of that.
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
            from mdt.models.mdtv_agent import MDTVAgent
        except ImportError as e:
            raise ImportError(
                "MDTPolicyClient needs the mdt_policy package "
                "(github.com/intuitive-robots/mdt_policy, see "
                "docs/CALVIN_SETUP.md) -- not available in this environment. "
                "Use MockMDTPolicyClient for structural testing."
            ) from e
        self._model = MDTVAgent.load_from_checkpoint(checkpoint_path).to(device)
        self.n_candidates = n_candidates

    def propose(self, observation):
        """`observation`: dict with `rgb_obs["rgb_static"]` (uint8 HxWx3, and
        optionally `rgb_gripper`), `goal` (language-conditioned goal
        embedding) -- see module docstring. Returns a list of `n_candidates`
        (act_window_size, 7) action chunks -- calls `self._model(observation,
        goal)` (`MDTVAgent.forward`, confirmed to return the raw chunk, see
        module docstring) once per candidate, each a genuinely fresh
        diffusion sample (confirmed: fresh `torch.randn` noise per call).
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
