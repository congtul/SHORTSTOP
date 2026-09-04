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

`ForwardOnlyPolicy` below batches this same draw (`len(latent_goal)`
becomes `n_candidates` instead of 1, in one forward call) rather than
looping `n_candidates` calls of batch size 1 -- see its own docstring.
Same source of diversity either way; batching only parallelizes it.
"""
import numpy as np
import torch


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


class ForwardOnlyPolicy:
    """Wraps an already-configured `MDTVAgent` (loaded + overridden by the
    caller -- see scripts/run_calvin_unshielded.py / run_calvin_shielded.py's
    own main(): `get_default_beso_and_env` then sampler_type/
    num_sampling_steps/sigma/EMA-weight overrides applied) and draws all
    `n_candidates` diffusion samples in ONE batched forward call, instead of
    looping `model(observation, goal)` `n_candidates` times the way
    MDTPolicyClient.propose() above does.

    That loop is genuinely wasteful for a real run: every iteration re-runs
    the vision encoder (`compute_voltron_embeddings`) on the *same*
    rgb_static/rgb_gripper, and `MDTVAgent.forward()`'s own preprocessing
    (mdtv_agent.py:688) has no batching path at all, so nothing amortizes
    across candidates.

    Fix: replicate `forward()`'s own preprocessing verbatim for the single
    real observation (batch=1 -- `language_goal`/`compute_voltron_
    embeddings`, unchanged from `forward()`'s own code), then
    `.expand(n_candidates, ...).contiguous()` the resulting conditioning
    tensors (`latent_goal`, `perceptual_emb`) before calling
    `denoise_actions()` once. `denoise_actions()` draws `x = torch.randn((
    len(latent_goal), act_window_size, 7))` internally (mdtv_agent.py:546)
    -- with `latent_goal`'s batch dim now `n_candidates`, this draws
    `n_candidates` independent noise seeds and denoises all of them in one
    batched pass, so every returned candidate is still a genuinely
    independent sample (see module docstring's "Diversity across repeated
    calls" note -- batching only parallelizes the same mechanism, it does
    not change what makes candidates differ).

    Deliberately does NOT call `model(observation, goal)` / `model.
    forward()` for the batched path: `forward()`'s own lang-goal branch
    does `self.language_goal(goal["lang"]).unsqueeze(0)`, which *hardcodes*
    a batch-of-1 result -- feeding it an already-batched goal would
    silently produce the wrong shape. Bypassed by computing latent_goal/
    perceptual_emb for the one real observation first (batch=1, exactly as
    forward() does) and expanding *after* -- never re-deriving
    `language_goal`'s own batching behavior, which has not been verified.

    Only supports `goal={"lang": ...}` (CALVIN's own conditioning, per
    module docstring) -- raises NotImplementedError for the visual-goal
    branch (`forward()`'s `else` case), whose `rgb_static.squeeze(0))`
    handling was not verified to generalize to batch > 1 and is not
    exercised by this project's actual checkpoint/harness anyway.
    """

    def __init__(self, model, n_candidates=1):
        self.model = model
        self.n_candidates = n_candidates

    def propose(self, observation):
        goal = observation["goal"]
        model = self.model
        k = self.n_candidates

        if "lang" not in goal:
            raise NotImplementedError(
                "ForwardOnlyPolicy's batched propose only supports "
                "goal={'lang': ...} -- see class docstring."
            )

        rgb_static = observation["rgb_obs"]["rgb_static"]
        rgb_gripper = observation["rgb_obs"]["rgb_gripper"]

        # Verbatim copy of MDTVAgent.forward()'s own lang-goal preprocessing
        # (mdtv_agent.py:692-698) for the single real observation (batch=1).
        if model.use_text_not_embedding:
            latent_goal = model.language_goal(goal["lang_text"])
            latent_goal = latent_goal.to(torch.float32)
        else:
            latent_goal = model.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(rgb_static.device)

        perceptual_emb = model.compute_voltron_embeddings(rgb_static, rgb_gripper)
        perceptual_emb["modality"] = "lang"

        # batch=1 -> batch=k. .contiguous() (not just .expand()) so nothing
        # downstream trips over a stride-0 view if it calls .view() instead
        # of .reshape() on these tensors.
        latent_goal_k = latent_goal.expand(k, *latent_goal.shape[1:]).contiguous()
        perceptual_emb_k = {
            key: (value.expand(k, *value.shape[1:]).contiguous() if torch.is_tensor(value) else value)
            for key, value in perceptual_emb.items()
        }
        latent_plan_k = torch.zeros_like(latent_goal_k)

        action_seq = model.denoise_actions(latent_plan_k, perceptual_emb_k, latent_goal_k, inference=True)
        return [np.asarray(action_seq[i].detach().cpu()) for i in range(k)]


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
