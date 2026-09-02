"""Client for a Mini Diffuser checkpoint (Stage 7c Propose step, RLBench).

I/O contract as best established from research (Mini Diffuser's own repo
README does not document the interface in detail -- see
docs/STAGE7C_ARM_PIPELINE_DESIGN.md's "confirm before real use" note),
following the PerAct/RLBench keyframe-policy convention every paper on this
benchmark (PerAct, RVT, Mini Diffuser included) evaluates against:
  - observation: multi-view RGB-D (RLBench's 4 fixed cameras -- front, left
    shoulder, right shoulder, wrist -- 128x128 per the RLBench paper),
    proprioception (joint positions/velocities/torques, 6D end-effector
    pose), and a language goal string.
  - output: one *keypose* -- 8D vector (3D translation + 4D quaternion +
    1D binary gripper), NOT a dense per-step action chunk. RLBench's own
    sampling-based motion planner is what actually moves the arm from its
    current pose to that keypose (an "observe -> predict one keypose ->
    motion-plan -> execute -> observe" loop, extracting training keyframes
    where joint velocity is near zero and the gripper state changes --
    PerAct's convention, which Mini Diffuser follows).

This is a materially different granularity than shortstop.pi_policy_client
(pi0.5/LIBERO's dense chunk) -- see shortstop/keypose_reach.py and
shortstop/keypose_shield.py for why Reach/Certify/Repair had to be
redesigned around it rather than reused as-is.
"""
import numpy as np


class MiniDiffuserClient:
    """Real client: NOT implemented here. Mini Diffuser's own repo
    (github.com/utomm/mini-diffuse-actor) does not document a served-model
    /network interface the way openpi does (see module docstring) -- it
    looks like a training+eval codebase you run in-process against RLBench,
    not a client/server split. Confirm this against the actual repo before
    building a client wrapper; a stub that guesses a wire format would be
    worse than no code at all here. Use MockMiniDiffuserClient below for
    structural testing until that's confirmed.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MiniDiffuserClient needs a confirmed serving interface for the real "
            "checkpoint -- see this class's docstring. Use MockMiniDiffuserClient "
            "for structural testing."
        )


class MockMiniDiffuserClient:
    """Structural stand-in: same `.propose(observation) -> list of K
    keyposes` interface, synthetic 8D keyposes (3D position + 4D
    quaternion (not normalized -- fine for shape/structure testing, a real
    client's output would be) + 1D gripper) instead of a real inference
    call.
    """

    def __init__(self, n_candidates=8, position_noise_std=0.05, rng=None):
        self.n_candidates = n_candidates
        self.position_noise_std = position_noise_std
        self.rng = rng if rng is not None else np.random.default_rng()

    def propose(self, observation):
        del observation
        candidates = []
        for _ in range(self.n_candidates):
            position = self.rng.normal(0.0, self.position_noise_std, size=3)
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # identity, placeholder
            gripper = np.array([1.0])  # open, placeholder
            candidates.append(np.concatenate([position, quaternion, gripper]))
        return candidates
