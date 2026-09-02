"""Unshielded baseline: violation_rate/success_rate with vs. without the
privileged obstacle, over `cfg.num_sequences` CALVIN sequences.

Run from WSL2, inside the `mdt_env` conda environment set up per
docs/CALVIN_SETUP.md (needs a real GPU + the mdt_policy checkpoint +
debug dataset -- NOT runnable/tested in the dev sandbox this was written
in; treat this as a carefully-reasoned first draft to debug against the
real checkout, not a guaranteed-working script):

    cd SHORTSTOP
    python scripts/run_calvin_unshielded.py

Reuses mdt_evaluate.py's own model/env setup (`get_default_beso_and_env`
+ the same sampler/EMA overrides `main()` applies) rather than
shortstop.mdt_policy_client.MDTPolicyClient -- that class only calls
`MDTVAgent.load_from_checkpoint()` directly and is missing the
sampler_type/num_sampling_steps/sigma/EMA-weight overrides the real eval
script applies afterwards; use it only for structural/mocked testing
(tests/test_mdt_policy_client.py), not for a real run.
"""
import sys
from pathlib import Path

import hydra
import numpy as np
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
MDT_POLICY_ROOT = REPO_ROOT / "mdt_policy"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MDT_POLICY_ROOT))

from mdt.evaluation.multistep_sequences import get_sequences  # noqa: E402
from mdt.evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition  # noqa: E402
from mdt.utils.utils import get_last_checkpoint  # noqa: E402

from shortstop.calvin_experiment import run_calvin_unshielded_sequence  # noqa: E402
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates  # noqa: E402
from shortstop.calvin_obstacle import sample_obstacle_from_reference_chunk  # noqa: E402


class _ForwardOnlyPolicy:
    """Adapter around the loaded MDTVAgent: `.propose(observation)` calls
    `model(obs, goal)` (forward(), not step() -- see
    shortstop/mdt_policy_client.py's docstring for why) `n_candidates`
    times, returning raw numpy chunks."""

    def __init__(self, model, n_candidates=1):
        self.model = model
        self.n_candidates = n_candidates

    def propose(self, observation):
        goal = observation["goal"]
        return [
            np.asarray(self.model(observation, goal).squeeze(0).detach().cpu())
            for _ in range(self.n_candidates)
        ]


@hydra.main(config_path="../mdt_policy/conf", config_name="mdt_evaluate")
def main(cfg):
    seed_everything(0, workers=True)

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    model, env, _, lang_embeddings = get_default_beso_and_env(
        cfg.train_folder, cfg.dataset_path, checkpoint,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite, device_id=cfg.device,
    )
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.multistep = cfg.multistep
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler
    model.eval()

    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    policy = _ForwardOnlyPolicy(model, n_candidates=1)

    eval_sequences = get_sequences(cfg.num_sequences)

    for label, obstacle_fn in [
        ("without obstacle", None),
        ("with obstacle", lambda joint_angles, reference_chunk: sample_obstacle_from_reference_chunk(
            joint_angles, reference_chunk, radius=0.05,
        )),
    ]:
        sequence_results = []
        for initial_state, eval_sequence in eval_sequences:
            attempts = run_calvin_unshielded_sequence(
                env, policy, task_oracle, lang_embeddings, initial_state, eval_sequence, val_annotations,
                get_env_state_for_initial_condition, ep_len=cfg.ep_len, replan_steps=cfg.multistep,
                obstacle_fn=obstacle_fn,
            )
            sequence_results.append(attempts)

        slots = build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5)
        violation_rate, success_rate = fixed_cohort_rates(slots)
        print(f"[{label}] violation_rate={violation_rate:.3f}  success_rate={success_rate:.3f}"
              f"  (avg_seq_len={success_rate * 5:.2f}/5, n_sequences={cfg.num_sequences})")


if __name__ == "__main__":
    main()
