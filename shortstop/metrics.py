import numpy as np


def aggregate(episode_logs):
    """Aggregate per-episode logs into the paper's headline metrics.

    Each log needs 'violated', 'reached', 'shield_activations', 'steps'.
    Optional (present only for shielded runs): 'latencies_ms',
    'rejected_total', 'rejected_truly_unsafe' -- see run_phase1.py.

    Recovery rate (7th metric) is intentionally not computed here: it
    compares plain reject vs. repair, and there is no repair mechanism
    until Phase 4.
    """
    violation_rate = float(np.mean([e["violated"] for e in episode_logs]))
    success_rate = float(np.mean([e["reached"] for e in episode_logs]))
    activation_rate = float(np.mean([e["shield_activations"] / e["steps"] for e in episode_logs]))

    all_latencies = [t for e in episode_logs for t in e.get("latencies_ms", [])]
    latency_ms_median = float(np.median(all_latencies)) if all_latencies else None

    total_rejected = sum(e.get("rejected_total", 0) for e in episode_logs)
    total_truly_unsafe = sum(e.get("rejected_truly_unsafe", 0) for e in episode_logs)
    intervention_precision = (
        total_truly_unsafe / total_rejected if total_rejected > 0 else None
    )

    return {
        "n_episodes": len(episode_logs),
        "violation_rate": violation_rate,
        "success_rate": success_rate,
        "shield_activation_rate": activation_rate,
        "latency_ms_median": latency_ms_median,
        "intervention_precision": intervention_precision,
    }


def conservatism_cost(unshielded_logs, shielded_logs):
    """Success-rate drop the shield causes on episodes with no real unsafe event.

    "Real unsafe event" is read off the paired Unshielded run (same seed, so
    same scenario/noise): an episode is "benign" if the Unshielded rollout
    never violated. Requires unshielded_logs[i] and shielded_logs[i] to come
    from the same seed for every i.
    """
    benign_idx = [i for i, e in enumerate(unshielded_logs) if not e["violated"]]
    if not benign_idx:
        return None
    unshielded_success = np.mean([unshielded_logs[i]["reached"] for i in benign_idx])
    shielded_success = np.mean([shielded_logs[i]["reached"] for i in benign_idx])
    return float(unshielded_success - shielded_success)
