from shortstop.metrics import aggregate, conservatism_cost


def _log(violated, reached, activations, steps, latencies_ms=None, **extra):
    return {
        "violated": violated,
        "reached": reached,
        "shield_activations": activations,
        "steps": steps,
        "latencies_ms": latencies_ms or [],
        **extra,
    }


def test_latency_mean_and_p95_expose_a_heavy_tail_that_median_hides():
    """9 cheap steps (1ms) + 1 expensive step (100ms): median stays cheap,
    mean/p95 must show the tail median alone would hide -- the exact
    scenario that made the old Stage3-vs-Stage4 latency comparison
    misleading (see shortstop/shield.py's RepairShield docstring).
    """
    logs = [_log(False, True, 0, 1, latencies_ms=[1.0] * 9 + [100.0])]
    m = aggregate(logs)

    assert m["latency_ms_median"] == 1.0
    assert m["latency_ms_mean"] > 10.0
    assert m["latency_ms_p95"] > 50.0


def test_conservatism_cost_only_counts_benign_episodes():
    """Episode 0 is benign under Unshielded (never violated) -- the shield's
    failure to reach goal there is a real conservatism cost. Episode 1 is
    NOT benign (unshielded violated), so its outcome must not count.
    """
    unshielded_logs = [
        _log(violated=False, reached=True, activations=0, steps=5),
        _log(violated=True, reached=False, activations=0, steps=5),
    ]
    shielded_logs = [
        _log(violated=False, reached=False, activations=1, steps=5),
        _log(violated=False, reached=True, activations=1, steps=5),
    ]

    cost = conservatism_cost(unshielded_logs, shielded_logs)

    assert cost == 1.0  # 100% success -> 0% success, only episode 0 counted
