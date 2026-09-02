from shortstop.calvin_metrics import attempted_only, build_fixed_cohort_slots, fixed_cohort_rates


def _ok(n):
    return [{"violated": False, "reached": True} for _ in range(n)]


def test_build_fixed_cohort_slots_pads_truncated_sequences_with_untried_placeholders():
    sequence_results = [_ok(5), _ok(3), []]  # full success, fail at #4, fail at #1
    slots = build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5)

    assert len(slots) == 15  # 3 sequences * 5, fixed regardless of where each stopped
    assert sum(s["attempted"] for s in slots) == 5 + 3 + 0
    # untried slots (beyond where a sequence stopped) count as neither reached nor violated
    untried = [s for s in slots if not s["attempted"]]
    assert len(untried) == 15 - 8
    assert all(not s["reached"] and not s["violated"] for s in untried)


def test_build_fixed_cohort_slots_rejects_more_attempts_than_the_cap():
    import pytest
    with pytest.raises(ValueError):
        build_fixed_cohort_slots([_ok(6)], subtasks_per_sequence=5)


def test_success_rate_equals_avg_seq_len_over_five():
    # sequences completing 5, 4, 3 subtasks -> avg_seq_len = 4.0 -> success_rate = 4.0/5
    sequence_results = [_ok(5), _ok(4) + [{"violated": False, "reached": False}], _ok(3) + [{"violated": False, "reached": False}] * 2]
    slots = build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5)
    _, success_rate = fixed_cohort_rates(slots)
    assert abs(success_rate - 4.0 / 5) < 1e-9


def test_obstacle_present_raises_violation_rate_and_lowers_success_rate_vs_absent():
    # Same underlying policy behavior; "without obstacle" -> nothing ever
    # flagged, subtasks run to natural completion. "With obstacle" -> the
    # 2nd subtask of 2 of the 3 sequences gets flagged mid-way, truncating
    # that sequence right there (violated=True implies reached=False for
    # that attempt, and every later subtask in that sequence is untried).
    without_obstacle = [_ok(5), _ok(5), _ok(5)]

    with_obstacle = [
        _ok(5),
        [{"violated": False, "reached": True}, {"violated": True, "reached": False}],
        [{"violated": False, "reached": True}, {"violated": True, "reached": False}],
    ]

    slots_without = build_fixed_cohort_slots(without_obstacle)
    slots_with = build_fixed_cohort_slots(with_obstacle)

    violation_without, success_without = fixed_cohort_rates(slots_without)
    violation_with, success_with = fixed_cohort_rates(slots_with)

    assert violation_without == 0.0
    assert success_without == 1.0
    assert violation_with > violation_without
    assert success_with < success_without


def test_attempted_only_drops_untried_slots():
    slots = build_fixed_cohort_slots([_ok(2)], subtasks_per_sequence=5)
    assert len(attempted_only(slots)) == 2
