"""Fixed-cohort violation/success rate for CALVIN's chained-sequence eval.

CALVIN's own `evaluate_sequence()` (mdt_policy/mdt/evaluation/mdt_evaluate.py)
stops a sequence at the first failed subtask -- so a naive per-subtask
average computed only over "subtasks that actually got attempted" has a
baseline-dependent denominator: a baseline that fails early contributes
fewer, early-subtask-skewed attempts than one that survives longer. That
is not a fair comparison across shield arms/baselines (same concern raised
in conversation -- a baseline failing mostly at subtask 3 vs one failing
mostly at subtask 4 would otherwise be scored on different, differently-
composed samples).

CALVIN's own literature avoids exactly this by always normalizing by the
FIXED number of *launched* sequences (the "1 | 2 | 3 | 4 | 5 | Avg. Len."
table in any CALVIN paper) -- a sequence that fails early contributes 0 to
every later position; it is never dropped from the denominator. This
module applies the same fixed-cohort convention at subtask-slot
granularity: `n_sequences * subtasks_per_sequence` fixed slots, so
`success_rate` computed this way equals `avg_seq_len / subtasks_per_sequence`
exactly (verified in tests/test_calvin_metrics.py).
"""
import numpy as np


def build_fixed_cohort_slots(sequence_results, subtasks_per_sequence=5):
    """`sequence_results`: list of per-sequence subtask-attempt lists, one
    list per launched sequence, each containing 0..subtasks_per_sequence
    dicts with 'violated'/'reached' -- one per subtask actually attempted,
    in order, before that sequence stopped (success all the way through,
    or a failure/violation that ends the chain early).

    Returns a flat list of exactly `len(sequence_results) *
    subtasks_per_sequence` dicts: attempted subtasks keep their real
    'violated'/'reached'/'attempted': True; slots beyond where a sequence
    stopped are filled with 'violated': False, 'reached': False,
    'attempted': False (never run, so neither could have happened).
    """
    slots = []
    for attempts in sequence_results:
        if len(attempts) > subtasks_per_sequence:
            raise ValueError(f"got {len(attempts)} attempts, expected at most {subtasks_per_sequence}")
        for a in attempts:
            slots.append({"violated": bool(a["violated"]), "reached": bool(a["reached"]), "attempted": True})
        for _ in range(subtasks_per_sequence - len(attempts)):
            slots.append({"violated": False, "reached": False, "attempted": False})
    return slots


def fixed_cohort_rates(slots):
    """(violation_rate, success_rate) over ALL slots -- fixed denominator
    (len(slots)), untried slots count as 0 for both, per
    build_fixed_cohort_slots. success_rate here equals avg_seq_len /
    subtasks_per_sequence -- same information as CALVIN's own metric,
    expressed as a fraction so it lines up with the other 6 ShortStop
    metrics (all fractions/rates).
    """
    violation_rate = float(np.mean([s["violated"] for s in slots]))
    success_rate = float(np.mean([s["reached"] for s in slots]))
    return violation_rate, success_rate


def attempted_only(slots):
    """The subset of slots that were actually run. Feed this into
    shortstop.metrics.aggregate() for the *conditional* metrics (latency,
    shield-activation rate, ...) that have no sensible value for a slot
    that never ran -- unlike violation/success, those are not cohort-level
    incidence rates, so they should not use the fixed-500 denominator.
    """
    return [s for s in slots if s["attempted"]]
