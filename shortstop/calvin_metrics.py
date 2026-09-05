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


def recovery_rate(sequence_results):
    """Fraction of subtask attempts where the shield activated
    (`n_activated > 0` -- at least one decision rejected something) that
    still ended in `reached=True`. The paper's own definition ("fraction
    of rejected-chunk situations from which the task is still completed,
    isolating the value of repair vs bare rejection") -- deliberately
    GENERAL, not gated on a repair mechanism existing, unlike
    shortstop.metrics.aggregate's narrower `repair_successes/
    repair_attempts` ratio (which stays `None` for any shield whose
    select() doesn't report those specific keys, e.g. ArmSTLMonitorShield/
    ArmConfThreshShield -- yet the paper reports a real recovery_rate for
    both of those in Table II, so that narrower definition can't be what's
    meant there). `sequence_results`: list of per-sequence subtask-attempt
    lists, each dict needs 'n_activated'/'reached' (see
    shortstop.calvin_experiment.run_calvin_shielded_subtask's result).
    `None` if no subtask attempt ever activated the shield at all (nothing
    to compute a rate over)."""
    activated = [
        a for attempts in sequence_results for a in attempts
        if a.get("n_activated", 0) > 0
    ]
    if not activated:
        return None
    return float(np.mean([a["reached"] for a in activated]))


def conservatism_cost(unshielded_sequence_results, shielded_sequence_results):
    """Success-rate drop the shield causes on subtask attempts where the
    PAIRED unshielded run had no true violation -- mirrors
    shortstop.metrics.conservatism_cost's definition (see its docstring)
    at CALVIN's subtask-attempt granularity. Requires
    unshielded_sequence_results[i] and shielded_sequence_results[i] to
    come from the SAME sequence (same seed/cohort index) for every i --
    true for any 2 CALVIN scripts sharing the tuning/eval cohort
    convention (see docs/TUNING_WORKFLOW.md muc 0). A sequence can stop
    at a different subtask in each run (early failure/violation) -- only
    subtask positions BOTH runs actually attempted are compared; extra
    trailing attempts on either side are simply not part of the pairing.
    `None` if no paired, benign subtask attempt exists at all."""
    benign_pairs = []
    for seq_idx, unshielded_attempts in enumerate(unshielded_sequence_results):
        shielded_attempts = shielded_sequence_results[seq_idx]
        for subtask_idx, u in enumerate(unshielded_attempts):
            if subtask_idx >= len(shielded_attempts) or u["violated"]:
                continue
            benign_pairs.append((u, shielded_attempts[subtask_idx]))
    if not benign_pairs:
        return None
    unshielded_success = np.mean([u["reached"] for u, _ in benign_pairs])
    shielded_success = np.mean([s["reached"] for _, s in benign_pairs])
    return float(unshielded_success - shielded_success)


def bootstrap_ci(sequence_results, statistic_fn, n_resamples=10_000, seed=0):
    """Paper's own bootstrap recipe (docs/PARAMETERS_REFERENCE.md's
    "num_sequences" entry: "bootstrap 10^4 resamples") -- estimates the
    sampling variability of any metric from a SINGLE existing run's saved
    `sequence_results` (e.g. loaded back from a `results.json` already on
    disk), no extra CALVIN env/policy execution needed.

    Resamples at the SEQUENCE level (draws `len(sequence_results)`
    sequences WITH replacement, `n_resamples` times), not per-subtask-
    attempt -- subtasks within one sequence are correlated (CALVIN's own
    chained structure: subtask 2 only runs if subtask 1 succeeded, see
    build_fixed_cohort_slots's own docstring), so resampling individual
    attempts would break that correlation and understate the true
    variance.

    `statistic_fn(sequence_results_subset) -> float`: computes the metric
    of interest on a resampled list of per-sequence attempt-lists, e.g.
    `lambda seqs: fixed_cohort_rates(build_fixed_cohort_slots(seqs))[0]`
    for violation_rate, or `[1]` for success_rate. Any of this module's
    other metrics (recovery_rate, conservatism_cost -- the latter needs a
    closure capturing the PAIRED unshielded sequence_results, resampled
    with the SAME drawn indices) can be plugged in the same way.

    Returns `(mean, std, values)`: `mean`/`std` are the ones to report as
    "mean +/- std" (matching the paper's own Table II style); `values` is
    the full length-`n_resamples` array, e.g. for a percentile CI
    (`np.percentile(values, [2.5, 97.5])`) instead of a symmetric std.

    This is NOT a substitute for the paper's own 5-seed protocol (running
    the real CALVIN env/policy 5 times with different seeds) -- it only
    quantifies the sampling uncertainty of ONE already-collected run, not
    the additional variability a different seed's checkpoint/environment
    draw would show. Use both together, not one instead of the other."""
    rng = np.random.default_rng(seed)
    n = len(sequence_results)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled = [sequence_results[j] for j in idx]
        values[i] = statistic_fn(resampled)
    return float(values.mean()), float(values.std()), values
